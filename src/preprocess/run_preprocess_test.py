from __future__ import annotations

import argparse
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.utils import load_config, set_seed, setup_logger
from src.preprocess.clip_cache import CLIPEncoder
from src.preprocess.faster_rcnn_extractor import FasterRCNNExtractor
from src.preprocess.ner_extractor import (
    KeywordExtractor,
    NERExtractor,
    SemanticUnitExtractor,
)

def get_image_key(image_path: str) -> str:
    return hashlib.md5(image_path.encode()).hexdigest()[:16]

def extract_image_filename(image_path: str) -> str:
    return Path(image_path).name

def find_image_path(image_dir: Path, image_path: str) -> Path:
    full_path = image_dir / image_path
    if full_path.exists():
        return full_path

    filename = extract_image_filename(image_path)
    full_path = image_dir / filename
    if full_path.exists():
        return full_path

    full_path = image_dir / "images" / filename
    if full_path.exists():
        return full_path

    full_path = image_dir / "images" / image_path
    if full_path.exists():
        return full_path

    return None

def process_unique_images(
    image_paths: List[str],
    image_dir: Path,
    clip: CLIPEncoder,
    frcnn: FasterRCNNExtractor,
    ocr,
    cfg,
    image_cache_dir: Path,
    logger,
) -> Dict[str, str]:
    image_cache_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = image_cache_dir / "features"
    meta_dir = image_cache_dir / "meta"
    feat_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cached_keys = set()
    for npz_file in feat_dir.glob("*.npz"):
        cached_keys.add(npz_file.stem)

    path_to_key = {}
    pending_images = []
    pending_full_paths = []

    for img_path in image_paths:
        key = get_image_key(img_path)
        path_to_key[img_path] = key

        npz_ok = (feat_dir / f"{key}.npz").exists()
        meta_ok = (meta_dir / f"{key}.json").exists()

        if npz_ok and meta_ok:
            continue

        full_path = find_image_path(image_dir, img_path)
        if full_path is None:
            logger.warning(f"图片不存在: {img_path}")
            continue

        pending_images.append(img_path)
        pending_full_paths.append(full_path)

    if not pending_images:
        logger.info(f"所有 {len(image_paths)} 张图片已缓存或不存在")
        return path_to_key

    logger.info(f"开始处理 {len(pending_images)} 张未缓存的图片（共 {len(image_paths)} 张唯一图片）")

    fg = cfg.fine_grained
    d_c = clip.d_c

    pbar = tqdm(total=len(pending_images), desc="Processing images", ncols=100)

    for img_path, full_path in zip(pending_images, pending_full_paths):
        try:
            image = Image.open(full_path).convert("RGB")
            key = get_image_key(img_path)

            h_v_global = clip.encode_image([image])[0]

            boxes, scores = frcnn.detect(image)
            crops = frcnn.crop_regions(image, boxes)
            region_feats = clip.encode_image(crops)
            boxes = boxes[: region_feats.shape[0]]
            scores = scores[: region_feats.shape[0]]

            ocr_blocks = ocr.extract(str(full_path))
            ocr_text = ocr.concat_text(ocr_blocks)

            np.savez_compressed(
                feat_dir / f"{key}.npz",
                h_v_global=h_v_global.astype(np.float32),
                region_feats=region_feats.astype(np.float32),
                region_boxes=boxes.astype(np.float32),
                region_scores=scores.astype(np.float32),
            )

            meta = {
                "image_path": img_path,
                "ocr_text": ocr_text,
                "ocr_blocks": ocr_blocks,
                "region_boxes": boxes.tolist(),
                "region_scores": scores.tolist(),
                "n_regions": int(region_feats.shape[0]),
                "has_ocr": len(ocr_blocks) > 0,
            }
            with open(meta_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"图片处理失败 {img_path}: {e}")

        pbar.update(1)

    pbar.close()
    logger.info(f"图片特征缓存完成")
    return path_to_key

def process_samples(
    items: List[Dict],
    image_dir: Path,
    clip: CLIPEncoder,
    units: SemanticUnitExtractor,
    cfg,
    output_cache: Path,
    image_cache_dir: Path,
    path_to_key: Dict[str, str],
    logger,
):
    feat_dir = output_cache / "features"
    meta_dir = output_cache / "meta"
    feat_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    fg = cfg.fine_grained
    d_c = clip.d_c
    done = 0
    failed = []

    pbar = tqdm(total=len(items), desc="Processing samples", ncols=100)

    for idx, item in enumerate(items):
        sample_id = str(idx)
        try:
            image_path = item["image_path"]
            caption = item.get("caption", "") or ""
            label = int(item.get("label", 0))

            if image_path not in path_to_key:
                failed.append(sample_id)
                pbar.update(1)
                continue

            image_key = path_to_key[image_path]
            img_feat_path = image_cache_dir / "features" / f"{image_key}.npz"
            img_meta_path = image_cache_dir / "meta" / f"{image_key}.json"

            if not img_feat_path.exists():
                failed.append(sample_id)
                pbar.update(1)
                continue

            img_feat = np.load(img_feat_path)
            with open(img_meta_path, "r", encoding="utf-8") as f:
                img_meta = json.load(f)

            h_v_global = img_feat["h_v_global"]
            region_feats = img_feat["region_feats"]
            region_boxes = img_feat["region_boxes"]
            region_scores = img_feat["region_scores"]
            ocr_text = img_meta["ocr_text"]
            ocr_blocks = img_meta["ocr_blocks"]

            h_t_global = clip.encode_text([caption])[0]
            text_entities = units.extract(caption, max_units=fg.N_t, tfidf_topk=fg.tfidf_topk)
            text_entity_feats = clip.encode_text(text_entities)
            ocr_entities = units.extract(ocr_text, max_units=fg.N_r, tfidf_topk=fg.tfidf_topk)
            ocr_entity_feats = clip.encode_text(ocr_entities)

            n_t = int(text_entity_feats.shape[0])
            n_o = int(region_feats.shape[0])
            n_r = int(ocr_entity_feats.shape[0])
            n_v = n_o + n_r
            has_local = (n_t > 0) and (n_v > 0)

            np.savez_compressed(
                feat_dir / f"{sample_id}.npz",
                h_t_global=h_t_global.astype(np.float32),
                h_v_global=h_v_global.astype(np.float32),
                text_entity_feats=text_entity_feats.astype(np.float32),
                region_feats=region_feats.astype(np.float32),
                region_boxes=region_boxes.astype(np.float32),
                region_scores=region_scores.astype(np.float32),
                ocr_entity_feats=ocr_entity_feats.astype(np.float32),
                label=np.int64(label),
            )

            meta = {
                "post_id": item.get("post_id", sample_id),
                "image_path": image_path,
                "caption": caption,
                "label": label,
                "split": "test",
                "d_c": d_c,
                "text_entities": text_entities,
                "ocr_entities": ocr_entities,
                "ocr_blocks": ocr_blocks,
                "region_boxes": region_boxes.tolist(),
                "region_scores": region_scores.tolist(),
                "shape": {"N_t": n_t, "N_o": n_o, "N_r": n_r, "N_v": n_v},
                "flags": {
                    "has_text": n_t > 0,
                    "has_region": n_o > 0,
                    "has_ocr": len(ocr_blocks) > 0,
                    "has_ocr_entity": n_r > 0,
                    "has_local": has_local,
                },
            }
            with open(meta_dir / f"{sample_id}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            done += 1

        except Exception as e:
            failed.append(sample_id)
            logger.warning(f"样本 {sample_id} 处理失败: {e}")

        pbar.update(1)

    pbar.close()

    if failed:
        with open(output_cache / "failed.json", "w", encoding="utf-8") as f:
            json.dump({"failed": failed}, f, ensure_ascii=False, indent=2)

    logger.info(f"样本处理完成: done={done}, failed={len(failed)}")

def main():
    parser = argparse.ArgumentParser(description="预处理新测试集（图片去重优化版）")
    parser.add_argument("--test_json", required=True, help="测试集JSON文件路径")
    parser.add_argument("--image_dir", required=True, help="测试集图片目录路径")
    parser.add_argument("--output_cache", required=True, help="输出缓存目录")
    parser.add_argument("--config", default="config.yaml", help="配置文件")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.project.seed)

    device = torch.device(
        "cuda" if (cfg.project.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    with open(args.test_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    if isinstance(items, list):
        print(f"测试集样本数: {len(items)} (列表格式)")
    else:
        print(f"测试集样本数: {len(items)} (字典格式)")

    if isinstance(items, list):
        all_image_paths = list(set(item["image_path"] for item in items))
    else:
        all_image_paths = list(set(item["image_path"] for item in items.values()))

    print(f"唯一图片数: {len(all_image_paths)}")

    clip = CLIPEncoder(cfg.clip_path("Twitter"), "en", device)
    print(f"CLIP 加载完成")

    frcnn = FasterRCNNExtractor(
        cfg.faster_rcnn_path(), device,
        top_n=cfg.preprocess.faster_rcnn.top_n,
        score_thresh=cfg.preprocess.faster_rcnn.score_thresh,
        min_box_size=cfg.preprocess.faster_rcnn.min_box_size,
    )

    from src.preprocess.ocr_extractor import OCRExtractor
    ocr_dirs = cfg.paddleocr_paths("Twitter")
    ocr = OCRExtractor(
        det_model_dir=ocr_dirs["det_model_dir"],
        rec_model_dir=ocr_dirs["rec_model_dir"],
        cls_model_dir=ocr_dirs["cls_model_dir"],
        language="en",
        use_gpu=bool(cfg.preprocess.ocr.use_gpu) and device.type == "cuda",
        drop_score=cfg.preprocess.ocr.drop_score,
        max_text_len=cfg.preprocess.ocr.max_text_len,
    )

    ner = NERExtractor(cfg.ner_path("Twitter"), "en", device)
    if isinstance(items, list):
        corpus = [item.get("caption", "") or "" for item in items]
    else:
        corpus = [item.get("caption", "") or "" for item in items.values()]
    keyword = KeywordExtractor("en", corpus)
    units = SemanticUnitExtractor(ner, keyword)

    output_cache = Path(args.output_cache)
    image_cache_dir = output_cache / "image_cache"

    class SimpleLogger:
        def info(self, msg):
            print(f"INFO: {msg}")
        def warning(self, msg):
            print(f"WARNING: {msg}")

    logger = SimpleLogger()

    path_to_key = process_unique_images(
        all_image_paths,
        Path(args.image_dir),
        clip, frcnn, ocr,
        cfg, image_cache_dir,
        logger,
    )

    process_samples(
        items,
        Path(args.image_dir),
        clip, units,
        cfg, output_cache,
        image_cache_dir,
        path_to_key,
        logger,
    )

    print(f"测试集预处理完成，缓存目录: {output_cache}")

if __name__ == "__main__":
    main()