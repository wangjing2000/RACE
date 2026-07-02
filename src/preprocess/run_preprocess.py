from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.utils import load_config, set_seed, setup_logger, get_run_log_path
from src.preprocess.clip_cache import CLIPEncoder
from src.preprocess.faster_rcnn_extractor import FasterRCNNExtractor
from src.preprocess.ner_extractor import (
    KeywordExtractor,
    NERExtractor,
    SemanticUnitExtractor,
)


def build_split(
    ids: List[str],
    labels: List[int],
    ratios: List[float],
    seed: int,
    stratified: bool,
) -> Dict[str, str]:
    from sklearn.model_selection import train_test_split

    train_r, val_r, test_r = ratios
    strat = labels if stratified else None

    ids_train, ids_tmp, y_train, y_tmp = train_test_split(
        ids, labels,
        test_size=(val_r + test_r),
        random_state=seed,
        stratify=strat,
        shuffle=True,
    )
    rel_test = test_r / (val_r + test_r)
    strat_tmp = y_tmp if stratified else None
    ids_val, ids_test = train_test_split(
        ids_tmp,
        test_size=rel_test,
        random_state=seed,
        stratify=strat_tmp,
        shuffle=True,
    )

    mapping = {}
    for i in ids_train:
        mapping[i] = "train"
    for i in ids_val:
        mapping[i] = "val"
    for i in ids_test:
        mapping[i] = "test"
    return mapping

def load_or_build_split(cfg, dataset, items, logger, force=False) -> Dict[str, str]:
    split_path = cfg.cache_dir(dataset) / "split.json"
    if split_path.exists() and not force:
        with open(split_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        logger.info(f"加载已有切分: {split_path} (共 {len(mapping)} 条)")
        return mapping

    ids = list(items.keys())
    labels = [int(items[i]["label"]) for i in ids]
    mapping = build_split(
        ids, labels,
        ratios=list(cfg.split.ratios),
        seed=int(cfg.split.seed),
        stratified=bool(cfg.split.stratified),
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    n_tr = sum(v == "train" for v in mapping.values())
    n_va = sum(v == "val" for v in mapping.values())
    n_te = sum(v == "test" for v in mapping.values())
    logger.info(f"生成切分并落盘: train={n_tr} val={n_va} test={n_te} -> {split_path}")
    return mapping


def process_sample(
    sample_id: str,
    item: Dict,
    split_label: str,
    tools: Dict,
    cfg,
    feat_dir: Path,
    meta_dir: Path,
) -> None:
    dataset_dir = cfg.dataset_dir(tools["dataset"])
    image_path = dataset_dir / item["image_path"]
    caption = item.get("caption", "") or ""
    label = int(item["label"])

    image = Image.open(image_path).convert("RGB")

    clip: CLIPEncoder = tools["clip"]
    frcnn: FasterRCNNExtractor = tools["frcnn"]
    ocr = tools["ocr"]
    units: SemanticUnitExtractor = tools["units"]
    fg = cfg.fine_grained

    h_v_global = clip.encode_image([image])[0]
    h_t_global = clip.encode_text([caption])[0]

    text_entities = units.extract(caption, max_units=fg.N_t, tfidf_topk=fg.tfidf_topk)
    text_entity_feats = clip.encode_text(text_entities)

    boxes, scores = frcnn.detect(image)
    crops = frcnn.crop_regions(image, boxes)
    region_feats = clip.encode_image(crops)

    boxes = boxes[: region_feats.shape[0]]
    scores = scores[: region_feats.shape[0]]

    ocr_blocks = ocr.extract(image_path)
    ocr_text = ocr.concat_text(ocr_blocks)
    ocr_entities = units.extract(ocr_text, max_units=fg.N_r, tfidf_topk=fg.tfidf_topk)
    ocr_entity_feats = clip.encode_text(ocr_entities)

    d_c = clip.d_c
    for arr, name in [
        (h_t_global, "h_t_global"), (h_v_global, "h_v_global"),
    ]:
        assert arr.shape == (d_c,), f"{name} 维度应为 ({d_c},)，实际 {arr.shape}"
    for arr, name in [
        (text_entity_feats, "text_entity_feats"),
        (region_feats, "region_feats"),
        (ocr_entity_feats, "ocr_entity_feats"),
    ]:
        assert arr.ndim == 2 and arr.shape[1] == d_c, f"{name} 末维应为 {d_c}"

    n_t = int(text_entity_feats.shape[0])
    n_o = int(region_feats.shape[0])
    n_r = int(ocr_entity_feats.shape[0])
    n_v = n_o + n_r
    has_local = (n_t > 0) and (n_v > 0)

    feat_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feat_dir / f"{sample_id}.npz",
        h_t_global=h_t_global.astype(np.float32),
        h_v_global=h_v_global.astype(np.float32),
        text_entity_feats=text_entity_feats.astype(np.float32),
        region_feats=region_feats.astype(np.float32),
        region_boxes=boxes.astype(np.float32),
        region_scores=scores.astype(np.float32),
        ocr_entity_feats=ocr_entity_feats.astype(np.float32),
        label=np.int64(label),
    )

    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "post_id": item.get("post_id", sample_id),
        "image_path": item["image_path"],
        "caption": caption,
        "label": label,
        "split": split_label,
        "d_c": d_c,
        "text_entities": text_entities,
        "ocr_entities": ocr_entities,
        "ocr_blocks": ocr_blocks,
        "region_boxes": boxes.tolist(),
        "region_scores": scores.tolist(),
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


def process_samples_batch(
    sample_ids: List[str],
    items: List[Dict],
    split_labels: List[str],
    tools: Dict,
    cfg,
    feat_dir: Path,
    meta_dir: Path,
) -> List[str]:
    if not sample_ids:
        return []

    dataset_dir = cfg.dataset_dir(tools["dataset"])
    clip: CLIPEncoder = tools["clip"]
    frcnn: FasterRCNNExtractor = tools["frcnn"]
    ocr = tools["ocr"]
    units: SemanticUnitExtractor = tools["units"]
    fg = cfg.fine_grained
    d_c = clip.d_c

    failed = []

    try:
        images = []
        captions = []
        labels = []
        image_paths = []

        for sid, item in zip(sample_ids, items):
            try:
                image_path = dataset_dir / item["image_path"]
                images.append(Image.open(image_path).convert("RGB"))
                captions.append(item.get("caption", "") or "")
                labels.append(int(item["label"]))
                image_paths.append(image_path)
            except Exception as e:
                failed.append(sid)
                images.append(None)
                captions.append("")
                labels.append(0)
                image_paths.append(None)

        valid_mask = [img is not None for img in images]
        valid_images = [img for img in images if img is not None]
        valid_captions = [c for c, v in zip(captions, valid_mask) if v]

        h_v_global_all = clip.encode_image(valid_images)
        h_t_global_all = clip.encode_text(valid_captions)

        text_entities_all = units.extract(valid_captions, max_units=fg.N_t, tfidf_topk=fg.tfidf_topk)

        all_text_entities = []
        entity_offsets = []
        for ents in text_entities_all:
            entity_offsets.append(len(all_text_entities))
            all_text_entities.extend(ents)

        text_entity_feats_all = clip.encode_text(all_text_entities)

        frcnn_results = frcnn.detect(valid_images)

        all_crops = []
        boxes_all = []
        scores_all = []
        box_offsets = []

        for img, (boxes, scores) in zip(valid_images, frcnn_results):
            crops = frcnn.crop_regions(img, boxes)
            box_offsets.append(len(all_crops))
            all_crops.extend(crops)
            boxes_all.append(boxes)
            scores_all.append(scores)

        region_feats_all = clip.encode_image(all_crops)

        ocr_blocks_all = []
        ocr_texts = []

        for img_path in image_paths:
            if img_path is None:
                ocr_blocks_all.append([])
                ocr_texts.append("")
            else:
                blocks = ocr.extract(img_path)
                ocr_blocks_all.append(blocks)
                ocr_texts.append(ocr.concat_text(blocks))

        valid_ocr_texts = [t for t, v in zip(ocr_texts, valid_mask) if v]
        ocr_entities_all = units.extract(valid_ocr_texts, max_units=fg.N_r, tfidf_topk=fg.tfidf_topk)

        all_ocr_entities = []
        ocr_offsets = []
        for ents in ocr_entities_all:
            ocr_offsets.append(len(all_ocr_entities))
            all_ocr_entities.extend(ents)

        ocr_entity_feats_all = clip.encode_text(all_ocr_entities)

        feat_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        valid_idx = 0
        for i, (sid, item, split_label) in enumerate(zip(sample_ids, items, split_labels)):
            if not valid_mask[i]:
                continue

            h_t_global = h_t_global_all[valid_idx]
            h_v_global = h_v_global_all[valid_idx]

            ent_start = entity_offsets[valid_idx]
            ent_end = entity_offsets[valid_idx + 1] if valid_idx + 1 < len(entity_offsets) else len(text_entity_feats_all)
            text_entities = text_entities_all[valid_idx]
            text_entity_feats = text_entity_feats_all[ent_start:ent_end]

            boxes = boxes_all[valid_idx]
            scores = scores_all[valid_idx]
            box_start = box_offsets[valid_idx]
            box_end = box_offsets[valid_idx + 1] if valid_idx + 1 < len(box_offsets) else len(region_feats_all)
            region_feats = region_feats_all[box_start:box_end]
            boxes = boxes[:region_feats.shape[0]]
            scores = scores[:region_feats.shape[0]]

            ocr_start = ocr_offsets[valid_idx]
            ocr_end = ocr_offsets[valid_idx + 1] if valid_idx + 1 < len(ocr_offsets) else len(ocr_entity_feats_all)
            ocr_entities = ocr_entities_all[valid_idx]
            ocr_entity_feats = ocr_entity_feats_all[ocr_start:ocr_end]
            ocr_blocks = ocr_blocks_all[i]

            n_t = int(text_entity_feats.shape[0])
            n_o = int(region_feats.shape[0])
            n_r = int(ocr_entity_feats.shape[0])
            n_v = n_o + n_r
            has_local = (n_t > 0) and (n_v > 0)

            np.savez_compressed(
                feat_dir / f"{sid}.npz",
                h_t_global=h_t_global.astype(np.float32),
                h_v_global=h_v_global.astype(np.float32),
                text_entity_feats=text_entity_feats.astype(np.float32),
                region_feats=region_feats.astype(np.float32),
                region_boxes=boxes.astype(np.float32),
                region_scores=scores.astype(np.float32),
                ocr_entity_feats=ocr_entity_feats.astype(np.float32),
                label=np.int64(labels[valid_idx]),
            )

            meta = {
                "post_id": item.get("post_id", sid),
                "image_path": item["image_path"],
                "caption": captions[i],
                "label": labels[valid_idx],
                "split": split_label,
                "d_c": d_c,
                "text_entities": text_entities,
                "ocr_entities": ocr_entities,
                "ocr_blocks": ocr_blocks,
                "region_boxes": boxes.tolist(),
                "region_scores": scores.tolist(),
                "shape": {"N_t": n_t, "N_o": n_o, "N_r": n_r, "N_v": n_v},
                "flags": {
                    "has_text": n_t > 0,
                    "has_region": n_o > 0,
                    "has_ocr": len(ocr_blocks) > 0,
                    "has_ocr_entity": n_r > 0,
                    "has_local": has_local,
                },
            }
            with open(meta_dir / f"{sid}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            valid_idx += 1

    except Exception as e:
        for sid, item, split_label in zip(sample_ids, items, split_labels):
            try:
                process_sample(sid, item, split_label, tools, cfg, feat_dir, meta_dir)
            except Exception as ex:
                failed.append(sid)

    return failed


def process_dataset(dataset: str, config_path: str, force_split: bool = False, limit: int = None) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)

    log_path = get_run_log_path(cfg.path("log_dir"), dataset, "preprocess")
    logger = setup_logger("preprocess", log_path, resume=True)
    logger.info("=" * 60)
    logger.info(f"开始预处理数据集: {dataset}")

    missing = cfg.validate_pretrained(dataset)
    if missing:
        for m in missing:
            logger.error(f"缺少预训练资源 -> {m}")
        raise FileNotFoundError("预训练资源不全，预处理终止。")

    json_path = cfg.json_path(dataset)
    with open(json_path, "r", encoding="utf-8") as f:
        items: Dict[str, Dict] = json.load(f)
    logger.info(f"样本总数: {len(items)} (源: {json_path})")

    split_map = load_or_build_split(cfg, dataset, items, logger, force=force_split)

    device = torch.device(
        "cuda" if (cfg.project.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    logger.info(f"使用设备: {device}")

    language = cfg.language(dataset)

    logger.info("初始化 CLIP / Faster R-CNN / PaddleOCR / NER ...")
    clip = CLIPEncoder(cfg.clip_path(dataset), language, device)
    logger.info(f"CLIP 加载完成, d_c={clip.d_c}")

    frcnn = FasterRCNNExtractor(
        cfg.faster_rcnn_path(), device,
        top_n=cfg.preprocess.faster_rcnn.top_n,
        score_thresh=cfg.preprocess.faster_rcnn.score_thresh,
        min_box_size=cfg.preprocess.faster_rcnn.min_box_size,
    )

    from src.preprocess.ocr_extractor import OCRExtractor
    ocr_dirs = cfg.paddleocr_paths(dataset)
    ocr = OCRExtractor(
        det_model_dir=ocr_dirs["det_model_dir"],
        rec_model_dir=ocr_dirs["rec_model_dir"],
        cls_model_dir=ocr_dirs["cls_model_dir"],
        language=language,
        use_gpu=bool(cfg.preprocess.ocr.use_gpu) and device.type == "cuda",
        drop_score=cfg.preprocess.ocr.drop_score,
        max_text_len=cfg.preprocess.ocr.max_text_len,
    )

    ner = NERExtractor(cfg.ner_path(dataset), language, device)
    corpus = [items[i].get("caption", "") or "" for i in items]
    keyword = KeywordExtractor(language, corpus)
    units = SemanticUnitExtractor(ner, keyword)

    tools = {
        "dataset": dataset, "clip": clip, "frcnn": frcnn,
        "ocr": ocr, "units": units,
    }

    cache_root = cfg.cache_dir(dataset)
    feat_dir = cache_root / "features"
    meta_dir = cache_root / "meta"
    resume = bool(cfg.preprocess.resume)
    batch_size = getattr(cfg.preprocess, "batch_size", 8)

    failed_path = cache_root / "failed.json"
    failed: Dict[str, str] = {}

    done, skipped, n_fail = 0, 0, 0
    t0 = time.time()
    ids = list(items.keys())

    if limit is not None and limit > 0:
        ids = ids[:limit]
        logger.info(f"测试模式：仅处理前 {limit} 条数据")

    pending_ids = []
    for sid in ids:
        npz_ok = (feat_dir / f"{sid}.npz").exists()
        meta_ok = (meta_dir / f"{sid}.json").exists()
        if resume and npz_ok and meta_ok:
            skipped += 1
        else:
            pending_ids.append(sid)

    pbar = tqdm(total=len(ids), desc=f"Preprocess[{dataset}]", ncols=100)
    pbar.set_postfix(done=done, skip=skipped, fail=n_fail)
    pbar.update(skipped)

    for i in range(0, len(pending_ids), batch_size):
        batch_ids = pending_ids[i:i + batch_size]
        batch_items = [items[sid] for sid in batch_ids]
        batch_splits = [split_map.get(sid, "train") for sid in batch_ids]

        try:
            batch_failed = process_samples_batch(
                batch_ids, batch_items, batch_splits,
                tools, cfg, feat_dir, meta_dir,
            )
            done += (len(batch_ids) - len(batch_failed))
            for sid in batch_failed:
                n_fail += 1
                failed[sid] = "批量处理失败"
                logger.warning(f"样本 {sid} 批量处理失败")
        except Exception as e:
            logger.warning(f"批量处理失败，降级为单样本模式: {e}")
            for sid, item, split_label in zip(batch_ids, batch_items, batch_splits):
                try:
                    process_sample(sid, item, split_label, tools, cfg, feat_dir, meta_dir)
                    done += 1
                except Exception as ex:
                    n_fail += 1
                    failed[sid] = repr(ex)
                    logger.warning(f"样本 {sid} 处理失败: {ex}")

        pbar.update(len(batch_ids))
        pbar.set_postfix(done=done, skip=skipped, fail=n_fail)
    pbar.close()

    if failed:
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)

    stats = {
        "dataset": dataset,
        "language": language,
        "d_c": clip.d_c,
        "total": len(ids),
        "done": done,
        "skipped": skipped,
        "failed": n_fail,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(cache_root / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"预处理完成: {stats}")

def main():
    parser = argparse.ArgumentParser(description="RACE 离线预处理")
    parser.add_argument("--dataset", required=True,
                        help="数据集名: Weibo / Twitter / GossipCop")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force-split", action="store_true",
                        help="强制重建切分（覆盖已有 split.json）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理样本数量（用于测试，如 --limit 10 只处理前10条）")
    args = parser.parse_args()
    process_dataset(args.dataset, args.config, force_split=args.force_split, limit=args.limit)

if __name__ == "__main__":
    main()