from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

class OCRExtractor:
    def __init__(self, det_model_dir: str, rec_model_dir: str, cls_model_dir: str, language: str = "zh"):
        self.language = language
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                det_model_dir=det_model_dir,
                rec_model_dir=rec_model_dir,
                cls_model_dir=cls_model_dir,
                use_angle_cls=True,
                lang="ch" if language == "zh" else "en",
                show_log=False,
                use_gpu=True,
            )
        except ImportError:
            self.ocr = None

    def extract(self, image: Image.Image) -> List[Tuple[str, float, List[List[int]]]]:
        if self.ocr is None:
            return []

        import numpy as np
        img_array = np.array(image)
        results = self.ocr.ocr(img_array, cls=True)

        if not results or not results[0]:
            return []

        ocr_results = []
        for line in results[0]:
            box = line[0]
            text = line[1][0]
            score = line[1][1]
            ocr_results.append((text, score, box))

        return ocr_results

    def cache_ocr(
        self,
        json_path: str | Path,
        images_dir: str | Path,
        cache_dir: str | Path,
    ) -> None:
        json_path = Path(json_path)
        images_dir = Path(images_dir)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for sample in tqdm(samples, desc="Caching OCR results"):
            sample_id = sample.get("id", "")
            cache_path = cache_dir / f"{sample_id}_ocr.json"
            if cache_path.exists():
                continue

            image_path = images_dir / sample.get("image", "")
            if image_path.exists():
                img = Image.open(image_path).convert("RGB")
                ocr_results = self.extract(img)
                ocr_data = [
                    {"text": t, "score": float(s), "box": b}
                    for t, s, b in ocr_results
                ]
            else:
                ocr_data = []

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(ocr_data, f, ensure_ascii=False, indent=2)