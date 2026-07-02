from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

class FasterRCNNExtractor:
    def __init__(self, model_dir: str | Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model_dir = str(model_dir)

        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()

        self.transform = weights.transforms()
        self.score_threshold = 0.7

    @torch.no_grad()
    def extract_regions(self, image: Image.Image, max_regions: int = 10) -> np.ndarray:
        img_tensor = self.transform(image).to(self.device)
        outputs = self.model(img_tensor.unsqueeze(0))[0]

        scores = outputs["scores"].cpu().numpy()
        boxes = outputs["boxes"].cpu().numpy()

        mask = scores >= self.score_threshold
        boxes = boxes[mask][:max_regions]

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        return boxes.astype(np.float32)

    def cache_regions(
        self,
        json_path: str | Path,
        images_dir: str | Path,
        cache_dir: str | Path,
        max_regions: int = 10,
    ) -> None:
        json_path = Path(json_path)
        images_dir = Path(images_dir)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for sample in tqdm(samples, desc="Caching region boxes"):
            sample_id = sample.get("id", "")
            cache_path = cache_dir / f"{sample_id}_regions.npz"
            if cache_path.exists():
                continue

            image_path = images_dir / sample.get("image", "")
            if image_path.exists():
                img = Image.open(image_path).convert("RGB")
                regions = self.extract_regions(img, max_regions)
            else:
                regions = np.zeros((0, 4), dtype=np.float32)

            np.savez(cache_path, regions=regions)