from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models.clip_encoder import FrozenCLIPEncoder

class CLIPEncoder:
    def __init__(self, model_dir: str | Path, language: str = "zh", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder = FrozenCLIPEncoder(model_dir, language).to(self.device)
        self.d_c = self.encoder.d_c

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_feats = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            feats = self.encoder.encode_text(batch, self.device)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            feats = self.encoder.encode_image(batch, self.device)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    def cache_dataset(
        self,
        json_path: str | Path,
        images_dir: str | Path,
        cache_dir: str | Path,
        text_batch_size: int = 64,
        image_batch_size: int = 32,
    ) -> None:
        json_path = Path(json_path)
        images_dir = Path(images_dir)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for sample in tqdm(samples, desc="Caching CLIP features"):
            sample_id = sample.get("id", "")
            cache_path = cache_dir / f"{sample_id}.npz"
            if cache_path.exists():
                continue

            text = sample.get("text", "")
            h_t_global = self.encode_texts([text])[0]

            image_path = images_dir / sample.get("image", "")
            if image_path.exists():
                img = Image.open(image_path).convert("RGB")
                h_v_global = self.encode_images([img])[0]
            else:
                h_v_global = np.zeros(self.d_c, dtype=np.float32)

            np.savez(cache_path, h_t_global=h_t_global, h_v_global=h_v_global)