from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image

class FrozenCLIPEncoder(nn.Module):
    def __init__(self, model_dir: str | Path, language: str):
        super().__init__()
        self.language = language
        model_dir = str(model_dir)

        if language == "zh":
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            self.processor = ChineseCLIPProcessor.from_pretrained(
                model_dir, local_files_only=True
            )
            self._model = ChineseCLIPModel.from_pretrained(
                model_dir, local_files_only=True
            )
            self._text_max_len = 52
        else:
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(
                model_dir, local_files_only=True
            )
            self._model = CLIPModel.from_pretrained(
                model_dir, local_files_only=True
            )
            self._text_max_len = 77

        self.d_c: int = int(self._model.config.projection_dim)

        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        return self

    @torch.no_grad()
    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        texts = [t if (t and t.strip()) else " " for t in texts]
        inputs = self.processor(
            text=texts, padding=True, truncation=True,
            max_length=self._text_max_len, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self._model.get_text_features(**inputs)
        return out.float()

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image], device: torch.device) -> torch.Tensor:
        images = [img.convert("RGB") for img in images]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self._model.get_image_features(**inputs)
        return out.float()