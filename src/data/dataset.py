from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

class RaceDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path,
        images_dir: str | Path,
        cache_dir: str | Path,
        split: str = "train",
        language: str = "zh",
        max_text_units: int = 20,
        max_ocr_units: int = 15,
        max_region_units: int = 10,
    ):
        super().__init__()
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.language = language
        self.max_text_units = max_text_units
        self.max_ocr_units = max_ocr_units
        self.max_region_units = max_region_units

        self.samples: List[Dict[str, Any]] = []
        self._load_json()

    def _load_json(self) -> None:
        if not self.json_path.exists():
            raise FileNotFoundError(f"数据集 JSON 不存在: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(f"JSON 应为列表格式，实际: {type(raw)}")

        for item in raw:
            if self.split == "train" and item.get("split", "train") != "train":
                continue
            if self.split == "val" and item.get("split", "train") != "val":
                continue
            if self.split == "test" and item.get("split", "train") != "test":
                continue
            self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sample_id = sample.get("id", str(idx))
        label = float(sample.get("label", 0))

        cache_path = self.cache_dir / f"{sample_id}.npz"
        if cache_path.exists():
            data = dict(np.load(cache_path, allow_pickle=True))
            h_t_global = torch.from_numpy(data["h_t_global"].astype(np.float32))
            h_v_global = torch.from_numpy(data["h_v_global"].astype(np.float32))

            text_feats_np = data.get("text_feats", np.zeros((1, 512), dtype=np.float32))
            text_feats = torch.from_numpy(text_feats_np.astype(np.float32))
            if text_feats.dim() == 1:
                text_feats = text_feats.unsqueeze(0)

            region_feats_np = data.get("region_feats", np.zeros((0, 512), dtype=np.float32))
            region_feats = torch.from_numpy(region_feats_np.astype(np.float32))
            if region_feats.dim() == 1:
                region_feats = region_feats.unsqueeze(0)

            ocr_feats_np = data.get("ocr_feats", np.zeros((0, 512), dtype=np.float32))
            ocr_feats = torch.from_numpy(ocr_feats_np.astype(np.float32))
            if ocr_feats.dim() == 1:
                ocr_feats = ocr_feats.unsqueeze(0)
        else:
            h_t_global = torch.zeros(512)
            h_v_global = torch.zeros(512)
            text_feats = torch.zeros(1, 512)
            region_feats = torch.zeros(0, 512)
            ocr_feats = torch.zeros(0, 512)

        text_feats = text_feats[: self.max_text_units]
        region_feats = region_feats[: self.max_region_units]
        ocr_feats = ocr_feats[: self.max_ocr_units]

        text_mask = torch.ones(text_feats.shape[0], dtype=torch.bool)
        region_mask = torch.ones(region_feats.shape[0], dtype=torch.bool) if region_feats.shape[0] > 0 else torch.zeros(0, dtype=torch.bool)
        ocr_mask = torch.ones(ocr_feats.shape[0], dtype=torch.bool) if ocr_feats.shape[0] > 0 else torch.zeros(0, dtype=torch.bool)

        return {
            "id": sample_id,
            "label": label,
            "h_t_global": h_t_global,
            "h_v_global": h_v_global,
            "text_feats": text_feats,
            "text_mask": text_mask,
            "region_feats": region_feats,
            "region_mask": region_mask,
            "ocr_feats": ocr_feats,
            "ocr_mask": ocr_mask,
        }