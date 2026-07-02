from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.dataset import RaceDataset

class KFoldRaceDataset(RaceDataset):
    def __init__(
        self,
        json_path: str | Path,
        images_dir: str | Path,
        cache_dir: str | Path,
        split: str = "train",
        language: str = "zh",
        n_folds: int = 5,
        current_fold: int = 0,
        seed: int = 42,
        max_text_units: int = 20,
        max_ocr_units: int = 15,
        max_region_units: int = 10,
    ):
        self.n_folds = n_folds
        self.current_fold = current_fold
        self.seed = seed
        self._fold_indices: Optional[List[int]] = None

        super().__init__(
            json_path=json_path,
            images_dir=images_dir,
            cache_dir=cache_dir,
            split=split,
            language=language,
            max_text_units=max_text_units,
            max_ocr_units=max_ocr_units,
            max_region_units=max_region_units,
        )

        self._setup_fold()

    def _setup_fold(self) -> None:
        n = len(self.samples)
        rng = np.random.RandomState(self.seed)
        indices = list(range(n))
        rng.shuffle(indices)

        fold_size = math.ceil(n / self.n_folds)
        start = self.current_fold * fold_size
        end = min(start + fold_size, n)

        val_indices = set(indices[start:end])
        train_indices = [i for i in indices if i not in val_indices]

        if self.split == "train":
            self.samples = [self.samples[i] for i in train_indices]
        elif self.split == "val":
            self.samples = [self.samples[i] for i in sorted(val_indices)]

        self._fold_indices = train_indices if self.split == "train" else sorted(val_indices)

def get_current_fold(config) -> int:
    return getattr(config, "current_fold", 0)