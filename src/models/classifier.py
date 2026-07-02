from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.dim_check import check_last_dim

class RumorClassifier(nn.Module):
    def __init__(self, d_f: int = 256, dropout: float = 0.2):
        super().__init__()
        self.d_f = d_f
        self.net = nn.Sequential(
            nn.Linear(d_f, d_f // 2),
            nn.LayerNorm(d_f // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_f // 2, 1),
        )

    def forward(self, f_fuse: torch.Tensor) -> torch.Tensor:
        check_last_dim(f_fuse, self.d_f, "f_fuse")
        return self.net(f_fuse).squeeze(-1)