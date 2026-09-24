from __future__ import annotations

import torch
import torch.nn as nn

def _make_mlp(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_out),
        nn.LayerNorm(d_out),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(d_out, d_out),
    )

class ConflictAwareGatedFusion(nn.Module):
    def __init__(self, f_g_dim=513, d_f=256, dropout=0.1):
        super().__init__()
        self.proj_g = _make_mlp(f_g_dim, d_f, dropout)
        self.W_g = nn.Linear(d_f, d_f, bias=False)
        self.W_c = nn.Linear(d_f, d_f, bias=False)
        self.b_s = nn.Parameter(torch.zeros(d_f))

    def forward(self, f_g, f_c, s_c):
        f_g_proj = self.proj_g(f_g)
        alpha = torch.sigmoid(
            self.W_g(f_g_proj)
            + self.W_c(f_c)
            + s_c.unsqueeze(-1) * self.b_s
        )
        f_fuse = (
            alpha * f_g_proj
            + (1.0 - alpha) * f_c
        )
        return f_fuse, alpha