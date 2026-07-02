from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.dim_check import check_last_dim

def _make_mlp(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_out),
        nn.LayerNorm(d_out),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(d_out, d_out),
    )

class GatedFusion(nn.Module):
    def __init__(self, d_g: int = 256, d_f: int = 256, dropout: float = 0.2):
        super().__init__()
        self.d_g = d_g
        self.d_f = d_f
        f_g_dim = 2 * d_g + 1

        self.proj_g = _make_mlp(f_g_dim, d_f, dropout)

        gate_in_dim = d_f + d_f + 1 + 1
        self.gate_linear = nn.Linear(gate_in_dim, 1)

    def forward(
        self,
        f_g: torch.Tensor,
        f_c: torch.Tensor,
        s_c: torch.Tensor,
        s_g: torch.Tensor,
        has_local: torch.Tensor,
        use_gate: bool = True,
    ):
        B = f_g.shape[0]
        check_last_dim(f_g, 2 * self.d_g + 1, "f_g")
        check_last_dim(f_c, self.d_f, "f_c")

        f_g_proj = self.proj_g(f_g)

        if use_gate:
            gate_in = torch.cat(
                [f_g_proj, f_c, s_c.unsqueeze(-1), s_g.unsqueeze(-1)], dim=-1
            )
            g_raw = torch.sigmoid(self.gate_linear(gate_in)).squeeze(-1)
        else:
            g_raw = torch.full((B,), 0.5, device=f_g.device, dtype=f_g.dtype)

        g = g_raw * has_local.to(f_g.dtype)

        f_fuse = g.unsqueeze(-1) * f_c + (1 - g).unsqueeze(-1) * f_g_proj

        check_last_dim(f_fuse, self.d_f, "f_fuse")
        return f_fuse, g