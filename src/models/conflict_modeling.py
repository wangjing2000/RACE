from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.dim_check import check_last_dim, check_shape

class ConflictModeling(nn.Module):
    def __init__(
        self,
        d_l: int = 256,
        d_r: int = 128,
        d_f: int = 256,
        gamma: float = 5.0,
        delta: float = 0.5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_l = d_l
        self.d_r = d_r
        self.d_f = d_f
        self.gamma = gamma
        self.delta = delta

        self.relevance_mlp = nn.Sequential(
            nn.Linear(2 * d_l, d_r),
            nn.LayerNorm(d_r),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_r, 1),
        )

        self.conflict_mlp = nn.Sequential(
            nn.Linear(2 * d_l, d_f),
            nn.LayerNorm(d_f),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_f, d_f),
        )

        self.gate_linear = nn.Linear(d_f + 1, 1)

    def forward(
        self,
        Z_t: torch.Tensor,
        text_mask: torch.Tensor,
        Z_v: torch.Tensor,
        visual_mask: torch.Tensor,
        use_relevance: bool = True,
        use_conflict: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        B = Z_t.shape[0]
        device = Z_t.device
        dtype = Z_t.dtype

        if Z_t.shape[1] > 0:
            check_last_dim(Z_t, self.d_l, "Z_t")
        if Z_v.shape[1] > 0:
            check_last_dim(Z_v, self.d_l, "Z_v")
        check_shape(text_mask, [B, Z_t.shape[1]], "text_mask")
        check_shape(visual_mask, [B, Z_v.shape[1]], "visual_mask")

        has_text_units = text_mask.any(dim=1)
        has_visual_units = visual_mask.any(dim=1)
        has_local = has_text_units & has_visual_units

        N_t, N_v = Z_t.shape[1], Z_v.shape[1]
        empty_cij = torch.zeros(B, N_t, N_v, device=device, dtype=dtype)
        empty_attn = torch.zeros(B, N_t, N_v, device=device, dtype=dtype)

        if not use_conflict or N_t == 0 or N_v == 0:
            f_c = torch.zeros(B, self.d_f, device=device, dtype=dtype)
            s_c = torch.zeros(B, device=device, dtype=dtype)
            rho = torch.zeros(B, device=device, dtype=dtype)
            return f_c, s_c, rho, empty_cij, empty_attn, has_local

        Zt_exp = Z_t.unsqueeze(2).expand(B, N_t, N_v, self.d_l)
        Zv_exp = Z_v.unsqueeze(1).expand(B, N_t, N_v, self.d_l)
        pair_feats = torch.cat([Zt_exp, Zv_exp], dim=-1)

        if use_relevance:
            r_ij = torch.sigmoid(
                self.relevance_mlp(pair_feats).squeeze(-1)
            )
        else:
            r_ij = torch.ones(B, N_t, N_v, device=device, dtype=dtype)

        Zt_norm = F.normalize(Z_t, dim=-1)
        Zv_norm = F.normalize(Z_v, dim=-1)
        cos_sim = torch.bmm(Zt_norm, Zv_norm.transpose(1, 2))
        e_ij = 1.0 - cos_sim

        c_ij = r_ij * e_ij

        pair_mask = (
            text_mask.unsqueeze(2) & visual_mask.unsqueeze(1)
        )

        NEG_INF = torch.finfo(dtype).min / 2
        c_masked = c_ij.masked_fill(~pair_mask, NEG_INF)

        c_scaled = self.gamma * c_masked
        c_max = c_scaled.flatten(1).max(dim=1).values
        c_max = c_max.clamp(min=NEG_INF + 1)

        exp_sum = (
            torch.exp(c_scaled - c_max.view(B, 1, 1))
            .masked_fill(~pair_mask, 0.0)
            .flatten(1).sum(dim=1)
        )
        exp_sum = exp_sum.clamp(min=1e-10)
        s_c_raw = c_max / self.gamma + torch.log(exp_sum) / self.gamma

        s_c = s_c_raw * has_local.to(dtype)

        c_for_attn = c_ij.masked_fill(~pair_mask, NEG_INF)
        attn_flat = torch.softmax(
            c_for_attn.flatten(1), dim=1
        ).reshape(B, N_t, N_v)
        attn = attn_flat.masked_fill(~pair_mask, 0.0)

        weighted = attn.unsqueeze(-1) * pair_feats
        agg = weighted.flatten(1, 2).sum(dim=1)
        f_c_raw = self.conflict_mlp(agg)

        f_c = f_c_raw * has_local.unsqueeze(-1).to(dtype)

        gate_in = torch.cat([f_c, s_c.unsqueeze(-1)], dim=-1)
        rho_raw = torch.sigmoid(self.gate_linear(gate_in)).squeeze(-1)
        rho = rho_raw * has_local.to(dtype)

        check_last_dim(f_c, self.d_f, "f_c")
        assert s_c.shape == (B,), f"s_c shape={s_c.shape}"
        assert rho.shape == (B,), f"rho shape={rho.shape}"

        return f_c, s_c, rho, c_ij, attn, has_local