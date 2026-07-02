from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.dim_check import check_last_dim, check_shape

def _make_mlp(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_out),
        nn.LayerNorm(d_out),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(d_out, d_out),
    )

class MultiGranularityExtractor(nn.Module):
    def __init__(
        self,
        d_c: int = 512,
        d_l: int = 256,
        d_g: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_c = d_c
        self.d_l = d_l
        self.d_g = d_g

        self.mlp_t = _make_mlp(d_c, d_l, dropout)
        self.mlp_obj = _make_mlp(d_c, d_l, dropout)
        self.mlp_ocr = _make_mlp(d_c, d_l, dropout)

        self.mlp_tg = _make_mlp(d_c, d_g, dropout)
        self.mlp_vg = _make_mlp(d_c, d_g, dropout)

    def forward(
        self,
        h_t_global: torch.Tensor,
        h_v_global: torch.Tensor,
        text_feats: torch.Tensor,
        text_mask: torch.Tensor,
        region_feats: torch.Tensor,
        region_mask: torch.Tensor,
        ocr_feats: torch.Tensor,
        ocr_mask: torch.Tensor,
        use_region: bool = True,
        use_ocr: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor, torch.Tensor, torch.Tensor]:
        B = h_t_global.shape[0]

        check_last_dim(h_t_global, self.d_c, "h_t_global")
        check_last_dim(h_v_global, self.d_c, "h_v_global")
        if text_feats.shape[1] > 0:
            check_last_dim(text_feats, self.d_c, "text_feats")
        if region_feats.shape[1] > 0:
            check_last_dim(region_feats, self.d_c, "region_feats")
        if ocr_feats.shape[1] > 0:
            check_last_dim(ocr_feats, self.d_c, "ocr_feats")

        ht_norm = F.normalize(h_t_global, dim=-1)
        hv_norm = F.normalize(h_v_global, dim=-1)
        s_g = (ht_norm * hv_norm).sum(dim=-1)

        h_tg = self.mlp_tg(h_t_global)
        h_vg = self.mlp_vg(h_v_global)
        f_g = torch.cat([h_tg, h_vg, s_g.unsqueeze(-1)], dim=-1)

        Z_t = self._project_seq(text_feats, self.mlp_t)

        parts_feat, parts_mask = [], []

        if use_region and region_feats.shape[1] > 0:
            Z_obj = self._project_seq(region_feats, self.mlp_obj)
            parts_feat.append(Z_obj)
            parts_mask.append(region_mask)

        if use_ocr and ocr_feats.shape[1] > 0:
            Z_ocr = self._project_seq(ocr_feats, self.mlp_ocr)
            parts_feat.append(Z_ocr)
            parts_mask.append(ocr_mask)

        if parts_feat:
            Z_v = torch.cat(parts_feat, dim=1)
            visual_mask = torch.cat(parts_mask, dim=1)
        else:
            Z_v = torch.zeros(B, 0, self.d_l, device=h_t_global.device,
                              dtype=h_t_global.dtype)
            visual_mask = torch.zeros(B, 0, device=h_t_global.device,
                                      dtype=torch.bool)

        check_last_dim(f_g, 2 * self.d_g + 1, "f_g")
        if Z_t.shape[1] > 0:
            check_last_dim(Z_t, self.d_l, "Z_t")
        if Z_v.shape[1] > 0:
            check_last_dim(Z_v, self.d_l, "Z_v")
        assert s_g.shape == (B,), f"s_g 形状应为 ({B},)，实际 {s_g.shape}"

        return f_g, Z_t, text_mask, Z_v, visual_mask, s_g

    @staticmethod
    def _project_seq(
        feats: torch.Tensor,
        mlp: nn.Sequential,
    ) -> torch.Tensor:
        B, N, d_in = feats.shape
        if N == 0:
            d_out = mlp[-1].out_features
            return torch.zeros(B, 0, d_out, device=feats.device, dtype=feats.dtype)
        out = mlp(feats.reshape(B * N, d_in))
        return out.reshape(B, N, -1)