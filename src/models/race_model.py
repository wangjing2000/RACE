from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.feature_extraction import MultiGranularityExtractor
from src.models.conflict_modeling import ConflictModeling
from src.models.gated_fusion import GatedFusion
from src.models.classifier import RumorClassifier

@dataclass
class AblationConfig:
    use_region: bool = True
    use_ocr: bool = True
    use_relevance: bool = True
    use_conflict: bool = True
    use_gate: bool = True
    use_align: bool = True

class RACEModel(nn.Module):
    def __init__(
        self,
        d_c: int = 512,
        d_l: int = 256,
        d_g: int = 256,
        d_f: int = 256,
        d_r: int = 128,
        gamma: float = 5.0,
        delta: float = 0.5,
        dropout: float = 0.2,
        ablation: Optional[AblationConfig] = None,
    ):
        super().__init__()
        self.delta = delta
        self.ablation = ablation or AblationConfig()

        self.extractor = MultiGranularityExtractor(
            d_c=d_c, d_l=d_l, d_g=d_g, dropout=dropout
        )
        self.conflict = ConflictModeling(
            d_l=d_l, d_r=d_r, d_f=d_f, gamma=gamma, delta=delta, dropout=dropout
        )
        self.fusion = GatedFusion(d_g=d_g, d_f=d_f, dropout=dropout)
        self.classifier = RumorClassifier(d_f=d_f, dropout=dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ab = self.ablation

        f_g, Z_t, text_mask, Z_v, visual_mask, s_g = self.extractor(
            h_t_global=batch["h_t_global"],
            h_v_global=batch["h_v_global"],
            text_feats=batch["text_feats"],
            text_mask=batch["text_mask"],
            region_feats=batch["region_feats"],
            region_mask=batch["region_mask"],
            ocr_feats=batch["ocr_feats"],
            ocr_mask=batch["ocr_mask"],
            use_region=ab.use_region,
            use_ocr=ab.use_ocr,)

        f_c, s_c, rho, c_ij, attn, has_local = self.conflict(
            Z_t=Z_t,
            text_mask=text_mask,
            Z_v=Z_v,
            visual_mask=visual_mask,
            use_relevance=ab.use_relevance,
            use_conflict=ab.use_conflict,
        )

        N_t, N_v = Z_t.shape[1], Z_v.shape[1]
        if N_t > 0 and N_v > 0:
            pair_mask = text_mask.unsqueeze(2) & visual_mask.unsqueeze(1)
        else:
            B = f_g.shape[0]
            pair_mask = torch.zeros(
                B, N_t, N_v, dtype=torch.bool, device=f_g.device
            )

        f_fuse, g = self.fusion(
            f_g=f_g,
            f_c=f_c,
            s_c=s_c,
            s_g=s_g,
            has_local=has_local,
            use_gate=ab.use_gate,
        )

        logit = self.classifier(f_fuse)

        return {
            "logit": logit,
            "s_c": s_c,
            "c_ij": c_ij,
            "pair_mask": pair_mask,
            "f_fuse": f_fuse,
            "g": g,
            "has_local": has_local,
            "attn": attn,
        }

    def predict(self, batch: Dict) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(batch)
        return torch.sigmoid(out["logit"])