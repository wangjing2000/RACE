from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class RACELoss(nn.Module):
    def __init__(
        self,
        delta: float = 0.5,
        beta: float = 0.5,
        lambda_real: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.delta = delta
        self.beta = beta
        self.lambda_real = lambda_real
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
        c_ij: torch.Tensor,
        pair_mask: torch.Tensor,
        s_c: torch.Tensor,
        has_local: torch.Tensor,
        use_align: bool = True,
    ) -> Dict[str, torch.Tensor]:
        y = label.float()

        pw = getattr(self, "pos_weight", None)
        if pw is not None:
            pw = pw.to(logit.device)
        L_cls = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)

        S0_mask = (label == 0) & has_local
        if S0_mask.any():
            penalty = torch.clamp(c_ij[S0_mask] - self.delta, min=0.0)
            valid = pair_mask[S0_mask].float()
            L_real = self.lambda_real * (penalty * valid).sum() / S0_mask.sum().float()
        else:
            L_real = logit.new_zeros(1).squeeze()

        S1_mask = (label == 1) & has_local
        if use_align and S1_mask.any():
            shortfall = torch.clamp(self.delta - s_c[S1_mask], min=0.0)
            L_align = self.beta * shortfall.mean()
        else:
            L_align = logit.new_zeros(1).squeeze()

        loss = L_cls + L_real + L_align
        return {
            "loss": loss,
            "L_cls": L_cls.detach(),
            "L_real": L_real.detach(),
            "L_align": L_align.detach(),
        }