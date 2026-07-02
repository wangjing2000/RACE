from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import torch

def save_checkpoint(
    path: Path,
    epoch: int,
    model_state: Dict,
    optim_state: Dict,
    sched_state: Optional[Dict],
    best_metric: float,
    config_dict: Dict,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "model":       model_state,
        "optimizer":   optim_state,
        "scheduler":   sched_state,
        "best_metric": best_metric,
        "config":      config_dict,
    }, path)

def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)