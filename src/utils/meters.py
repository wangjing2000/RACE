from __future__ import annotations
from collections import defaultdict
import torch

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

class MetricBag:
    def __init__(self):
        self._meters: dict[str, AverageMeter] = defaultdict(AverageMeter)

    def update(self, d: dict[str, float], n: int = 1):
        for k, v in d.items():
            self._meters[k].update(float(v), n)

    def reset(self):
        for m in self._meters.values():
            m.reset()

    def avg(self) -> dict[str, float]:
        return {k: m.avg for k, m in self._meters.items()}