"""Per-camera clear-day baseline = low percentile of recent, confident raw haze values."""

from __future__ import annotations
from collections import deque
import numpy as np


class Baseline:
    def __init__(self, min_samples: int = 50, pct: float = 5.0, maxlen: int = 4000):
        self.vals, self.min_samples, self.pct = deque(maxlen=maxlen), min_samples, pct

    def update(self, raw: float, confidence: float) -> None:
        if confidence >= 0.7:
            self.vals.append(raw)

    @property
    def value(self) -> float | None:
        return (
            float(np.percentile(self.vals, self.pct))
            if len(self.vals) >= self.min_samples
            else None
        )
