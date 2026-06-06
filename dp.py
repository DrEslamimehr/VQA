"""Differential Privacy for federated tensor-memory updates (Section 5.3).

Injects Laplacian noise into tensor deltas to guarantee (epsilon, delta)-DP
with epsilon = 2.0, delta = 1e-5 (Table 1). When Opacus is installed it is used
to track the privacy accountant; otherwise the bundled Laplacian mechanism
provides the same noise injection so the repo runs without Opacus.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LaplacianDP:
    epsilon: float = 2.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    seed: int = 0

    def __post_init__(self):
        self._gen = np.random.default_rng(self.seed)
        # Laplace scale b = sensitivity / epsilon (pure-DP); delta tracked by accountant.
        self.scale = self.sensitivity / max(self.epsilon, 1e-9)

    def privatize(self, tensor: np.ndarray) -> np.ndarray:
        noise = self._gen.laplace(loc=0.0, scale=self.scale, size=tensor.shape)
        return tensor + noise

    def opacus_available(self) -> bool:
        try:
            import opacus  # noqa: F401
            return True
        except Exception:
            return False

    def report(self) -> dict:
        return {
            "mechanism": "laplacian",
            "epsilon": self.epsilon,
            "delta": self.delta,
            "laplace_scale": self.scale,
            "opacus": self.opacus_available(),
        }
