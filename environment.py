"""Dec-POMDP anomaly-detection environment (Section 3.2).

Each episode streams windows from a :class:`DatasetBundle`. At every step the
agent receives a local, noisy observation o_{i,t} (a window's feature
trajectory), queries the tensor memory for context r_{i,t}, forms the state
s_{i,t} = [q_{i,t}, r_{i,t}], and emits an action a_{i,t} in {normal, anomaly}.

Reward is the multi-objective function of Eq. (1):
    R = w1 * Health - w2 * Energy - w3 * PrivacyLeakage
where ``Health`` rewards correct anomaly classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..data.datasets import Split


@dataclass
class RewardWeights:
    w1: float = 1.0   # health benefit
    w2: float = 0.05  # energy cost
    w3: float = 0.05  # privacy leakage


class DecPOMDPEnv:
    """A simple, deterministic Dec-POMDP wrapper around a labelled split."""

    def __init__(self, split: Split, weights: RewardWeights | None = None, seed: int = 0):
        self.split = split
        self.w = weights or RewardWeights()
        self.gen = np.random.default_rng(seed)
        self.n = len(split)
        self._order = self.gen.permutation(self.n)
        self._ptr = 0

    def reset(self) -> np.ndarray:
        self._ptr = 0
        self._order = self.gen.permutation(self.n)
        return self._observe()

    def _observe(self) -> np.ndarray:
        idx = self._order[self._ptr % self.n]
        # Local "current observation embedding" q_t = last-timestep features.
        return self.split.X[idx]  # (seq_len, d)

    def current_label(self) -> int:
        idx = self._order[self._ptr % self.n]
        return int(self.split.y[idx])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        label = self.current_label()
        health = 1.0 if action == label else -1.0
        energy = 0.2 if action == 1 else 0.1   # acting on anomaly costs more
        privacy = 0.1                          # constant leakage per transmission
        reward = (self.w.w1 * health - self.w.w2 * energy - self.w.w3 * privacy)
        self._ptr += 1
        done = self._ptr >= self.n
        next_obs = self._observe() if not done else None
        return next_obs, float(reward), done
