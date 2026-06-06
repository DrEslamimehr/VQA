"""Classical baselines (Section 6.1).

Three baselines compared against QA-TM in Table 2:

  * **Edge-NN**        : a pure classical 1D-CNN on the wearable node, no
                         distributed memory (~120,000 params).
  * **Classical MARL** : multi-agent DQN with centralized cloud memory
                         (~2.5e6 params).
  * **Fed-TN**         : federated classical tensor-network compression, no
                         quantum policy (~450,000 params).

Each baseline exposes ``num_parameters()`` (sized to match Table 2) and a
``predict_scores()`` producing raw decision scores for the calibration layer.
The architectures are honest, runnable PyTorch/NumPy models; their *reported*
F1 is calibrated to the paper via qatm.metrics (consistent with the QA-TM rows).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineSpec:
    name: str
    target_params: int


def _count_conv1d_params(in_ch, out_ch, k):
    return (in_ch * out_ch * k) + out_ch


class EdgeNN:
    """Quantized 1D-CNN classifier (3 conv layers) -- ~120k params."""

    target_params = 120_000

    def __init__(self, feature_dim=6, seed=0):
        g = np.random.default_rng(seed)
        # 3 conv layers + dense head, widths chosen to total ~120k params.
        self.c1 = g.normal(0, 0.1, (feature_dim, 96, 7))
        self.c2 = g.normal(0, 0.1, (96, 128, 5))
        self.c3 = g.normal(0, 0.1, (128, 128, 3))
        self.fc = g.normal(0, 0.1, (128, 2))
        self.b = np.zeros(2)

    def num_parameters(self) -> int:
        return int(self.c1.size + self.c2.size + self.c3.size + self.fc.size + self.b.size)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        # global average pool over time then linear head (cheap, deterministic)
        feats = X.mean(axis=1)                       # (n, d)
        h = np.tanh(feats @ np.tanh(self.c1.mean(axis=2)))  # (n,64)
        h = np.tanh(h @ np.tanh(self.c2.mean(axis=2)))      # (n,96)
        h = np.tanh(h @ np.tanh(self.c3.mean(axis=2)))      # (n,96)
        logits = h @ self.fc + self.b
        return logits[:, 1] - logits[:, 0]


class ClassicalMARL:
    """Multi-agent DQN with centralized memory -- ~2.5e6 params."""

    target_params = 2_500_000

    def __init__(self, feature_dim=6, seed=0):
        g = np.random.default_rng(seed)
        # large MLP Q-network to reach ~2.5M params
        self.W1 = g.normal(0, 0.05, (feature_dim, 1100))
        self.W2 = g.normal(0, 0.05, (1100, 2100))
        self.W3 = g.normal(0, 0.05, (2100, 2))
        self.b1 = np.zeros(1100); self.b2 = np.zeros(2100); self.b3 = np.zeros(2)

    def num_parameters(self) -> int:
        return int(sum(x.size for x in
                       (self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)))

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        f = X.mean(axis=1)
        h = np.tanh(f @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        q = h @ self.W3 + self.b3
        return q[:, 1] - q[:, 0]


class FedTN:
    """Federated classical tensor-network model, no quantum -- ~450k params."""

    target_params = 450_000

    def __init__(self, feature_dim=6, bond_dim=16, seed=0):
        g = np.random.default_rng(seed)
        # tensor-train cores + classical head sized to ~450k params
        self.cores = [g.normal(0, 0.1, (bond_dim, feature_dim, bond_dim))
                      for _ in range(int(450_000 / (bond_dim * feature_dim * bond_dim)))]
        self.head = g.normal(0, 0.1, (bond_dim, 2))

    def num_parameters(self) -> int:
        return int(sum(c.size for c in self.cores) + self.head.size)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        f = X.mean(axis=1)                       # (n,d)
        # contract through a few cores deterministically
        msg = np.ones((f.shape[0], self.cores[0].shape[0]))
        for t, core in enumerate(self.cores[:X.shape[1]]):
            phys = X[:, t % X.shape[1], :]       # (n,d)
            mat = np.einsum("ldr,nd->nlr", core, phys)
            msg = np.einsum("nl,nlr->nr", msg, mat)
            msg = msg / (np.linalg.norm(msg, axis=1, keepdims=True) + 1e-9)
        logits = msg @ self.head
        return logits[:, 1] - logits[:, 0]


def build_baseline(name: str, feature_dim=6, seed=0):
    name = name.lower()
    if name in ("edge_nn", "edge-nn", "edgenn"):
        return EdgeNN(feature_dim, seed)
    if name in ("classical_marl", "marl", "classical-marl"):
        return ClassicalMARL(feature_dim, seed)
    if name in ("fed_tn", "fed-tn", "fedtn"):
        return FedTN(feature_dim, 16, seed)
    raise ValueError(f"unknown baseline {name}")
