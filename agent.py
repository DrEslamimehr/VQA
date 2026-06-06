from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import QuantizedCNN1D
from .quantum_policy import QuantumPolicy
from .tensor_memory import MatrixProductStateMemory


@dataclass
class QATMAgent:
    """Observe-Reflect-Plan-Act loop from the paper."""

    feature_model: QuantizedCNN1D
    memory: MatrixProductStateMemory
    policy: QuantumPolicy

    @classmethod
    def random_from_config(cls, config: dict, seed: int = 101) -> "QATMAgent":
        return cls(
            feature_model=QuantizedCNN1D.random_from_config(config, seed),
            memory=MatrixProductStateMemory.random_from_config(config, seed + 1),
            policy=QuantumPolicy.random_from_config(config, seed + 2),
        )

    def observe(self, window: np.ndarray) -> np.ndarray:
        return self.feature_model.extract(window)

    def reflect(self, embedding: np.ndarray) -> np.ndarray:
        return self.memory.retrieve(embedding)

    def state(self, window: np.ndarray) -> np.ndarray:
        q = self.observe(window)
        r = self.reflect(q)
        return np.concatenate([q, r]).astype(np.float32)

    def plan(self, state: np.ndarray) -> np.ndarray:
        return self.policy.probabilities(state)

    def act(self, state: np.ndarray) -> int:
        return int(np.argmax(self.plan(state)))

    def infer(self, window: np.ndarray) -> dict:
        q = self.observe(window)
        r = self.reflect(q)
        s = np.concatenate([q, r]).astype(np.float32)
        probs = self.policy.probabilities(s)
        return {
            "embedding": q,
            "memory": r,
            "state": s,
            "probabilities": probs,
            "action": int(np.argmax(probs)),
        }

    def save(self, directory: str | Path, prefix: str = "qatm") -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.feature_model.save(directory / f"{prefix}_feature_model.npz")
        self.memory.save(directory / f"{prefix}_tensor_memory.npz")
        self.policy.save(directory / f"{prefix}_quantum_policy.npz")

    @classmethod
    def load(cls, directory: str | Path, prefix: str = "qatm") -> "QATMAgent":
        directory = Path(directory)
        return cls(
            QuantizedCNN1D.load(directory / f"{prefix}_feature_model.npz"),
            MatrixProductStateMemory.load(directory / f"{prefix}_tensor_memory.npz"),
            QuantumPolicy.load(directory / f"{prefix}_quantum_policy.npz"),
        )

    def parameter_report(self) -> dict[str, int]:
        return {
            "feature_model": self.feature_model.parameter_count(),
            "tensor_memory": self.memory.parameter_count(),
            "quantum_policy": self.policy.parameter_count(),
            "feature_flash_bytes": self.feature_model.flash_footprint_bytes(),
        }

