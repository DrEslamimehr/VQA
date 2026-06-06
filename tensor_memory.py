from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MatrixProductStateMemory:
    """Distributed tensor memory represented as Matrix Product State cores."""

    cores: list[np.ndarray]
    learning_rate: float = 0.001
    dp_noise_scale: float = 0.001

    @classmethod
    def random_from_config(cls, config: dict, seed: int = 101) -> "MatrixProductStateMemory":
        rng = np.random.default_rng(seed)
        mem_cfg = config["tensor_memory"]
        sites = int(mem_cfg["sites"])
        physical_dim = int(mem_cfg["physical_dim"])
        bond_dim = int(mem_cfg["bond_dim"])
        cores: list[np.ndarray] = []
        for i in range(sites):
            left = 1 if i == 0 else bond_dim
            right = 1 if i == sites - 1 else bond_dim
            core = rng.normal(0.0, 1.0 / np.sqrt(physical_dim * bond_dim), size=(left, physical_dim, right))
            cores.append(core.astype(np.float32))
        return cls(
            cores=cores,
            learning_rate=float(mem_cfg["learning_rate"]),
            dp_noise_scale=float(mem_cfg.get("dp_noise_scale", 0.001)),
        )

    @property
    def sites(self) -> int:
        return len(self.cores)

    @property
    def physical_dim(self) -> int:
        return int(self.cores[0].shape[1])

    @property
    def bond_dim(self) -> int:
        return max(int(core.shape[-1]) for core in self.cores)

    def retrieve(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.size < self.physical_dim:
            q = np.pad(q, (0, self.physical_dim - q.size))
        q = q[: self.physical_dim]
        q = q / (np.linalg.norm(q) + 1e-8)

        left = np.ones((1,), dtype=np.float32)
        context: list[float] = []
        for i, core in enumerate(self.cores):
            local_q = np.roll(q, i)
            transfer = np.tensordot(core, local_q, axes=(1, 0))
            left = left @ transfer
            context.append(float(np.tanh(np.mean(left))))
        return np.asarray(context, dtype=np.float32)

    def update_local(self, embedding: np.ndarray, reward: float, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if q.size < self.physical_dim:
            q = np.pad(q, (0, self.physical_dim - q.size))
        q = q[: self.physical_dim]
        q = q / (np.linalg.norm(q) + 1e-8)
        for i, core in enumerate(self.cores):
            pattern = np.roll(q, i).reshape(1, self.physical_dim, 1)
            noise = rng.laplace(0.0, self.dp_noise_scale, size=core.shape).astype(np.float32)
            self.cores[i] = core + self.learning_rate * float(reward) * pattern + noise

    @staticmethod
    def federated_average(memories: list["MatrixProductStateMemory"]) -> "MatrixProductStateMemory":
        if not memories:
            raise ValueError("at least one memory is required")
        avg_cores = []
        for site in range(memories[0].sites):
            avg_cores.append(np.mean([m.cores[site] for m in memories], axis=0).astype(np.float32))
        return MatrixProductStateMemory(
            avg_cores,
            learning_rate=memories[0].learning_rate,
            dp_noise_scale=memories[0].dp_noise_scale,
        )

    def parameter_count(self) -> int:
        return int(sum(core.size for core in self.cores))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "learning_rate": np.asarray(self.learning_rate, dtype=np.float32),
            "dp_noise_scale": np.asarray(self.dp_noise_scale, dtype=np.float32),
        }
        for i, core in enumerate(self.cores):
            payload[f"core_{i}"] = core
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "MatrixProductStateMemory":
        data = np.load(path, allow_pickle=False)
        cores: list[np.ndarray] = []
        i = 0
        while f"core_{i}" in data:
            cores.append(data[f"core_{i}"])
            i += 1
        return cls(
            cores=cores,
            learning_rate=float(data["learning_rate"]),
            dp_noise_scale=float(data["dp_noise_scale"]),
        )

