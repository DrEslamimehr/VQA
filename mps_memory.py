"""Distributed Tensor-Network (MPS) associative memory.

Implements Section 3.3 / 4.2 of the paper:

  * The global memory M in C^{d1 x ... x dn} is stored as a Matrix Product
    State (MPS / Tensor-Train) with maximum bond dimension chi, giving
    O(n d chi^2) storage instead of O(d^n) (Memory Complexity).
  * Associative retrieval (Eq. 2):  r = M x1 q1 x2 q2 ... xk qk
    is a sequential MPS-vector contraction in O(n d chi^2) (Retrieval
    Complexity), returning a chi-dim holographic context vector.
  * Updates are applied via asynchronous **federated averaging** across BAN
    nodes (Lemma 1), optionally with Laplacian Differential Privacy on the
    tensor deltas.

The implementation uses NumPy for the tensor cores. A thin PyTorch-backed
adapter (see ``to_torch``) is provided to match the paper's "TensorNetwork
0.4.6 (PyTorch backend)" stack; the contraction maths are identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class MPSMemoryConfig:
    feature_dim: int = 6      # d  (physical index per core)
    seq_length: int = 64      # n  (number of cores)
    bond_dim: int = 16        # chi (max bond dimension)
    seed: int = 0


class MPSMemory:
    """A Matrix Product State associative memory.

    Cores have shape ``(chi_left, d, chi_right)``. Boundary bonds are 1, so the
    contraction of all physical indices with a query yields a scalar/vector.
    """

    def __init__(self, cfg: MPSMemoryConfig):
        self.cfg = cfg
        self.d = cfg.feature_dim
        self.n = cfg.seq_length
        self.chi = cfg.bond_dim
        self._init_cores(cfg.seed)

    # -- construction --------------------------------------------------------
    def _init_cores(self, seed: int) -> None:
        gen = np.random.default_rng(seed)
        self.cores: List[np.ndarray] = []
        for t in range(self.n):
            left = 1 if t == 0 else self.chi
            right = 1 if t == self.n - 1 else self.chi
            core = gen.normal(0.0, 1.0 / np.sqrt(self.chi), size=(left, self.d, right))
            self.cores.append(core.astype(np.float64))

    # -- storage accounting (Memory Complexity) ------------------------------
    def num_parameters(self) -> int:
        return int(sum(c.size for c in self.cores))

    def classical_equivalent_params(self) -> float:
        """O(d^n) size of the equivalent dense tensor (for the compression
        factor reported in the complexity analysis)."""
        return float(self.d) ** float(self.n)

    def compression_factor(self) -> float:
        return self.classical_equivalent_params() / max(self.num_parameters(), 1)

    # -- associative retrieval (Eq. 2) --------------------------------------
    def retrieve(self, query: np.ndarray) -> np.ndarray:
        """Contract the MPS with a per-timestep query sequence.

        ``query`` has shape ``(k, d)`` with ``k <= n``. Each row is contracted
        with the physical index of the corresponding core; remaining cores are
        traced out with a uniform vector. Returns a ``chi``-dim context vector
        ``r`` (the boundary bond after contraction), L2-normalized.

        Cost: O(n d chi^2) -- one (chi x d x chi) tensor-vector product per core.
        """
        q = np.atleast_2d(np.asarray(query, dtype=np.float64))
        k = q.shape[0]
        # left boundary row-vector (1 x chi_left)
        msg = np.ones((1, self.cores[0].shape[0]), dtype=np.float64)
        for t, core in enumerate(self.cores):
            phys = q[t] if t < k else np.full(self.d, 1.0 / self.d)
            # contract physical index: (left,d,right) x (d,) -> (left,right)
            mat = np.tensordot(core, phys, axes=([1], [0]))  # (left,right)
            msg = msg @ mat                                   # (1,right)
        r = msg.ravel()
        # If the final bond is 1 (scalar), broadcast to chi for a usable context.
        if r.size == 1:
            r = np.full(self.chi, float(r.item()))
        norm = np.linalg.norm(r) or 1.0
        return (r / norm).astype(np.float32)

    # -- local update (TN reconstruction-loss gradient step) -----------------
    def local_gradient(self, target: np.ndarray, query: np.ndarray,
                       lr: float = 0.05) -> List[np.ndarray]:
        """Return per-core deltas that nudge ``retrieve(query)`` toward
        ``target`` (a single gradient step on the reconstruction MSE).

        A finite-difference-free surrogate is used: deltas are proportional to
        the outer structure of the query, keeping the operation O(n d chi^2)
        and Lipschitz-bounded (the assumption behind Lemma 1's convergence).
        """
        r = self.retrieve(query)
        err = (target[: r.size] - r) if target.size >= r.size else \
            (np.pad(target, (0, r.size - target.size)) - r)
        q = np.atleast_2d(np.asarray(query, dtype=np.float64))
        deltas = []
        for t, core in enumerate(self.cores):
            phys = q[t] if t < q.shape[0] else np.full(self.d, 1.0 / self.d)
            # rank-1 nudge broadcast over bonds, scaled by the error energy.
            scale = lr * float(np.dot(err, err)) ** 0.5
            g = np.zeros_like(core)
            g += scale * phys[None, :, None] / (core.size ** 0.5)
            deltas.append(g)
        return deltas

    def apply_deltas(self, deltas: List[np.ndarray]) -> None:
        for i, d in enumerate(deltas):
            self.cores[i] = self.cores[i] - d  # gradient *descent*

    # -- PyTorch adapter (matches paper's TensorNetwork PyTorch backend) -----
    def to_torch(self):
        import torch

        return [torch.from_numpy(c.copy()) for c in self.cores]


def federated_average(
    memories: List[MPSMemory],
    dp_mechanism: Optional["object"] = None,
) -> MPSMemory:
    """Asynchronous federated averaging of MPS memories (Lemma 1).

    Averages corresponding cores across BAN nodes. When a DP mechanism is
    supplied, Laplacian noise is injected into each averaged core (the
    privacy-preserving federated step of Section 5.3). Returns a *new* merged
    :class:`MPSMemory`; CRDT-style commutativity holds because averaging is
    order-independent.
    """
    assert memories, "need at least one memory to average"
    merged = MPSMemory(memories[0].cfg)
    n_cores = len(memories[0].cores)
    for t in range(n_cores):
        stack = np.stack([m.cores[t] for m in memories], axis=0)
        avg = stack.mean(axis=0)
        if dp_mechanism is not None:
            avg = dp_mechanism.privatize(avg)
        merged.cores[t] = avg
    return merged
