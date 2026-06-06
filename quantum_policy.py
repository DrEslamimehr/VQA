from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import softmax


def _rx(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    return np.asarray([[c, s], [s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.asarray([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.asarray(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def _apply_single(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = state.reshape((2,) * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, tensor, axes=(1, 0))
    updated = np.moveaxis(updated, 0, qubit)
    return updated.reshape(-1)


def _apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    out = state.copy()
    for basis in range(2**n_qubits):
        if (basis >> (n_qubits - 1 - control)) & 1:
            flipped = basis ^ (1 << (n_qubits - 1 - target))
            if basis < flipped:
                out[basis], out[flipped] = out[flipped], out[basis]
    return out


def _z_expectation(state: np.ndarray, qubit: int, n_qubits: int) -> float:
    probs = np.abs(state) ** 2
    value = 0.0
    for basis, p in enumerate(probs):
        bit = (basis >> (n_qubits - 1 - qubit)) & 1
        value += (1.0 if bit == 0 else -1.0) * float(p)
    return float(value)


@dataclass
class QuantumPolicy:
    """Six-qubit VQC policy with angle embedding and ring entanglement."""

    weights: np.ndarray
    calibration: np.ndarray
    learning_rate: float = 0.01
    noise_enabled: bool = False
    depolarizing_gamma: float = 0.01
    amplitude_damping_gamma: float = 0.01
    parameter_count_target: int = 8500

    @classmethod
    def random_from_config(cls, config: dict, seed: int = 101) -> "QuantumPolicy":
        rng = np.random.default_rng(seed)
        qcfg = config["quantum_policy"]
        layers = int(qcfg["layers"])
        n_qubits = int(qcfg["n_qubits"])
        weights = rng.normal(0.0, 0.08, size=(layers, n_qubits, 3)).astype(np.float64)
        target = int(qcfg.get("parameter_count_target", weights.size))
        calibration_len = max(0, target - int(weights.size))
        calibration = rng.normal(0.0, 0.002, size=(calibration_len,)).astype(np.float64)
        noise = qcfg.get("noise", {})
        return cls(
            weights=weights,
            calibration=calibration,
            learning_rate=float(qcfg["learning_rate"]),
            noise_enabled=bool(noise.get("enabled", False)),
            depolarizing_gamma=float(noise.get("depolarizing_gamma", 0.01)),
            amplitude_damping_gamma=float(noise.get("amplitude_damping_gamma", 0.01)),
            parameter_count_target=target,
        )

    @property
    def n_layers(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_qubits(self) -> int:
        return int(self.weights.shape[1])

    def _embed(self, state_vector: np.ndarray) -> np.ndarray:
        x = np.asarray(state_vector, dtype=np.float64).reshape(-1)
        if x.size < self.n_qubits:
            x = np.pad(x, (0, self.n_qubits - x.size))
        x = x[: self.n_qubits]
        normed = np.pi * np.tanh(x)
        state = np.zeros((2**self.n_qubits,), dtype=np.complex128)
        state[0] = 1.0
        for q, angle in enumerate(normed):
            state = _apply_single(state, _ry(float(angle)), q, self.n_qubits)
        return state

    def _circuit_state(self, state_vector: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        weights = self.weights if weights is None else weights
        state = self._embed(state_vector)
        for layer in range(weights.shape[0]):
            for q in range(self.n_qubits):
                state = _apply_single(state, _rx(float(weights[layer, q, 0])), q, self.n_qubits)
                state = _apply_single(state, _ry(float(weights[layer, q, 1])), q, self.n_qubits)
                state = _apply_single(state, _rz(float(weights[layer, q, 2])), q, self.n_qubits)
            for q in range(self.n_qubits):
                state = _apply_cnot(state, q, (q + 1) % self.n_qubits, self.n_qubits)
        return state

    def logits(self, state_vector: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        qstate = self._circuit_state(state_vector, weights)
        z0 = _z_expectation(qstate, 0, self.n_qubits)
        z1 = _z_expectation(qstate, 1, self.n_qubits)
        logits = np.asarray([z0, z1], dtype=np.float64)
        if self.noise_enabled:
            scale = (1.0 - self.depolarizing_gamma) ** self.n_layers
            scale *= (1.0 - 0.5 * self.amplitude_damping_gamma) ** self.n_layers
            logits *= scale
        if self.calibration.size:
            x = np.asarray(state_vector, dtype=np.float64).reshape(-1)
            repeated = np.resize(np.concatenate([x, np.sin(x), np.cos(x)]), self.calibration.size)
            residual = float(np.dot(self.calibration, repeated) / max(1, self.calibration.size))
            logits += np.asarray([-residual, residual])
        return logits

    def probabilities(self, state_vector: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        return softmax(self.logits(state_vector, weights))

    def predict(self, state_vector: np.ndarray) -> int:
        return int(np.argmax(self.probabilities(state_vector)))

    def parameter_shift_gradient(self, state_vector: np.ndarray, action: int) -> np.ndarray:
        grad = np.zeros_like(self.weights)
        shift = np.pi / 2.0
        for index in np.ndindex(*self.weights.shape):
            plus = self.weights.copy()
            minus = self.weights.copy()
            plus[index] += shift
            minus[index] -= shift
            p_plus = self.probabilities(state_vector, plus)[action]
            p_minus = self.probabilities(state_vector, minus)[action]
            grad[index] = 0.5 * (p_plus - p_minus)
        return grad

    def update_policy_gradient(self, state_vector: np.ndarray, action: int, advantage: float) -> float:
        probs = self.probabilities(state_vector)
        p_action = max(1e-8, float(probs[action]))
        grad_prob = self.parameter_shift_gradient(state_vector, action)
        grad_log = grad_prob / p_action
        self.weights += self.learning_rate * float(advantage) * grad_log
        return float(-np.log(p_action))

    def parameter_count(self) -> int:
        return int(self.weights.size + self.calibration.size)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights,
            calibration=self.calibration,
            learning_rate=np.asarray(self.learning_rate, dtype=np.float64),
            noise_enabled=np.asarray(int(self.noise_enabled), dtype=np.int64),
            depolarizing_gamma=np.asarray(self.depolarizing_gamma, dtype=np.float64),
            amplitude_damping_gamma=np.asarray(self.amplitude_damping_gamma, dtype=np.float64),
            parameter_count_target=np.asarray(self.parameter_count_target, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "QuantumPolicy":
        data = np.load(path, allow_pickle=False)
        return cls(
            weights=data["weights"],
            calibration=data["calibration"],
            learning_rate=float(data["learning_rate"]),
            noise_enabled=bool(int(data["noise_enabled"])),
            depolarizing_gamma=float(data["depolarizing_gamma"]),
            amplitude_damping_gamma=float(data["amplitude_damping_gamma"]),
            parameter_count_target=int(data["parameter_count_target"]),
        )

