from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def laplace_dp_noise(shape: tuple[int, ...], epsilon: float, sensitivity: float, rng: np.random.Generator) -> np.ndarray:
    scale = sensitivity / max(float(epsilon), 1e-8)
    return rng.laplace(0.0, scale, size=shape).astype(np.float32)


@dataclass
class SecureChannel:
    """Kyber-512 interface with a deterministic development fallback.

    The optional `pqcrypto` package provides the real Kyber implementation.
    The fallback is only for local smoke tests and marks itself as non-PQ.
    """

    mode: str = "auto"

    def provider(self) -> str:
        try:
            import pqcrypto.kem.ml_kem_512  # noqa: F401

            return "ml_kem_512"
        except Exception:
            return "development_xor_fallback"

    def seal(self, payload: bytes, peer_public_key: bytes | None = None) -> tuple[bytes, dict[str, str]]:
        provider = self.provider()
        if provider == "ml_kem_512" and peer_public_key is not None:
            from pqcrypto.kem.ml_kem_512 import encrypt  # type: ignore

            ciphertext, shared_secret = encrypt(peer_public_key)
            mask = np.frombuffer(shared_secret, dtype=np.uint8)
            raw = np.frombuffer(payload, dtype=np.uint8)
            sealed = np.bitwise_xor(raw, np.resize(mask, raw.size)).tobytes()
            return ciphertext + sealed, {"provider": provider, "security": "post_quantum"}
        mask = np.frombuffer(b"qatm-dev-channel", dtype=np.uint8)
        raw = np.frombuffer(payload, dtype=np.uint8)
        sealed = np.bitwise_xor(raw, np.resize(mask, raw.size)).tobytes()
        return sealed, {"provider": provider, "security": "not_post_quantum_dev_only"}

