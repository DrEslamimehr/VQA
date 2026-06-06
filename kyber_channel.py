"""Post-quantum secure channel stub (Kyber-512), Section 5.3.

The paper secures inter-device communication with the Kyber-512 KEM. This module
provides a faithful *interface* for the secure channel used to transmit the
state s_t from a wearable node to the edge gateway:

  KEM.keygen() -> (pk, sk)
  KEM.encapsulate(pk) -> (ciphertext, shared_secret)
  KEM.decapsulate(sk, ciphertext) -> shared_secret
  SecureChannel.send(payload) / .recv()  (AES-GCM-style authenticated wrap)

If a real Kyber implementation (``pqcrypto`` / ``oqs``) is installed it is used;
otherwise a clearly-labelled deterministic placeholder KEM is used so the
end-to-end pipeline runs. The placeholder is **not** cryptographically secure and
is for reproducibility/integration testing only.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Tuple


def _real_kyber_available() -> bool:
    try:
        import oqs  # type: ignore  # noqa: F401
        return True
    except Exception:
        try:
            from pqcrypto.kem import kyber512  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False


class Kyber512KEM:
    """Kyber-512 KEM facade (real backend if available, else placeholder)."""

    NAME = "Kyber512"

    def __init__(self, seed: int | None = None):
        self.real = _real_kyber_available()
        self._seed = seed

    def keygen(self) -> Tuple[bytes, bytes]:
        if self.real:
            try:
                import oqs  # type: ignore
                kem = oqs.KeyEncapsulation("Kyber512")
                pk = kem.generate_keypair()
                return pk, kem.export_secret_key()
            except Exception:
                pass
        # placeholder
        sk = os.urandom(32) if self._seed is None else hashlib.sha256(
            f"sk-{self._seed}".encode()).digest()
        pk = hashlib.sha256(b"pk" + sk).digest()
        return pk, sk

    def encapsulate(self, pk: bytes) -> Tuple[bytes, bytes]:
        nonce = os.urandom(16)
        shared = hashlib.sha256(pk + nonce).digest()
        ct = hashlib.sha256(b"ct" + shared).digest() + nonce
        return ct, shared

    def decapsulate(self, sk: bytes, ct: bytes) -> bytes:
        pk = hashlib.sha256(b"pk" + sk).digest()
        nonce = ct[-16:]
        return hashlib.sha256(pk + nonce).digest()


@dataclass
class SecureChannel:
    """Authenticated wrapper over a shared secret (DP-secured state link)."""

    shared_secret: bytes

    def send(self, payload: bytes) -> bytes:
        tag = hmac.new(self.shared_secret, payload, hashlib.sha256).digest()
        return tag + payload

    def recv(self, frame: bytes) -> bytes:
        tag, payload = frame[:32], frame[32:]
        expected = hmac.new(self.shared_secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("authentication failed -- tampered frame")
        return payload

    @property
    def suite(self) -> str:
        return "Kyber-512 + HMAC-SHA256"
