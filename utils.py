from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def zscore(x: np.ndarray, axis: int | tuple[int, ...] | None = None, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - np.mean(x, axis=axis, keepdims=True)) / (np.std(x, axis=axis, keepdims=True) + eps)


def minmax_unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.min(x)
    hi = np.max(x)
    return (x - lo) / (hi - lo + eps)


def sliding_windows(
    signal: np.ndarray,
    labels: np.ndarray,
    window: int,
    stride: int,
    positive_label: int,
    normal_labels: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    normal_set = set(int(v) for v in normal_labels)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for start in range(0, max(0, signal.shape[-1] - window + 1), stride):
        stop = start + window
        label_slice = labels[start:stop]
        if label_slice.size == 0:
            continue
        counts = np.bincount(label_slice.astype(np.int64), minlength=max(positive_label + 1, 8))
        majority = int(np.argmax(counts))
        if majority == positive_label:
            y = 1
        elif majority in normal_set:
            y = 0
        else:
            continue
        xs.append(signal[..., start:stop].astype(np.float32))
        ys.append(y)
    if not xs:
        return np.empty((0,) + tuple(signal.shape[:-1]) + (window,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def resample_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] == target_len:
        return x
    old = np.linspace(0.0, 1.0, x.shape[-1], dtype=np.float32)
    new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    flat = x.reshape((-1, x.shape[-1]))
    out = np.stack([np.interp(new, old, row) for row in flat]).astype(np.float32)
    return out.reshape(x.shape[:-1] + (target_len,))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_test_split_indices(n: int, seed: int, test_fraction: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    test_n = max(1, int(round(n * test_fraction)))
    return idx[test_n:], idx[:test_n]

