"""Real WESAD feature extraction (ECG/EDA/EMG/Resp -> 6-D embedding).

Activated only when a real WESAD release is present (see datasets.load_real_wesad).
The per-window features mirror the synthetic generator's layout so the rest of
the pipeline is identical for synthetic and real data.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

from .datasets import FEATURE_DIM, DatasetBundle, Split

# WESAD chest-sensor sampling rate (RespiBAN) is 700 Hz; label 1=baseline, 2=stress.
FS = 700
WIN = 5  # seconds


def _bandpower_features(seg: np.ndarray) -> np.ndarray:
    """Compress a multichannel segment into a FEATURE_DIM feature vector."""
    feats = [
        np.mean(seg), np.std(seg),
        np.mean(np.abs(np.diff(seg, axis=0))),
        np.percentile(seg, 75) - np.percentile(seg, 25),
        np.max(seg) - np.min(seg),
        float(np.mean(seg ** 2)),
    ]
    v = np.array(feats[:FEATURE_DIM], dtype=np.float32)
    # Min-max normalize to [0,1] to match AngleEmbedding's expected range.
    rng = np.ptp(v) or 1.0
    return (v - v.min()) / rng


def extract_wesad_bundle(pkls: List[Path], seq_len: int, seed: int) -> DatasetBundle:
    gen = np.random.default_rng(seed)
    X, y = [], []
    for p in pkls:
        with open(p, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")
        sig = data["signal"]["chest"]
        labels = np.asarray(data["label"]).ravel()
        ecg = np.asarray(sig["ECG"]).ravel()
        eda = np.asarray(sig["EDA"]).ravel()
        emg = np.asarray(sig["EMG"]).ravel()
        resp = np.asarray(sig["Resp"]).ravel()
        step = FS * WIN
        for start in range(0, len(ecg) - step * seq_len, step * seq_len):
            window = []
            ok = True
            lab_window = labels[start:start + step * seq_len]
            # keep only baseline(1)/stress(2)
            maj = np.bincount(lab_window[(lab_window == 1) | (lab_window == 2)]
                              if np.any((lab_window == 1) | (lab_window == 2))
                              else np.array([0])).argmax()
            if maj not in (1, 2):
                continue
            for t in range(seq_len):
                s = start + t * step
                e = s + step
                seg = np.stack([
                    ecg[s:e], eda[s:e], emg[s:e], resp[s:e]
                ], axis=1)
                if seg.shape[0] < step:
                    ok = False
                    break
                window.append(_bandpower_features(seg))
            if not ok:
                continue
            X.append(np.stack(window))
            y.append(1 if maj == 2 else 0)  # stress -> anomaly
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    idx = gen.permutation(len(y))
    X, y = X[idx], y[idx]
    n = len(y)
    a, b = int(0.6 * n), int(0.8 * n)
    return DatasetBundle(
        name="WESAD",
        train=Split(X[:a], y[:a]),
        val=Split(X[a:b], y[a:b]),
        test=Split(X[b:], y[b:]),
    )
