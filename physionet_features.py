"""Real PhysioNet PPG/ECG feature extraction (-> 6-D embedding).

Activated only when real WFDB records are present
(see datasets.load_real_physionet). Uses a light-weight header/signal reader so
that ``wfdb`` is an optional dependency; if ``wfdb`` is installed it is used,
otherwise a minimal ``.dat`` int16 reader is attempted.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .datasets import FEATURE_DIM, DatasetBundle, Split


def _read_record(hea: Path) -> np.ndarray:
    try:
        import wfdb  # type: ignore

        rec = wfdb.rdrecord(str(hea.with_suffix("")))
        return np.asarray(rec.p_signal, dtype=np.float32)
    except Exception:
        dat = hea.with_suffix(".dat")
        raw = np.fromfile(dat, dtype=np.int16).astype(np.float32)
        return raw.reshape(-1, 1)


def _features(seg: np.ndarray) -> np.ndarray:
    feats = [
        np.mean(seg), np.std(seg),
        np.mean(np.abs(np.diff(seg, axis=0))),
        np.percentile(seg, 90) - np.percentile(seg, 10),
        float(np.sqrt(np.mean(seg ** 2))),
        float(np.mean(np.sign(np.diff(seg.ravel())) != 0)),  # zero-cross-ish
    ]
    v = np.array(feats[:FEATURE_DIM], dtype=np.float32)
    rng = np.ptp(v) or 1.0
    return (v - v.min()) / rng


def extract_physionet_bundle(recs: List[Path], seq_len: int, seed: int) -> DatasetBundle:
    gen = np.random.default_rng(seed)
    X, y = [], []
    win = 700 * 5  # 5 s windows at ~700 Hz
    for hea in recs:
        sig = _read_record(hea)
        # Heuristic arrhythmia proxy label from the header comments if present.
        label = 0
        try:
            txt = hea.read_text().lower()
            if any(k in txt for k in ("af", "arrhythm", "abnormal", "pvc")):
                label = 1
        except Exception:
            pass
        flat = sig[:, 0]
        for start in range(0, len(flat) - win * seq_len, win * seq_len):
            window = []
            ok = True
            for t in range(seq_len):
                s = start + t * win
                seg = flat[s:s + win][:, None]
                if seg.shape[0] < win:
                    ok = False
                    break
                window.append(_features(seg))
            if ok:
                X.append(np.stack(window))
                y.append(label)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    idx = gen.permutation(len(y))
    X, y = X[idx], y[idx]
    n = len(y)
    a, b = int(0.6 * n), int(0.8 * n)
    return DatasetBundle(
        name="PhysioNet",
        train=Split(X[:a], y[:a]),
        val=Split(X[a:b], y[a:b]),
        test=Split(X[b:], y[b:]),
    )
