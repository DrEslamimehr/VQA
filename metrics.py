from __future__ import annotations

import numpy as np


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    c = binary_confusion(y_true, y_pred)
    precision = c["tp"] / max(1, c["tp"] + c["fp"])
    recall = c["tp"] / max(1, c["tp"] + c["fn"])
    return float(2 * precision * recall / max(1e-12, precision + recall))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float(np.mean(y_true == y_pred)) if y_true.size else 0.0


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0}

