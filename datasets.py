"""Biosignal datasets for QA-TM.

Two access modes (selected via ``config.data.synthetic`` / ``real_data_root``):

1. **Synthetic generators** (default) -- deterministic, dependency-free
   physiological signal simulators that produce class-separable feature
   embeddings. They let the whole repository run end-to-end with *zero*
   downloads while still exercising the exact same downstream pipeline
   (feature embedding -> MPS memory -> VQC policy) as the real data.

2. **Real-data loaders** -- hooks for the genuine WESAD and PhysioNet
   PPG/ECG datasets. WESAD is gated (~5 GB) and PhysioNet must be downloaded
   per its license; see ``docs/DATASETS.md``. When ``real_data_root`` is set
   the loaders read the user-provided files and emit the same feature tensors.

All synthetic signals model the paper's modalities:
  * WESAD     : ECG, EDA, EMG, Respiration (stress vs. baseline)
  * PhysioNet : multi-site PPG + ECG (arrhythmia vs. normal sinus rhythm)

The per-timestep feature embedding has dimension ``d = vqc.qubits`` (=6),
matching "the dimensionality of the compressed state vector s_t".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

FEATURE_DIM = 6  # d == Nq; per-window embedding fed to the quantum feature map


@dataclass
class Split:
    """A single (features, labels) split."""

    X: np.ndarray  # shape (n_samples, seq_len, FEATURE_DIM)
    y: np.ndarray  # shape (n_samples,)  binary {0: normal, 1: anomaly}

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class DatasetBundle:
    name: str
    train: Split
    val: Split
    test: Split
    feature_dim: int = FEATURE_DIM


# ---------------------------------------------------------------------------
# Synthetic biosignal generator
# ---------------------------------------------------------------------------
def _synthetic_windows(
    n: int,
    seq_len: int,
    anomaly_rate: float,
    dataset: str,
    seed: int,
) -> Split:
    """Generate ``n`` windows of synthetic multimodal biosignal *features*.

    Each window is a short trajectory of ``seq_len`` timesteps, each a
    ``FEATURE_DIM``-vector of band-power / morphology features. Anomalous
    windows (stress for WESAD, arrhythmia for PhysioNet) have systematically
    shifted feature statistics, producing a controlled, reproducible class
    separation.
    """
    gen = np.random.default_rng(seed)
    y = (gen.random(n) < anomaly_rate).astype(np.int64)

    # Dataset-specific class-mean shifts in feature space. These are fixed
    # constants (not learned) so the generator is fully deterministic.
    if dataset.upper().startswith("WESAD"):
        normal_mean = np.array([0.30, 0.20, 0.15, 0.55, 0.25, 0.40])
        anom_mean = np.array([0.62, 0.55, 0.48, 0.30, 0.58, 0.20])
        noise_scale = 0.18
    else:  # PhysioNet PPG/ECG arrhythmia
        normal_mean = np.array([0.45, 0.50, 0.40, 0.35, 0.30, 0.52])
        anom_mean = np.array([0.20, 0.78, 0.66, 0.62, 0.58, 0.22])
        noise_scale = 0.20  # arrhythmia is intrinsically noisier -> lower F1

    X = np.empty((n, seq_len, FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        base = anom_mean if y[i] == 1 else normal_mean
        # Temporal structure: a slow drift across the window plus per-step noise.
        drift = np.linspace(-0.05, 0.05, seq_len)[:, None]
        win = base[None, :] + drift + gen.normal(0.0, noise_scale, size=(seq_len, FEATURE_DIM))
        X[i] = np.clip(win, 0.0, 1.0).astype(np.float32)
    return Split(X=X, y=y)


def make_synthetic(
    dataset: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seq_len: int,
    anomaly_rate: float,
    seed: int,
) -> DatasetBundle:
    """Build a deterministic synthetic dataset bundle for ``dataset``."""
    return DatasetBundle(
        name=dataset,
        train=_synthetic_windows(n_train, seq_len, anomaly_rate, dataset, seed + 11),
        val=_synthetic_windows(n_val, seq_len, anomaly_rate, dataset, seed + 22),
        test=_synthetic_windows(n_test, seq_len, anomaly_rate, dataset, seed + 33),
    )


# ---------------------------------------------------------------------------
# Real-data loader hooks (WESAD / PhysioNet)
# ---------------------------------------------------------------------------
def load_real_wesad(root: Path, seq_len: int, seed: int) -> DatasetBundle:
    """Load the real WESAD dataset.

    Expects the official WESAD release laid out as ``<root>/WESAD/S*/S*.pkl``
    (the per-subject pickles distributed by Schmidt et al., 2018). This hook
    extracts ECG/EDA/EMG/Resp windows, computes the ``FEATURE_DIM`` embedding,
    and labels stress (condition 2) vs. baseline (condition 1).

    Implemented as a documented stub: it raises a clear, actionable error if
    the data is absent so that the synthetic path remains the zero-setup
    default. See ``docs/DATASETS.md`` for the download procedure.
    """
    root = Path(root)
    pkls = sorted(root.glob("WESAD/S*/S*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"Real WESAD not found under {root}/WESAD/S*/S*.pkl. "
            "Download WESAD (Schmidt et al., 2018) and point "
            "config.data.real_data_root at the parent directory, or keep "
            "config.data.synthetic: true. See docs/DATASETS.md."
        )
    # NOTE: full RESP/EDA/EMG/ECG feature extraction is provided in
    # qatm/data/wesad_features.py; this branch wires it in when data exists.
    from .wesad_features import extract_wesad_bundle

    return extract_wesad_bundle(pkls, seq_len=seq_len, seed=seed)


def load_real_physionet(root: Path, seq_len: int, seed: int) -> DatasetBundle:
    """Load the real PhysioNet PPG/ECG arrhythmia dataset.

    Expects WFDB records under ``<root>/physionet/`` (``.dat``/``.hea`` pairs).
    Raises an actionable error when absent; see ``docs/DATASETS.md``.
    """
    root = Path(root)
    recs = sorted(root.glob("physionet*/**/*.hea"))
    if not recs:
        raise FileNotFoundError(
            f"Real PhysioNet not found under {root}/physionet*/. "
            "Download the PhysioNet PPG/ECG challenge data and set "
            "config.data.real_data_root, or keep config.data.synthetic: true. "
            "See docs/DATASETS.md."
        )
    from .physionet_features import extract_physionet_bundle

    return extract_physionet_bundle(recs, seq_len=seq_len, seed=seed)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def get_dataset(cfg, dataset: str, seed: int) -> DatasetBundle:
    """Return a :class:`DatasetBundle` for ``dataset`` honoring the config."""
    d = cfg.data
    seq_len = cfg.memory["sequence_length"]
    if d.get("synthetic", True) or not d.get("real_data_root"):
        return make_synthetic(
            dataset=dataset,
            n_train=d["n_train"],
            n_val=d["n_val"],
            n_test=d["n_test"],
            seq_len=seq_len,
            anomaly_rate=d["anomaly_rate"],
            seed=seed,
        )
    root = Path(d["real_data_root"])
    if dataset.upper().startswith("WESAD"):
        return load_real_wesad(root, seq_len, seed)
    return load_real_physionet(root, seq_len, seed)
