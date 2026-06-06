from __future__ import annotations

import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import resolve_path
from .utils import resample_linear, sliding_windows, train_test_split_indices, zscore


@dataclass
class WindowDataset:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            name=np.asarray(self.name),
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
        )

    @classmethod
    def load(cls, path: str | Path) -> "WindowDataset":
        data = np.load(path, allow_pickle=False)
        return cls(
            name=str(data["name"]),
            x_train=data["x_train"],
            y_train=data["y_train"],
            x_test=data["x_test"],
            y_test=data["y_test"],
        )


def synthetic_task(name: str, config: dict, seed: int) -> WindowDataset:
    cfg = config["dataset"]["synthetic"]
    rng = np.random.default_rng(seed + (0 if name == "wesad" else 1000))
    n = int(cfg["samples_per_task"])
    channels = int(cfg["channels"])
    length = int(cfg["window_samples"])
    positive_fraction = float(cfg.get("positive_fraction", 0.5))
    y = (rng.random(n) < positive_fraction).astype(np.int64)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    x = np.empty((n, channels, length), dtype=np.float32)
    for i in range(n):
        label = y[i]
        base_freq = 4.0 + 0.8 * rng.normal()
        anomaly_boost = 2.0 if label else 0.0
        for c in range(channels):
            phase = rng.uniform(0, 2 * np.pi)
            freq = base_freq + c * 0.4 + anomaly_boost * (1.0 if c in (0, 1) else 0.25)
            waveform = np.sin(2 * np.pi * freq * t + phase)
            harmonic = 0.35 * np.sin(2 * np.pi * (freq * 0.5) * t + phase / 2)
            drift = (0.6 if label and c in (2, 3) else 0.15) * (t - 0.5)
            noise = rng.normal(0.0, 0.12 + 0.04 * label, size=length)
            x[i, c] = waveform + harmonic + drift + noise
        if label:
            start = rng.integers(length // 5, max(length // 5 + 1, length // 2))
            stop = min(length, start + length // 8)
            x[i, 0, start:stop] += rng.normal(1.5, 0.2)
            x[i, 1, start:stop] -= rng.normal(1.0, 0.2)
    x = zscore(x, axis=2)
    train_idx, test_idx = train_test_split_indices(n, seed, test_fraction=0.25)
    return WindowDataset(name, x[train_idx], y[train_idx], x[test_idx], y[test_idx])


def _wesad_subject_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def prepare_wesad(config: dict, seed: int) -> WindowDataset:
    raw_root = resolve_path(config, "raw_root") / "WESAD"
    if not raw_root.exists():
        raise FileNotFoundError(f"WESAD root not found: {raw_root}")
    ds_cfg = config["dataset"]
    wesad_cfg = ds_cfg["wesad"]
    sample_rate = int(wesad_cfg["sample_rate_hz"])
    window = int(round(float(ds_cfg["window_seconds"]) * sample_rate))
    stride = int(round(float(ds_cfg["stride_seconds"]) * sample_rate))
    subjects = list(wesad_cfg["subjects"])
    test_subjects = set(wesad_cfg["test_subjects"])
    normal_labels: Iterable[int] = wesad_cfg["normal_labels"]

    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    test_x: list[np.ndarray] = []
    test_y: list[np.ndarray] = []

    for subject in subjects:
        pkl = raw_root / subject / f"{subject}.pkl"
        if not pkl.exists():
            continue
        payload = _wesad_subject_pickle(pkl)
        chest = payload["signal"]["chest"]
        label = np.asarray(payload["label"], dtype=np.int64)
        channels = [
            np.asarray(chest["ECG"]).reshape(-1),
            np.asarray(chest["EDA"]).reshape(-1),
            np.asarray(chest["EMG"]).reshape(-1),
            np.asarray(chest["Resp"]).reshape(-1),
            np.asarray(chest["Temp"]).reshape(-1),
        ]
        length = min(min(c.size for c in channels), label.size)
        signal = np.stack([c[:length] for c in channels]).astype(np.float32)
        x, y = sliding_windows(
            signal,
            label[:length],
            window=window,
            stride=stride,
            positive_label=int(wesad_cfg["stress_label"]),
            normal_labels=normal_labels,
        )
        if subject in test_subjects:
            test_x.append(x)
            test_y.append(y)
        else:
            train_x.append(x)
            train_y.append(y)

    if not train_x or not test_x:
        raise ValueError("WESAD files were found, but no windows were produced")
    return WindowDataset(
        "wesad",
        zscore(np.concatenate(train_x), axis=2),
        np.concatenate(train_y),
        zscore(np.concatenate(test_x), axis=2),
        np.concatenate(test_y),
    )


def _read_physionet_records(root: Path) -> list[tuple[Path, int]]:
    labels: dict[str, int] = {}
    for label_file in [root / "ALARMS", root / "alarms.csv", root / "answers.txt"]:
        if not label_file.exists():
            continue
        for line in label_file.read_text(errors="ignore").splitlines():
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            record = parts[0]
            tail = " ".join(parts[1:]).lower()
            labels[record] = 1 if ("true" in tail or tail in {"1", "t"}) else 0
    records = []
    for header in root.rglob("*.hea"):
        record = header.stem
        if record in labels:
            records.append((header, labels[record]))
    return records


def _read_wfdb_window(header: Path, target_len: int, channels: int = 5) -> np.ndarray | None:
    try:
        import wfdb  # type: ignore

        rec = wfdb.rdrecord(str(header.with_suffix("")))
        data = np.asarray(rec.p_signal, dtype=np.float32).T
    except Exception:
        return None
    if data.ndim != 2 or data.size == 0:
        return None
    if data.shape[0] < channels:
        data = np.pad(data, ((0, channels - data.shape[0]), (0, 0)))
    data = data[:channels]
    center = data.shape[1] // 2
    half = min(data.shape[1] // 2, target_len // 2)
    start = max(0, center - half)
    stop = min(data.shape[1], start + target_len)
    return resample_linear(data[:, start:stop], target_len)


def prepare_physionet2015(config: dict, seed: int) -> WindowDataset:
    raw_root = resolve_path(config, "raw_root") / "physionet_challenge_2015"
    if not raw_root.exists():
        raise FileNotFoundError(f"PhysioNet 2015 root not found: {raw_root}")
    if (raw_root / "training.zip").exists() and not (raw_root / "training").exists():
        with zipfile.ZipFile(raw_root / "training.zip") as zf:
            zf.extractall(raw_root / "training")
    root = raw_root / "training" if (raw_root / "training").exists() else raw_root
    records = _read_physionet_records(root)
    if not records:
        raise ValueError("no labeled PhysioNet 2015 records found")

    cfg = config["dataset"]["synthetic"]
    target_len = int(cfg["window_samples"])
    channels = int(cfg["channels"])
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for header, label in records:
        window = _read_wfdb_window(header, target_len=target_len, channels=channels)
        if window is None:
            continue
        xs.append(window)
        ys.append(label)
    if not xs:
        raise RuntimeError("WFDB was unavailable or no PhysioNet records could be loaded")
    x = zscore(np.stack(xs), axis=2)
    y = np.asarray(ys, dtype=np.int64)
    train_idx, test_idx = train_test_split_indices(len(y), seed, test_fraction=0.25)
    return WindowDataset("physionet2015", x[train_idx], y[train_idx], x[test_idx], y[test_idx])


def processed_path(config: dict, task: str) -> Path:
    experiment = str(config.get("experiment", {}).get("name", "default"))
    return resolve_path(config, "processed_root") / experiment / f"{task}.npz"


def prepare_task(config: dict, task: str, seed: int) -> WindowDataset:
    mode = config["dataset"]["mode"]
    if mode == "synthetic":
        return synthetic_task(task, config, seed)
    if task == "wesad":
        try:
            return prepare_wesad(config, seed)
        except Exception:
            if mode == "real_or_synthetic":
                return synthetic_task(task, config, seed)
            raise
    if task == "physionet2015":
        try:
            return prepare_physionet2015(config, seed)
        except Exception:
            if mode == "real_or_synthetic":
                return synthetic_task(task, config, seed)
            raise
    raise ValueError(f"unknown task: {task}")


def load_or_prepare_task(config: dict, task: str, seed: int) -> WindowDataset:
    path = processed_path(config, task)
    if path.exists():
        return WindowDataset.load(path)
    dataset = prepare_task(config, task, seed)
    dataset.save(path)
    return dataset
