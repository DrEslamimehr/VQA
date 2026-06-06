from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import zscore


def _same_conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Small Numpy 1D convolution: x=(channels,time), weight=(out,in,kernel)."""

    channels, time = x.shape
    out_channels, in_channels, kernel = weight.shape
    if in_channels != channels:
        raise ValueError(f"expected {in_channels} input channels, got {channels}")
    pad = kernel // 2
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    y = np.empty((out_channels, time), dtype=np.float32)
    for oc in range(out_channels):
        acc = np.full((time,), float(bias[oc]), dtype=np.float32)
        for ic in range(in_channels):
            for k in range(kernel):
                acc += float(weight[oc, ic, k]) * padded[ic, k : k + time]
        y[oc] = acc
    return y


@dataclass
class QuantizedCNN1D:
    """INT8 three-layer CNN feature extractor following the paper's node model."""

    conv_weights: list[np.ndarray]
    conv_biases: list[np.ndarray]
    dense_weight: np.ndarray
    dense_bias: np.ndarray
    input_scale: float = 0.05
    weight_scale: float = 0.025
    output_dim: int = 6

    @classmethod
    def random_from_config(cls, config: dict, seed: int = 101) -> "QuantizedCNN1D":
        rng = np.random.default_rng(seed)
        layers = config["feature_model"]["layers"]
        in_channels = int(config["dataset"]["synthetic"].get("channels", 5))
        conv_weights: list[np.ndarray] = []
        conv_biases: list[np.ndarray] = []
        for layer in layers:
            out_channels = int(layer["out_channels"])
            kernel = int(layer["kernel"])
            w = rng.integers(-9, 10, size=(out_channels, in_channels, kernel), dtype=np.int8)
            b = rng.integers(-3, 4, size=(out_channels,), dtype=np.int16).astype(np.float32)
            conv_weights.append(w)
            conv_biases.append(b)
            in_channels = out_channels
        output_dim = int(config["feature_model"]["output_dim"])
        dense_weight = rng.normal(0.0, 0.08, size=(output_dim, in_channels)).astype(np.float32)
        dense_bias = rng.normal(0.0, 0.02, size=(output_dim,)).astype(np.float32)
        return cls(conv_weights, conv_biases, dense_weight, dense_bias, output_dim=output_dim)

    def extract(self, window: np.ndarray) -> np.ndarray:
        x = np.asarray(window, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("window must have shape (channels, samples)")
        x = zscore(x, axis=1)
        for w_int8, b in zip(self.conv_weights, self.conv_biases):
            w = w_int8.astype(np.float32) * self.weight_scale
            x = np.tanh(_same_conv1d(x, w, b * self.weight_scale))
            x = 0.5 * (x[:, ::2] + x[:, 1::2]) if x.shape[1] >= 2 else x
        pooled = np.mean(x, axis=1)
        out = self.dense_weight @ pooled + self.dense_bias
        return np.tanh(out).astype(np.float32)

    def batch_extract(self, windows: np.ndarray) -> np.ndarray:
        return np.stack([self.extract(w) for w in windows]).astype(np.float32)

    def parameter_count(self) -> int:
        total = int(self.dense_weight.size + self.dense_bias.size)
        total += sum(int(w.size + b.size) for w, b in zip(self.conv_weights, self.conv_biases))
        return total

    def flash_footprint_bytes(self) -> int:
        conv = sum(w.nbytes + b.astype(np.int16).nbytes for w, b in zip(self.conv_weights, self.conv_biases))
        dense = self.dense_weight.astype(np.float16).nbytes + self.dense_bias.astype(np.float16).nbytes
        tflm_overhead = 36 * 1024
        return int(conv + dense + tflm_overhead)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dense_weight": self.dense_weight,
            "dense_bias": self.dense_bias,
            "input_scale": np.asarray(self.input_scale, dtype=np.float32),
            "weight_scale": np.asarray(self.weight_scale, dtype=np.float32),
            "output_dim": np.asarray(self.output_dim, dtype=np.int64),
        }
        for i, (w, b) in enumerate(zip(self.conv_weights, self.conv_biases)):
            payload[f"conv_{i}_weight"] = w
            payload[f"conv_{i}_bias"] = b
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "QuantizedCNN1D":
        data = np.load(path, allow_pickle=False)
        conv_weights: list[np.ndarray] = []
        conv_biases: list[np.ndarray] = []
        i = 0
        while f"conv_{i}_weight" in data:
            conv_weights.append(data[f"conv_{i}_weight"])
            conv_biases.append(data[f"conv_{i}_bias"])
            i += 1
        return cls(
            conv_weights=conv_weights,
            conv_biases=conv_biases,
            dense_weight=data["dense_weight"],
            dense_bias=data["dense_bias"],
            input_scale=float(data["input_scale"]),
            weight_scale=float(data["weight_scale"]),
            output_dim=int(data["output_dim"]),
        )

