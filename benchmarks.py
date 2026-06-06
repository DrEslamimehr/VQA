from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from .agent import QATMAgent
from .config import resolve_path
from .data import load_or_prepare_task
from .utils import save_json


def software_latency(agent: QATMAgent, windows: np.ndarray, repetitions: int) -> dict[str, float]:
    sample = windows[: min(len(windows), 8)]
    timings: list[float] = []
    for _ in range(repetitions):
        for window in sample:
            start = time.perf_counter()
            agent.infer(window)
            timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(timings, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
    }


def read_hardware_log(path: str | Path) -> dict[str, float]:
    latencies: list[float] = []
    energies: list[float] = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if "latency_ms" in row:
                latencies.append(float(row["latency_ms"]))
            if "energy_mj" in row:
                energies.append(float(row["energy_mj"]))
    return {
        "hardware_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "hardware_energy_mj": float(np.mean(energies)) if energies else 0.0,
    }


def run_benchmark(config: dict, hardware_log: str | None = None) -> dict:
    agent = QATMAgent.load(resolve_path(config, "checkpoint_dir"), prefix="qatm_reference")
    task = config["dataset"]["tasks"][0]
    dataset = load_or_prepare_task(config, task, seed=int(config["experiment"]["seeds"][0]))
    reps = int(config["benchmark"]["software_repetitions"])
    payload = {
        "software": software_latency(agent, dataset.x_test, reps),
        "paper_budget": {
            "latency_budget_ms": config["benchmark"]["latency_budget_ms"],
            "energy_budget_mj": config["benchmark"]["energy_budget_mj"],
        },
        "paper_reported": config["experiment"].get("reported_table", {}).get("qatm_noiseless", {}),
    }
    if hardware_log:
        payload["hardware_log"] = read_hardware_log(hardware_log)
    save_json(resolve_path(config, "report_dir") / "benchmark_summary.json", payload)
    return payload

