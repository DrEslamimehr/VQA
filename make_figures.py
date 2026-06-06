#!/usr/bin/env python3
"""Regenerate Figures 3, 4 and 5 from results/results.json.

Run ``evaluate_qatm.py`` first to produce results/results.json, then:
    python scripts/make_figures.py
Figures are written to results/figure3_ablation.png, figure4_scalability.png,
and figure5_energy_latency.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_results():
    with open(RESULTS / "results.json") as fh:
        return json.load(fh)


def figure3(res):
    ab = res["figure3_ablation_wesad"]
    labels = ["Full Model", "w/o TN Memory", "w/o Quantum Policy"]
    keys = ["full_model", "wo_tn_memory", "wo_quantum_policy"]
    vals = [ab[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color="#b3b3f0", edgecolor="#5a5ad6", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                ha="center", color="#1a1a8c", fontsize=11)
    ax.set_ylim(0.78, 1.0)
    ax.set_ylabel("F1-Score")
    ax.set_title("Fig. 3 — Ablation study (WESAD)")
    fig.tight_layout()
    fig.savefig(RESULTS / "figure3_ablation.png", dpi=160)
    plt.close(fig)


def figure4(res):
    s = res["figure4_scalability"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s["nodes"], s["qatm_ms"], "s-", color="blue", label="QA-TM Gateway Latency")
    ax.plot(s["nodes"], s["classical_marl_ms"], "^--", color="red", label="Classical MARL Latency")
    ax.set_xlabel("Number of Wearable Nodes (N)")
    ax.set_ylabel("Average Gateway Latency (ms)")
    ax.set_title("Fig. 4 — Scalability analysis")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure4_scalability.png", dpi=160)
    plt.close(fig)


def figure5(res):
    el = res["figure5_energy_latency"]
    methods = ["edge_nn", "classical_marl", "fed_tn", "qatm"]
    labels = ["Edge-NN", "Class-MARL", "Fed-TN", "QA-TM"]
    lat = [el[m]["latency_ms"] for m in methods]
    eng = [el[m]["energy_mj"] for m in methods]
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    b1 = ax.bar(x - w / 2, lat, w, color="#8c8cf0", label="Latency (ms)")
    b2 = ax.bar(x + w / 2, eng, w, color="#f08c8c", label="Energy (mJ)")
    for bars, vals in ((b1, lat), (b2, eng)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{int(v)}",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms) / Energy (mJ)")
    ax.set_title("Fig. 5 — Per-inference latency & energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure5_energy_latency.png", dpi=160)
    plt.close(fig)


def main():
    res = load_results()
    figure3(res)
    figure4(res)
    figure5(res)
    print(f"Figures written to {RESULTS}/figure3_ablation.png, "
          f"figure4_scalability.png, figure5_energy_latency.png")


if __name__ == "__main__":
    main()
