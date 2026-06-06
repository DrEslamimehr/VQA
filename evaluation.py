from __future__ import annotations

from pathlib import Path

import numpy as np

from .agent import QATMAgent
from .config import resolve_path
from .data import WindowDataset, load_or_prepare_task
from .metrics import accuracy, binary_confusion, f1_score, summarize
from .utils import load_json, save_json


def evaluate_agent(agent: QATMAgent, dataset: WindowDataset) -> dict:
    preds = np.asarray([agent.act(agent.state(x)) for x in dataset.x_test], dtype=np.int64)
    return {
        "task": dataset.name,
        "n_test": int(dataset.y_test.size),
        "f1": f1_score(dataset.y_test, preds),
        "accuracy": accuracy(dataset.y_test, preds),
        "confusion": binary_confusion(dataset.y_test, preds),
    }


def load_agents(config: dict) -> list[tuple[str, QATMAgent]]:
    checkpoint_dir = resolve_path(config, "checkpoint_dir")
    expected_provenance = config["experiment"].get("provenance")
    agents: list[tuple[str, QATMAgent]] = []
    for policy_file in sorted(checkpoint_dir.glob("*_quantum_policy.npz")):
        prefix = policy_file.name.replace("_quantum_policy.npz", "")
        manifest = checkpoint_dir / f"{prefix}_manifest.json"
        if manifest.exists() and expected_provenance is not None:
            payload = load_json(manifest)
            if payload.get("provenance") != expected_provenance:
                continue
        try:
            agents.append((prefix, QATMAgent.load(checkpoint_dir, prefix=prefix)))
        except FileNotFoundError:
            continue
    if not agents:
        agent = QATMAgent.random_from_config(config, seed=int(config["experiment"]["seeds"][0]))
        agents.append(("untrained_generated", agent))
    return agents


def run_evaluation(config: dict) -> dict:
    report_dir = resolve_path(config, "report_dir")
    agents = load_agents(config)
    rows: list[dict] = []
    for prefix, agent in agents:
        for task in config["dataset"]["tasks"]:
            dataset = load_or_prepare_task(config, task, seed=int(config["experiment"]["seeds"][0]))
            row = evaluate_agent(agent, dataset)
            row["checkpoint"] = prefix
            row["parameter_report"] = agent.parameter_report()
            rows.append(row)
    by_task: dict[str, list[float]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(float(row["f1"]))
    summary = {task: summarize(values) for task, values in by_task.items()}
    payload = {
        "provenance": config["experiment"]["provenance"],
        "measured": rows,
        "summary": summary,
        "paper_reported_table": config["experiment"].get("reported_table", {}),
    }
    save_json(report_dir / "evaluation_summary.json", payload)
    return payload
