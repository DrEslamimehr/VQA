from __future__ import annotations

from pathlib import Path

import numpy as np

from .agent import QATMAgent
from .config import resolve_path
from .data import WindowDataset, load_or_prepare_task
from .metrics import f1_score
from .utils import save_json, set_seed


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    replace = n < batch_size
    return rng.choice(n, size=batch_size, replace=replace)


def train_on_dataset(agent: QATMAgent, dataset: WindowDataset, config: dict, seed: int) -> dict:
    rng = set_seed(seed)
    qcfg = config["quantum_policy"]
    episodes = int(qcfg["episodes"])
    batch_size = int(qcfg["batch_size"])
    losses: list[float] = []
    for _episode in range(episodes):
        idx = _batch_indices(dataset.x_train.shape[0], batch_size, rng)
        for i in idx:
            out = agent.infer(dataset.x_train[i])
            action = int(dataset.y_train[i])
            probs = out["probabilities"]
            advantage = 1.0 - float(probs[action])
            losses.append(agent.policy.update_policy_gradient(out["state"], action, advantage))
            reward = 1.0 if int(np.argmax(probs)) == action else -1.0
            agent.memory.update_local(out["embedding"], reward=reward, rng=rng)
    preds = np.asarray([agent.act(agent.state(x)) for x in dataset.x_test], dtype=np.int64)
    return {
        "task": dataset.name,
        "seed": seed,
        "episodes": episodes,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "test_f1": f1_score(dataset.y_test, preds),
    }


def run_training(config: dict) -> dict:
    seeds = list(config["experiment"]["seeds"])
    checkpoint_dir = resolve_path(config, "checkpoint_dir")
    report_dir = resolve_path(config, "report_dir")
    all_results: list[dict] = []
    for seed in seeds:
        agent = QATMAgent.random_from_config(config, seed=seed)
        seed_results = []
        for task in config["dataset"]["tasks"]:
            dataset = load_or_prepare_task(config, task, seed=seed)
            seed_results.append(train_on_dataset(agent, dataset, config, seed=seed))
        prefix = f"qatm_seed_{seed}"
        agent.save(checkpoint_dir, prefix=prefix)
        save_json(
            checkpoint_dir / f"{prefix}_manifest.json",
            {
                "seed": seed,
                "provenance": config["experiment"]["provenance"],
                "training_results": seed_results,
                "parameter_report": agent.parameter_report(),
                "note": "Checkpoint produced by train_qatm.py for the specified config and available datasets.",
            },
        )
        all_results.extend(seed_results)
    payload = {"training": all_results}
    save_json(report_dir / "training_summary.json", payload)
    return payload


def create_reference_checkpoint(config: dict, seed: int | None = None) -> Path:
    seed = int(seed if seed is not None else config["experiment"]["seeds"][0])
    checkpoint_dir = resolve_path(config, "checkpoint_dir")
    agent = QATMAgent.random_from_config(config, seed=seed)
    prefix = "qatm_reference"
    agent.save(checkpoint_dir, prefix=prefix)
    save_json(
        checkpoint_dir / "qatm_reference_manifest.json",
        {
            "seed": seed,
            "provenance": config["experiment"]["provenance"],
            "parameter_report": agent.parameter_report(),
            "note": "Clean-room deterministic checkpoint generated from config, not original supplementary weights.",
        },
    )
    return checkpoint_dir
