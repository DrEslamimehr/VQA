#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qatm.config import load_config
from qatm.data import prepare_task, processed_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QA-TM datasets.")
    parser.add_argument("--config", default="configs/paper_qatm.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"]["seeds"][0])
    for task in config["dataset"]["tasks"]:
        dataset = prepare_task(config, task, seed=seed)
        out = processed_path(config, task)
        dataset.save(out)
        print(f"{task}: train={len(dataset.y_train)} test={len(dataset.y_test)} -> {out}")


if __name__ == "__main__":
    main()

