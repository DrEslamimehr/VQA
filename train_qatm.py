#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qatm.config import load_config
from qatm.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QA-TM using the paper-style policy-gradient loop.")
    parser.add_argument("--config", default="configs/paper_qatm.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_training(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

