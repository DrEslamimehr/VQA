#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qatm.config import load_config
from qatm.training import create_reference_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic QA-TM reference checkpoints.")
    parser.add_argument("--config", default="configs/paper_qatm.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    out = create_reference_checkpoint(config, seed=args.seed)
    print(f"created checkpoint bundle in {out}")


if __name__ == "__main__":
    main()

