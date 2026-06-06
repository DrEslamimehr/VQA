#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qatm.benchmarks import run_benchmark
from qatm.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark QA-TM latency and optional hardware logs.")
    parser.add_argument("--config", default="configs/paper_qatm.yaml")
    parser.add_argument("--hardware-log", help="CSV with latency_ms and energy_mj columns")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_benchmark(config, hardware_log=args.hardware_log)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

