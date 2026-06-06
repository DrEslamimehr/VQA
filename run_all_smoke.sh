#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_datasets.py --config configs/synthetic_smoke.yaml
python scripts/create_reference_checkpoints.py --config configs/synthetic_smoke.yaml
python scripts/evaluate_qatm.py --config configs/synthetic_smoke.yaml
python scripts/benchmark_edge.py --config configs/synthetic_smoke.yaml

