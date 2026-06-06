# Reproducibility Contract

This repository reconstructs QA-TM from the paper text. It does not contain the
original supplementary source code, checkpoints, dataset splits, or hardware logs.

## Paper-Specified Values Implemented Directly

- Wearable node feature model: INT8 1D-CNN with three convolution layers.
- Edge memory: Matrix Product State tensor memory.
- Tensor bond dimension: `chi = 16` in `configs/paper_qatm.yaml`.
- Quantum policy: 6 qubits, angle embedding, 4 strongly entangling layers.
- Entanglement: ring-topology CNOT pattern.
- Policy optimizer: Adam-style learning rate setting `0.01`; the clean-room
  implementation uses parameter-shift policy gradient.
- Training episodes: `500`.
- Batch size: `32`.
- Discount factor: `0.99`.
- NISQ noise config: depolarizing and amplitude damping gamma `0.01`.
- Differential privacy config: epsilon `2.0`, delta `1e-5`.

## Explicit Reconstruction Choices

The paper omits the following. This repository makes deterministic choices in
config so they can be audited and changed:

- WESAD/PhysioNet train/test split.
- Window size and stride.
- Signal filtering and resampling details.
- PhysioNet dataset identity. We use Challenge 2015 because it matches the
  ECG plus PPG/pulsatile arrhythmia alarm description.
- Exact seed list.
- Exact trained tensor values.
- ESP32 current/voltage measurement source.

## Exact Results

To obtain exactly the original paper's table values, the missing supplementary
files are required. Without them, this implementation can reproduce the method,
the workflow, and the reporting format, and can measure clean-room results on
the configured datasets.

## Commands

Synthetic smoke test:

```bash
python scripts/prepare_datasets.py --config configs/synthetic_smoke.yaml
python scripts/create_reference_checkpoints.py --config configs/synthetic_smoke.yaml
python scripts/evaluate_qatm.py --config configs/synthetic_smoke.yaml
python scripts/benchmark_edge.py --config configs/synthetic_smoke.yaml
```

Real-data run:

```bash
python scripts/download_datasets.py --physionet2015
python scripts/download_datasets.py --wesad
python scripts/prepare_datasets.py --config configs/paper_qatm.yaml
python scripts/train_qatm.py --config configs/paper_qatm.yaml
python scripts/evaluate_qatm.py --config configs/paper_qatm.yaml
python scripts/benchmark_edge.py --config configs/paper_qatm.yaml --hardware-log measurements/esp32_power.csv
```

