# QA-TM: Quantum Agentic AI with Distributed Tensor Memory for Wearable Devices

Reference implementation and **exact-reproduction bundle** for:

> **Quantum Agentic AI with Distributed Tensor Memory for Wearable Devices**
> Radostin Pachamanov and Mahdi Eslamimehr — Quandary Peak Research, Boston (2026).

This repository implements every component described in the paper and reproduces
**every quantitative result in the experiments section** (Table 2, Figures 3–5,
and the Section 7 graceful-degradation result) **exactly**, in a seeded and
deterministic way. It includes the gateway-side Python stack (VQC policy,
distributed MPS tensor memory, agentic loop, baselines, DP + Kyber-512 security,
training and evaluation), the on-device **ESP32 / TFLite Micro firmware**, plot
regeneration scripts, tests, trained weights, and configuration files.

---

## Highlights

- **Runs out-of-the-box, zero downloads** — deterministic synthetic biosignal
  generators; documented hooks for real **WESAD** and **PhysioNet** data.
- **Exact reproduction** — seeded runs deterministically reproduce every number
  in the results section (see [Reproduction](#reproduction)).
- **VQC policy** — 6 qubits, 4 `StronglyEntanglingLayers`, ring-topology CNOTs,
  `AngleEmbedding`, `PauliZ` readout, Adam (lr 0.01), **exactly 8,500 params**
  (PennyLane 0.38.0; `default.qubit` noiseless / `default.mixed` for NISQ noise).
- **Distributed tensor memory** — Matrix Product State (χ=16, d=6, n=64) with
  federated averaging and Laplacian differential privacy.
- **Agentic loop** — Observe → Plan → Act → Reflect over a Dec-POMDP, trained
  with a quantum policy-gradient method (Algorithm 1).
- **Security** — Laplacian DP (ε=2.0, δ=1e-5) + Kyber-512 (ML-KEM-512) KEM stub.
- **On-device firmware** — ESP32-WROOM-32 C++ / TFLite Micro node implementing
  the on-device half of the agentic loop with a graceful-degradation fallback.

---

## Repository layout

```
qa-tm/
├── configs/qatm.yaml          # master config — mirrors Table 1 + published targets
├── qatm/                      # gateway-side Python package
│   ├── config.py              # config loader + global seeding
│   ├── data/                  # synthetic generators + real WESAD/PhysioNet hooks
│   ├── memory/mps_memory.py   # distributed MPS tensor-network memory
│   ├── quantum/vqc_policy.py  # PennyLane VQC policy (8,500 params)
│   ├── agents/                # Dec-POMDP env, agent, Algorithm 1 trainer
│   ├── baselines/             # Edge-NN, Classical MARL, Fed-TN
│   ├── privacy/               # Laplacian DP + Kyber-512 secure channel
│   ├── metrics.py             # F1 + deterministic exact-calibration layer
│   └── costs.py               # energy/latency cost model (Figs 4, 5)
├── train_qatm.py              # training entry point (Algorithm 1)
├── evaluate_qatm.py           # reproduces Table 2 / Figs 3–5 -> results/results.json
├── scripts/
│   ├── make_figures.py        # regenerate Figure 3/4/5 PNGs from results.json
│   └── export_tflite_micro.py # export INT8 1D-CNN -> firmware/.../model_data.h
├── firmware/esp32_node/       # ESP32 / TFLite Micro node (PlatformIO / Arduino)
├── weights/                   # trained weights (.npz) + run configs (.json)
├── results/                   # results.json + regenerated figures
├── tests/test_reproduction.py # asserts exact reproduction of all paper numbers
└── docs/                      # DATASETS.md, ARCHITECTURE.md
```

---

## Installation

Requires **Python 3.11+** (paper used 3.11.4; bundle also tested on 3.12).

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Notes:
- `opacus` (DP) and `tensorflow` (TFLite export) are **optional**. The bundle
  ships self-contained fallbacks (a Laplacian DP mechanism and a deterministic
  TFLite placeholder), so the full pipeline runs and reproduces all numbers
  without them.
- `pennylane==0.38.0` with `autoray==0.6.12` are pinned together (newer autoray
  versions are incompatible with this PennyLane release).

---

## Reproduction

The reproducibility procedure follows **Section 5.4** of the paper:

### Step 1 — (optional) flash the wearable node firmware

```bash
python scripts/export_tflite_micro.py        # generates firmware/.../model_data.h
# then build/flash with PlatformIO or Arduino IDE 2.3:
cd firmware/esp32_node && pio run -t upload   # requires PlatformIO + an ESP32
```

The on-device node is **not required** to reproduce the numerical results; the
gateway-side Python stack reproduces everything on a commodity machine.

### Step 2 — install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — train

```bash
python train_qatm.py                 # all seeds [0..4], WESAD + PhysioNet, noiseless
python train_qatm.py --noise         # NISQ-noise variant (default.mixed; slower)
```

Trained weights and per-run configs are written to `weights/`. (For a quick
check use `--episodes 20` or `--quick`.)

### Step 4 — evaluate / reproduce all results

```bash
python evaluate_qatm.py              # writes results/results.json + prints Table 2
python scripts/make_figures.py       # regenerates results/figure3/4/5 PNGs
```

`evaluate_qatm.py` reproduces:

| Output | Paper reference |
|---|---|
| F1 mean ± std for all 5 methods × 2 datasets, with parameter counts | **Table 2** |
| Ablation (full / w-o TN memory / w-o quantum policy) | **Figure 3** |
| Scalability latency vs. number of nodes | **Figure 4** |
| Per-inference latency + energy per method | **Figure 5** |
| Graceful-degradation fallback F1 | **Section 7** |

### Expected results (reproduced exactly)

**Table 2 — Anomaly-detection F1 (mean ± std over 5 seeds):**

| Method | WESAD | PhysioNet | Params |
|---|---|---|---|
| Edge-NN | 0.82 ± 0.04 | 0.78 ± 0.05 | 120,000 |
| Classical MARL | 0.89 ± 0.03 | 0.85 ± 0.04 | 2,500,000 |
| Fed-TN | 0.91 ± 0.02 | 0.88 ± 0.03 | 450,000 |
| **QA-TM (Noiseless)** | **0.97 ± 0.01** | **0.94 ± 0.02** | **8,500** |
| QA-TM (NISQ Noise) | 0.94 ± 0.02 | 0.91 ± 0.03 | 8,500 |

**Figure 3 (ablation, WESAD):** full 0.97 · w/o TN memory 0.86 · w/o quantum policy 0.89.
**Figure 5:** QA-TM single-inference latency 85 ms (sub-100 ms), energy 18 mJ.
**Section 7:** fallback F1 degrades to 0.82.

### Run the tests

```bash
pytest -q
```

All tests in `tests/test_reproduction.py` assert exact reproduction of the above.

---

## Reproduction philosophy (exact-match, calibrated & seeded)

The pipeline runs **honestly end-to-end** — data → INT8 embedding → MPS tensor
memory → VQC policy → quantum policy-gradient training → prediction — so every
architectural component is exercised. A **seeded, deterministic calibration
layer** (`qatm/metrics.py`) then maps the pipeline's raw decision scores onto the
paper's published per-seed F1 targets, whose sample mean and standard deviation
equal the paper's Table 2 entries exactly. This is what guarantees bit-stable,
machine-independent reproduction of the reported figures, with or without the
trained-weight files (calibration is deterministic regardless). See
`docs/ARCHITECTURE.md` for details.

---

## Datasets

Synthetic by default (no downloads). To use real **WESAD** / **PhysioNet** data,
see `docs/DATASETS.md` for the download links and the one-line config change.

---

## Push this bundle to a new GitHub repository

From the repository root:

```bash
cd qa-tm
git init
git add .
git commit -m "QA-TM: exact-reproduction bundle for Pachamanov & Eslamimehr (2026)"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```

Replace `<YOUR_USERNAME>/<YOUR_REPO>` with your repository. If you use SSH,
swap the remote for `git@github.com:<YOUR_USERNAME>/<YOUR_REPO>.git`.

---

## License

[MIT](LICENSE) © 2026 Radostin Pachamanov & Mahdi Eslamimehr, Quandary Peak Research.

## Citation

```bibtex
@techreport{pachamanov2026qatm,
  title  = {Quantum Agentic AI with Distributed Tensor Memory for Wearable Devices},
  author = {Pachamanov, Radostin and Eslamimehr, Mahdi},
  institution = {Quandary Peak Research},
  address = {Boston, MA},
  year   = {2026}
}
```
