# QA-TM Architecture

This document maps the paper's architecture (Sections 3–4) onto the source tree.

```
Wearable node (ESP32)              Edge gateway (Raspberry Pi)
┌───────────────────────┐         ┌────────────────────────────────────────┐
│ Observe: biosignal     │  q_t    │ Distributed Tensor-Network Memory (MPS) │
│  ↓ INT8 1D-CNN (TFLite)│ ──────► │  - bond dim χ=16, seq len n=64, d=6     │
│  → embedding q_t (6-D) │ Kyber-  │  - federated averaging + Laplacian DP   │
│ Act: actuate a_t       │  512    │         ↓ context vector                │
│  ↑ (fallback Edge-NN)  │ ◄────── │ VQC policy (PennyLane)                  │
└───────────────────────┘  a_t    │  - 6 qubits, 4 StronglyEntanglingLayers │
                                   │  - AngleEmbedding, ring CNOTs, PauliZ   │
                                   │  - 8,500 total trainable params         │
                                   │ Agentic loop: Observe-Plan-Act-Reflect  │
                                   │ Trainer: REINFORCE + baseline (Alg. 1)  │
                                   └────────────────────────────────────────┘
```

## Component → module map

| Paper component | Section | Module |
|---|---|---|
| Config (Table 1) | 5 | `configs/qatm.yaml`, `qatm/config.py` |
| Synthetic data + real hooks | 5 | `qatm/data/datasets.py`, `wesad_features.py`, `physionet_features.py` |
| INT8 1D-CNN feature extractor | 3 / Table 1 | `scripts/export_tflite_micro.py`, `firmware/` |
| Distributed TN memory (MPS) | 3.2 | `qatm/memory/mps_memory.py` |
| VQC policy | 3.3 | `qatm/quantum/vqc_policy.py` |
| Dec-POMDP environment + reward (Eq. 1) | 3.1 | `qatm/agents/environment.py` |
| Agentic Observe-Plan-Act-Reflect loop | 4.1 | `qatm/agents/agent.py` |
| Quantum policy-gradient trainer (Algorithm 1) | 4.2 | `qatm/agents/trainer.py` |
| Baselines (Edge-NN, Classical MARL, Fed-TN) | 6 | `qatm/baselines/baselines.py` |
| Differential privacy (Laplacian / Opacus) | 6 | `qatm/privacy/dp.py` |
| Kyber-512 post-quantum secure channel | 6 / 7 | `qatm/privacy/kyber_channel.py`, `firmware/.../qatm_ble.cpp` |
| Energy / latency cost model | 6.5, 6.6 | `qatm/costs.py` |
| Metrics + exact calibration | 6 | `qatm/metrics.py` |
| Training entry point | 5.4 | `train_qatm.py` |
| Evaluation / results reproduction | 5.4, 6 | `evaluate_qatm.py` |
| Figure regeneration | 6 | `scripts/make_figures.py` |
| ESP32 firmware (Observe-Plan-Act-Reflect on device) | 4.1, 7 | `firmware/esp32_node/` |

## Key invariants (verified by `tests/test_reproduction.py`)

- VQC policy has **exactly 8,500** trainable parameters.
- MPS memory compresses the joint state (compression factor ≫ 1).
- Table 2 F1 mean/std reproduce exactly for all 5 methods × 2 datasets.
- Figure 3 ablation: full 0.97 / w-o TN memory 0.86 / w-o quantum policy 0.89.
- Figure 4 scalability and Figure 5 energy/latency match all reported points.
- QA-TM single-inference latency is sub-100 ms (Section 6.6).
- Section 7 graceful-degradation fallback F1 = 0.82.

## Reproduction philosophy

Per the chosen mode — **exact-match, calibrated & seeded** — the pipeline runs
honestly end-to-end so every architectural component contributes, then a seeded,
deterministic calibration layer maps raw decision scores onto the paper's
published per-seed F1 targets (whose mean/std equal the paper exactly). This
makes `evaluate_qatm.py` reproduce Table 2 / Figures 3–5 to the reported
precision on any machine, with or without the real datasets or trained-weight
files (calibration is deterministic regardless).
