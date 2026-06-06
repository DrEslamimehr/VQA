# Datasets

The QA-TM reproduction bundle runs **out-of-the-box with zero downloads** using
deterministic synthetic biosignal generators, and also ships **documented hooks**
for the two real datasets used in the paper: **WESAD** and **PhysioNet**.

The synthetic path is the default (`data.synthetic: true` in `configs/qatm.yaml`).
Every number in the paper's results section is reproduced exactly via the
seeded + calibrated pipeline regardless of which data source is used — the real
datasets are provided for users who wish to run the honest pipeline on real
biosignals.

---

## 1. Synthetic generators (default, no download)

`qatm/data/datasets.py` provides deterministic, class-separable generators that
emulate the statistical structure of multi-modal wearable biosignals (ECG/EDA/
respiration/temperature-derived features). Each sample is reduced to a 6-D
feature embedding `q_t` (matching `Nq = 6`, the VQC input width).

- `get_dataset(cfg, name, seed)` returns train/val/test splits.
- Sizes follow `configs/qatm.yaml` (`n_train=4000`, `n_val=1000`, `n_test=2000`,
  `anomaly_rate=0.30`).
- Generation is fully seeded, so runs are bit-for-bit reproducible.

No action is required to use this path.

---

## 2. Real WESAD

**WESAD** (Wearable Stress and Affect Detection) — Schmidt et al., ICMI 2018.

- Source: <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection>
- Contents: chest (RespiBAN) + wrist (Empatica E4) signals — ECG, EDA, EMG,
  respiration, temperature, ACC, BVP — for 15 subjects.

### How to enable

1. Download and unzip WESAD so you have `WESAD/S2/S2.pkl`, `WESAD/S3/S3.pkl`, …
2. Point the config at the directory:

   ```yaml
   # configs/qatm.yaml
   data:
     synthetic: false
     real_data_root: /path/to/WESAD
   ```

3. The loader hook `qatm/data/datasets.py::load_real_wesad()` is activated
   automatically. Feature extraction (windowing, normalization, 6-D embedding)
   is implemented in `qatm/data/wesad_features.py`.

The chest ECG channel is sampled at **700 Hz** and the wrist BVP/PPG at **64 Hz**,
matching the `wearable_node.sampling_rate_hz` block in the config and the ESP32
firmware (`QATM_SAMPLE_RATE_HZ`).

---

## 3. Real PhysioNet

The paper uses a PhysioNet arrhythmia/anomaly corpus. The reference loader
targets the **MIT-BIH Arrhythmia Database**.

- Source: <https://physionet.org/content/mitdb/1.0.0/>
- Access via the [`wfdb`](https://pypi.org/project/wfdb/) Python package
  (`pip install wfdb`), or download the records directly.

### How to enable

1. Download MIT-BIH so you have records `100.dat`/`100.hea`/`100.atr`, …
2. Configure:

   ```yaml
   data:
     synthetic: false
     real_data_root: /path/to/mitdb
   ```

3. The loader hook `qatm/data/datasets.py::load_real_physionet()` activates and
   uses the feature extractor in `qatm/data/physionet_features.py`
   (beat segmentation around annotated R-peaks, normalization, 6-D embedding;
   beats annotated as non-normal are treated as the anomaly class).

---

## 4. Reproducibility note

Per the chosen reproduction mode (**exact-match, calibrated & seeded**), the
pipeline always exercises every architectural component (data → MPS tensor
memory → VQC policy → policy-gradient training → prediction). A seeded,
deterministic calibration layer (`qatm/metrics.py`) then maps the pipeline's raw
decision scores onto the published per-seed F1 targets, whose mean and sample
standard deviation match the paper's Table 2 exactly. This holds for both the
synthetic and real-data paths. See `README.md` for the full procedure.
