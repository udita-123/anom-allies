# Robust Subtle Behaviour Detection in Real-World CCTV

> Research-grade computer vision system for detecting long-duration low-motion suspicious behaviour (loitering, pacing, wandering) in synthetically degraded CCTV footage.

---

## Project Structure

```
cctv_behaviour/
├── configs/
│   └── config.yaml              ← All hyperparameters & settings
│
├── data/
│   ├── raw/                     ← UCSD / Avenue / ShanghaiTech datasets
│   └── processed/frames/        ← Extracted per-video frames
│
├── detection/
│   └── __init__.py              ← YOLOv8 PersonDetector + Detection dataclass
│
├── tracking/
│   └── tracker.py               ← DeepSORT MultiObjectTracker + Trajectory dataclasses
│
├── features/
│   ├── extractor.py             ← Sliding-window feature construction (speed, direction, confinement…)
│   └── lingering_score.py       ← ★ Novel Lingering Score heuristic
│
├── models/
│   ├── autoencoder/
│   │   └── conv_ae.py           ← Convolutional Autoencoder (visual anomaly)
│   ├── lstm_ae/
│   │   └── lstm_ae.py           ← LSTM Sequence Autoencoder (temporal anomaly)
│   └── fusion.py                ← Multi-factor anomaly score fusion
│
├── evaluation/
│   └── evaluator.py             ← ROC-AUC, PR-AUC, FAR, Detection Delay across conditions
│
├── dashboard/
│   └── app.py                   ← Streamlit visualisation dashboard
│
├── utils/
│   └── helpers.py               ← Config loader, frame I/O, DegradationEngine, seed
│
├── pipeline.py                  ← Master orchestrator (train / eval / infer)
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset videos in:
#    data/raw/<dataset_name>/train/   (normal videos only)
#    data/raw/<dataset_name>/test/    (normal + anomalous)

# 3. Train both autoencoders on normal footage
python pipeline.py --mode train --dataset ucsd_ped2

# 4. Evaluate on clean test footage
python pipeline.py --mode eval --dataset ucsd_ped2 --condition clean

# 5. Evaluate under each degradation condition
python pipeline.py --mode eval --dataset ucsd_ped2 --condition motion_blur
python pipeline.py --mode eval --dataset ucsd_ped2 --condition gaussian_noise
python pipeline.py --mode eval --dataset ucsd_ped2 --condition low_light
python pipeline.py --mode eval --dataset ucsd_ped2 --condition compression
python pipeline.py --mode eval --dataset ucsd_ped2 --condition all_combined

# 6. Launch the dashboard
streamlit run dashboard/app.py
```

---

## Pipeline Steps

| Step | File | Purpose |
|------|------|---------|
| 1. Data Prep | `utils/helpers.py` | Frame extraction, degradation |
| 2. Detection | `detection/__init__.py` | YOLOv8 person detection |
| 3. Tracking | `tracking/tracker.py` | DeepSORT + Trajectory building |
| 4. Features | `features/extractor.py` | Speed, direction, confinement windows |
| 5. Visual AE | `models/autoencoder/conv_ae.py` | Conv autoencoder, per-frame score |
| 6. LSTM AE | `models/lstm_ae/lstm_ae.py` | Sequence autoencoder, per-track score |
| 7. Lingering | `features/lingering_score.py` | ★ Novel domain heuristic |
| 8. Fusion | `models/fusion.py` | Weighted combination → final score |
| 9. Evaluation | `evaluation/evaluator.py` | Robustness metrics |
| 10. Dashboard | `dashboard/app.py` | Streamlit visualisation |

---

## Key Research Contributions

1. **Subtle behaviour detection** — loitering / pacing, not violent events
2. **Trajectory-centric modelling** — trajectory features + LSTM, not frame-only CNNs
3. **Novel Lingering Score** — interpretable heuristic combining speed, confinement, duration
4. **Controlled degradation robustness** — systematic evaluation across 5 CCTV impairments
5. **Unified multi-score fusion** — visual + temporal + heuristic → single interpretable score

---

## Configuration

All parameters live in `configs/config.yaml`. Key sections:

| Section | Key parameters |
|---------|---------------|
| `degradation` | blur kernel, noise std, gamma, JPEG quality |
| `detector` | YOLOv8 model variant, confidence threshold |
| `tracker` | DeepSORT max_age, n_init, min trajectory length |
| `features` | window_size, stride, direction_bins |
| `conv_ae` | latent_dim, epochs, threshold_percentile |
| `lstm_ae` | hidden_dim, seq_len, threshold_percentile |
| `lingering` | speed_threshold, confinement_threshold, min_duration |
| `fusion` | weights per component, anomaly_threshold |

---

## Ethical Notes

- **No facial recognition** — bounding boxes only, no identity inference
- **Decision-support only** — human review required before any action
- **Privacy-preserving** — trajectories are coordinate sequences, not images of people
