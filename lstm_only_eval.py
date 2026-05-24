"""
lstm_only_eval.py  —  Run this from your project root:
    PYTHONPATH=. python lstm_only_eval.py
"""
import multiprocessing
multiprocessing.freeze_support()

import json
import cv2
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

from models.lstm_ae.lstm_ae import LSTMAETrainer
from detection import PersonDetector
from tracking.tracker import MultiObjectTracker
from utils.helpers import DegradationEngine

# ── config ─────────────────────────────────────────────────────────────────
DATASETS = {
    "ucsd_ped2": {
        "test_dir": "data/frames/ucsd_ped2/test",
        "ckpt":     "models/lstm_ae/lstm_ae_ucsd_ped2.pt",
    },
    "ucsd_ped1": {
        "test_dir": "data/frames/ucsd_ped1/test",
        "ckpt":     "models/lstm_ae/lstm_ae_ucsd_ped1.pt",
    },
}

CONDITIONS = ["clean", "motion_blur", "gaussian_noise", "low_light", "compression", "all_combined"]
SEQ_LEN = 12
OUTPUT_FILE = "outputs/lstm_only_eval.json"

# ── ground truth builders ──────────────────────────────────────────────────
# frame_paths are Path objects like .../Test001/0061.jpg
# clip = p.parent.name, fidx = int(p.stem)

def build_ped2_gt(frame_paths):
    RANGES = {
        "Test001": [(61,180)], "Test002": [(95,180)], "Test003": [(1,146)],
        "Test004": [(31,180)], "Test005": [(1,129)],  "Test006": [(1,159)],
        "Test007": [(46,180)], "Test008": [(1,180)],  "Test009": [(1,120)],
        "Test010": [(1,150)], "Test011": [(1,180)],  "Test012": [(88,180)],
    }
    labels = {}
    for p in frame_paths:
        clip = p.parent.name
        fidx = int(p.stem)
        anom = 0
        for (s, e) in RANGES.get(clip, []):
            if s <= fidx <= e:
                anom = 1
                break
        labels[str(p)] = anom
    return labels

def build_ped1_gt(frame_paths, gt_root="data/raw/ucsd_ped1/Test"):
    labels = {}
    gt_root = Path(gt_root)
    for p in frame_paths:
        clip = p.parent.name
        fidx = int(p.stem)
        gt_dir = gt_root / f"{clip}_gt"
        bmp_files = list(gt_dir.glob(f"{str(fidx).zfill(3)}*.bmp")) if gt_dir.exists() else []
        if not bmp_files:
            bmp_files = list(gt_dir.glob(f"{fidx:04d}*.bmp")) if gt_dir.exists() else []
        labels[str(p)] = 1 if bmp_files else 0
    return labels

# ── helpers ────────────────────────────────────────────────────────────────

def load_frame(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ── main eval ──────────────────────────────────────────────────────────────

def evaluate_lstm_only(dataset_key, cfg):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_key}")
    print(f"{'='*60}")

    test_dir = Path(cfg["test_dir"])
    frame_paths = sorted(
        list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png")),
        key=lambda p: (p.parent.name, int(p.stem))
    )
    print(f"  Found {len(frame_paths)} test frames")
    print(f"  Sample: {frame_paths[0]}  ...  {frame_paths[-1]}")

    trainer = LSTMAETrainer()
    trainer.load(cfg["ckpt"])
    print(f"  LSTM-AE loaded, threshold={trainer.threshold:.6f}")

    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    engine = DegradationEngine(cfg_yaml)
    detector = PersonDetector()

    if dataset_key == "ucsd_ped2":
        gt = build_ped2_gt(frame_paths)
    else:
        gt = build_ped1_gt(frame_paths)

    anomaly_count = sum(gt.values())
    print(f"  GT: {anomaly_count} anomalous / {len(frame_paths)} total frames")

    results = {}

    for condition in CONDITIONS:
        print(f"\n  Condition: {condition} ...", flush=True)
        tracker = MultiObjectTracker()
        tracker.reset()

        for idx, fpath in enumerate(frame_paths):
            frame = load_frame(fpath)
            if frame is None:
                continue
            if condition != "clean":
                if condition == "all_combined":
                    for cond in ["motion_blur", "gaussian_noise", "low_light", "compression"]:
                        frame = engine.apply_single(frame, cond)
                else:
                    frame = engine.apply_single(frame, condition)
            dets = detector.detect(frame, idx)
            tracker.update(dets, idx, frame_rgb=frame)

        trajectories = tracker.get_all_trajectories()
        print(f"    Tracked {len(trajectories)} trajectories")

        # score each frame: max LSTM-AE error of any sequence window ending at that frame
        frame_scores = {str(p): 0.0 for p in frame_paths}

        for traj in trajectories:
            cents = traj.centroids
            fidxs = traj.frame_indices
            if len(cents) < SEQ_LEN:
                continue
            for i in range(len(cents) - SEQ_LEN + 1):
                seq = cents[i:i + SEQ_LEN].astype(np.float32)
                score = float(trainer.score(seq))
                last_abs_idx = fidxs[i + SEQ_LEN - 1]
                if last_abs_idx < len(frame_paths):
                    key = str(frame_paths[last_abs_idx])
                    if key in frame_scores:
                        frame_scores[key] = max(frame_scores[key], score)

        y_true  = np.array([gt[str(p)]          for p in frame_paths])
        y_score = np.array([frame_scores[str(p)] for p in frame_paths])

        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = float("nan")
        try:
            pr_auc = average_precision_score(y_true, y_score)
        except Exception:
            pr_auc = float("nan")

        preds       = (y_score >= trainer.threshold).astype(int)
        normal_mask = (y_true == 0)
        far = float(preds[normal_mask].sum()) / max(normal_mask.sum(), 1)

        print(f"    AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  FAR={far:.4f}")
        results[condition] = {"auc": round(auc, 4), "pr_auc": round(pr_auc, 4), "far": round(far, 4)}

    return results

# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = {}
    for dataset_key, cfg in DATASETS.items():
        all_results[dataset_key] = evaluate_lstm_only(dataset_key, cfg)

    Path("outputs").mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to {OUTPUT_FILE}")
    print("\n=== SUMMARY ===")
    for ds, conds in all_results.items():
        print(f"\n{ds}:")
        print(f"  {'Condition':<20} {'AUC':>7} {'PR-AUC':>8} {'FAR':>7}")
        print(f"  {'-'*44}")
        for cond, m in conds.items():
            print(f"  {cond:<20} {m['auc']:>7.4f} {m['pr_auc']:>8.4f} {m['far']:>7.4f}")