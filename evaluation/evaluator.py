"""
evaluation/evaluator.py — Robustness Evaluation Framework.
Supports UCSD Ped1 (bmp mask GT) and UCSD Ped2 (frame range GT).
Uses per-condition calibrated thresholds for accurate FAR computation.
Saves PR curves to JSON for plotting.
"""

from __future__ import annotations

import json
import logging
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

logger = logging.getLogger(__name__)

DEFAULT_CFG = {
    "conditions": ["clean", "motion_blur", "gaussian_noise", "low_light", "compression", "all_combined"],
    "save_curves": True,
    "output_dir": "outputs",
    "frame_smoothing": 5,
}

# Per-condition p99 thresholds calibrated on UCSD Ped2 training frames
PER_CONDITION_THRESHOLDS = {
    "clean":          0.007655,
    "motion_blur":    0.007655,
    "compression":    0.007655,
    "gaussian_noise": 0.009500,
    "low_light":      0.136510,
    "all_combined":   0.135620,
}


@dataclass
class EvalMetrics:
    condition:        str
    roc_auc:          float
    pr_auc:           float
    false_alarm_rate: float
    detection_delay:  float
    n_samples:        int
    threshold_used:   float

    def as_dict(self) -> dict:
        return asdict(self)


class RobustnessEvaluator:

    def __init__(self, cfg: dict = None) -> None:
        self.cfg        = {**DEFAULT_CFG, **(cfg or {})}
        self.conditions = self.cfg["conditions"]
        self.output_dir = Path(self.cfg.get("output_dir", "outputs"))
        self._data: dict[str, dict] = {
            c: {"y_true": [], "y_score": [], "delays": []}
            for c in self.conditions
        }
        self._calibrated_threshold: Optional[float] = None
        self._pr_curves: dict = {}

    def evaluate(
        self,
        conv_trainer,
        lstm_trainer,
        frame_paths:        list,
        degradation_engine,
        dataset:            str,
        results:            list = None,
    ) -> list[EvalMetrics]:
        logger.info("[Eval] Starting robustness evaluation ...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not frame_paths:
            logger.warning("[Eval] No frame paths provided.")
            return []

        if hasattr(conv_trainer, "threshold") and conv_trainer.threshold is not None:
            self._calibrated_threshold = conv_trainer.threshold
            logger.info(f"[Eval] Using calibrated threshold: {self._calibrated_threshold:.6f}")
        else:
            logger.warning("[Eval] No calibrated threshold found — FAR will use Youden's J fallback.")

        y_true = self._load_ground_truth(frame_paths, dataset)
        if y_true is None:
            logger.warning("[Eval] No ground-truth labels found. Saving scores only.")
            self._save_scores_only(conv_trainer, frame_paths, dataset)
            return []

        all_metrics: list[EvalMetrics] = []

        for condition in self.conditions:
            logger.info(f"[Eval] Condition: {condition}")
            try:
                if condition == "clean":
                    score_dict = conv_trainer.score(frame_paths)
                else:
                    deg_fn = degradation_engine.get(condition)
                    if deg_fn is None:
                        logger.warning(f"[Eval] No degradation fn for '{condition}' — skipping.")
                        continue
                    score_dict = conv_trainer.score_degraded(frame_paths, deg_fn)

                y_score = np.array(score_dict["scores"], dtype=np.float32)
                smoothing = self.cfg.get("frame_smoothing", 5)
                if smoothing > 1:
                    y_score = self._smooth(y_score, smoothing)

                n        = min(len(y_true), len(y_score))
                y_true_  = y_true[:n]
                y_score_ = y_score[:n]

                self.add(condition, y_true_, y_score_)
                metrics = self.compute(condition)
                all_metrics.append(metrics)

                # Save PR curve data
                if self.cfg.get("save_curves", True):
                    self._save_pr_curve(condition, y_true_, y_score_, dataset)

            except Exception as e:
                logger.error(f"[Eval] Error on condition '{condition}': {e}")
                continue

        if all_metrics:
            report_path = str(self.output_dir / f"{dataset}_eval_report.json")
            self.save(all_metrics, report_path)
            delta = self.degradation_delta(all_metrics)
            if delta:
                logger.info("[Eval] AUC drops vs clean:")
                for cond, drop in delta.items():
                    logger.info(f"       {cond:20s}  ΔAUC = {drop:+.4f}")

        logger.info("[Eval] Evaluation complete.")
        return all_metrics

    # ── Ground Truth Loaders ───────────────────────────────────────────────

    def _load_ground_truth(self, frame_paths, dataset):
        # Try npy first
        candidates = [
            Path(f"data/frames/{dataset}/test/gt_labels.npy"),
            Path(f"data/raw/{dataset}/gt_labels.npy"),
            Path(f"outputs/{dataset}_gt.npy"),
        ]
        for c in candidates:
            if c.exists():
                labels = np.load(str(c))
                logger.info(f"[Eval] Loaded GT from '{c}'")
                return labels.astype(np.float32)

        # Dataset-specific GT builders
        if "ped2" in dataset.lower():
            return self._build_ucsd_ped2_gt(frame_paths)
        elif "ped1" in dataset.lower():
            return self._build_ucsd_ped1_gt(frame_paths, dataset)

        logger.warning(f"[Eval] No GT builder for dataset '{dataset}'")
        return None

    def _build_ucsd_ped2_gt(self, frame_paths):
        """Ped2 GT: hardcoded anomaly frame ranges."""
        RANGES = {
            "Test001": [(61,180)],  "Test002": [(95,180)],
            "Test003": [(1,146)],   "Test004": [(31,180)],
            "Test005": [(1,129)],   "Test006": [(1,159)],
            "Test007": [(46,180)],  "Test008": [(1,180)],
            "Test009": [(1,120)],   "Test010": [(1,150)],
            "Test011": [(1,180)],   "Test012": [(88,180)],
        }
        labels = []
        for fp in frame_paths:
            fp    = Path(fp)
            clip  = fp.parent.name
            try:
                frame_num = int(fp.stem)
            except ValueError:
                frame_num = len(labels) + 1
            if clip in RANGES:
                is_anom = any(s <= frame_num <= e for s, e in RANGES[clip])
                labels.append(1.0 if is_anom else 0.0)
            else:
                labels.append(0.0)
        arr = np.array(labels, dtype=np.float32)
        logger.info(f"[Eval] Built GT from UCSD Ped2 annotations: {int(arr.sum())} anomaly / {len(arr)} total frames")
        return arr

    def _build_ucsd_ped1_gt(self, frame_paths, dataset):
        """
        Ped1 GT: presence of a .bmp file in TestXXX_gt/ means that frame is anomalous.
        Clips without a _gt folder are entirely normal.
        """
        # Build lookup: clip_name → set of anomalous frame numbers
        gt_root = Path(f"data/raw/{dataset}/Test")
        anomaly_frames: dict[str, set] = {}

        gt_dirs = sorted(gt_root.glob("*_gt"))
        if not gt_dirs:
            logger.warning("[Eval] No _gt folders found for Ped1 — check data/raw/ucsd_ped1/Test/")
            return None

        for gt_dir in gt_dirs:
            clip_name = gt_dir.name.replace("_gt", "")  # e.g. Test003_gt → Test003
            frame_nums = set()
            for bmp in gt_dir.glob("*.bmp"):
                try:
                    frame_nums.add(int(bmp.stem))
                except ValueError:
                    pass
            anomaly_frames[clip_name] = frame_nums
            logger.info(f"[Eval] Ped1 GT {clip_name}: {len(frame_nums)} anomalous frames")

        labels = []
        for fp in frame_paths:
            fp    = Path(fp)
            clip  = fp.parent.name
            try:
                frame_num = int(fp.stem)
            except ValueError:
                frame_num = len(labels) + 1

            if clip in anomaly_frames:
                labels.append(1.0 if frame_num in anomaly_frames[clip] else 0.0)
            else:
                labels.append(0.0)  # no GT folder = normal clip

        arr = np.array(labels, dtype=np.float32)
        logger.info(f"[Eval] Built GT from UCSD Ped1 bmp masks: {int(arr.sum())} anomaly / {len(arr)} total frames")
        return arr

    def _save_scores_only(self, conv_trainer, frame_paths, dataset):
        score_dict = conv_trainer.score(frame_paths)
        out = self.output_dir / f"{dataset}_scores.npy"
        np.save(str(out), score_dict["scores"])
        logger.info(f"[Eval] Scores saved → {out}")

    # ── PR Curve Saving ────────────────────────────────────────────────────

    def _save_pr_curve(self, condition: str, y_true: np.ndarray,
                       y_score: np.ndarray, dataset: str) -> None:
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            curve_data = {
                "condition": condition,
                "precision": precision.tolist(),
                "recall":    recall.tolist(),
                "thresholds": thresholds.tolist(),
                "pr_auc":    float(average_precision_score(y_true, y_score)),
            }
            out_path = self.output_dir / f"{dataset}_pr_{condition}.json"
            with open(out_path, "w") as f:
                json.dump(curve_data, f)
        except Exception as e:
            logger.warning(f"[Eval] Could not save PR curve for {condition}: {e}")

    # ── Core Methods ───────────────────────────────────────────────────────

    def add(self, condition, y_true, y_score, event_starts=None, first_detections=None):
        if condition not in self._data:
            self._data[condition] = {"y_true": [], "y_score": [], "delays": []}
        self._data[condition]["y_true"].extend(y_true.tolist())
        self._data[condition]["y_score"].extend(y_score.tolist())
        if event_starts and first_detections:
            delays = [fd - es for es, fd in zip(event_starts, first_detections) if fd >= es]
            self._data[condition]["delays"].extend(delays)

    def compute(self, condition: str) -> EvalMetrics:
        d  = self._data[condition]
        yt = np.array(d["y_true"],  dtype=np.float32)
        ys = np.array(d["y_score"], dtype=np.float32)

        if len(yt) == 0 or yt.sum() == 0:
            return EvalMetrics(condition=condition, roc_auc=0.0, pr_auc=0.0,
                               false_alarm_rate=0.0, detection_delay=0.0,
                               n_samples=len(yt), threshold_used=0.0)

        roc_auc = float(roc_auc_score(yt, ys))
        pr_auc  = float(average_precision_score(yt, ys))

        if condition in PER_CONDITION_THRESHOLDS:
            thr    = PER_CONDITION_THRESHOLDS[condition]
            y_pred = (ys >= thr).astype(int)
        elif self._calibrated_threshold is not None:
            thr    = self._calibrated_threshold
            y_pred = (ys >= thr).astype(int)
        else:
            fpr, tpr, thresholds = roc_curve(yt, ys)
            thr    = float(thresholds[np.argmax(tpr - fpr)])
            y_pred = (ys >= thr).astype(int)

        tn  = int(((y_pred == 0) & (yt == 0)).sum())
        fp  = int(((y_pred == 1) & (yt == 0)).sum())
        far = fp / max(fp + tn, 1)

        delays    = d["delays"]
        avg_delay = float(np.mean(delays)) if delays else float("nan")

        logger.info(
            f"[Eval] {condition:20s}  AUC={roc_auc:.4f}  "
            f"PR={pr_auc:.4f}  FAR={far:.4f}  thr={thr:.6f}"
        )

        return EvalMetrics(
            condition=condition,
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            false_alarm_rate=round(far, 4),
            detection_delay=round(avg_delay, 2),
            n_samples=len(yt),
            threshold_used=round(thr, 6),
        )

    def compute_all(self) -> list[EvalMetrics]:
        return [self.compute(c) for c in self.conditions]

    def save(self, metrics: list[EvalMetrics], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        report = {
            "results": [m.as_dict() for m in metrics],
            "summary": {
                "best_auc_condition":  max(metrics, key=lambda m: m.roc_auc).condition,
                "worst_auc_condition": min(metrics, key=lambda m: m.roc_auc).condition,
                "mean_roc_auc":        round(np.mean([m.roc_auc for m in metrics]), 4),
                "mean_pr_auc":         round(np.mean([m.pr_auc  for m in metrics]), 4),
                "per_condition_thresholds": PER_CONDITION_THRESHOLDS,
            },
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"[Eval] Report saved → {path}")

    def degradation_delta(self, metrics: list[EvalMetrics]) -> dict[str, float]:
        clean = next((m for m in metrics if m.condition == "clean"), None)
        if clean is None:
            return {}
        return {
            m.condition: round(clean.roc_auc - m.roc_auc, 4)
            for m in metrics if m.condition != "clean"
        }

    @staticmethod
    def _smooth(scores: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return scores
        kernel = np.ones(window) / window
        return np.convolve(scores, kernel, mode="same")
