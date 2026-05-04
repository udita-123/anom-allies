"""
pipeline.py — Robust CCTV Subtle Behaviour Detection Pipeline
=============================================================
Modes   : train | test
Datasets: ucsd_ped1, ucsd_ped2 (frame-based) + generic video datasets

LSTM-AE is trained on raw (x, y) centroid sequences from DeepSORT tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
from glob import glob
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

UCSD_DATASETS = {"ucsd_ped1", "ucsd_ped2"}


# ─────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────

from utils.helpers import load_config, set_seed, extract_frames, DegradationEngine
from detection import PersonDetector
from tracking.tracker import MultiObjectTracker
from features.extractor import extract_all
from features.lingering_score import LingeringScorer
from models.autoencoder.conv_ae import ConvAETrainer
from models.lstm_ae.lstm_ae import LSTMAETrainer
from models.fusion import AnomalyFusion, FusionResult
from evaluation.evaluator import RobustnessEvaluator


# ─────────────────────────────────────────────────────────────
#  UCSD frame collector
# ─────────────────────────────────────────────────────────────

def collect_ucsd_frames(dataset: str, split: str = "train") -> list[Path]:
    root   = f"data/frames/{dataset}/{split}"
    frames = sorted(glob(os.path.join(root, "**", "*.jpg"), recursive=True))
    if not frames:
        frames = sorted(glob(os.path.join(root, "*.jpg")))
    logger.info(f"Collected {len(frames)} frames from '{root}'")
    return [Path(f) for f in frames]


# ─────────────────────────────────────────────────────────────
#  Video dataset helpers
# ─────────────────────────────────────────────────────────────

def get_video_paths(cfg: dict, dataset: str, split: str) -> list[Path]:
    root  = Path(cfg["datasets"]["root"]) / dataset / split
    exts  = {".mp4", ".avi", ".mov", ".mpeg"}
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    logger.info(f"Found {len(paths)} videos in {root}")
    return paths


def get_frame_paths_for_video(video_path: Path, cfg: dict) -> list[Path]:
    frames_dir = Path("data/processed/frames") / video_path.stem
    if not frames_dir.exists() or not any(frames_dir.iterdir()):
        return extract_frames(
            str(video_path), str(frames_dir),
            fps=cfg["datasets"]["fps_extract"],
        )
    return sorted(frames_dir.glob("*.jpg"))


# ─────────────────────────────────────────────────────────────
#  Detection + tracking helper
# ─────────────────────────────────────────────────────────────

def run_detection_tracking(
    frame_paths: list[Path],
    detector:    PersonDetector,
    tracker:     MultiObjectTracker,
    cfg:         dict,
) -> list:
    from utils.helpers import load_frame
    tracker.reset()
    for idx, fp in enumerate(frame_paths):
        frame = load_frame(fp, size=cfg["datasets"]["resize"])
        dets  = detector.detect(frame, frame_idx=idx)
        tracker.update(dets, frame_idx=idx, frame_rgb=frame)
    return tracker.get_completed_trajectories(
        cfg["tracker"]["min_trajectory_length"]
    )


# ─────────────────────────────────────────────────────────────
#  Centroid sequence builder for LSTM
# ─────────────────────────────────────────────────────────────

def build_centroid_sequences(
    trajectories: list,
    seq_len:      int,
) -> list[np.ndarray]:
    """
    Convert trajectories into (seq_len, 2) centroid sequences for LSTM-AE.
    Normalises each trajectory to [0,1] and splits into non-overlapping windows.
    Returns list of (seq_len, 2) float32 arrays.
    """
    sequences = []
    for traj in trajectories:
        pts = traj.centroids          # (N, 2)
        if len(pts) < seq_len:
            continue
        mn  = pts.min(axis=0)
        mx  = pts.max(axis=0)
        rng = np.where((mx - mn) < 1e-6, 1.0, mx - mn)
        pts_norm = (pts - mn) / rng

        for start in range(0, len(pts_norm) - seq_len + 1, seq_len):
            sequences.append(
                pts_norm[start:start + seq_len].astype(np.float32)
            )
    return sequences


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────

def run_training(cfg: dict, dataset: str) -> None:
    logger.info("=" * 60)
    logger.info(f"PHASE: TRAINING — {dataset}")
    logger.info("=" * 60)

    set_seed(cfg["project"]["seed"])
    os.makedirs("models/autoencoder", exist_ok=True)
    os.makedirs("models/lstm_ae",     exist_ok=True)

    lstm_cfg = cfg.get("lstm_ae", {})
    seq_len  = int(lstm_cfg.get("seq_len", 12))

    # ── UCSD frame-based datasets ─────────────────────────────
    if dataset in UCSD_DATASETS:
        train_frames = collect_ucsd_frames(dataset, split="train")
        if not train_frames:
            raise RuntimeError(
                f"No training frames for {dataset}. Run prepare_ucsd.py first."
            )

        # Conv-AE
        conv_ckpt = f"models/autoencoder/conv_ae_{dataset}.pt"
        conv_trainer = ConvAETrainer(cfg.get("conv_ae", {}))
        if Path(conv_ckpt).exists():
            logger.info(f"Conv-AE checkpoint found, loading from {conv_ckpt}")
            conv_trainer.load(conv_ckpt)
        else:
            logger.info(f"Training Conv-AE on {len(train_frames)} frames ...")
            conv_trainer.fit(train_frames, frame_size=tuple(cfg["datasets"]["resize"]))
            conv_trainer.save(conv_ckpt)
        logger.info("Conv-AE training complete.")

        # LSTM-AE via detection + tracking
        logger.info("Running detection + tracking on train frames for LSTM-AE ...")
        try:
            detector   = PersonDetector(cfg.get("detector", {}))
            tracker    = MultiObjectTracker(cfg.get("tracker", {}))
            all_seqs:  list[np.ndarray] = []

            clip_dirs = sorted(set(f.parent for f in train_frames))
            for clip_dir in clip_dirs:
                clip_frames = sorted(clip_dir.glob("*.jpg"))
                if not clip_frames:
                    continue
                trajs = run_detection_tracking(clip_frames, detector, tracker, cfg)
                seqs  = build_centroid_sequences(trajs, seq_len)
                all_seqs.extend(seqs)
                logger.info(
                    f"  {clip_dir.name}: {len(trajs)} trajectories "
                    f"→ {len(seqs)} sequences (seq_len={seq_len})"
                )

            if all_seqs:
                logger.info(
                    f"Training LSTM-AE on {len(all_seqs)} centroid sequences "
                    f"(seq_len={seq_len}, input_dim=2) ..."
                )
                lstm_trainer = LSTMAETrainer(lstm_cfg)
                lstm_trainer.fit(all_seqs)
                lstm_trainer.save(f"models/lstm_ae/lstm_ae_{dataset}.pt")
                logger.info("LSTM-AE training complete.")
            else:
                logger.warning(
                    "No centroid sequences extracted. "
                    "Try lowering min_trajectory_length in config."
                )

        except Exception as e:
            logger.warning(f"Detection/tracking failed ({e}). LSTM-AE skipped.")

        logger.info("Training complete.")
        return

    # ── Generic video datasets ─────────────────────────────────
    video_paths = get_video_paths(cfg, dataset, split="train")
    detector    = PersonDetector(cfg.get("detector", {}))
    tracker     = MultiObjectTracker(cfg.get("tracker", {}))
    all_frames: list[Path]       = []
    all_seqs:   list[np.ndarray] = []

    for vid_path in video_paths:
        logger.info(f"Processing: {vid_path.name}")
        frame_paths = get_frame_paths_for_video(vid_path, cfg)
        all_frames.extend(frame_paths)
        trajs = run_detection_tracking(frame_paths, detector, tracker, cfg)
        all_seqs.extend(build_centroid_sequences(trajs, seq_len))

    logger.info(f"Training Conv-AE on {len(all_frames)} frames ...")
    conv_trainer = ConvAETrainer(cfg.get("conv_ae", {}))
    conv_trainer.fit(all_frames, frame_size=tuple(cfg["datasets"]["resize"]))
    conv_trainer.save(f"models/autoencoder/conv_ae_{dataset}.pt")

    if all_seqs:
        logger.info(f"Training LSTM-AE on {len(all_seqs)} sequences ...")
        lstm_trainer = LSTMAETrainer(lstm_cfg)
        lstm_trainer.fit(all_seqs)
        lstm_trainer.save(f"models/lstm_ae/lstm_ae_{dataset}.pt")

    logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────
#  TESTING
# ─────────────────────────────────────────────────────────────

def run_testing(cfg: dict, dataset: str) -> None:
    logger.info("=" * 60)
    logger.info(f"PHASE: TESTING + ROBUSTNESS EVALUATION — {dataset}")
    logger.info("=" * 60)

    set_seed(cfg["project"]["seed"])

    lstm_cfg = cfg.get("lstm_ae", {})
    seq_len  = int(lstm_cfg.get("seq_len", 12))

    # Load Conv-AE
    conv_trainer = ConvAETrainer(cfg.get("conv_ae", {}))
    conv_trainer.load(f"models/autoencoder/conv_ae_{dataset}.pt")

    # Load LSTM-AE if checkpoint exists
    lstm_path    = f"models/lstm_ae/lstm_ae_{dataset}.pt"
    lstm_trainer = None
    use_lstm     = False
    if Path(lstm_path).exists():
        try:
            lstm_trainer = LSTMAETrainer(lstm_cfg)
            lstm_trainer.load(lstm_path)
            use_lstm = True
            logger.info("LSTM-AE loaded.")
        except Exception as e:
            logger.warning(f"LSTM-AE load failed: {e} — visual only.")
    else:
        logger.info("No LSTM-AE checkpoint found — visual only mode.")

    fusion             = AnomalyFusion(cfg.get("fusion", {}))
    degradation_engine = DegradationEngine(cfg.get("degradation", {}))
    evaluator          = RobustnessEvaluator(cfg.get("evaluation", {}))

    # ── UCSD frame-based datasets ─────────────────────────────
    if dataset in UCSD_DATASETS:
        test_frames = collect_ucsd_frames(dataset, split="test")
        if not test_frames:
            raise RuntimeError(f"No test frames found for {dataset}.")

        logger.info(f"Scoring {len(test_frames)} test frames ...")

        evaluator.evaluate(
            conv_trainer       = conv_trainer,
            lstm_trainer       = lstm_trainer,
            frame_paths        = test_frames,
            degradation_engine = degradation_engine,
            dataset            = dataset,
        )

        if use_lstm:
            logger.info("Running tracking on test frames for behaviour scoring ...")
            try:
                detector  = PersonDetector(cfg.get("detector", {}))
                tracker   = MultiObjectTracker(cfg.get("tracker", {}))
                lingering = LingeringScorer(cfg.get("lingering", {}))
                all_results: list[FusionResult] = []

                clip_dirs = sorted(set(f.parent for f in test_frames))
                for clip_dir in clip_dirs:
                    clip_frames = sorted(clip_dir.glob("*.jpg"))
                    if not clip_frames:
                        continue
                    trajs         = run_detection_tracking(clip_frames, detector, tracker, cfg)
                    seqs          = build_centroid_sequences(trajs, seq_len)
                    visual_scores = conv_trainer.score(clip_frames)

                    behav_scores = {}
                    for i, seq in enumerate(seqs):
                        try:
                            behav_scores[i] = float(lstm_trainer.score(seq))
                        except Exception:
                            pass

                    ling_scores = lingering.score(trajs) if trajs else {}

                    result = fusion.fuse(
                        visual_scores = visual_scores,
                        behav_scores  = behav_scores,
                        ling_scores   = ling_scores,
                        video_name    = clip_dir.name,
                    )
                    all_results.append(result)
                    logger.info(
                        f"  {clip_dir.name}: score={result.anomaly_score:.3f}  "
                        f"tier={result.alert_tier}  lstm_seqs={len(seqs)}"
                    )

                _save_fusion_results(all_results, dataset)

            except Exception as e:
                logger.warning(f"Behaviour scoring failed: {e}")

        logger.info("Testing complete. Results saved to outputs/")
        return

    # ── Generic video datasets ─────────────────────────────────
    video_paths  = get_video_paths(cfg, dataset, split="test")
    detector     = PersonDetector(cfg.get("detector", {}))
    tracker      = MultiObjectTracker(cfg.get("tracker", {}))
    lingering    = LingeringScorer(cfg.get("lingering", {}))
    all_results: list[FusionResult] = []

    for vid_path in video_paths:
        logger.info(f"Testing: {vid_path.name}")
        frame_paths   = get_frame_paths_for_video(vid_path, cfg)
        trajs         = run_detection_tracking(frame_paths, detector, tracker, cfg)
        seqs          = build_centroid_sequences(trajs, seq_len)
        visual_scores = conv_trainer.score(frame_paths)

        behav_scores = {}
        if use_lstm:
            for i, seq in enumerate(seqs):
                try:
                    behav_scores[i] = float(lstm_trainer.score(seq))
                except Exception:
                    pass

        ling_scores = lingering.score(trajs) if trajs else {}

        result = fusion.fuse(
            visual_scores = visual_scores,
            behav_scores  = behav_scores,
            ling_scores   = ling_scores,
            video_name    = vid_path.stem,
        )
        all_results.append(result)

    evaluator.evaluate(
        conv_trainer       = conv_trainer,
        lstm_trainer       = lstm_trainer,
        frame_paths        = [],
        degradation_engine = degradation_engine,
        dataset            = dataset,
        results            = all_results,
    )
    _save_fusion_results(all_results, dataset)
    logger.info("Testing complete. Results saved to outputs/")


# ─────────────────────────────────────────────────────────────
#  Save helpers
# ─────────────────────────────────────────────────────────────

def _save_fusion_results(results: list[FusionResult], dataset: str) -> None:
    os.makedirs("outputs", exist_ok=True)
    out = [
        {
            "video_name":    r.video_name,
            "anomaly_score": round(r.anomaly_score, 4),
            "is_anomalous":  r.is_anomalous,
            "alert_tier":    r.alert_tier,
            "explanation":   r.explanation,
            "visual_score":  round(r.visual_score_norm, 4),
            "lstm_score":    round(r.lstm_score_norm, 4),
            "ling_score":    round(r.lingering_score_norm, 4),
        }
        for r in results
    ]
    path = f"outputs/{dataset}_fusion_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Fusion results saved → {path}")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust CCTV Anomaly Detection Pipeline"
    )
    parser.add_argument("--mode",    required=True, choices=["train", "test"])
    parser.add_argument("--dataset", required=True,
                        help="e.g. ucsd_ped2, ucsd_ped1")
    parser.add_argument("--config",  default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Mode    : {args.mode}")
    logger.info(f"Dataset : {args.dataset}")
    cfg = load_config(args.config)

    if args.mode == "train":
        run_training(cfg, args.dataset)
    elif args.mode == "test":
        run_testing(cfg, args.dataset)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()