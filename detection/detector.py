"""
detection/detector.py
YOLOv8 + DeepSORT person detector for the CCTV behavioural anomaly pipeline.

Install:
    pip install ultralytics deep_sort_realtime --break-system-packages

Usage (standalone test):
    PYTHONPATH=. python detection/detector.py --frames data/frames/ucsd_ped2/test/Test001

Pipeline interface (called from pipeline.py):
    from detection.detector import Detector
    det = Detector(conf=0.20, iou=0.45, device='cpu')
    tracks = det.process_frame_dir(frame_dir)   # -> list[TrackResult]
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ── lazy imports so the module is importable even if deps are missing ──────────
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False
    print("[detector] WARNING: ultralytics not installed. Run: "
          "pip install ultralytics --break-system-packages", file=sys.stderr)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    _DS_OK = True
except ImportError:
    _DS_OK = False
    print("[detector] WARNING: deep_sort_realtime not installed. Run: "
          "pip install deep_sort_realtime --break-system-packages", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single YOLO person detection on one frame."""
    frame_idx: int
    bbox_ltwh: tuple[float, float, float, float]   # left, top, width, height
    confidence: float

    @property
    def centroid(self) -> tuple[float, float]:
        l, t, w, h = self.bbox_ltwh
        return (l + w / 2, t + h / 2)

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        l, t, w, h = self.bbox_ltwh
        return (l, t, l + w, t + h)


@dataclass
class TrackResult:
    """Accumulated trajectory for one DeepSORT track across a clip."""
    track_id: int
    frame_indices: list[int] = field(default_factory=list)
    centroids:    list[tuple[float, float]] = field(default_factory=list)
    confidences:  list[float] = field(default_factory=list)

    def to_numpy(self) -> np.ndarray:
        """Return (N, 2) array of (cx, cy) centroids."""
        return np.array(self.centroids, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame_indices)


# ─────────────────────────────────────────────────────────────────────────────
# Detector class
# ─────────────────────────────────────────────────────────────────────────────

class Detector:
    """
    YOLOv8 person detector + DeepSORT multi-object tracker.

    Parameters
    ----------
    model_name : str
        YOLOv8 model variant.  'yolov8n.pt' is fastest; 'yolov8s.pt' is more
        accurate.  Weights are auto-downloaded on first use.
    conf      : float  Minimum YOLO confidence to accept a detection.
                       Use 0.20 for low-res grayscale datasets like UCSD Ped2.
    iou       : float  NMS IoU threshold.
    device    : str    'cpu', '0' (first GPU), etc.
    max_age   : int    DeepSORT frames to keep a lost track alive.
    min_hits  : int    Minimum consecutive detections before confirming a track.
    frame_size: tuple  (W, H) used to normalise centroids to [0, 1].
                       If None, raw pixel coordinates are returned.
    """

    PERSON_CLASS_ID = 0   # COCO class 0 = person

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.20,           # FIX 5: lowered from 0.35 for low-res grayscale
        iou: float = 0.45,
        device: str = "cpu",
        max_age: int = 30,
        min_hits: int = 3,
        frame_size: Optional[tuple[int, int]] = None,
    ):
        if not _YOLO_OK:
            raise RuntimeError("ultralytics not installed.")
        if not _DS_OK:
            raise RuntimeError("deep_sort_realtime not installed.")

        self.conf = conf
        self.iou = iou
        self.device = device
        self.frame_size = frame_size   # (W, H) or None

        # Load YOLOv8 model
        self.model = YOLO(model_name)
        self.model.to(device)

        # DeepSORT tracker — one instance per clip (reset between clips)
        self._ds_kwargs = dict(max_age=max_age, min_hits=min_hits, max_iou_distance=0.7)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _new_tracker(self) -> "DeepSort":
        return DeepSort(**self._ds_kwargs)

    def _detect_frame(self, img_path: Path, frame_idx: int) -> list[Detection]:
        """Run YOLO on a single frame, return person detections."""
        # FIX 1: read with cv2 and convert grayscale → BGR so YOLO always
        # receives a proper 3-channel image (UCSD Ped frames are grayscale)
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        results = self.model(
            img,                              # pass numpy array, not path string
            conf=self.conf,
            iou=self.iou,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )
        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf[0].cpu().numpy())
                ltwh = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                detections.append(Detection(frame_idx, ltwh, conf_val))
        return detections

    def _normalise_centroid(
        self, cx: float, cy: float, img_w: int, img_h: int
    ) -> tuple[float, float]:
        if self.frame_size is not None:
            w, h = self.frame_size
        else:
            w, h = img_w, img_h
        return (cx / w, cy / h)

    # ── public API ────────────────────────────────────────────────────────────

    def process_frame_dir(
        self,
        frame_dir: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        max_frames: Optional[int] = None,
    ) -> list[TrackResult]:
        """
        Run detection + tracking on all frames in *frame_dir*.

        Parameters
        ----------
        frame_dir   : Path to directory containing frame images.
        extensions  : Accepted image file extensions.
        max_frames  : Optional cap on number of frames to process.

        Returns
        -------
        List of TrackResult objects, one per confirmed track.
        """
        frame_dir = Path(frame_dir)
        frame_paths = sorted(
            p for p in frame_dir.iterdir()
            if p.suffix.lower() in extensions
        )
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        if not frame_paths:
            print(f"[detector] No frames found in {frame_dir}", file=sys.stderr)
            return []

        tracker = self._new_tracker()
        track_dict: dict[int, TrackResult] = {}

        # Infer frame dimensions from first image
        first = cv2.imread(str(frame_paths[0]))
        if first is not None:
            img_h, img_w = first.shape[:2]
        else:
            img_w, img_h = 320, 240   # UCSD Ped default

        print(f"[detector] Processing {len(frame_paths)} frames in {frame_dir.name} …")

        for idx, fpath in enumerate(frame_paths):
            detections = self._detect_frame(fpath, idx)

            # Format for DeepSORT: list of ([l, t, w, h], confidence, class_id)
            ds_input = [
                (list(d.bbox_ltwh), d.confidence, "person")
                for d in detections
            ]

            # FIX 1 continued: pass the numpy frame to DeepSORT so its
            # MobileNet embedder also gets a proper BGR image
            frame_bgr = cv2.imread(str(fpath))
            if frame_bgr is not None and (
                len(frame_bgr.shape) == 2 or frame_bgr.shape[2] == 1
            ):
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

            tracks = tracker.update_tracks(ds_input, frame=frame_bgr)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = int(track.track_id)
                ltwh = track.to_ltwh()
                cx = ltwh[0] + ltwh[2] / 2
                cy = ltwh[1] + ltwh[3] / 2
                cx_n, cy_n = self._normalise_centroid(cx, cy, img_w, img_h)

                if tid not in track_dict:
                    track_dict[tid] = TrackResult(track_id=tid)
                track_dict[tid].frame_indices.append(idx)
                track_dict[tid].centroids.append((cx_n, cy_n))
                track_dict[tid].confidences.append(
                    float(track.get_det_conf() or 0.0)
                )

        results = list(track_dict.values())
        print(f"[detector] → {len(results)} tracks confirmed.")
        return results

    def process_clip_list(
        self,
        clip_dirs: list[str | Path],
        **kwargs,
    ) -> dict[str, list[TrackResult]]:
        """
        Run detection on multiple clip directories.

        Returns
        -------
        dict mapping clip name → list[TrackResult]
        """
        out: dict[str, list[TrackResult]] = {}
        for cd in clip_dirs:
            name = Path(cd).name
            out[name] = self.process_frame_dir(cd, **kwargs)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integration helpers
# ─────────────────────────────────────────────────────────────────────────────

def tracks_to_centroid_sequences(
    tracks: list[TrackResult],
    seq_len: int = 12,
    stride: Optional[int] = None,
) -> dict[int, list[np.ndarray]]:
    """
    Convert TrackResult list into LSTM-AE input sequences.

    For each track, sliding-window the centroid array into non-overlapping
    (or strided) chunks of shape (seq_len, 2).

    Returns
    -------
    dict mapping track_id -> list of np.ndarray (seq_len, 2)
    """
    if stride is None:
        stride = seq_len   # non-overlapping (matches pipeline default)

    out: dict[int, list[np.ndarray]] = {}
    for tr in tracks:
        if len(tr) < seq_len:
            continue   # too short to form even one sequence
        pts = tr.to_numpy()
        seqs: list[np.ndarray] = []
        for start in range(0, len(pts) - seq_len + 1, stride):
            seqs.append(pts[start : start + seq_len])
        if seqs:
            out[tr.track_id] = seqs
    return out


def tracks_to_trajectory_dict(
    tracks: list[TrackResult],
) -> dict[int, np.ndarray]:
    """
    Return {track_id: np.ndarray (N, 2)} for use with LingeringScorer.
    Matches the Trajectory namedtuple format expected by lingering_score.py.
    """
    return {tr.track_id: tr.to_numpy() for tr in tracks if len(tr) >= 2}


# ─────────────────────────────────────────────────────────────────────────────
# CLI — quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="YOLOv8 + DeepSORT detector smoke test")
    ap.add_argument("--frames", required=True, help="Path to frame directory")
    ap.add_argument("--model",  default="yolov8n.pt", help="YOLOv8 model name")
    ap.add_argument("--conf",   type=float, default=0.20)
    ap.add_argument("--iou",    type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seq_len",type=int,   default=12)
    ap.add_argument("--max_frames", type=int, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # Windows requirement

    args = _parse_args()

    det = Detector(
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    tracks = det.process_frame_dir(args.frames, max_frames=args.max_frames)

    print(f"\n{'─'*50}")
    print(f"  Confirmed tracks : {len(tracks)}")
    for tr in tracks[:10]:
        print(f"  Track {tr.track_id:>3d} | frames={len(tr):>4d} | "
              f"cx range [{min(c[0] for c in tr.centroids):.3f}, "
              f"{max(c[0] for c in tr.centroids):.3f}]")
    if len(tracks) > 10:
        print(f"  … and {len(tracks)-10} more")

    seqs = tracks_to_centroid_sequences(tracks, seq_len=args.seq_len)
    total_seqs = sum(len(v) for v in seqs.values())
    print(f"\n  LSTM sequences (seq_len={args.seq_len}) : {total_seqs}")
    print(f"{'─'*50}\n")