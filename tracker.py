"""
tracking/tracker.py — DeepSORT multi-object tracker + trajectory management.

Trajectories are the central data structure for all downstream modelling.
Each trajectory is a Trajectory object keyed by track_id.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)

DEFAULT_CFG = {
    "max_age":               30,
    "min_hits":              3,
    "iou_threshold":         0.3,
    "min_trajectory_length": 12,
}


# ─────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    frame_idx: int
    cx:        float
    cy:        float
    bbox:      np.ndarray   # [x1, y1, x2, y2]


@dataclass
class Trajectory:
    track_id: int
    points:   list[TrajectoryPoint] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.points)

    @property
    def centroids(self) -> np.ndarray:
        """(N, 2) array of (cx, cy) ordered by frame."""
        return np.array([[p.cx, p.cy] for p in self.points])

    @property
    def frame_indices(self) -> np.ndarray:
        return np.array([p.frame_idx for p in self.points])

    @property
    def bbox_areas(self) -> np.ndarray:
        return np.array([
            (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])
            for p in self.points
        ])

    def is_long_enough(self, min_len: int) -> bool:
        return self.length >= min_len


# ─────────────────────────────────────────────────────────────
#  Tracker
# ─────────────────────────────────────────────────────────────

class MultiObjectTracker:
    """
    Wraps deep_sort_realtime. Call update() frame-by-frame.

    Detection input format (from PersonDetector):
        list of [x1, y1, x2, y2, confidence]

    Usage:
        tracker = MultiObjectTracker(cfg["tracker"])
        for frame_idx, frame in enumerate(frames):
            dets = detector.detect(frame, frame_idx)
            tracker.update(dets, frame_idx=frame_idx, frame_rgb=frame)
        trajs = tracker.get_completed_trajectories(min_length=12)
    """

    def __init__(self, cfg: dict = None) -> None:
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.trajectories: dict[int, Trajectory] = {}
        self._build_tracker()

    def _build_tracker(self) -> None:
        self._ds = DeepSort(
            max_age=self.cfg["max_age"],
            n_init=self.cfg["min_hits"],        # min_hits → n_init
            max_iou_distance=self.cfg["iou_threshold"],
        )

    # ── update ────────────────────────────────────────────────

    def update(
        self,
        detections: list[list[float]],          # [[x1,y1,x2,y2,conf], ...]
        frame_idx:  int,
        frame_rgb:  Optional[np.ndarray] = None,
    ) -> list[tuple[int, np.ndarray]]:
        """
        Feed one frame's detections into DeepSORT.

        Returns list of (track_id, bbox_ltrb) for confirmed tracks.
        """
        # Convert [x1,y1,x2,y2,conf] → DeepSort format: ([l,t,w,h], conf, "person")
        raw = []
        for d in detections:
            x1, y1, x2, y2, conf = d
            l, t = x1, y1
            w, h = x2 - x1, y2 - y1
            raw.append(([l, t, w, h], conf, "person"))

        tracks = self._ds.update_tracks(raw, frame=frame_rgb)

        active = []
        for t in tracks:
            if not t.is_confirmed():
                continue

            tid  = int(t.track_id)
            bbox = t.to_ltrb()          # [x1, y1, x2, y2]
            cx   = (bbox[0] + bbox[2]) / 2
            cy   = (bbox[1] + bbox[3]) / 2

            if tid not in self.trajectories:
                self.trajectories[tid] = Trajectory(track_id=tid)

            self.trajectories[tid].points.append(
                TrajectoryPoint(
                    frame_idx=frame_idx,
                    cx=cx, cy=cy,
                    bbox=np.array(bbox),
                )
            )
            active.append((tid, np.array(bbox)))

        return active

    # ── accessors ─────────────────────────────────────────────

    def get_all_trajectories(self) -> list[Trajectory]:
        return list(self.trajectories.values())

    def get_completed_trajectories(
        self, min_length: Optional[int] = None
    ) -> list[Trajectory]:
        ml = min_length or self.cfg.get("min_trajectory_length", 12)
        return [t for t in self.trajectories.values() if t.is_long_enough(ml)]

    def reset(self) -> None:
        """Clear all state between videos."""
        self.trajectories = {}
        self._build_tracker()
        logger.debug("[Tracker] Reset.")