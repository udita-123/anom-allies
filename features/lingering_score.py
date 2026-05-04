"""
features/lingering_score.py — Lingering Behaviour Heuristic.

Score in [0,1]. Higher = stronger loitering signal.
Called by pipeline.py as:
    lingering = LingeringScorer(cfg["lingering"])
    ling_scores = lingering.score(trajs)   # trajs = list[Trajectory]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from tracking.tracker import Trajectory


DEFAULT_CFG = {
    "speed_threshold":        2.5,
    "confinement_threshold":  0.05,
    "min_duration_frames":    60,
    "low_motion_threshold":   2.5,   # alias used in config.yaml
    "weights": {
        "speed_inv":    0.4,
        "confinement":  0.4,
        "duration":     0.2,
    },
}


@dataclass
class LingeringResult:
    track_id:          int
    score:             float
    is_loitering:      bool
    mean_speed:        float
    confinement_ratio: float
    duration_frames:   int
    label:             str


class LingeringScorer:
    """
    Computes per-trajectory lingering scores.

    Pipeline calls:
        lingering.score(trajs)   → dict[track_id, float]
    """

    def __init__(
        self,
        cfg: dict = None,
        frame_size: tuple[int, int] = (240, 360),
    ) -> None:
        c = {**DEFAULT_CFG, **(cfg or {})}
        w = c.get("weights", DEFAULT_CFG["weights"])

        # support both config key names
        self.speed_thr = c.get("speed_threshold",
                         c.get("low_motion_threshold", 2.5))
        self.conf_thr  = c.get("confinement_threshold", 0.05)
        self.min_dur   = c.get("min_duration_frames", 60)
        self.w_speed   = w.get("speed_inv",   0.4)
        self.w_conf    = w.get("confinement", 0.4)
        self.w_dur     = w.get("duration",    0.2)
        self.frame_h, self.frame_w = frame_size

    # ── pipeline.py calls this with a list ───────────────────────────────

    def score(
        self,
        trajectories: list[Trajectory] | Trajectory,
    ) -> dict[int, float]:
        """
        Score one or many trajectories.
        Returns dict: track_id → float score [0,1]

        Accepts both a single Trajectory and a list.
        """
        if isinstance(trajectories, Trajectory):
            trajectories = [trajectories]
        return {
            t.track_id: self._score_one(t).score
            for t in trajectories
        }

    def score_all(
        self, trajectories: list[Trajectory]
    ) -> dict[int, LingeringResult]:
        """Full results with labels. Returns dict: track_id → LingeringResult."""
        return {t.track_id: self._score_one(t) for t in trajectories}

    # ── internal ─────────────────────────────────────────────────────────

    def _score_one(self, traj: Trajectory) -> LingeringResult:
        pts = traj.centroids
        n   = len(pts)

        if n > 1:
            diffs      = np.diff(pts, axis=0)
            speeds     = np.sqrt((diffs**2).sum(axis=1))
            mean_speed = float(speeds.mean())
        else:
            mean_speed = 0.0

        speed_inv_score = float(
            np.clip(1.0 - mean_speed / max(self.speed_thr, 1e-6), 0.0, 1.0)
        )

        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()
        confinement_ratio = float(
            (x_range * y_range) / max(self.frame_w * self.frame_h, 1)
        )
        confinement_score = float(
            np.clip(1.0 - confinement_ratio / max(self.conf_thr, 1e-6), 0.0, 1.0)
        )

        duration       = traj.length
        duration_score = float(
            np.clip((duration - self.min_dur) / max(2 * self.min_dur, 1), 0.0, 1.0)
        )

        composite = float(np.clip(
            self.w_speed * speed_inv_score
            + self.w_conf  * confinement_score
            + self.w_dur   * duration_score,
            0.0, 1.0,
        ))

        is_loitering = (
            mean_speed        < self.speed_thr
            and confinement_ratio < self.conf_thr
            and duration          >= self.min_dur
        )

        return LingeringResult(
            track_id=traj.track_id,
            score=composite,
            is_loitering=is_loitering,
            mean_speed=mean_speed,
            confinement_ratio=confinement_ratio,
            duration_frames=duration,
            label=self._label(composite, mean_speed, confinement_ratio, duration),
        )

    @staticmethod
    def _label(score, speed, confinement, duration) -> str:
        if score > 0.75:
            return "Possible Loitering — sustained, confined, slow movement"
        elif score > 0.50:
            return "Irregular Movement — partially confined or prolonged presence"
        elif speed < 0.5 and duration > 30:
            return "Stationary Person — standing still for extended period"
        return "Normal Behaviour"