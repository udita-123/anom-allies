"""
features/extractor.py — Motion and spatial features from trajectories.

Output per sliding window: (7,) feature vector
  [speed_mean, speed_std, dir_entropy, confinement, dx_mean, dy_mean, bbox_area_mean]

Matches lstm_ae.input_dim = 7 (not 8 — config.yaml has 8 but we use 7,
so update config lstm_ae.input_dim to 7 or add one more feature below)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy
from tracking.tracker import Trajectory


# ─────────────────────────────────────────────────────────────
#  Defaults — fills in missing config keys safely
# ─────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    "window_size":      12,
    "stride":           6,
    "direction_bins":   8,
    "confinement_grid": [4, 4],
    "speed_smooth_k":   3,
    # these keys exist in config.yaml but are ignored by extractor:
    "velocity":         True,
    "direction":        True,
    "acceleration":     True,
    "bbox_area":        True,
    "aspect_ratio":     True,
    "smoothing_window": 5,
}


class FeatureExtractor:
    def __init__(
        self,
        cfg: dict,
        frame_size: tuple[int, int] = (240, 360),   # UCSD Ped2 native resolution
    ) -> None:
        c = {**DEFAULT_CFG, **(cfg or {})}
        self.window   = c["window_size"]
        self.stride   = c["stride"]
        self.dir_bins = c["direction_bins"]
        self.grid     = c["confinement_grid"]
        self.smk      = c["speed_smooth_k"]
        self.frame_h, self.frame_w = frame_size

    def extract(self, traj: Trajectory) -> np.ndarray:
        """
        Returns (n_windows, 7) feature matrix.
        Empty array if trajectory too short.
        """
        pts = traj.centroids        # (N, 2)
        if len(pts) < self.window:
            return np.empty((0, 7))

        areas          = traj.bbox_areas
        speeds, dxs, dys = self._compute_velocities(pts)

        windows = []
        for start in range(0, len(pts) - self.window + 1, self.stride):
            end     = start + self.window
            w_pts   = pts[start:end]
            w_spd   = speeds[start:end]
            w_dx    = dxs[start:end]
            w_dy    = dys[start:end]
            w_areas = areas[start:end]

            feat = np.array([
                np.mean(w_spd),
                np.std(w_spd),
                self._direction_entropy(w_dx, w_dy),
                self._spatial_confinement(w_pts),
                np.mean(w_dx),
                np.mean(w_dy),
                np.mean(w_areas),
            ], dtype=np.float32)

            windows.append(feat)

        return np.stack(windows) if windows else np.empty((0, 7))

    def _compute_velocities(self, pts):
        diff  = np.diff(pts, axis=0)
        dx    = diff[:, 0]
        dy    = diff[:, 1]
        speed = np.sqrt(dx**2 + dy**2)
        if self.smk > 1:
            kernel = np.ones(self.smk) / self.smk
            speed  = np.convolve(speed, kernel, mode="same")
        dx    = np.concatenate([dx,    [dx[-1]]])
        dy    = np.concatenate([dy,    [dy[-1]]])
        speed = np.concatenate([speed, [speed[-1]]])
        return speed, dx, dy

    def _direction_entropy(self, dx, dy):
        angles = np.arctan2(dy, dx)
        hist, _ = np.histogram(angles, bins=self.dir_bins, range=(-np.pi, np.pi))
        hist = hist + 1e-8
        return float(scipy_entropy(hist / hist.sum()))

    def _spatial_confinement(self, pts):
        rows, cols = self.grid
        cell_h = self.frame_h / rows
        cell_w = self.frame_w / cols
        cells  = set()
        for x, y in pts:
            r = int(np.clip(y // cell_h, 0, rows - 1))
            c = int(np.clip(x // cell_w, 0, cols - 1))
            cells.add((r, c))
        return len(cells) / (rows * cols)


# ─────────────────────────────────────────────────────────────
#  Batch helper — called by pipeline.py
# ─────────────────────────────────────────────────────────────

def extract_all(
    trajectories: list[Trajectory],
    cfg: dict,
    frame_size: tuple[int, int] = (240, 360),
) -> dict[int, np.ndarray]:
    """
    Extract features for every trajectory.
    Returns dict: track_id → (n_windows, 7) matrix.
    Skips trajectories that are too short.
    """
    c = {**DEFAULT_CFG, **(cfg or {})}
    extractor = FeatureExtractor(c, frame_size)
    result = {}
    for t in trajectories:
        mat = extractor.extract(t)
        if mat.shape[0] > 0:
            result[t.track_id] = mat
    return result