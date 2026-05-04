"""
models/fusion.py — Multi-Factor Anomaly Score Fusion.

Combines three complementary signals:
  1. Visual reconstruction error    (Conv Autoencoder)
  2. Trajectory sequence deviation  (LSTM Autoencoder)
  3. Lingering heuristic score      (Domain-informed)

Outputs a single normalised anomaly score in [0, 1] per person-track,
plus a human-readable explanation and alert tier.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
#  Output dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    video_name: str = ""
    track_id:   int = -1
    frame_idx:  Optional[int] = None

    # Raw component scores
    visual_error:    float = 0.0
    lstm_error:      float = 0.0
    lingering_score: float = 0.0

    # Normalised [0, 1] component scores
    visual_score_norm:    float = 0.0
    lstm_score_norm:      float = 0.0
    lingering_score_norm: float = 0.0

    # Fused output
    anomaly_score: float = 0.0
    is_anomalous:  bool  = False
    alert_tier:    str   = "low"
    explanation:   str   = ""

    # Frame-level scores array (when fusing across a whole video)
    frame_scores: list[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
#  Default config
# ─────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    "weights": {
        "visual_ae": 0.5,
        "lstm_ae":   0.3,
        "lingering": 0.2,
    },
    "anomaly_threshold": 0.5,
    "alert_labels": {
        "high":   "Possible Loitering",
        "medium": "Irregular Movement",
        "low":    "Normal Behaviour",
    },
}


# ─────────────────────────────────────────────────────────────
#  Fusion engine
# ─────────────────────────────────────────────────────────────

class AnomalyFusion:
    """
    Fuses visual, temporal, and heuristic anomaly signals.

    Called by pipeline.py as:
        fusion = AnomalyFusion(cfg["fusion"])
        result = fusion.fuse(
            visual_scores = {"paths": [...], "scores": np.ndarray},
            behav_scores  = {"track_id": score, ...} or {},
            ling_scores   = {"track_id": score, ...} or {},
            video_name    = "Test001",
        )
    """

    def __init__(self, cfg: dict = None) -> None:
        cfg = {**DEFAULT_CFG, **(cfg or {})}

        weights       = cfg.get("weights", DEFAULT_CFG["weights"])
        self.w_visual = weights.get("visual_ae", 0.5)
        self.w_lstm   = weights.get("lstm_ae",   0.3)
        self.w_linger = weights.get("lingering", 0.2)

        self.threshold    = cfg.get("anomaly_threshold", 0.5)
        self.alert_labels = cfg.get("alert_labels", DEFAULT_CFG["alert_labels"])

        # Calibration stats (set via calibrate())
        self._v_mean: Optional[float] = None
        self._v_std:  Optional[float] = None
        self._l_mean: Optional[float] = None
        self._l_std:  Optional[float] = None

    # ─────────────────────────────────────────────────────────
    #  Pipeline-facing: fuse() accepts dicts from pipeline.py
    # ─────────────────────────────────────────────────────────

    def fuse(
        self,
        visual_scores: dict,               # {"paths": [...], "scores": np.ndarray}
        behav_scores:  dict = None,        # {"track_id": float, ...} or {}
        ling_scores:   dict = None,        # {"track_id": float, ...} or {}
        video_name:    str  = "",
        # Per-track direct call support:
        track_id:      int           = -1,
        frame_idx:     Optional[int] = None,
        visual_error:  float         = 0.0,
        lstm_error:    float         = 0.0,
        lingering_score: float       = 0.0,
    ) -> FusionResult:
        """
        Two calling modes:

        Mode 1 — Pipeline batch mode (called from run_testing):
            fusion.fuse(
                visual_scores={"paths": [...], "scores": np.ndarray},
                behav_scores={...},
                ling_scores={...},
                video_name="Test001",
            )

        Mode 2 — Per-track mode (direct use):
            fusion.fuse(
                track_id=42, frame_idx=100,
                visual_error=0.03, lstm_error=0.02, lingering_score=0.7,
            )
        """
        behav_scores = behav_scores or {}
        ling_scores  = ling_scores  or {}

        # ── Mode 1: dict-based batch from pipeline ────────────────────────
        if isinstance(visual_scores, dict) and "scores" in visual_scores:
            return self._fuse_video(
                visual_scores, behav_scores, ling_scores, video_name
            )

        # ── Mode 2: per-track scalar call ─────────────────────────────────
        return self._fuse_track(
            track_id=track_id,
            frame_idx=frame_idx,
            visual_error=visual_error,
            lstm_error=lstm_error,
            lingering_score=lingering_score,
            video_name=video_name,
        )

    # ─────────────────────────────────────────────────────────
    #  Internal: video-level fusion
    # ─────────────────────────────────────────────────────────

    def _fuse_video(
        self,
        visual_scores: dict,
        behav_scores:  dict,
        ling_scores:   dict,
        video_name:    str,
    ) -> FusionResult:
        """
        Fuse across all frames of one video.
        Returns a single FusionResult summarising the video.
        """
        v_scores = np.array(visual_scores["scores"], dtype=np.float32)

        # Normalise visual scores
        v_norm = self._normalise_array(v_scores, self._v_mean, self._v_std)

        # Aggregate behavioural scores (mean over tracks, or 0 if none)
        if behav_scores:
            l_raw  = np.array(list(behav_scores.values()), dtype=np.float32)
            l_norm = float(self._normalise_array(l_raw, self._l_mean, self._l_std).mean())
        else:
            l_norm = 0.0

        # Aggregate lingering scores (mean over tracks, or 0 if none)
        if ling_scores:
            g_raw  = np.array(list(ling_scores.values()), dtype=np.float32)
            g_norm = float(np.clip(g_raw, 0.0, 1.0).mean())
        else:
            g_norm = 0.0

        # Frame-level fused scores
        frame_scores = (
            self.w_visual * v_norm
            + self.w_lstm  * l_norm
            + self.w_linger * g_norm
        )
        frame_scores = np.clip(frame_scores, 0.0, 1.0)

        # Video-level summary score = mean of frame scores
        summary_score = float(frame_scores.mean())
        is_anom       = summary_score >= self.threshold
        tier, expl    = self._classify(summary_score, v_norm.mean(), l_norm, g_norm)

        return FusionResult(
            video_name=video_name,
            visual_error=float(v_scores.mean()),
            lstm_error=float(np.mean(list(behav_scores.values()))) if behav_scores else 0.0,
            lingering_score=float(np.mean(list(ling_scores.values()))) if ling_scores else 0.0,
            visual_score_norm=float(v_norm.mean()),
            lstm_score_norm=l_norm,
            lingering_score_norm=g_norm,
            anomaly_score=summary_score,
            is_anomalous=is_anom,
            alert_tier=tier,
            explanation=expl,
            frame_scores=frame_scores.tolist(),
        )

    # ─────────────────────────────────────────────────────────
    #  Internal: per-track fusion
    # ─────────────────────────────────────────────────────────

    def _fuse_track(
        self,
        track_id:        int,
        frame_idx:       Optional[int],
        visual_error:    float,
        lstm_error:      float,
        lingering_score: float,
        video_name:      str = "",
    ) -> FusionResult:
        v_norm = self._zscore_norm(visual_error, self._v_mean, self._v_std)
        l_norm = self._zscore_norm(lstm_error,   self._l_mean, self._l_std)
        g_norm = float(np.clip(lingering_score, 0.0, 1.0))

        score   = float(np.clip(
            self.w_visual * v_norm + self.w_lstm * l_norm + self.w_linger * g_norm,
            0.0, 1.0,
        ))
        is_anom = score >= self.threshold
        tier, expl = self._classify(score, v_norm, l_norm, g_norm)

        return FusionResult(
            video_name=video_name,
            track_id=track_id,
            frame_idx=frame_idx,
            visual_error=visual_error,
            lstm_error=lstm_error,
            lingering_score=lingering_score,
            visual_score_norm=v_norm,
            lstm_score_norm=l_norm,
            lingering_score_norm=g_norm,
            anomaly_score=score,
            is_anomalous=is_anom,
            alert_tier=tier,
            explanation=expl,
        )

    def fuse_batch(self, records: list[dict]) -> list[FusionResult]:
        """Fuse a list of per-track dicts."""
        return [self._fuse_track(**r) for r in records]

    # ─────────────────────────────────────────────────────────
    #  Calibration
    # ─────────────────────────────────────────────────────────

    def calibrate(
        self,
        train_visual_errors: np.ndarray,
        train_lstm_errors:   np.ndarray,
    ) -> None:
        """
        Compute mean/std from training errors for z-score normalisation.
        Call once after both autoencoders have been trained.
        """
        self._v_mean = float(train_visual_errors.mean())
        self._v_std  = float(train_visual_errors.std() + 1e-8)
        self._l_mean = float(train_lstm_errors.mean())
        self._l_std  = float(train_lstm_errors.std() + 1e-8)

    # ─────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────

    def _zscore_norm(
        self, value: float, mean: Optional[float], std: Optional[float]
    ) -> float:
        if mean is None or std is None:
            return float(np.clip(value, 0.0, 1.0))
        z = (value - mean) / std
        return float(1.0 / (1.0 + np.exp(-z)))

    def _normalise_array(
        self, arr: np.ndarray, mean: Optional[float], std: Optional[float]
    ) -> np.ndarray:
        if mean is None or std is None:
            return np.clip(arr, 0.0, 1.0)
        z = (arr - mean) / (std + 1e-8)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def _classify(
        self, score: float, v: float, l: float, g: float
    ) -> tuple[str, str]:
        if score >= self.threshold:
            dominant = max(
                [("visual appearance", v), ("motion pattern", l), ("lingering behaviour", g)],
                key=lambda x: x[1],
            )[0]
            if score >= 0.80:
                return "high", (
                    f"High anomaly confidence ({score:.2f}) driven by "
                    f"{dominant}. Possible loitering detected."
                )
            return "medium", (
                f"Moderate anomaly ({score:.2f}) — unusual {dominant} observed."
            )
        return "low", f"Normal behaviour (score={score:.2f})"