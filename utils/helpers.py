"""
utils/helpers.py — Shared utilities for the CCTV Behaviour Detection pipeline.

Degradations use OpenCV for physically accurate CCTV simulation:
- Motion blur:    directional horizontal kernel via cv2.filter2D
- Gaussian noise: cv2.randn additive noise
- Low light:      cv2.convertScaleAbs with alpha scaling
- Compression:    cv2.imencode/imdecode JPEG round-trip
"""

import cv2
import yaml
import random
import logging
import numpy as np
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")


# ─────────────────────────────────────────────────────────────
#  Frame I/O
# ─────────────────────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str, fps: int = 10) -> list[Path]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    interval   = max(1, round(native_fps / fps))
    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved, frame_idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            fpath = out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(fpath), frame)
            saved.append(fpath)
        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(saved)} frames -> {out_dir}")
    return sorted(saved)


def load_frame(path: str | Path, size: tuple | None = None) -> np.ndarray:
    """Load a frame as RGB numpy array, optionally resized to (H, W)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, (size[1], size[0]))
    return img


# ─────────────────────────────────────────────────────────────
#  PIL <-> numpy conversion helpers
# ─────────────────────────────────────────────────────────────

def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """PIL RGB -> numpy BGR (OpenCV format)."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    """numpy BGR -> PIL RGB."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────────────────────
#  Degradation Engine — OpenCV based
# ─────────────────────────────────────────────────────────────

class DegradationEngine:
    """
    Apply realistic CCTV degradations using OpenCV.

    All methods accept and return PIL.Image to integrate seamlessly
    with conv_trainer.score_degraded().

    Usage:
        engine = DegradationEngine(cfg["degradation"])
        deg_fn = engine.get("motion_blur")
        result = conv_trainer.score_degraded(frame_paths, deg_fn)
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg or {}

    # ── Motion Blur ────────────────────────────────────────────────────────
    # Directional horizontal kernel simulates camera pan/shake.
    # More physically accurate than PIL's isotropic BoxBlur.

    def motion_blur(self, img: Image.Image) -> Image.Image:
        cfg     = self.cfg.get("motion_blur", {})
        k_range = cfg.get("kernel_range", [5, 15])
        k       = random.randint(k_range[0], k_range[1])
        if k % 2 == 0:
            k += 1  # kernel size must be odd

        kernel            = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k  # horizontal blur

        arr = _pil_to_bgr(img)
        arr = cv2.filter2D(arr, -1, kernel)
        return _bgr_to_pil(arr)

    # ── Gaussian Noise ─────────────────────────────────────────────────────
    # cv2.randn fills array with Gaussian random values.
    # std=8 calibrated to keep normal frame errors below threshold.

    def gaussian_noise(self, img: Image.Image) -> Image.Image:
        cfg   = self.cfg.get("gaussian_noise", {})
        sigma = cfg.get("std", 8)

        arr   = _pil_to_bgr(img).astype(np.float32)
        noise = np.zeros_like(arr, dtype=np.float32)
        cv2.randn(noise, mean=0, stddev=sigma)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return _bgr_to_pil(arr)

    # ── Low Light ──────────────────────────────────────────────────────────
    # cv2.convertScaleAbs: output = |alpha * input + beta|
    # alpha < 1 darkens. Simulates night or poorly lit environments.

    def low_light(self, img: Image.Image) -> Image.Image:
        cfg     = self.cfg.get("low_light", {})
        b_range = cfg.get("brightness_factor", [0.2, 0.6])
        alpha   = random.uniform(b_range[0], b_range[1])

        arr = _pil_to_bgr(img)
        arr = cv2.convertScaleAbs(arr, alpha=alpha, beta=0)
        return _bgr_to_pil(arr)

    # ── Compression ────────────────────────────────────────────────────────
    # JPEG encode to in-memory buffer then decode.
    # Simulates heavily compressed CCTV streams or low-bitrate storage.

    def compression(self, img: Image.Image) -> Image.Image:
        cfg     = self.cfg.get("compression", {})
        q_range = cfg.get("jpeg_quality_range", [10, 40])
        quality = random.randint(q_range[0], q_range[1])

        arr    = _pil_to_bgr(img)
        _, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        arr    = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return _bgr_to_pil(arr)

    # ── Combined ───────────────────────────────────────────────────────────

    def combined(self, img: Image.Image) -> Image.Image:
        """Apply all four degradations sequentially."""
        for fn in [self.motion_blur, self.gaussian_noise,
                   self.low_light, self.compression]:
            img = fn(img)
        return img

    # ── get() — called by evaluator ────────────────────────────────────────

    def get(self, condition: str) -> Callable | None:
        mapping = {
            "motion_blur":    self.motion_blur,
            "gaussian_noise": self.gaussian_noise,
            "low_light":      self.low_light,
            "compression":    self.compression,
            "combined":       self.combined,
            "all_combined":   self.combined,
        }
        return mapping.get(condition, None)

    # ── Legacy numpy interface ─────────────────────────────────────────────

    def apply_single(self, img: np.ndarray, name: str) -> np.ndarray:
        """Apply a single degradation to a BGR numpy array."""
        pil = _bgr_to_pil(img)
        out = self.get(name)(pil)
        return _pil_to_bgr(out)

    def apply_all(self, img: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """Randomly apply each degradation with given probability."""
        pil = _bgr_to_pil(img)
        for fn in [self.motion_blur, self.gaussian_noise,
                   self.low_light, self.compression]:
            if random.random() < prob:
                pil = fn(pil)
        return _pil_to_bgr(pil)


# ─────────────────────────────────────────────────────────────
#  Normalisation helpers
# ─────────────────────────────────────────────────────────────

def frame_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 [0,255] -> CHW float32 [0,1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def normalise_score(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise an array of scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-8:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)
