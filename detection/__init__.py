"""
detection/__init__.py — YOLOv8 Person Detector
================================================
Wraps Ultralytics YOLOv8 for person-only detection.
Returns bounding boxes in [x1, y1, x2, y2, confidence] format
compatible with DeepSORT tracker input.
"""

from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── lazy import so pipeline doesn't crash if ultralytics not installed ──
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    logger.warning("[Detector] ultralytics not installed. Run: pip install ultralytics")


DEFAULT_CFG = {
    "model_path":     "yolov8n.pt",
    "conf_threshold": 0.35,
    "iou_threshold":  0.5,
    "device":         "cpu",
}

PERSON_CLASS_ID = 0   # COCO class 0 = person


class PersonDetector:
    """
    YOLOv8-based person detector.

    Usage (called by pipeline.py):
        detector = PersonDetector(cfg["detector"])
        dets = detector.detect(frame_rgb, frame_idx=0)
        # dets: list of [x1, y1, x2, y2, confidence]

    Compatible with deep_sort_realtime DeepSORT tracker input format.
    """

    def __init__(self, cfg: dict = None) -> None:
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}

        device = self.cfg.get("device", "cpu")
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.conf  = self.cfg["conf_threshold"]
        self.iou   = self.cfg["iou_threshold"]

        if not _YOLO_AVAILABLE:
            raise RuntimeError(
                "ultralytics is not installed.\n"
                "Run: pip install ultralytics"
            )

        logger.info(f"[Detector] Loading YOLOv8 from '{self.cfg['model_path']}' on {self.device}")
        self.model = YOLO(self.cfg["model_path"])
        self.model.to(self.device)
        logger.info("[Detector] YOLOv8 ready.")

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> list[list[float]]:
        """
        Detect persons in a single RGB frame.

        Args:
            frame     : np.ndarray HxWx3 RGB uint8
            frame_idx : frame number (for logging only)

        Returns:
            list of detections, each: [x1, y1, x2, y2, confidence]
            Empty list if no persons detected.
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append([float(x1), float(y1),
                                    float(x2), float(y2), conf])

        return detections

    def detect_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[list[list[float]]]:
        """
        Detect persons in a batch of frames.

        Returns:
            list of detection lists, one per frame.
        """
        results = self.model(
            frames,
            conf=self.conf,
            iou=self.iou,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        batch_dets = []
        for r in results:
            dets = []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    dets.append([float(x1), float(y1),
                                  float(x2), float(y2), conf])
            batch_dets.append(dets)

        return batch_dets


# ── expose at package level ───────────────────────────────
__all__ = ["PersonDetector"]