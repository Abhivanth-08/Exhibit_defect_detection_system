"""
preprocessing/human_mask.py
============================
YOLOv8-based human masking filter.

Detects 'person' bounding boxes in a frame and blacks them out so that
the ViT encoder / JEPA model never sees human pixels.

Usage
-----
    from preprocessing.human_mask import HumanMaskFilter
    masker = HumanMaskFilter()          # loads YOLOv8n once
    clean  = masker.mask(frame_bgr)     # returns masked BGR frame (same size)
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Optional

_MODEL = None   # module-level singleton


def _get_model(weights: str = "yolov8n.pt"):
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(weights)
        _MODEL.fuse()          # speed optimisation
    return _MODEL


class HumanMaskFilter:
    """
    Masks human (person) regions in a frame using YOLOv8.

    Parameters
    ----------
    weights     : YOLO checkpoint — 'yolov8n.pt' is tiny + fast (auto-downloaded).
    conf        : confidence threshold for person detection (default 0.35).
    use_mask    : if True, use segmentation masks when available; otherwise use bbox.
    fill_value  : pixel value used for masked regions (default 0 = black).
    """

    PERSON_CLASS = 0   # COCO class 0 = person

    def __init__(
        self,
        weights: str   = "yolov8n.pt",
        conf: float    = 0.35,
        use_mask: bool = False,
        fill_value: int = 0,
    ):
        self.model      = _get_model(weights)
        self.conf       = conf
        self.use_mask   = use_mask
        self.fill_value = fill_value

    # ── public API ────────────────────────────────────────────────────────────

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply person masking to a BGR or RGB frame.

        Parameters
        ----------
        frame : np.ndarray  H×W×3, uint8

        Returns
        -------
        np.ndarray  same shape/dtype as input, with person regions filled.
        """
        out = frame.copy()
        boxes = self._detect_persons(frame)
        for x1, y1, x2, y2 in boxes:
            out[y1:y2, x1:x2] = self.fill_value
        return out

    def mask_with_alpha(self, frame: np.ndarray, alpha: float = 0.15) -> np.ndarray:
        """
        Semi-transparent masking — useful for debugging (keeps scene visible).
        alpha=0 → fully black, alpha=1 → unchanged.
        """
        out      = frame.copy().astype(np.float32)
        overlay  = np.full_like(out, self.fill_value, dtype=np.float32)
        boxes    = self._detect_persons(frame)
        for x1, y1, x2, y2 in boxes:
            out[y1:y2, x1:x2] = (
                alpha * out[y1:y2, x1:x2] + (1 - alpha) * overlay[y1:y2, x1:x2]
            )
        return out.astype(np.uint8)

    def has_person(self, frame: np.ndarray) -> bool:
        """Return True if at least one person is detected."""
        return len(self._detect_persons(frame)) > 0

    def person_area_fraction(self, frame: np.ndarray) -> float:
        """
        Fraction of frame area covered by detected persons (0.0 – 1.0).
        Can be used to skip or down-weight heavily occluded frames.
        """
        h, w   = frame.shape[:2]
        total  = h * w
        masked = 0
        for x1, y1, x2, y2 in self._detect_persons(frame):
            masked += (x2 - x1) * (y2 - y1)
        return min(masked / total, 1.0) if total > 0 else 0.0

    # ── internals ─────────────────────────────────────────────────────────────

    def _detect_persons(self, frame: np.ndarray) -> list[tuple[int,int,int,int]]:
        """Run YOLO and return list of (x1,y1,x2,y2) integer boxes for persons."""
        results = self.model(frame, conf=self.conf, classes=[self.PERSON_CLASS],
                             verbose=False)
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if int(box.cls[0]) != self.PERSON_CLASS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                h, w = frame.shape[:2]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                boxes.append((x1, y1, x2, y2))
        return boxes
