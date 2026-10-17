"""YOLO detector wrapper (inference only).

Loads an existing Ultralytics YOLO *detection* checkpoint supplied by the
user (e.g. ``best.pt``) and returns standardized :class:`Detections` that the
rest of FailureForge can use without touching Ultralytics result objects.

No training or fine-tuning happens here. The user must provide the weights
file; nothing is ever downloaded.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_TASK = "detect"
UNSUPPORTED_TASKS = ("segment", "pose", "obb", "classify")


class DetectorError(Exception):
    """Raised for any detector loading, validation, or inference failure."""


@dataclass
class Detections:
    """Standardized prediction result.

    - ``boxes_xyxy``: float32 array, shape ``(N, 4)``, pixel coords ``[x1, y1, x2, y2]``.
    - ``scores``: float32 array, shape ``(N,)``, confidence values.
    - ``classes``: int64 array, shape ``(N,)``, integer class IDs.
    - ``class_names``: model class names (informational, not per-box).

    Empty detections are valid: shapes ``(0, 4)``, ``(0,)``, ``(0,)``.
    """

    boxes_xyxy: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    class_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.boxes_xyxy = np.asarray(self.boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)
        n = len(self.scores)
        if len(self.boxes_xyxy) != n or len(self.classes) != n:
            raise ValueError(
                f"inconsistent detection array lengths: boxes={len(self.boxes_xyxy)}, "
                f"scores={n}, classes={len(self.classes)}"
            )

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0


def _empty_detections(class_names: list[str] | None = None) -> Detections:
    return Detections(
        boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        classes=np.zeros((0,), dtype=np.int64),
        class_names=list(class_names or []),
    )


def _validate_image(image_bgr: Any) -> tuple[int, int]:
    """Validate an OpenCV-style BGR image; return (height, width)."""
    if not isinstance(image_bgr, np.ndarray):
        raise DetectorError(
            f"image must be a NumPy BGR array, got {type(image_bgr).__name__}"
        )
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise DetectorError(
            f"image must have shape (H, W, 3), got {tuple(image_bgr.shape)}"
        )
    h, w, _ = image_bgr.shape
    if h < 1 or w < 1:
        raise DetectorError(f"image must be non-empty, got shape {(h, w, 3)}")
    if image_bgr.dtype != np.uint8:
        raise DetectorError(
            f"image dtype must be uint8 (OpenCV-style BGR), got {image_bgr.dtype}"
        )
    return h, w


class YOLODetector:
    """Ultralytics YOLO detection checkpoint wrapper (inference only).

    Parameters:
        weights: path to an existing detection checkpoint (e.g. ``best.pt``).
        conf: confidence threshold in [0, 1] (default 0.25).
        imgsz: inference image size in pixels, positive int (default 640).
        device: inference device, e.g. ``"cpu"`` or ``"cuda:0"``.
    """

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {path}")
        if not path.is_file():
            raise DetectorError(f"weights path is a directory, not a file: {path}")
        if not (0.0 <= float(conf) <= 1.0):
            raise DetectorError(f"conf must be in [0, 1], got {conf}")
        if int(imgsz) < 1:
            raise DetectorError(f"imgsz must be a positive int, got {imgsz}")

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Install it with: "
                'pip install "failureforge[yolo]"'
            ) from e

        try:
            self._model = YOLO(str(path))
        except Exception as e:
            raise DetectorError(f"failed to load YOLO weights {path}: {e}") from e

        task = getattr(self._model, "task", None)
        if task != SUPPORTED_TASK:
            raise DetectorError(
                f"unsupported model task {task!r}: FailureForge v0.1 supports "
                f"object detection ({SUPPORTED_TASK!r}) only; "
                f"rejected tasks include {', '.join(UNSUPPORTED_TASKS)}"
            )

        self.weights = str(path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.task = task
        self.class_names = self._normalize_names(getattr(self._model, "names", {}))
        self._sha256: str | None = None

    @staticmethod
    def _normalize_names(names: Any) -> list[str]:
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        return [str(n) for n in (names or [])]

    @property
    def sha256(self) -> str:
        """SHA-256 fingerprint of the weights file (computed once, cached)."""
        if self._sha256 is None:
            h = hashlib.sha256()
            with open(self.weights, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            self._sha256 = h.hexdigest()
        return self._sha256

    @property
    def metadata(self) -> dict[str, Any]:
        """Detector metadata (weights, config, task, classes, fingerprint)."""
        return {
            "weights": self.weights,
            "sha256": self.sha256,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "device": self.device,
            "task": self.task,
            "class_names": list(self.class_names),
        }

    def predict(self, image_bgr: np.ndarray) -> Detections:
        """Run inference on an OpenCV-style BGR ``uint8`` array ``(H, W, 3)``.

        Returns standardized :class:`Detections` in pixel coordinates.
        Zero detections yields valid empty arrays (not an error).
        """
        _validate_image(image_bgr)
        try:
            results = self._model.predict(
                image_bgr, conf=self.conf, imgsz=self.imgsz,
                device=self.device, verbose=False,
            )
        except Exception as e:
            raise DetectorError(f"YOLO inference failed: {e}") from e
        if not results:
            return _empty_detections(self.class_names)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return _empty_detections(self.class_names)
        try:
            xyxy = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(boxes.conf.cpu().numpy(), dtype=np.float32).reshape(-1)
            classes = np.asarray(boxes.cls.cpu().numpy(), dtype=np.int64).reshape(-1)
        except Exception as e:
            raise DetectorError(f"failed to normalize YOLO predictions: {e}") from e
        return Detections(
            boxes_xyxy=xyxy, scores=scores, classes=classes,
            class_names=list(self.class_names),
        )


class DummyDetector:
    """Deterministic stub for tests; returns fixed boxes regardless of input.

    Each box dict is expected to have at minimum ``class_id``; ``confidence``,
    ``x_center``, ``y_center``, ``width``, ``height`` are accepted but ignored
    by downstream code that only needs class identity.
    """

    def __init__(self, boxes: list[dict] | None = None):
        self.boxes = boxes or []

    def predict(self, image: object) -> list[dict]:
        return [dict(b) for b in self.boxes]


class ConstantRelativeDetector(DummyDetector):
    """Predicts the boxes it was constructed with (used to simulate baseline)."""
