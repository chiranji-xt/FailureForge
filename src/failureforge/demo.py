"""Self-contained demo data and a tiny rule-based detector (no weights needed).

``ColorBlobDetector`` is a deterministic colour-threshold "detector" that finds
one coloured square. It exists so the full pipeline (baseline, perturbations,
search, reports) can be exercised end to end without a model download. It is
**not** a YOLO model and its results say nothing about real detectors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .detector import Detections

SQUARE_RGB = (60, 120, 200)
BACKGROUND_RGB = (220, 220, 220)


def make_demo_dataset(root: str | Path, n: int = 8, size: int = 128) -> Path:
    """Write a tiny YOLO dataset (``test`` split) and return its ``data.yaml``.

    Each image holds one square at a deterministic, image-dependent position.
    """
    root = Path(root)
    images = root / "images" / "test"
    labels = root / "labels" / "test"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    side = size // 3
    for i in range(n):
        cx = size // 2 + ((i * 17) % 21) - 10
        cy = size // 2 + ((i * 11) % 21) - 10
        x1, y1 = cx - side // 2, cy - side // 2
        img = Image.new("RGB", (size, size), BACKGROUND_RGB)
        ImageDraw.Draw(img).rectangle([x1, y1, x1 + side - 1, y1 + side - 1], fill=SQUARE_RGB)
        img.save(images / f"demo{i}.png")
        (labels / f"demo{i}.txt").write_text(
            f"0 {(x1 + side / 2) / size:.6f} {(y1 + side / 2) / size:.6f} "
            f"{side / size:.6f} {side / size:.6f}\n",
            encoding="utf-8",
        )
    yaml_path = root / "data.yaml"
    yaml_path.write_text("path: .\ntest: images/test\nnames: [square]\n", encoding="utf-8")
    return yaml_path


class ColorBlobDetector:
    """Detects pixels near ``SQUARE_RGB`` and returns their bounding box."""

    def __init__(self, tolerance: int = 40, min_pixels: int = 50) -> None:
        self.tolerance = int(tolerance)
        self.min_pixels = int(min_pixels)
        self.class_names = ["square"]
        self.weights = None
        self.sha256 = None
        self.task = "detect"
        self.imgsz = None
        self.device = "cpu"
        self.conf = 0.25

    def predict(self, image_bgr: np.ndarray) -> Detections:
        target = np.array(SQUARE_RGB[::-1], dtype=np.int16)
        diff = np.abs(image_bgr.astype(np.int16) - target).max(axis=2)
        mask = diff <= self.tolerance
        ys, xs = np.nonzero(mask)
        if xs.size < self.min_pixels:
            return Detections(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                classes=np.zeros((0,), dtype=np.int64),
            )
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        fill = xs.size / float((x2 - x1) * (y2 - y1))
        return Detections(
            boxes_xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float32),
            scores=np.array([min(max(fill, 0.0), 1.0)], dtype=np.float32),
            classes=np.array([0], dtype=np.int64),
        )
