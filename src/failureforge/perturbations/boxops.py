"""Pixel-space box mapping helpers for geometric/cropping transforms.

A *forward* affine is a 2x3 matrix ``M`` mapping source pixel ``(x, y)`` to
destination ``(x', y')``::

    [x']   [a b tx] [x]
    [y'] = [c d ty] [y]
    [1 ]   [0 0 1 ] [1]

Images are warped with the *inverse* (PIL's ``Image.transform`` samples the
source for each destination pixel), while boxes are mapped with the forward
matrix, so image content and labels stay consistent.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from ..dataset import GroundTruthBox
from .base import MIN_BOX_SIDE_PX


def _box_coords(box) -> tuple[int, float, float, float, float]:
    if hasattr(box, "x1"):
        return (int(getattr(box, "class_id", 0)), float(box.x1), float(box.y1),
                float(box.x2), float(box.y2))
    seq = list(box)
    if len(seq) == 5:
        cid, x1, y1, x2, y2 = seq
        return (int(cid), float(x1), float(y1), float(x2), float(y2))
    x1, y1, x2, y2 = seq
    return (0, float(x1), float(y1), float(x2), float(y2))


def _removed_entry(cid, x1, y1, x2, y2, reason) -> dict:
    return {"class_id": cid, "box": [x1, y1, x2, y2], "reason": reason}


def map_boxes(
    boxes: list | None,
    point_fn: Callable[[float, float], tuple[float, float]],
    out_w: int,
    out_h: int,
    min_side: float = MIN_BOX_SIDE_PX,
) -> tuple[list, list]:
    """Map box corners through ``point_fn``, rebuild AABBs, clip, drop tiny ones.

    Returns ``(kept, removed)`` where ``kept`` are ``GroundTruthBox`` objects and
    ``removed`` records the dropped boxes with a reason.
    """
    kept: list[GroundTruthBox] = []
    removed: list[dict] = []
    for box in boxes or []:
        cid, x1, y1, x2, y2 = _box_coords(box)
        xs, ys = [], []
        for (px, py) in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
            nx, ny = point_fn(px, py)
            xs.append(nx)
            ys.append(ny)
        nx1, ny1, nx2, ny2 = min(xs), min(ys), max(xs), max(ys)
        cx1, cy1 = max(0.0, nx1), max(0.0, ny1)
        cx2, cy2 = min(float(out_w), nx2), min(float(out_h), ny2)
        if (cx2 - cx1) < min_side or (cy2 - cy1) < min_side:
            removed.append(_removed_entry(cid, x1, y1, x2, y2, "degenerate_after_transform"))
            continue
        kept.append(GroundTruthBox(cid, cx1, cy1, cx2, cy2))
    return kept, removed


# ---------------------------------------------------------------------------
# Affine helpers
# ---------------------------------------------------------------------------

def rotation_forward(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, cx - c * cx + s * cy],
                     [s, c, cy - s * cx - c * cy]], dtype=np.float64)


def scale_forward(scale: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[scale, 0.0, cx - scale * cx],
                     [0.0, scale, cy - scale * cy]], dtype=np.float64)


def translate_forward(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx],
                     [0.0, 1.0, ty]], dtype=np.float64)


def invert_affine(m: np.ndarray) -> np.ndarray:
    a, b, tx = m[0]
    c, d, ty = m[1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("non-invertible affine transform")
    ia, ib = d / det, -b / det
    ic, id_ = -c / det, a / det
    itx = -(ia * tx + ib * ty)
    ity = -(ic * tx + id_ * ty)
    return np.array([[ia, ib, itx], [ic, id_, ity]], dtype=np.float64)


def apply_affine(boxes: list | None, m: np.ndarray, out_w: int, out_h: int):
    def point(x: float, y: float) -> tuple[float, float]:
        return (m[0, 0] * x + m[0, 1] * y + m[0, 2],
                m[1, 0] * x + m[1, 1] * y + m[1, 2])

    return map_boxes(boxes, point, out_w, out_h)


def warp_affine(pil: Image.Image, forward: np.ndarray, size: tuple[int, int],
                fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    inv = invert_affine(forward)
    data = (inv[0, 0], inv[0, 1], inv[0, 2], inv[1, 0], inv[1, 1], inv[1, 2])
    return pil.transform(size, Image.AFFINE, data, resample=Image.BILINEAR, fillcolor=fill)


# ---------------------------------------------------------------------------
# Perspective helpers
# ---------------------------------------------------------------------------

def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping 4 ``src`` points to 4 ``dst`` points."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(a, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography")
    return h / h[2, 2]


def perspective_from_homography(boxes: list | None, h: np.ndarray,
                                out_w: int, out_h: int):
    def point(x: float, y: float) -> tuple[float, float]:
        v = h @ np.array([x, y, 1.0])
        return (v[0] / v[2], v[1] / v[2])

    return map_boxes(boxes, point, out_w, out_h)


def warp_perspective(pil: Image.Image, h_src_to_dst: np.ndarray,
                     size: tuple[int, int], fill=(0, 0, 0)) -> Image.Image:
    h_inv = np.linalg.inv(h_src_to_dst)
    data = (h_inv[0, 0], h_inv[0, 1], h_inv[0, 2],
            h_inv[1, 0], h_inv[1, 1], h_inv[1, 2],
            h_inv[2, 0], h_inv[2, 1])
    return pil.transform(size, Image.PERSPECTIVE, data, resample=Image.BILINEAR, fillcolor=fill)
