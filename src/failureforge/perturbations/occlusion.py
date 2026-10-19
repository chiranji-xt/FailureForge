"""Occlusion family (6): cutout, random patches, partial obstruction.

Occluders are painted over the image. GT box geometry is unchanged by default
(the object is still where it was); set ``remove_fully_occluded=True`` to drop
boxes whose covered fraction reaches ``full_threshold`` (legacy behaviour).
"""
from __future__ import annotations

import numpy as np

from .base import Context, ParamSpec, Transform, register


def _covered_fraction(box, rects) -> float:
    if hasattr(box, "x1"):
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    else:
        x1, y1, x2, y2 = list(box)[:4]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0:
        return 1.0
    inter = 0.0
    for (rx1, ry1, rx2, ry2) in rects:
        ix1, iy1 = max(x1, rx1), max(y1, ry1)
        ix2, iy2 = min(x2, rx2), min(y2, ry2)
        inter += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return min(1.0, inter / area)


def _apply_box_policy(ctx: Context, rects: list) -> tuple[list | None, list, dict]:
    if ctx.boxes is None:
        return None, [], {}
    if not ctx.params.get("remove_fully_occluded", False):
        return ctx.boxes, [], {}
    threshold = float(ctx.params.get("full_threshold", 0.85))
    kept, removed = [], []
    for box in ctx.boxes:
        frac = _covered_fraction(box, rects)
        if frac >= threshold:
            removed.append({"class_id": int(getattr(box, "class_id", 0)),
                            "reason": "fully_occluded", "occluded_fraction": frac})
        else:
            kept.append(box)
    return kept, removed, {"removed_boxes": removed}


def _paint(image: np.ndarray, rects_and_colors) -> np.ndarray:
    out = image.copy()
    for (x0, y0, x1, y1), color in rects_and_colors:
        out[y0:y1, x0:x1] = color
    return out


def _occlusion(ctx: Context):
    ratio = float(ctx.params["ratio"])
    h, w = ctx.image.shape[:2]
    side = int(round(ratio * min(w, h)))
    if side <= 0:
        return ctx.image.copy(), ctx.boxes, {"geometry_changed": False, "occluders": []}
    max_x, max_y = max(0, w - side), max(0, h - side)
    x0 = int(ctx.rng.integers(0, max_x + 1))
    y0 = int(ctx.rng.integers(0, max_y + 1))
    rect = (x0, y0, min(w, x0 + side), min(h, y0 + side))
    out = _paint(ctx.image, [(rect, np.array([0, 0, 0], dtype=np.uint8))])
    kept, removed, extra = _apply_box_policy(ctx, [rect])
    extra.update({"occluders": [list(rect)], "cutout_xyxy": list(rect)})
    return out, kept, {"geometry_changed": False, **extra}


def _random_patch(ctx: Context):
    ratio = float(ctx.params["ratio"])
    n = int(ctx.params["num_patches"])
    h, w = ctx.image.shape[:2]
    side = int(round(ratio * min(w, h)))
    if side <= 0:
        return ctx.image.copy(), ctx.boxes, {"geometry_changed": False, "occluders": []}
    rects, paints = [], []
    for _ in range(n):
        x0 = int(ctx.rng.integers(0, max(1, w)))
        y0 = int(ctx.rng.integers(0, max(1, h)))
        rect = (x0, y0, min(w, x0 + side), min(h, y0 + side))
        color = ctx.rng.integers(0, 256, size=3).astype(np.uint8)
        rects.append(rect)
        paints.append((rect, color))
    out = _paint(ctx.image, paints)
    kept, removed, extra = _apply_box_policy(ctx, rects)
    extra.update({"occluders": [list(r) for r in rects]})
    return out, kept, {"geometry_changed": False, **extra}


def _partial_obstruction(ctx: Context):
    ratio = float(ctx.params["ratio"])
    side = str(ctx.params["side"])
    h, w = ctx.image.shape[:2]
    thickness = int(round(ratio * min(w, h)))
    if thickness <= 0:
        return ctx.image.copy(), ctx.boxes, {"geometry_changed": False, "occluders": []}
    if side == "top":
        rect = (0, 0, w, min(h, thickness))
    elif side == "bottom":
        rect = (0, max(0, h - thickness), w, h)
    elif side == "left":
        rect = (0, 0, min(w, thickness), h)
    else:  # right
        rect = (max(0, w - thickness), 0, w, h)
    out = _paint(ctx.image, [(rect, np.array([0, 0, 0], dtype=np.uint8))])
    kept, removed, extra = _apply_box_policy(ctx, [rect])
    extra.update({"occluders": [list(rect)], "obstruction_side": side})
    return out, kept, {"geometry_changed": False, **extra}


def register_occlusion() -> None:
    register(Transform(
        "occlusion", "occlusion",
        {"ratio": ParamSpec(0.0, 0.6, default=0.2, doc="cutout side as a fraction of min(W,H)"),
         "remove_fully_occluded": ParamSpec(choices=(True, False), default=False, search=False,
                                            doc="drop boxes covered past the threshold"),
         "full_threshold": ParamSpec(0.0, 1.0, default=0.85, search=False,
                                     doc="covered fraction counted as removed")},
        _occlusion, doc="Single random black square cutout (legacy-compatible)."),
        aliases=("cutout",))
    register(Transform(
        "random_patch", "occlusion",
        {"ratio": ParamSpec(0.0, 0.6, default=0.15, doc="patch side as a fraction of min(W,H)"),
         "num_patches": ParamSpec(1, 10, default=4, integer=True, doc="number of patches")},
        _random_patch, doc="Several random filled rectangular patches."))
    register(Transform(
        "partial_obstruction", "occlusion",
        {"ratio": ParamSpec(0.0, 0.6, default=0.2, doc="band thickness as a fraction of min(W,H)"),
         "side": ParamSpec(choices=("top", "bottom", "left", "right"), default="bottom",
                           search=False, doc="image edge the band covers")},
        _partial_obstruction, doc="A band obscuring one image edge."))
