"""Geometric family (7): rotation, scaling, translation, perspective.

Ground-truth boxes are always transformed with the same matrix used for the
image, then clipped and rebuilt as axis-aligned boxes.
"""
from __future__ import annotations

import numpy as np

from .base import Context, ParamSpec, Transform, register
from .boxops import (
    apply_affine, homography, perspective_from_homography,
    rotation_forward, scale_forward, translate_forward, warp_affine,
    warp_perspective,
)


def _geometry_result(ctx: Context, out_bgr: np.ndarray, kept, removed, **extra):
    return out_bgr, kept, {"geometry_changed": True, "removed_boxes": removed, **extra}


def _rotation(ctx: Context):
    angle = float(ctx.params["angle"])
    h, w = ctx.image.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    forward = rotation_forward(angle, cx, cy)
    out = warp_affine(ctx.pil(), forward, (w, h))
    kept, removed = apply_affine(ctx.boxes, forward, w, h)
    return _geometry_result(ctx, np.asarray(out, dtype=np.uint8)[:, :, ::-1].copy(),
                            kept, removed, angle=angle)


def _scaling(ctx: Context):
    scale = float(ctx.params["scale"])
    h, w = ctx.image.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    forward = scale_forward(scale, cx, cy)
    out = warp_affine(ctx.pil(), forward, (w, h))
    kept, removed = apply_affine(ctx.boxes, forward, w, h)
    return _geometry_result(ctx, np.asarray(out, dtype=np.uint8)[:, :, ::-1].copy(),
                            kept, removed, scale=scale)


def _translation(ctx: Context):
    h, w = ctx.image.shape[:2]
    dx = float(ctx.params["dx"]) * w
    dy = float(ctx.params["dy"]) * h
    forward = translate_forward(dx, dy)
    out = warp_affine(ctx.pil(), forward, (w, h))
    kept, removed = apply_affine(ctx.boxes, forward, w, h)
    return _geometry_result(ctx, np.asarray(out, dtype=np.uint8)[:, :, ::-1].copy(),
                            kept, removed, dx=float(ctx.params["dx"]), dy=float(ctx.params["dy"]))


def _perspective(ctx: Context):
    jitter = float(ctx.params["jitter"])
    h, w = ctx.image.shape[:2]
    if min(w, h) < 4:
        return ctx.image.copy(), ctx.boxes, {
            "geometry_changed": True, "removed_boxes": [], "jitter": jitter,
            "note": "image too small for perspective; returned unchanged"}
    span = jitter * min(w, h)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    offsets = ctx.rng.uniform(-span, span, size=(4, 2))
    dst = src + offsets
    hmat = homography(src, dst)
    out = warp_perspective(ctx.pil(), hmat, (w, h))
    kept, removed = perspective_from_homography(ctx.boxes, hmat, w, h)
    return _geometry_result(ctx, np.asarray(out, dtype=np.uint8)[:, :, ::-1].copy(),
                            kept, removed, jitter=jitter)


def register_geometry() -> None:
    register(Transform(
        "rotation", "geometric",
        {"angle": ParamSpec(-90.0, 90.0, default=15.0, doc="CCW degrees; canvas size kept")},
        _rotation, geometry=True, doc="Rotate about the image centre; boxes rotated + clipped."))
    register(Transform(
        "scaling", "geometric",
        {"scale": ParamSpec(0.5, 2.0, default=1.2, doc="zoom about centre; canvas kept")},
        _scaling, geometry=True, doc="Uniform zoom about the centre."))
    register(Transform(
        "translation", "geometric",
        {"dx": ParamSpec(-0.5, 0.5, default=0.1, doc="x shift as a fraction of width"),
         "dy": ParamSpec(-0.5, 0.5, default=0.0, doc="y shift as a fraction of height")},
        _translation, geometry=True, doc="Shift content; canvas kept, exposed area black."))
    register(Transform(
        "perspective", "geometric",
        {"jitter": ParamSpec(0.0, 0.15, default=0.05, doc="corner jitter as a fraction of min(W,H)")},
        _perspective, geometry=True, doc="Random 4-corner perspective warp; boxes re-mapped."))
