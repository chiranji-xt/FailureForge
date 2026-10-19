"""Shadows / lighting family (9): cast shadows, uneven illumination, local darkening.

Approximate synthetic lighting - documented as not physically accurate, but
seed-reproducible and geometry-preserving.
"""
from __future__ import annotations

import numpy as np

from .base import Context, ParamSpec, Transform, clip_u8, register, to_float


def _coords(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    return xx.astype(np.float32), yy.astype(np.float32)


def _keep(ctx: Context, out_bgr: np.ndarray, **extra) -> tuple:
    return out_bgr, ctx.boxes, {"geometry_changed": False, **extra}


def _cast_shadow(ctx: Context):
    intensity = float(ctx.params["intensity"])
    coverage = float(ctx.params["coverage"])
    h, w = ctx.image.shape[:2]
    xx, yy = _coords(h, w)
    t = np.deg2rad(float(ctx.params["angle"]))
    v = -xx * np.sin(t) + yy * np.cos(t)
    center = float(ctx.rng.uniform(v.min(), v.max()))
    width = coverage * min(w, h)
    mask = (np.abs(v - center) < width / 2.0).astype(np.float32)
    out = to_float(ctx.image) * (1.0 - intensity * mask[:, :, None])
    return _keep(ctx, clip_u8(out), intensity=intensity, coverage=coverage,
                 angle=float(ctx.params["angle"]))


def _uneven_illumination(ctx: Context):
    strength = float(ctx.params["strength"])
    h, w = ctx.image.shape[:2]
    xx, yy = _coords(h, w)
    t = np.deg2rad(float(ctx.params["angle"]))
    proj = xx * np.cos(t) + yy * np.sin(t)
    proj = (proj - proj.min()) / max(1e-6, (proj.max() - proj.min()))
    factor = 1.0 - strength * proj
    out = to_float(ctx.image) * factor[:, :, None]
    return _keep(ctx, clip_u8(out), strength=strength, angle=float(ctx.params["angle"]))


def _local_darkening(ctx: Context):
    intensity = float(ctx.params["intensity"])
    radius = float(ctx.params["radius"])
    h, w = ctx.image.shape[:2]
    xx, yy = _coords(h, w)
    cx = float(ctx.rng.uniform(0, w))
    cy = float(ctx.rng.uniform(0, h))
    r = max(1.0, radius * min(w, h))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(1.0 - dist / r, 0.0, 1.0)
    out = to_float(ctx.image) * (1.0 - intensity * mask[:, :, None])
    return _keep(ctx, clip_u8(out), intensity=intensity, radius=radius,
                 center=[cx, cy])


def register_lighting() -> None:
    register(Transform(
        "cast_shadow", "lighting",
        {"intensity": ParamSpec(0.0, 1.0, default=0.5, doc="darkening strength"),
         "coverage": ParamSpec(0.05, 1.0, default=0.4, doc="shadow band width (fraction of min side)"),
         "angle": ParamSpec(0.0, 180.0, default=45.0, search=False, doc="band orientation (deg)")},
        _cast_shadow, doc="A straight cast-shadow band across the image."))
    register(Transform(
        "uneven_illumination", "lighting",
        {"strength": ParamSpec(0.0, 1.0, default=0.4, doc="gradient darkening strength"),
         "angle": ParamSpec(0.0, 180.0, default=0.0, search=False, doc="gradient direction (deg)")},
        _uneven_illumination, doc="Linear illumination gradient (bright to dark)."))
    register(Transform(
        "local_darkening", "lighting",
        {"intensity": ParamSpec(0.0, 1.0, default=0.5, doc="darkening strength"),
         "radius": ParamSpec(0.05, 1.0, default=0.3, doc="dark spot radius (fraction of min side)")},
        _local_darkening, doc="A soft round dark spot at a random location."))
