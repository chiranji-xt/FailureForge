"""Lens-effects family (10): glare, lens flare approximation, vignetting.

Simple, explainable image-processing approximations - explicitly *not* a
physically accurate optical simulation. Geometry is unchanged.
"""
from __future__ import annotations

import numpy as np

from .base import Context, ParamSpec, Transform, clip_u8, register, to_float


def _keep(ctx: Context, out_bgr: np.ndarray, **extra) -> tuple:
    return out_bgr, ctx.boxes, {"geometry_changed": False, **extra}


def _circle_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(1.0 - dist / max(1e-6, r), 0.0, 1.0).astype(np.float32)


def _glare(ctx: Context):
    intensity = float(ctx.params["intensity"])
    radius = float(ctx.params["radius"])
    h, w = ctx.image.shape[:2]
    cx = float(ctx.rng.uniform(0, w))
    cy = float(ctx.rng.uniform(0, h))
    r = max(1.0, radius * min(w, h))
    mask = _circle_mask(h, w, cx, cy, r)
    out = to_float(ctx.image) + intensity * 255.0 * mask[:, :, None]
    return _keep(ctx, clip_u8(out), intensity=intensity, radius=radius,
                 center=[cx, cy])


def _lens_flare(ctx: Context):
    intensity = float(ctx.params["intensity"])
    h, w = ctx.image.shape[:2]
    out = to_float(ctx.image)
    lx = float(ctx.rng.uniform(0, w))
    ly = float(ctx.rng.uniform(0, h))
    n = 5
    for i in range(n):
        f = (i + 1) / (n + 1)
        cx = lx + (w * 0.5 - lx) * (2 * f)
        cy = ly + (h * 0.5 - ly) * (2 * f)
        r = (0.05 + 0.1 * (1 - f)) * min(w, h)
        mask = _circle_mask(h, w, cx, cy, max(1.0, r))
        out = out + intensity * 120.0 * (1 - f) * mask[:, :, None]
    return _keep(ctx, clip_u8(out), intensity=intensity, light_source=[lx, ly],
                 spots=n)


def _vignetting(ctx: Context):
    strength = float(ctx.params["strength"])
    h, w = ctx.image.shape[:2]
    if strength == 0.0 or w < 2 or h < 2:
        return _keep(ctx, ctx.image.copy(), strength=strength)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    norm = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2.0)
    factor = 1.0 - strength * np.clip(norm, 0.0, 1.0) ** 2
    out = to_float(ctx.image) * factor[:, :, None]
    return _keep(ctx, clip_u8(out), strength=strength)


def register_lens() -> None:
    register(Transform(
        "glare", "lens",
        {"intensity": ParamSpec(0.0, 1.0, default=0.6, doc="brightness added at the spot"),
         "radius": ParamSpec(0.05, 1.0, default=0.25, doc="spot radius (fraction of min side)")},
        _glare, doc="A bright soft spot (glare) at a random location."))
    register(Transform(
        "lens_flare", "lens",
        {"intensity": ParamSpec(0.0, 1.0, default=0.4, doc="flare brightness")},
        _lens_flare, doc="A short chain of soft bright spots (flare approximation)."))
    register(Transform(
        "vignetting", "lens",
        {"strength": ParamSpec(0.0, 1.0, default=0.5, doc="corner darkening strength")},
        _vignetting, doc="Radial corner darkening (vignette)."))
