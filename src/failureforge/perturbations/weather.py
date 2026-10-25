"""Weather / visibility family (4): fog, haze, rain, snow.

These are lightweight approximations, not physically accurate weather models,
but they are seed-reproducible and cheap. Effects do not change geometry.
"""
from __future__ import annotations

import numpy as np

from .base import Context, ParamSpec, Transform, clip_u8, register, to_float


def _keep(ctx: Context, out_bgr: np.ndarray, **extra) -> tuple:
    return out_bgr, ctx.boxes, {"geometry_changed": False, **extra}


def _fog(ctx: Context):
    intensity = float(ctx.params["intensity"])
    out = to_float(ctx.image) * (1.0 - intensity) + 255.0 * intensity
    return _keep(ctx, clip_u8(out), intensity=intensity)


def _haze(ctx: Context):
    intensity = float(ctx.params["intensity"])
    blue_gray = np.array([215.0, 205.0, 190.0], dtype=np.float32)  # BGR tint
    x = to_float(ctx.image) / 255.0
    x = np.power(x, 1.0 + 0.5 * intensity)  # mild contrast compression
    out = (x * 255.0) * (1.0 - intensity) + blue_gray * intensity
    return _keep(ctx, clip_u8(out), intensity=intensity)


def _rain(ctx: Context):
    intensity = float(ctx.params["intensity"])
    h, w = ctx.image.shape[:2]
    out = to_float(ctx.image) * (1.0 - 0.2 * intensity)
    n = int(intensity * min(w, h))
    max_len = max(6, min(w, h) // 4)
    hits = np.zeros(h * w, dtype=np.int64)
    for _ in range(n):  # RNG draw order is part of the seeded contract
        x0 = int(ctx.rng.integers(0, w))
        y0 = int(ctx.rng.integers(0, h))
        length = int(ctx.rng.integers(5, max_len + 1))
        slant = max(1, length // 3)
        t = np.arange(length)
        x = x0 + (t * slant) // length
        y = y0 + t
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        np.add.at(hits, y[ok] * w + x[ok], 1)
    # A pixel hit once gains +170 (capped at 255); hit twice or more is always 255
    # (255 + 170 caps at 255 and 0 + 340 > 255), matching the sequential result.
    hits = hits.reshape(h, w)
    once, many = hits == 1, hits >= 2
    out[once] = np.minimum(255.0, out[once] + 170.0)
    out[many] = 255.0
    return _keep(ctx, clip_u8(out), intensity=intensity, streaks=n)


def _snow(ctx: Context):
    intensity = float(ctx.params["intensity"])
    h, w = ctx.image.shape[:2]
    out = to_float(ctx.image) * (1.0 + 0.1 * intensity)
    n = int(intensity * (w * h) / 40.0)
    for _ in range(n):
        x = int(ctx.rng.integers(0, w))
        y = int(ctx.rng.integers(0, h))
        r = int(ctx.rng.integers(1, 3))
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(w, x + r + 1), min(h, y + r + 1)
        out[y1:y2, x1:x2] = 255.0
    return _keep(ctx, clip_u8(out), intensity=intensity, flakes=n)


def register_weather() -> None:
    register(Transform(
        "fog", "weather",
        {"intensity": ParamSpec(0.0, 1.0, default=0.4, doc="blend toward white")},
        _fog, doc="Synthetic fog: uniform blend toward white."))
    register(Transform(
        "haze", "weather",
        {"intensity": ParamSpec(0.0, 1.0, default=0.3, doc="blend + contrast compression")},
        _haze, doc="Synthetic haze with a light blue-grey tint (approximation)."))
    register(Transform(
        "rain", "weather",
        {"intensity": ParamSpec(0.0, 1.0, default=0.4, doc="density of streaks")},
        _rain, doc="Synthetic diagonal rain streaks (fixed rng, not physical)."))
    register(Transform(
        "snow", "weather",
        {"intensity": ParamSpec(0.0, 1.0, default=0.4, doc="density of flakes")},
        _snow, doc="Synthetic snow flakes (fixed rng, not physical)."))
