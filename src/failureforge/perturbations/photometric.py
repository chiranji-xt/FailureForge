"""Photometric families: noise (1), brightness/exposure (2), color (5)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance

from .base import (
    Context, ParamSpec, Transform, bgr_to_pil, clip_u8, pil_to_bgr, register,
    to_float,
)


def _keep_geometry(ctx: Context, out_bgr: np.ndarray, **extra) -> tuple:
    return out_bgr, ctx.boxes, {"geometry_changed": False, **extra}


# ---------------------------------------------------------------------------
# Family 1 - Noise
# ---------------------------------------------------------------------------

def _gaussian_noise(ctx: Context):
    sigma = float(ctx.params["sigma"])
    noise = ctx.rng.normal(0.0, sigma, size=ctx.image.shape)
    out = clip_u8(to_float(ctx.image) + noise)
    return _keep_geometry(ctx, out, noise_sigma=sigma)


def _salt_pepper_noise(ctx: Context):
    amount = float(ctx.params["amount"])
    salt_ratio = float(ctx.params["salt_ratio"])
    h, w = ctx.image.shape[:2]
    out = ctx.image.copy()
    selected = ctx.rng.random((h, w)) < amount
    salt = selected & (ctx.rng.random((h, w)) < salt_ratio)
    pepper = selected & ~salt
    out[salt] = 255
    out[pepper] = 0
    return _keep_geometry(ctx, out, noise_amount=amount, salt_ratio=salt_ratio)


def _speckle_noise(ctx: Context):
    sigma = float(ctx.params["sigma"])
    noise = ctx.rng.normal(0.0, 1.0, size=ctx.image.shape) * sigma
    out = clip_u8(to_float(ctx.image) * (1.0 + noise))
    return _keep_geometry(ctx, out, noise_sigma=sigma)


# ---------------------------------------------------------------------------
# Family 2 - Brightness / exposure
# ---------------------------------------------------------------------------

def _darken(ctx: Context):
    factor = float(ctx.params["factor"])
    out = clip_u8(to_float(ctx.image) * factor)
    return _keep_geometry(ctx, out, factor=factor)


def _brighten(ctx: Context):
    factor = float(ctx.params["factor"])
    out = clip_u8(to_float(ctx.image) * factor)
    return _keep_geometry(ctx, out, factor=factor)


def _overexposure(ctx: Context):
    strength = float(ctx.params["strength"])
    if strength == 0.0:
        return _keep_geometry(ctx, ctx.image.copy(), strength=0.0)
    x = to_float(ctx.image) / 255.0
    k = 1.0 + 4.0 * strength  # >1 pushes highlights toward white
    out = clip_u8((1.0 - np.power(1.0 - x, k)) * 255.0)
    return _keep_geometry(ctx, out, strength=strength)


def _brightness(ctx: Context):
    factor = float(ctx.params["factor"])
    out = clip_u8(to_float(ctx.image) * factor)
    return _keep_geometry(ctx, out, factor=factor)


# ---------------------------------------------------------------------------
# Family 5 - Colour distortion
# ---------------------------------------------------------------------------

def _hue_shift(ctx: Context):
    degrees = float(ctx.params["degrees"])
    if degrees == 0.0:
        return _keep_geometry(ctx, ctx.image.copy(), degrees=0.0)
    pil = bgr_to_pil(ctx.image).convert("HSV")
    h, s, v = pil.split()
    arr = np.asarray(h, dtype=np.int32)
    arr = (arr + int(round(degrees / 360.0 * 255.0))) % 256
    h = Image.fromarray(arr.astype(np.uint8), mode="L")
    out = pil_to_bgr(Image.merge("HSV", (h, s, v)).convert("RGB"))
    return _keep_geometry(ctx, out, degrees=degrees)


def _saturation(ctx: Context):
    factor = float(ctx.params["factor"])
    out = pil_to_bgr(ImageEnhance.Color(bgr_to_pil(ctx.image)).enhance(factor))
    return _keep_geometry(ctx, out, factor=factor)


def _contrast(ctx: Context):
    factor = float(ctx.params["factor"])
    out = pil_to_bgr(ImageEnhance.Contrast(bgr_to_pil(ctx.image)).enhance(factor))
    return _keep_geometry(ctx, out, factor=factor)


def _color_temperature(ctx: Context):
    warmth = float(ctx.params["warmth"])  # -1 cool .. +1 warm
    out = to_float(ctx.image)
    out[:, :, 0] *= 1.0 - 0.3 * warmth   # blue
    out[:, :, 2] *= 1.0 + 0.3 * warmth   # red
    return _keep_geometry(ctx, clip_u8(out), warmth=warmth)


def register_photometric() -> None:
    register(Transform(
        "gaussian_noise", "noise",
        {"sigma": ParamSpec(0.0, 50.0, default=15.0, doc="std-dev of additive Gaussian noise")},
        _gaussian_noise, doc="Additive zero-mean Gaussian noise (pixel amplitude)."))

    register(Transform(
        "salt_pepper_noise", "noise",
        {"amount": ParamSpec(0.0, 0.2, default=0.02, doc="fraction of affected pixels"),
         "salt_ratio": ParamSpec(0.0, 1.0, default=0.5, search=False,
                                 doc="share of affected pixels set white")},
        _salt_pepper_noise, doc="Impulse noise: random white/black pixels."))

    register(Transform(
        "speckle_noise", "noise",
        {"sigma": ParamSpec(0.0, 0.5, default=0.1, doc="multiplicative noise strength")},
        _speckle_noise, doc="Multiplicative speckle noise (x * (1 + N(0, sigma)))."))

    register(Transform(
        "darken", "exposure",
        {"factor": ParamSpec(0.0, 1.0, default=0.5, doc="multiplicative gain <= 1")},
        _darken, doc="Uniform darkening (scale pixel values down)."))

    register(Transform(
        "brighten", "exposure",
        {"factor": ParamSpec(1.0, 3.0, default=1.5, doc="multiplicative gain >= 1")},
        _brighten, doc="Uniform brightening (scale pixel values up, then clip)."))

    register(Transform(
        "overexposure", "exposure",
        {"strength": ParamSpec(0.0, 1.0, default=0.5, doc="highlight saturation strength")},
        _overexposure, doc="Saturating gain that blows highlights toward white."))

    register(Transform(
        "brightness", "exposure",
        {"factor": ParamSpec(0.3, 2.0, default=1.0, doc="generic multiplicative gain")},
        _brightness, doc="Generic brightness gain (legacy-compatible)."))

    register(Transform(
        "hue_shift", "color",
        {"degrees": ParamSpec(-180.0, 180.0, default=30.0, doc="hue rotation in degrees")},
        _hue_shift, doc="Rotate hue channel (HSV), preserving S/V."))

    register(Transform(
        "saturation", "color",
        {"factor": ParamSpec(0.0, 2.0, default=1.5, doc="1 = unchanged, 0 = grayscale")},
        _saturation, doc="Scale colour saturation."))

    register(Transform(
        "contrast", "color",
        {"factor": ParamSpec(0.0, 2.0, default=1.5, doc="1 = unchanged")},
        _contrast, doc="Scale contrast."))

    register(Transform(
        "color_temperature", "color",
        {"warmth": ParamSpec(-1.0, 1.0, default=0.3, doc="-1 cool .. +1 warm")},
        _color_temperature, doc="Warm/cool tint by scaling blue and red channels."))
