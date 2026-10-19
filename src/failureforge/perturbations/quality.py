"""Image-quality family (8): JPEG artifacts, downscaling, pixelation.

Output images keep the original size (downscaled/pxielated variants are
upsampled back), so GT geometry is unchanged.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .base import (
    Context, ParamSpec, PerturbationError, Transform, bgr_to_pil, pil_to_bgr,
    register,
)


def _keep(ctx: Context, out_bgr: np.ndarray, **extra) -> tuple:
    return out_bgr, ctx.boxes, {"geometry_changed": False, **extra}


def _jpeg_compression(ctx: Context):
    quality = int(ctx.params["quality"])
    pil = ctx.pil()
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return _keep(ctx, pil_to_bgr(out), quality=quality)


def _downscale(ctx: Context):
    scale = float(ctx.params["scale"])
    pil = ctx.pil()
    w, h = pil.size
    small = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    out = pil.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return _keep(ctx, pil_to_bgr(out), scale=scale)


def _pixelate(ctx: Context):
    size = int(ctx.params["pixel_size"])
    pil = ctx.pil()
    w, h = pil.size
    small = (max(1, w // size), max(1, h // size))
    out = pil.resize(small, Image.NEAREST).resize((w, h), Image.NEAREST)
    return _keep(ctx, pil_to_bgr(out), pixel_size=size)


def register_quality() -> None:
    register(Transform(
        "jpeg_compression", "quality",
        {"quality": ParamSpec(1, 100, default=25, integer=True, doc="JPEG quality 1-100")},
        _jpeg_compression, doc="Re-encode as low-quality JPEG (blocking artifacts)."))
    register(Transform(
        "downscale", "quality",
        {"scale": ParamSpec(0.1, 0.9, default=0.4, doc="downscale factor before upsampling back")},
        _downscale, doc="Simulate low resolution (down then up-sample)."))
    register(Transform(
        "pixelate", "quality",
        {"pixel_size": ParamSpec(2, 32, default=8, integer=True, doc="block side in px")},
        _pixelate, doc="Mosaic/pixelation via nearest-neighbour down/up-sampling."))
