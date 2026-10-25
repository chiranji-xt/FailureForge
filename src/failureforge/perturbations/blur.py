"""Blur family (3): Gaussian, motion, defocus."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .base import (
    Context, ParamSpec, PerturbationError, Transform, bgr_to_pil, clip_u8,
    pil_to_bgr, register,
)
from .boxops import translate_forward, warp_affine


def _keep(ctx: Context, out_bgr: np.ndarray, **extra) -> tuple:
    return out_bgr, ctx.boxes, {"geometry_changed": False, **extra}


def _gaussian_blur(ctx: Context):
    radius = float(ctx.params["radius"])
    out = pil_to_bgr(bgr_to_pil(ctx.image).filter(ImageFilter.GaussianBlur(radius)))
    return _keep(ctx, out, radius=radius)


def _blur(ctx: Context):
    radius = float(ctx.params["radius"])
    out = bgr_to_pil(ctx.image)
    out = out.filter(ImageFilter.GaussianBlur(radius)) if radius > 0 else out.copy()
    return _keep(ctx, pil_to_bgr(out), radius=radius)


def _defocus_blur(ctx: Context):
    radius = float(ctx.params["radius"])
    out = bgr_to_pil(ctx.image)
    out = out.filter(ImageFilter.BoxBlur(radius)) if radius > 0 else out.copy()
    return _keep(ctx, pil_to_bgr(out), radius=radius)


def _motion_blur(ctx: Context):
    kernel_size = int(ctx.params["kernel_size"])
    if kernel_size % 2 == 0:
        raise PerturbationError(f"'motion_blur.kernel_size' must be odd, got {kernel_size}")
    angle = float(ctx.params["angle"])
    pil = bgr_to_pil(ctx.image)
    w, h = pil.size
    t = np.deg2rad(angle)
    dx, dy = float(np.cos(t)), float(np.sin(t))
    acc = np.zeros((h, w, 3), dtype=np.uint32)  # exact integer sums (<= 31 * 255)
    half = kernel_size // 2
    for i in range(kernel_size):
        off = i - half
        shifted = warp_affine(pil, translate_forward(off * dx, off * dy), (w, h))
        acc += np.asarray(shifted)
    mean = acc / float(kernel_size)
    return _keep(ctx, pil_to_bgr(Image.fromarray(clip_u8(mean), mode="RGB")),
                 kernel_size=kernel_size, angle=angle)


def register_blur() -> None:
    register(Transform(
        "gaussian_blur", "blur",
        {"radius": ParamSpec(0.0, 10.0, default=2.0, doc="Gaussian sigma radius (px)")},
        _gaussian_blur, doc="Isotropic Gaussian blur."))
    register(Transform(
        "defocus_blur", "blur",
        {"radius": ParamSpec(0.0, 10.0, default=3.0, doc="box (disk-like) radius (px)")},
        _defocus_blur, doc="Uniform box blur approximating defocus/out-of-focus."))
    register(Transform(
        "motion_blur", "blur",
        {"kernel_size": ParamSpec(3, 31, default=9, integer=True, doc="odd streak length"),
         "angle": ParamSpec(0.0, 180.0, default=0.0, doc="direction in degrees")},
        _motion_blur, doc="Linear motion blur (average of shifted copies along a direction)."))
    register(Transform(
        "blur", "blur",
        {"radius": ParamSpec(0.0, 5.0, default=2.0, doc="Gaussian sigma radius (px)")},
        _blur, doc="Gaussian blur (legacy-compatible name)."))
