"""Cropping / framing family (11): random crop, edge clip, shifted framing.

Unlike ``translation`` (same canvas, black fill), cropping **removes** pixels
and changes the output size; ``shifted_framing`` keeps the canvas but pads the
exposed area by edge replication. All three remap GT boxes and drop any box
that is cropped away or becomes smaller than ``MIN_BOX_SIDE_PX``.
"""
from __future__ import annotations

import numpy as np

from .base import Context, ParamSpec, Transform, bgr_to_pil, pil_to_bgr, register
from .boxops import map_boxes


def _result(ctx: Context, out_bgr: np.ndarray, kept, removed, **extra):
    return out_bgr, kept, {"geometry_changed": True, "removed_boxes": removed, **extra}


def _random_crop(ctx: Context):
    frac = float(ctx.params["crop_fraction"])
    h, w = ctx.image.shape[:2]
    cw = max(1, int(round(w * frac)))
    ch = max(1, int(round(h * frac)))
    x0 = int(ctx.rng.integers(0, max(1, w - cw + 1)))
    y0 = int(ctx.rng.integers(0, max(1, h - ch + 1)))
    out = bgr_to_pil(ctx.image).crop((x0, y0, x0 + cw, y0 + ch))
    kept, removed = map_boxes(ctx.boxes, lambda x, y: (x - x0, y - y0), cw, ch)
    return _result(ctx, pil_to_bgr(out), kept, removed,
                   crop_box=[x0, y0, x0 + cw, y0 + ch], crop_fraction=frac)


def _edge_clip(ctx: Context):
    amount = float(ctx.params["amount"])
    side = str(ctx.params["side"])
    h, w = ctx.image.shape[:2]
    if side == "left":
        x0, y0, cw, ch = int(round(w * amount)), 0, max(1, w - int(round(w * amount))), h
    elif side == "right":
        cw, ch, x0, y0 = max(1, w - int(round(w * amount))), h, 0, 0
    elif side == "top":
        x0, y0, cw, ch = 0, int(round(h * amount)), w, max(1, h - int(round(h * amount)))
    else:  # bottom
        cw, ch, x0, y0 = w, max(1, h - int(round(h * amount))), 0, 0
    out = bgr_to_pil(ctx.image).crop((x0, y0, x0 + cw, y0 + ch))
    kept, removed = map_boxes(ctx.boxes, lambda x, y: (x - x0, y - y0), cw, ch)
    return _result(ctx, pil_to_bgr(out), kept, removed,
                   crop_box=[x0, y0, x0 + cw, y0 + ch], clipped_side=side, amount=amount)


def _shifted_framing(ctx: Context):
    h, w = ctx.image.shape[:2]
    tx = float(ctx.params["dx"]) * w
    ty = float(ctx.params["dy"]) * h
    arr = np.roll(np.roll(ctx.image, int(round(ty)), axis=0), int(round(tx)), axis=1)
    kept, removed = map_boxes(ctx.boxes, lambda x, y: (x + tx, y + ty), w, h)
    return _result(ctx, np.ascontiguousarray(arr), kept, removed,
                   dx=float(ctx.params["dx"]), dy=float(ctx.params["dy"]))


def register_cropping() -> None:
    register(Transform(
        "random_crop", "cropping",
        {"crop_fraction": ParamSpec(0.1, 0.9, default=0.7, doc="kept fraction of each side")},
        _random_crop, geometry=True, doc="Random crop keeping crop_fraction of both sides."))
    register(Transform(
        "edge_clip", "cropping",
        {"amount": ParamSpec(0.0, 0.3, default=0.1, doc="fraction removed from one side"),
         "side": ParamSpec(choices=("top", "bottom", "left", "right"), default="bottom",
                           search=False, doc="side to crop")},
        _edge_clip, geometry=True, doc="Crop a band off one edge (framing/clipping)."))
    register(Transform(
        "shifted_framing", "cropping",
        {"dx": ParamSpec(-0.5, 0.5, default=0.1, doc="x shift as a fraction of width"),
         "dy": ParamSpec(-0.5, 0.5, default=0.0, doc="y shift as a fraction of height")},
        _shifted_framing, geometry=True,
        doc="Shift framing within the canvas, wrapping exposed edges (no black fill)."))
