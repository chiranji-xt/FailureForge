"""Compatibility layer over the perturbation engine.

The canonical engine lives in :mod:`failureforge.perturbations` and works with
BGR ``uint8`` arrays and pixel-space ``GroundTruthBox`` boxes. This module keeps
the original ``failureforge.transforms`` interface working for earlier preview
modules:

- :func:`apply_perturbation` - PIL image + *normalized* YOLO dict boxes.
- :data:`PARAM_RANGES` - legacy ``{name: {param: (low, high)}}`` search space for
  the five original transforms, **derived from the engine** so there is a single
  source of truth.

New code should import from :mod:`failureforge.perturbations` directly.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .dataset import GroundTruthBox
from .perturbations import (
    PerturbationError,
    PerturbationResult,
    apply,
    available,
    families,
    param_specs,
    validate_params,
)

#: Covered fraction at which a box counts as fully occluded (legacy default).
FULL_OCCLUSION_THRESHOLD = 0.85

_LEGACY_NAMES = ("brightness", "blur", "rotation", "scaling", "occlusion")


def _build_param_ranges() -> dict[str, dict[str, tuple[float, float]]]:
    specs = param_specs()
    ranges: dict[str, dict[str, tuple[float, float]]] = {}
    for name in _LEGACY_NAMES:
        ranges[name] = {
            p: (float(s.low), float(s.high))
            for p, s in specs[name].items()
            if s.is_searchable
        }
    return ranges


PARAM_RANGES = _build_param_ranges()


def _normalized_to_boxes(boxes: list[dict], width: int, height: int) -> list[GroundTruthBox]:
    out = []
    for b in boxes:
        xc, yc, w, h = b["x_center"], b["y_center"], b["width"], b["height"]
        out.append(GroundTruthBox(
            int(b["class_id"]),
            (xc - w / 2) * width, (yc - h / 2) * height,
            (xc + w / 2) * width, (yc + h / 2) * height,
        ))
    return out


def _boxes_to_normalized(boxes, width: int, height: int) -> list[dict]:
    out = []
    for b in boxes or []:
        out.append({
            "class_id": int(b.class_id),
            "x_center": ((b.x1 + b.x2) / 2) / width,
            "y_center": ((b.y1 + b.y2) / 2) / height,
            "width": (b.x2 - b.x1) / width,
            "height": (b.y2 - b.y1) / height,
        })
    return out


def apply_perturbation(
    image: Image.Image,
    boxes: list[dict],
    name: str,
    params: dict[str, Any],
    seed: int = 0,
) -> tuple[Image.Image, list[dict], dict[str, Any]]:
    """Legacy PIL + normalized-box wrapper around the engine.

    Returns ``(PIL.Image, normalized_boxes, info)``. ``name`` must be one of
    :data:`PARAM_RANGES` (the five original transforms).
    """
    if name not in PARAM_RANGES:
        raise ValueError(
            f"unknown perturbation '{name}'. Valid: {sorted(PARAM_RANGES)}"
        )
    width, height = image.size
    bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8)[:, :, ::-1])
    gt_boxes = _normalized_to_boxes(boxes, width, height)

    engine_params = dict(params)
    if name == "occlusion":
        engine_params.setdefault("remove_fully_occluded", True)

    result = apply(name, bgr, engine_params, boxes=gt_boxes, seed=seed)
    out_pil = Image.fromarray(np.ascontiguousarray(result.image[:, :, ::-1]), "RGB")
    out_boxes = _boxes_to_normalized(result.boxes, width, height)

    meta = result.metadata
    info: dict[str, Any] = {"geometry_changed": meta.get("geometry_changed", False)}
    removed = meta.get("removed_boxes", [])
    info["removed"] = [
        {"class_id": r.get("class_id", 0), "reason": r.get("reason", "removed")}
        for r in removed
    ]
    if name == "occlusion" and "cutout_xyxy" in meta:
        cx1, cy1, cx2, cy2 = meta["cutout_xyxy"]
        info["cutout_xyxy"] = [cx1 / width, cy1 / height, cx2 / width, cy2 / height]
    info["metadata"] = meta
    return out_pil, out_boxes, info


__all__ = [
    "apply",
    "apply_perturbation",
    "available",
    "families",
    "param_specs",
    "validate_params",
    "PerturbationError",
    "PerturbationResult",
    "PARAM_RANGES",
    "FULL_OCCLUSION_THRESHOLD",
]
