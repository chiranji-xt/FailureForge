"""Core of the FailureForge perturbation engine.

Conventions
-----------
- Images are **OpenCV-style BGR ``uint8`` NumPy arrays** shaped ``(H, W, 3)``,
  matching :meth:`failureforge.detector.YOLODetector.predict`. Every transform
  accepts such an array and returns a new array of the same dtype (shape may
  change only for documented crop/quality transforms).
- Ground-truth boxes are pixel ``[x1, y1, x2, y2]`` and are represented by
  :class:`failureforge.dataset.GroundTruthBox`. Non-geometric transforms never
  change box geometry; geometric/cropping transforms remap corners, rebuild the
  enclosing axis-aligned box, clip to the canvas, and drop boxes that become too
  small (documented rule: side < ``MIN_BOX_SIDE_PX``).
- Stochastic transforms draw from a NumPy ``Generator`` seeded from the request
  seed, so identical ``(image, params, seed)`` always yield identical output.
- The caller's input array is never mutated.

A transform is registered with :func:`register` and invoked via :func:`apply`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

#: Side (px) below which a transformed box is considered degenerate and dropped.
MIN_BOX_SIDE_PX = 1.0


class PerturbationError(ValueError):
    """Raised for unknown transforms, invalid parameters, or invalid images."""


# ---------------------------------------------------------------------------
# Parameter specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    """Declarative parameter description used for validation and search."""

    low: float | None = None
    high: float | None = None
    default: Any = None
    integer: bool = False
    choices: tuple | None = None
    required: bool = False
    search: bool = True
    passthrough: bool = False
    doc: str = ""

    def validate(self, name: str, param: str, value: Any) -> Any:
        if self.passthrough:
            return value
        if self.choices is not None:
            if value not in self.choices:
                raise PerturbationError(
                    f"'{name}.{param}' must be one of {list(self.choices)}, got {value!r}"
                )
            return value
        try:
            v = int(value) if self.integer else float(value)
        except (TypeError, ValueError):
            raise PerturbationError(f"'{name}.{param}' must be numeric, got {value!r}")
        if self.integer and float(v) != float(value):
            raise PerturbationError(f"'{name}.{param}' must be an integer, got {value!r}")
        if self.low is not None and v < self.low:
            raise PerturbationError(f"'{name}.{param}'={v} below minimum {self.low}")
        if self.high is not None and v > self.high:
            raise PerturbationError(f"'{name}.{param}'={v} above maximum {self.high}")
        return v

    @property
    def is_searchable(self) -> bool:
        return self.search and self.choices is None and self.low is not None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TransformFn = Callable[["Context"], "tuple[np.ndarray, list | None, dict]"]


@dataclass
class Transform:
    name: str
    family: str
    params: dict[str, ParamSpec]
    fn: TransformFn
    geometry: bool = False
    doc: str = ""


@dataclass
class _Registry:
    transforms: dict[str, Transform] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)


_REGISTRY = _Registry()


def register(transform: Transform, aliases: Iterable[str] = ()) -> Transform:
    """Register ``transform``; ``aliases`` resolve to the same implementation."""
    if transform.name in _REGISTRY.transforms or transform.name in _REGISTRY.aliases:
        raise PerturbationError(f"transform '{transform.name}' already registered")
    _REGISTRY.transforms[transform.name] = transform
    for alias in aliases:
        if alias in _REGISTRY.transforms or alias in _REGISTRY.aliases:
            raise PerturbationError(f"name '{alias}' already registered")
        _REGISTRY.aliases[alias] = transform.name
    return transform


def get(name: str) -> Transform:
    if name in _REGISTRY.transforms:
        return _REGISTRY.transforms[name]
    if name in _REGISTRY.aliases:
        return _REGISTRY.transforms[_REGISTRY.aliases[name]]
    raise PerturbationError(
        f"unknown transform '{name}'. Available: {', '.join(available())}"
    )


def available() -> list[str]:
    """Sorted canonical transform names (aliases excluded)."""
    return sorted(_REGISTRY.transforms)


def families() -> dict[str, list[str]]:
    """Map family name -> sorted canonical transform names."""
    out: dict[str, list[str]] = {}
    for name in available():
        out.setdefault(_REGISTRY.transforms[name].family, []).append(name)
    return {k: sorted(v) for k, v in out.items()}


def param_specs() -> dict[str, dict[str, ParamSpec]]:
    return {n: dict(t.params) for n, t in _REGISTRY.transforms.items()}


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def validate_params(name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Validate/normalize ``params`` for transform ``name``.

    Returns a new dict with optional parameters filled from defaults. Unknown
    keys, missing required keys, out-of-range values, and bad choices raise
    :class:`PerturbationError`.
    """
    transform = get(name)
    params = dict(params or {})
    unknown = set(params) - set(transform.params)
    if unknown:
        raise PerturbationError(
            f"unknown parameter(s) for '{name}': {sorted(unknown)}; "
            f"expected {sorted(transform.params)}"
        )
    out: dict[str, Any] = {}
    for pname, spec in transform.params.items():
        if pname in params:
            out[pname] = spec.validate(name, pname, params[pname])
        elif spec.required:
            raise PerturbationError(f"'{name}' requires parameter '{pname}'")
        else:
            out[pname] = spec.default
    return out


# ---------------------------------------------------------------------------
# Image helpers (BGR <-> PIL)
# ---------------------------------------------------------------------------

def validate_bgr(image: Any) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise PerturbationError(f"image must be a NumPy array, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise PerturbationError(f"image must have shape (H, W, 3), got {tuple(image.shape)}")
    if image.dtype != np.uint8:
        raise PerturbationError(f"image dtype must be uint8, got {image.dtype}")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise PerturbationError(f"image must be non-empty, got shape {tuple(image.shape)}")
    return image


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """BGR uint8 ndarray -> RGB ``PIL.Image`` (always a copy)."""
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    return Image.fromarray(rgb, mode="RGB")


def pil_to_bgr(pil: Image.Image) -> np.ndarray:
    """``PIL.Image`` -> BGR uint8 ndarray (always a copy)."""
    rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32)


def clip_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Context and execution
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """Execution context passed to a transform function."""

    image: np.ndarray          # BGR uint8, already a private copy
    boxes: list | None         # GroundTruthBox list, or None (Quick Mode)
    params: dict[str, Any]
    rng: np.random.Generator
    seed: int

    def pil(self) -> Image.Image:
        return bgr_to_pil(self.image)


def _as_box_list(boxes: Any) -> list | None:
    if boxes is None:
        return None
    if isinstance(boxes, np.ndarray):
        if boxes.size == 0:
            return []
        raise PerturbationError(
            "boxes must be GroundTruthBox objects (pass a list, not a raw array)"
        )
    return list(boxes)


@dataclass
class PerturbationResult:
    """Result of applying one transform (or a combination)."""

    image: np.ndarray          # BGR uint8
    boxes: list | None         # transformed GroundTruthBox list, or None
    metadata: dict


def apply(
    name: str,
    image: np.ndarray,
    params: dict[str, Any] | None = None,
    boxes: list | None = None,
    seed: int = 0,
) -> PerturbationResult:
    """Apply transform ``name`` to ``image``.

    Parameters
    ----------
    name:
        Registered transform name (canonical or alias).
    image:
        BGR ``uint8`` ``(H, W, 3)`` array. Not mutated.
    params:
        Transform parameters (validated; defaults filled in).
    boxes:
        Optional ground-truth boxes (``GroundTruthBox`` list). ``None`` means
        "no labels" (Quick Mode) and is returned as ``None``.
    seed:
        Seed for stochastic transforms and random placement.

    Returns
    -------
    PerturbationResult
        ``.image`` (new BGR uint8 array), ``.boxes`` (transformed list or
        ``None``), ``.metadata`` (name, family, params, seed, diagnostics).
    """
    transform = get(name)
    validate_bgr(image)
    normalized = validate_params(name, params)
    ctx = Context(
        image=np.array(image, dtype=np.uint8, copy=True),
        boxes=_as_box_list(boxes),
        params=normalized,
        rng=np.random.default_rng(int(seed)),
        seed=int(seed),
    )
    out_image, out_boxes, info = transform.fn(ctx)
    validate_bgr(out_image)
    metadata = {
        "name": transform.name,
        "family": transform.family,
        "params": normalized,
        "seed": int(seed),
        "geometry_changed": bool(transform.geometry),
        "input_shape": list(image.shape),
        "output_shape": list(out_image.shape),
    }
    metadata.update(info or {})
    return PerturbationResult(image=out_image, boxes=out_boxes, metadata=metadata)
