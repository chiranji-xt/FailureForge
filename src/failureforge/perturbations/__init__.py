"""FailureForge perturbation engine.

Twelve perturbation families over OpenCV-style BGR ``uint8`` images with
seeded reproducibility and (for geometric/cropping transforms) correct
ground-truth box remapping. See :mod:`failureforge.perturbations.base` for the
core contract.

Quick start::

    import numpy as np
    from failureforge.perturbations import apply, available

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = apply("gaussian_noise", img, {"sigma": 20}, seed=0)
    result.image.shape, result.metadata["family"]  # (64,64,3), 'noise'

Geometric example (boxes are ``GroundTruthBox`` in pixel pixels)::

    from failureforge.dataset import GroundTruthBox
    boxes = [GroundTruthBox(0, 10, 10, 40, 40)]
    result = apply("rotation", img, {"angle": 30}, boxes=boxes, seed=0)
    result.boxes  # re-mapped, clipped GroundTruthBox list
"""
from __future__ import annotations

from .base import (
    MIN_BOX_SIDE_PX,
    Context,
    ParamSpec,
    PerturbationError,
    PerturbationResult,
    Transform,
    apply,
    available,
    families,
    get,
    param_specs,
    register,
    validate_params,
)

__all__ = [
    "apply",
    "available",
    "families",
    "get",
    "param_specs",
    "validate_params",
    "register",
    "Context",
    "ParamSpec",
    "Transform",
    "PerturbationError",
    "PerturbationResult",
    "MIN_BOX_SIDE_PX",
]


def _bootstrap() -> None:
    from .blur import register_blur
    from .combined import register_combined
    from .cropping import register_cropping
    from .geometry import register_geometry
    from .lens import register_lens
    from .lighting import register_lighting
    from .occlusion import register_occlusion
    from .photometric import register_photometric
    from .quality import register_quality
    from .weather import register_weather

    register_photometric()
    register_blur()
    register_weather()
    register_occlusion()
    register_geometry()
    register_quality()
    register_lighting()
    register_lens()
    register_cropping()
    register_combined()


_bootstrap()
