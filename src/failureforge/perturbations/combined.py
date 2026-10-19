"""Combined-corruption family (12): configurable ordered compositions.

A ``combined`` transform is a list of ``{"name", "params"}`` steps applied in
the given order (order matters). Each step gets a deterministic child seed drawn
from the request RNG, so a whole composition is reproducible from
``(steps, seed)``. Individual logic is reused from the other families - nothing
is reimplemented here.
"""
from __future__ import annotations

from .base import (
    Context, ParamSpec, PerturbationError, Transform, apply, register,
)

_EXAMPLES = ("gaussian_noise+blur", "darken+gaussian_noise", "rain+contrast",
             "gaussian_blur+jpeg_compression")


def _normalize_steps(steps) -> list[dict]:
    if not isinstance(steps, (list, tuple)) or not steps:
        raise PerturbationError("'combined.steps' must be a non-empty list of steps")
    out = []
    for i, step in enumerate(steps):
        if isinstance(step, str):
            out.append({"name": step, "params": {}})
        elif isinstance(step, dict) and "name" in step:
            out.append({"name": step["name"], "params": dict(step.get("params", {}))})
        else:
            raise PerturbationError(
                f"combined step {i} must be a name or {{'name','params'}}, got {step!r}"
            )
    return out


def _combined(ctx: Context):
    steps = _normalize_steps(ctx.params["steps"])
    child_seeds = [int(ctx.rng.integers(0, 2**31 - 1)) for _ in steps]
    image = ctx.image
    boxes = ctx.boxes
    components = []
    for step, seed in zip(steps, child_seeds):
        result = apply(step["name"], image, step["params"], boxes=boxes, seed=seed)
        image, boxes = result.image, result.boxes
        components.append({
            "name": result.metadata["name"],
            "family": result.metadata["family"],
            "params": result.metadata["params"],
            "seed": seed,
            "removed_boxes": result.metadata.get("removed_boxes", []),
        })
    removed = [r for c in components for r in c["removed_boxes"]]
    return image, boxes, {
        "geometry_changed": any(c["family"] in ("geometric", "cropping") for c in components),
        "components": components,
        "order": [c["name"] for c in components],
        "removed_boxes": removed,
    }


def register_combined() -> None:
    register(Transform(
        "combined", "combined",
        {"steps": ParamSpec(passthrough=True, required=True, search=False,
                            doc="ordered list of {name, params} steps")},
        _combined, doc=f"Ordered composition; e.g. {', '.join(_EXAMPLES)}."))
