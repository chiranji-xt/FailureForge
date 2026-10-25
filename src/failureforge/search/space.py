"""Search space derived from the perturbation registry's parameter specs.

Every searchable numeric parameter is sampled uniformly within its registered
(or caller-narrowed) range. Parameters that are not searchable (choices,
``search=False``) keep their registered defaults, so a sampled configuration is
always accepted by :func:`failureforge.perturbations.validate_params`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..perturbations import available, get, validate_params

#: Excluded by default: needs a caller-supplied ordered ``steps`` list.
DEFAULT_EXCLUDED = ("combined",)

#: Integer parameters the engine only accepts as odd values (not expressible in
#: ``ParamSpec``), so they are sampled on a step-2 grid.
ODD_PARAMS = frozenset({("motion_blur", "kernel_size")})


class SearchSpaceError(ValueError):
    """Raised for an empty or invalid search space."""


@dataclass(frozen=True)
class ParamRange:
    name: str
    low: float
    high: float
    integer: bool = False
    step: int = 1

    def sample(self, rng: random.Random) -> float | int:
        if self.integer:
            steps = (int(self.high) - int(self.low)) // self.step
            return int(self.low) + self.step * rng.randint(0, steps)
        value = round(rng.uniform(self.low, self.high), 6)
        return min(max(value, self.low), self.high)

    def to_dict(self) -> dict:
        out = {"low": self.low, "high": self.high, "integer": self.integer}
        if self.step != 1:
            out["step"] = self.step
        return out


class SearchSpace:
    """A set of perturbations with sampleable numeric parameter ranges."""

    def __init__(self, ranges: Mapping[str, tuple[ParamRange, ...]]) -> None:
        if not ranges:
            raise SearchSpaceError("search space contains no perturbations")
        self._ranges = {name: tuple(r) for name, r in sorted(ranges.items())}

    @classmethod
    def default(
        cls,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] = DEFAULT_EXCLUDED,
        overrides: Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
    ) -> "SearchSpace":
        """Build a space from the registry.

        ``include`` restricts to named transforms (unknown names raise);
        ``overrides`` narrows ``{transform: {param: (low, high)}}`` and must stay
        inside the registered bounds.
        """
        overrides = overrides or {}
        if include is None:
            names = [n for n in available() if n not in set(exclude)]
        else:
            names = sorted({get(raw).name for raw in include})  # unknown -> error
        unknown = set(overrides) - set(names)
        if unknown:
            raise SearchSpaceError(
                f"overrides for transforms not in the space: {sorted(unknown)}"
            )
        ranges: dict[str, tuple[ParamRange, ...]] = {}
        for name in names:
            transform = get(name)
            if any(s.required and not s.is_searchable for s in transform.params.values()):
                if include is not None:
                    raise SearchSpaceError(
                        f"'{name}' has a required non-searchable parameter and "
                        "cannot be searched"
                    )
                continue
            params = []
            for pname, spec in sorted(transform.params.items()):
                if not spec.is_searchable or spec.high is None:
                    continue
                low, high = float(spec.low), float(spec.high)
                if pname in overrides.get(name, {}):
                    low, high = (float(v) for v in overrides[name][pname])
                    if not (spec.low <= low <= high <= spec.high):
                        raise SearchSpaceError(
                            f"override for {name}.{pname} must satisfy "
                            f"{spec.low} <= low <= high <= {spec.high}"
                        )
                step = 2 if (name, pname) in ODD_PARAMS else 1
                if step == 2:  # snap bounds inward onto odd integers
                    low = low if int(low) % 2 else low + 1
                    high = high if int(high) % 2 else high - 1
                    if low > high:
                        raise SearchSpaceError(f"no odd value in {name}.{pname} range")
                params.append(
                    ParamRange(pname, low, high, bool(spec.integer), step)
                )
            bad = set(overrides.get(name, {})) - {p.name for p in params}
            if bad:
                raise SearchSpaceError(
                    f"override for non-searchable parameter(s) of {name}: {sorted(bad)}"
                )
            ranges[name] = tuple(params)
        return cls(ranges)

    @property
    def names(self) -> list[str]:
        return list(self._ranges)

    def ranges(self, name: str) -> tuple[ParamRange, ...]:
        return self._ranges[name]

    def sample(self, rng: random.Random) -> tuple[str, dict[str, Any]]:
        name = rng.choice(self.names)
        params = {r.name: r.sample(rng) for r in self._ranges[name]}
        return name, validate_params(name, params)

    def normalize(self, name: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if name not in self._ranges:
            raise SearchSpaceError(f"'{name}' is not part of this search space")
        return validate_params(name, dict(params))

    def describe(self) -> dict:
        return {n: {r.name: r.to_dict() for r in rs} for n, rs in self._ranges.items()}
