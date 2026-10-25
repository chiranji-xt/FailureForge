"""Trial records, ranking, and result containers for robustness search."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from ..objective import ObjectiveConfig

STATUS_OK = "ok"
STATUS_INVALID = "invalid"   # evaluated, but objective not comparable/finite
STATUS_ERROR = "error"       # evaluation raised


def config_key(name: str, params: dict[str, Any], ndigits: int = 4) -> str:
    """Stable identity for near-identical configurations (used to deduplicate)."""
    rounded = {
        k: (round(v, ndigits) if isinstance(v, float) else v)
        for k, v in sorted(params.items())
    }
    return f"{name}|{json.dumps(rounded, sort_keys=True, default=str)}"


@dataclass
class SearchTrial:
    index: int
    name: str
    family: str
    params: dict[str, Any]
    seed: int
    status: str
    score: float | None = None            # degradation; only set when status == ok
    baseline_value: float | None = None
    perturbed_value: float | None = None
    duration_s: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return (
            self.status == STATUS_OK
            and self.score is not None
            and math.isfinite(self.score)
        )

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "transformation": {
                "name": self.name, "family": self.family,
                "params": dict(self.params), "seed": self.seed,
            },
            "status": self.status,
            "score": self.score,
            "baseline_value": self.baseline_value,
            "perturbed_value": self.perturbed_value,
            "duration_s": self.duration_s,
            "error": self.error,
            "warnings": list(self.warnings),
        }


def rank_trials(trials: list[SearchTrial], *, unique: bool = False) -> list[SearchTrial]:
    """Valid trials by descending score (ties: lower index first).

    Invalid and errored trials are never ranked. With ``unique=True`` only the
    best trial of each near-identical configuration is kept.
    """
    ranked = sorted((t for t in trials if t.valid), key=lambda t: (-t.score, t.index))
    if not unique:
        return ranked
    seen: set[str] = set()
    out = []
    for t in ranked:
        key = config_key(t.name, t.params)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


@dataclass
class SearchResult:
    method: str
    seed: int
    budget: int
    objective: ObjectiveConfig
    iou_threshold: float
    space: dict
    trials: list[SearchTrial]
    baseline_metrics: dict
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def ranked(self, *, unique: bool = False) -> list[SearchTrial]:
        return rank_trials(self.trials, unique=unique)

    @property
    def best(self) -> SearchTrial | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def counts(self) -> dict:
        return {
            "total": len(self.trials),
            "valid": sum(t.valid for t in self.trials),
            "invalid": sum(t.status == STATUS_INVALID for t in self.trials),
            "error": sum(t.status == STATUS_ERROR for t in self.trials),
        }

    def to_dict(self) -> dict:
        best = self.best
        return {
            "method": self.method,
            "seed": self.seed,
            "budget": self.budget,
            "objective": {
                "metric": self.objective.metric,
                "definition": "baseline - perturbed (larger is worse; negative is improvement)",
            },
            "iou_threshold": self.iou_threshold,
            "space": self.space,
            "baseline_metrics": self.baseline_metrics,
            "counts": self.counts(),
            "best": best.to_dict() if best else None,
            "trials": [t.to_dict() for t in self.trials],
            "metadata": self.metadata,
            "warnings": list(self.warnings),
        }
