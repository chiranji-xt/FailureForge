"""Multi-seed confirmation of top search candidates on held-out images.

Search tunes on a discovery split; confirmation re-evaluates the best distinct
configurations on a *different* split with fresh seeds, so a high score caused
by one lucky seed or by over-fitting to the discovery images is exposed.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from ..dataset import YoloDataset
from ..evaluation import DEFAULT_IOU, PerturbedResult, derive_trial_seed, evaluate_perturbed
from ..evaluator import EvaluationReport
from ..objective import ObjectiveConfig, evaluate_objective
from .trials import SearchTrial, rank_trials


@dataclass
class ConfirmationRecord:
    name: str
    family: str
    params: dict[str, Any]
    search_index: int
    search_score: float
    seeds: list[int]
    scores: list[float | None]          # None = invalid/errored seed
    errors: list[str | None]
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    n_valid: int = 0
    n_invalid: int = 0
    degrades_in_all_seeds: bool = False

    def to_dict(self) -> dict:
        return {
            "transformation": {
                "name": self.name, "family": self.family, "params": dict(self.params),
            },
            "search_index": self.search_index,
            "search_score": self.search_score,
            "seeds": list(self.seeds),
            "scores": list(self.scores),
            "errors": list(self.errors),
            "mean": self.mean, "std": self.std, "min": self.min, "max": self.max,
            "n_valid": self.n_valid, "n_invalid": self.n_invalid,
            "degrades_in_all_seeds": self.degrades_in_all_seeds,
        }


@dataclass
class ConfirmationResult:
    objective: ObjectiveConfig
    records: list[ConfirmationRecord]
    baseline: EvaluationReport
    #: First valid per-candidate result, for failure analysis/galleries (not serialized).
    results: list[PerturbedResult] = field(default_factory=list, repr=False)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "objective": self.objective.metric,
            "split": self.baseline.split,
            "num_images": len(self.baseline.per_image),
            "records": [r.to_dict() for r in self.records],
            "metadata": self.metadata,
        }


def confirm_trials(
    detector: Any,
    holdout: YoloDataset,
    trials: list[SearchTrial],
    *,
    top_k: int = 3,
    n_seeds: int = 3,
    seed: int = 1000,
    objective: ObjectiveConfig | None = None,
    iou_threshold: float = DEFAULT_IOU,
    baseline: EvaluationReport | None = None,
) -> ConfirmationResult:
    """Re-score the ``top_k`` distinct valid configurations over ``n_seeds`` seeds."""
    if top_k < 1 or n_seeds < 1:
        raise ValueError("top_k and n_seeds must be >= 1")
    objective = objective or ObjectiveConfig()
    from ..evaluator import evaluate_baseline

    base = baseline or evaluate_baseline(detector, holdout, iou_threshold=iou_threshold)
    candidates = rank_trials(trials, unique=True)[:top_k]
    records: list[ConfirmationRecord] = []
    kept: list[PerturbedResult] = []
    for cand in candidates:
        seeds, scores, errors = [], [], []
        first: PerturbedResult | None = None
        for k in range(n_seeds):
            s = derive_trial_seed(seed, k, cand.name, cand.params)
            seeds.append(s)
            try:
                res = evaluate_perturbed(
                    detector, holdout, cand.name, cand.params, seed=s,
                    iou_threshold=iou_threshold, baseline=base,
                )
                scored = evaluate_objective(base, res.perturbed, config=objective)
            except Exception as exc:
                scores.append(None)
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            if scored.valid:
                scores.append(scored.degradation)
                errors.append(None)
                first = first or res
            else:
                scores.append(None)
                errors.append("; ".join(scored.warnings) or "invalid objective")
        valid = [x for x in scores if x is not None and math.isfinite(x)]
        rec = ConfirmationRecord(
            name=cand.name, family=cand.family, params=dict(cand.params),
            search_index=cand.index, search_score=cand.score,
            seeds=seeds, scores=scores, errors=errors,
            n_valid=len(valid), n_invalid=len(scores) - len(valid),
        )
        if valid:
            rec.mean = statistics.fmean(valid)
            rec.std = statistics.stdev(valid) if len(valid) > 1 else None
            rec.min, rec.max = min(valid), max(valid)
            rec.degrades_in_all_seeds = len(valid) == len(scores) and all(x > 0 for x in valid)
        records.append(rec)
        if first is not None:
            kept.append(first)
    return ConfirmationResult(
        objective=objective, records=records, baseline=base, results=kept,
        metadata={
            "top_k": top_k, "n_seeds": n_seeds, "confirmation_seed": seed,
            "iou_threshold": float(iou_threshold),
            "note": "confirmation uses images disjoint from the search split "
                    "only when the caller supplies a held-out dataset",
        },
    )
