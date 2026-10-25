"""Shared trial evaluation for every search algorithm.

Search never re-implements perturbation, matching, or scoring: a candidate is
evaluated with :func:`failureforge.evaluation.evaluate_perturbed` against one
shared baseline and scored by :func:`failureforge.objective.evaluate_objective`.
"""
from __future__ import annotations

import time
from typing import Any

from ..dataset import YoloDataset
from ..evaluation import DEFAULT_IOU, derive_trial_seed, evaluate_perturbed
from ..evaluator import EvaluationReport, evaluate_baseline
from ..objective import ObjectiveConfig, evaluate_objective
from ..perturbations import get
from .space import SearchSpace
from .trials import STATUS_ERROR, STATUS_INVALID, STATUS_OK, SearchResult, SearchTrial


def baseline_metrics(report: EvaluationReport) -> dict:
    return {
        "precision": report.aggregate.get("precision"),
        "recall": report.aggregate.get("recall"),
        "f1": report.aggregate.get("f1"),
        "map50": report.map50,
    }


def search_warnings(result: SearchResult) -> list[str]:
    """Conditions under which scores cannot support a robustness conclusion."""
    metric = result.objective.metric
    base = result.baseline_metrics.get(metric)
    warnings: list[str] = []
    if base is None or base <= 0.0:
        warnings.append(
            f"baseline {metric} is {base}: the detector has nothing to lose on this "
            "split, so every score is <= 0 and the search is uninformative "
            "(check the weights, class mapping, and labels)"
        )
    elif 0 < base < 0.1:
        warnings.append(
            f"baseline {metric} is very low ({base:.3f}); degradation is bounded by "
            "that value and scores are hard to compare"
        )
    scores = [t.score for t in result.trials if t.valid]
    if scores and max(scores) <= 0.0 and base and base > 0:
        warnings.append("no trial degraded the metric (best score <= 0)")
    if not scores:
        warnings.append("no valid trial: nothing is ranked")
    return warnings


def check_budget(budget: int) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"budget must be a positive integer, got {budget!r}")
    return budget


class TrialRunner:
    """Evaluate configurations on one dataset with a single shared baseline."""

    def __init__(
        self,
        detector: Any,
        dataset: YoloDataset,
        *,
        objective: ObjectiveConfig | None = None,
        iou_threshold: float = DEFAULT_IOU,
        baseline: EvaluationReport | None = None,
    ) -> None:
        self.detector = detector
        self.dataset = dataset
        self.objective = objective or ObjectiveConfig()
        self.iou_threshold = float(iou_threshold)
        self.baseline = baseline or evaluate_baseline(
            detector, dataset, iou_threshold=self.iou_threshold
        )

    def run(self, index: int, name: str, params: dict, base_seed: int) -> SearchTrial:
        transform = get(name)
        seed = derive_trial_seed(base_seed, index, transform.name, params)
        start = time.perf_counter()
        try:
            result = evaluate_perturbed(
                self.detector, self.dataset, transform.name, params,
                seed=seed, iou_threshold=self.iou_threshold, baseline=self.baseline,
            )
            scored = evaluate_objective(
                self.baseline, result.perturbed, config=self.objective
            )
        except Exception as exc:  # attributable; never scored as zero
            return SearchTrial(
                index=index, name=transform.name, family=transform.family,
                params=dict(params), seed=seed, status=STATUS_ERROR,
                duration_s=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
        return SearchTrial(
            index=index, name=transform.name, family=transform.family,
            params=dict(result.perturbation["params"]), seed=seed,
            status=STATUS_OK if scored.valid else STATUS_INVALID,
            score=scored.degradation if scored.valid else None,
            baseline_value=scored.baseline_value,
            perturbed_value=scored.perturbed_value,
            duration_s=time.perf_counter() - start,
            warnings=list(scored.warnings),
        )

    def package(self, method: str, seed: int, budget: int, space: SearchSpace,
                trials: list[SearchTrial], extra: dict | None = None) -> SearchResult:
        meta = {
            "split": self.baseline.split,
            "num_images": len(self.baseline.per_image),
            "model": self.baseline.metadata.get("model"),
            "seed_derivation": "sha256(base_seed|index|name|params)",
        }
        meta.update(extra or {})
        result = SearchResult(
            method=method, seed=int(seed), budget=int(budget),
            objective=self.objective, iou_threshold=self.iou_threshold,
            space=space.describe(), trials=trials,
            baseline_metrics=baseline_metrics(self.baseline), metadata=meta,
        )
        result.warnings = search_warnings(result)
        return result
