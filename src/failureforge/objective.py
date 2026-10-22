"""Deterministic performance-degradation objective for robustness search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .evaluator import EvaluationReport
from .failures import DEFAULT_RANK_METRICS, MetricComparison, compare_reports

SUPPORTED_METRICS = tuple(DEFAULT_RANK_METRICS)
DEFAULT_METRIC = "f1"
DIRECTION_HIGHER_IS_BETTER = "higher_is_better"


@dataclass(frozen=True)
class ObjectiveConfig:
    """Select the single evaluation metric used by the objective."""

    metric: str = DEFAULT_METRIC

    def __post_init__(self) -> None:
        if self.metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"metric must be one of {SUPPORTED_METRICS}, got {self.metric!r}"
            )


@dataclass(frozen=True)
class ObjectiveResult:
    """Signed degradation for one baseline/perturbed report pair.

    ``degradation`` is ``baseline - perturbed`` and is only populated for a
    valid comparison. It is negative when the perturbed metric improves.
    """

    metric: str
    direction: str
    baseline_value: float | None
    perturbed_value: float | None
    degradation: float | None
    compatible: bool
    valid: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "baseline_value": self.baseline_value,
            "perturbed_value": self.perturbed_value,
            "degradation": self.degradation,
            "compatible": bool(self.compatible),
            "valid": bool(self.valid),
            "warnings": list(self.warnings),
        }


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_value(report: EvaluationReport, metric: str) -> float | None:
    if metric == "map50":
        return _finite(report.map50)
    return _finite(report.aggregate.get(metric))


def _comparison_warnings(
    comparison: MetricComparison, metric: str
) -> tuple[str, ...]:
    warnings = list(comparison.warnings)
    if comparison.drops.get(metric) is None and not comparison.compatible:
        warnings.append(f"{metric}: no valid degradation for incompatible reports")
    return tuple(dict.fromkeys(warnings))


def evaluate_objective(
    baseline: EvaluationReport,
    perturbed: EvaluationReport,
    *,
    config: ObjectiveConfig | None = None,
) -> ObjectiveResult:
    """Evaluate signed metric degradation without running inference.

    A larger score means worse performance for a higher-is-better metric.
    Missing/non-finite metric values or incompatible reports produce an invalid
    result with ``degradation=None`` rather than a fabricated zero.
    """
    objective_config = config if config is not None else ObjectiveConfig()
    metric = objective_config.metric
    baseline_value = _metric_value(baseline, metric)
    perturbed_value = _metric_value(perturbed, metric)
    comparison = compare_reports(baseline, perturbed, metrics=(metric,))
    raw_degradation = comparison.drops.get(metric)
    valid = (
        comparison.compatible
        and baseline_value is not None
        and perturbed_value is not None
        and raw_degradation is not None
        and math.isfinite(float(raw_degradation))
    )
    return ObjectiveResult(
        metric=metric,
        direction=DIRECTION_HIGHER_IS_BETTER,
        baseline_value=baseline_value,
        perturbed_value=perturbed_value,
        degradation=float(raw_degradation) if valid else None,
        compatible=bool(comparison.compatible),
        valid=bool(valid),
        warnings=_comparison_warnings(comparison, metric),
    )
