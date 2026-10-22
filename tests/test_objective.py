"""Deterministic robustness-search objective tests."""

from __future__ import annotations

import dataclasses
import json

import pytest

from failureforge.evaluator import EvaluationReport
from failureforge.objective import (
    DEFAULT_METRIC,
    DIRECTION_HIGHER_IS_BETTER,
    SUPPORTED_METRICS,
    ObjectiveConfig,
    evaluate_objective,
)


def make_report(
    *,
    precision: float = 1.0,
    recall: float = 1.0,
    f1: float = 1.0,
    map50: float = 1.0,
    iou_threshold: float = 0.5,
    aggregation_method: str = "micro",
    split: str = "test",
    per_image: list | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        aggregate={
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        per_class={},
        per_image=list(per_image or []),
        map50=map50,
        error_records=[],
        metadata={},
        conf_threshold=0.25,
        iou_threshold=iou_threshold,
        split=split,
        aggregation_method=aggregation_method,
    )


def test_default_metric_and_explicit_direction():
    assert DEFAULT_METRIC == "f1"
    assert SUPPORTED_METRICS == ("precision", "recall", "f1", "map50")
    assert ObjectiveConfig().metric == "f1"
    result = evaluate_objective(make_report(), make_report())
    assert result.metric == "f1"
    assert result.direction == DIRECTION_HIGHER_IS_BETTER == "higher_is_better"


def test_equal_metrics_are_valid_zero_degradation():
    result = evaluate_objective(make_report(f1=0.7), make_report(f1=0.7))
    assert result.valid is True
    assert result.compatible is True
    assert result.baseline_value == pytest.approx(0.7)
    assert result.perturbed_value == pytest.approx(0.7)
    assert result.degradation == pytest.approx(0.0)
    assert result.warnings == ()


def test_lower_perturbed_metric_is_positive_degradation():
    result = evaluate_objective(make_report(f1=0.8), make_report(f1=0.5))
    assert result.valid is True
    assert result.degradation == pytest.approx(0.3)


def test_improvement_is_negative_degradation():
    result = evaluate_objective(make_report(f1=0.5), make_report(f1=0.8))
    assert result.valid is True
    assert result.degradation == pytest.approx(-0.3)


@pytest.mark.parametrize("metric", SUPPORTED_METRICS)
def test_each_supported_metric_is_compared(metric):
    baseline = make_report(precision=1.0, recall=1.0, f1=1.0, map50=1.0)
    perturbed = make_report(
        precision=0.9 if metric == "precision" else 1.0,
        recall=0.9 if metric == "recall" else 1.0,
        f1=0.9 if metric == "f1" else 1.0,
        map50=0.9 if metric == "map50" else 1.0,
    )
    result = evaluate_objective(
        baseline, perturbed, config=ObjectiveConfig(metric=metric)
    )
    assert result.metric == metric
    assert result.direction == DIRECTION_HIGHER_IS_BETTER
    assert result.valid is True
    assert result.baseline_value == pytest.approx(1.0)
    assert result.perturbed_value == pytest.approx(0.9)
    assert result.degradation == pytest.approx(0.1)


def test_only_the_selected_metric_affects_the_objective():
    baseline = make_report(precision=0.4, f1=1.0)
    perturbed = make_report(precision=0.4, f1=0.1)
    result = evaluate_objective(
        baseline, perturbed, config=ObjectiveConfig(metric="precision")
    )
    assert result.valid is True
    assert result.degradation == pytest.approx(0.0)


def test_invalid_metric_name_is_rejected():
    with pytest.raises(ValueError, match="metric must be one of"):
        ObjectiveConfig(metric="accuracy")


def test_missing_metric_is_invalid_not_zero():
    perturbed = make_report()
    perturbed.aggregate.pop("f1")
    result = evaluate_objective(make_report(), perturbed)
    assert result.valid is False
    assert result.compatible is True
    assert result.baseline_value == pytest.approx(1.0)
    assert result.perturbed_value is None
    assert result.degradation is None
    assert any("missing or non-finite" in warning for warning in result.warnings)


def test_missing_map50_is_invalid_not_zero():
    result = evaluate_objective(
        make_report(map50=float("nan")),
        make_report(map50=0.5),
        config=ObjectiveConfig(metric="map50"),
    )
    assert result.valid is False
    assert result.baseline_value is None
    assert result.perturbed_value == pytest.approx(0.5)
    assert result.degradation is None


def test_non_finite_metric_is_invalid():
    result = evaluate_objective(
        make_report(f1=1.0), make_report(f1=float("inf"))
    )
    assert result.valid is False
    assert result.baseline_value == pytest.approx(1.0)
    assert result.perturbed_value is None
    assert result.degradation is None


@pytest.mark.parametrize(
    "changes",
    [
        {"iou_threshold": 0.3},
        {"aggregation_method": "macro"},
    ],
)
def test_incompatible_reports_are_invalid(changes):
    result = evaluate_objective(
        make_report(f1=0.8), make_report(f1=0.5, **changes)
    )
    assert result.valid is False
    assert result.compatible is False
    assert result.baseline_value == pytest.approx(0.8)
    assert result.perturbed_value == pytest.approx(0.5)
    assert result.degradation is None
    assert any("incomparable" in warning for warning in result.warnings)


def test_missing_metric_and_incompatible_reports_are_invalid():
    perturbed = make_report(iou_threshold=0.3)
    perturbed.aggregate.pop("f1")
    result = evaluate_objective(make_report(), perturbed)
    assert result.valid is False
    assert result.compatible is False
    assert result.degradation is None
    assert any("incomparable" in warning for warning in result.warnings)
    assert any("no valid degradation" in warning for warning in result.warnings)


def test_warning_only_differences_remain_comparable():
    baseline = make_report(split="test", per_image=[object()])
    perturbed = make_report(split="val", per_image=[])
    result = evaluate_objective(baseline, perturbed)
    assert result.valid is True
    assert result.compatible is True
    assert result.degradation == pytest.approx(0.0)
    assert any("splits differ" in warning for warning in result.warnings)
    assert any("image counts differ" in warning for warning in result.warnings)


def test_invalid_result_is_distinct_from_valid_zero():
    zero = evaluate_objective(make_report(), make_report())
    invalid = evaluate_objective(make_report(), dataclasses.replace(
        make_report(), iou_threshold=0.3
    ))
    assert zero.valid is True and zero.degradation == 0.0
    assert invalid.valid is False and invalid.degradation is None


def test_repeated_evaluation_is_deterministic_and_json_safe():
    baseline = make_report(f1=0.75)
    perturbed = make_report(f1=0.5)
    first = evaluate_objective(baseline, perturbed).to_dict()
    second = evaluate_objective(baseline, perturbed).to_dict()
    assert first == second
    assert json.loads(json.dumps(first)) == first
    assert first == {
        "metric": "f1",
        "direction": "higher_is_better",
        "baseline_value": 0.75,
        "perturbed_value": 0.5,
        "degradation": 0.25,
        "compatible": True,
        "valid": True,
        "warnings": [],
    }
