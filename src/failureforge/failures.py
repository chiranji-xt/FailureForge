"""Verified failure analysis and prediction-instability records.

Two deliberately separate layers
--------------------------------
1. **Benchmark analysis** (:func:`analyze_report`) derives *verified* failure
   categories from an :class:`~failureforge.evaluator.EvaluationReport` that was
   produced by matching predictions against ground truth. Categories never change
   TP/FP/FN counts; they explain them.
2. **Quick Mode analysis** (:func:`analyze_quick`) turns a label-free
   :class:`~failureforge.evaluation.QuickModeResult` into instability records
   (disappeared, appeared, class change, confidence change, box drift). These are
   **not** verified false positives/negatives and carry no accuracy metrics.

Verified category rules (IoU thresholds are documented on each record)
-----------------------------------------------------------------------
- ``false_negative``: an unmatched ground-truth box (no class-correct prediction
  reached ``iou_threshold``). ``iou`` is the best overlap with *any* prediction,
  for diagnostics only.
- ``class_confusion``: an unmatched prediction with the evaluator's
  ``wrong_class`` diagnostic: it overlaps an *unmatched* GT box of a different
  class at IoU >= ``iou_threshold``.
- ``localization_error``: either (a) a matched prediction whose IoU is in
  ``[iou_threshold, localization_iou_threshold)`` (``counts_as="tp"``), or (b) an
  unmatched prediction with the ``poor_localization`` diagnostic (same class,
  best unmatched overlap in ``(0, iou_threshold)``; ``counts_as="fp"``).
- ``duplicate_detection``: an additional same-class prediction that also covers a
  GT box at IoU >= ``iou_threshold`` when another prediction is already the
  primary detection for that GT. Duplicates are the lowest-confidence extras.
- ``false_positive``: a prediction with no spatial overlap with any GT box.
- ``unmatched``: a prediction that cannot be attributed reliably, e.g. low
  overlap with a GT of another class, overlap with a GT already matched to a
  different-class prediction, or a predicted class absent from the dataset
  vocabulary.

Priority for unmatched predictions: duplicate > class confusion > localization >
background false positive > ambiguous/unmatched. Each prediction contributes at
most one failure record; each unmatched GT contributes at most one. A matched
``localization_error`` is an aspect of a true positive, not an extra FP/FN.
"""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import PIL

from . import __version__
from .evaluation import PerturbedResult, QuickModeResult
from .evaluator import EvaluationReport, PerImageResult
from .metrics import LOCALIZATION_IOU_THRESH, iou_xyxy

SCHEMA_VERSION = "failureforge.failure-analysis.v1"

FAILURE_CATEGORIES = (
    "false_negative",
    "false_positive",
    "class_confusion",
    "localization_error",
    "duplicate_detection",
    "unmatched",
)

QUICK_INSTABILITY_KINDS = (
    "disappeared",
    "appeared",
    "class_changed",
    "confidence_changed",
    "box_drift",
)

DEFAULT_RANK_METRICS = ("precision", "recall", "f1", "map50")


class AnalysisError(ValueError):
    """Raised when an evaluation report lacks the detail needed for analysis."""


# ---------------------------------------------------------------------------
# Configuration / thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformationConfig:
    """A reproducible perturbation configuration (name/params/seed/family)."""

    name: str
    params: dict = field(default_factory=dict)
    seed: int = 0
    family: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("transformation name must be non-empty")
        object.__setattr__(self, "params", dict(self.params or {}))
        object.__setattr__(self, "seed", int(self.seed))

    @classmethod
    def from_dict(cls, data: dict) -> "TransformationConfig":
        return cls(
            name=str(data["name"]),
            params=dict(data.get("params") or {}),
            seed=int(data.get("seed", 0)),
            family=data.get("family"),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "params": dict(self.params),
            "seed": int(self.seed),
        }


# ---------------------------------------------------------------------------
# Verified failure records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerifiedFailure:
    """One verified benchmark failure or diagnostic aspect.

    ``counts_as`` states the metric role of the underlying detection
    (``tp``/``fp``/``fn``) so consumers never mistake diagnostics for changed
    metrics.
    """

    category: str
    reason: str
    image: str
    counts_as: str
    transformation: dict | None = None
    gt_index: int | None = None
    pred_index: int | None = None
    gt_class_id: int | None = None
    gt_class_name: str | None = None
    pred_class_id: int | None = None
    pred_class_name: str | None = None
    gt_box: list[float] | None = None
    pred_box: list[float] | None = None
    score: float | None = None
    iou: float | None = None
    iou_threshold: float = 0.5
    localization_iou_threshold: float = LOCALIZATION_IOU_THRESH

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "reason": self.reason,
            "image": self.image,
            "counts_as": self.counts_as,
            "transformation": self.transformation,
            "gt_index": self.gt_index,
            "pred_index": self.pred_index,
            "gt_class_id": self.gt_class_id,
            "gt_class_name": self.gt_class_name,
            "pred_class_id": self.pred_class_id,
            "pred_class_name": self.pred_class_name,
            "gt_box": list(self.gt_box) if self.gt_box is not None else None,
            "pred_box": list(self.pred_box) if self.pred_box is not None else None,
            "score": self.score,
            "iou": self.iou,
            "iou_threshold": self.iou_threshold,
            "localization_iou_threshold": self.localization_iou_threshold,
        }


@dataclass
class SampleFailureSummary:
    image: str
    num_gt: int
    num_pred: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    counts: dict

    def to_dict(self) -> dict:
        return {
            "image": self.image, "num_gt": self.num_gt, "num_pred": self.num_pred,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "counts": dict(self.counts),
        }


@dataclass
class FailureAnalysis:
    failures: list[VerifiedFailure]
    counts: dict
    per_sample: list[SampleFailureSummary]
    thresholds: dict
    transformation: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": "benchmark",
            "transformation": self.transformation,
            "thresholds": dict(self.thresholds),
            "counts": dict(self.counts),
            "per_sample": [s.to_dict() for s in self.per_sample],
            "failures": [f.to_dict() for f in self.failures],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Report reconstruction helpers
# ---------------------------------------------------------------------------

def _best_iou(box: list[float], candidates: Iterable[list[float]]) -> tuple[float, int | None]:
    best, best_idx = 0.0, None
    for idx, other in candidates:
        value = iou_xyxy(tuple(box), tuple(other))
        if value > best:
            best, best_idx = value, idx
    return best, best_idx


def _image_ground_truth(per: PerImageResult) -> dict[int, dict]:
    gts: dict[int, dict] = {}
    for match in per.matches:
        gi = int(match["gt_index"])
        if gi in gts:
            raise AnalysisError(f"{per.image}: duplicate gt_index {gi} in match details")
        gts[gi] = {
            "box": list(match.get("gt_box") or []),
            "class_id": int(match["class_id"]),
            "class_name": match.get("class_name"),
            "matched": True,
        }
    for missed in per.missed:
        gi = int(missed["gt_index"])
        if gi in gts:
            raise AnalysisError(f"{per.image}: gt_index {gi} both matched and missed")
        gts[gi] = {
            "box": list(missed["box"]),
            "class_id": int(missed["class_id"]),
            "class_name": missed.get("class_name"),
            "matched": False,
        }
    return gts


def _image_predictions(per: PerImageResult) -> dict[int, dict]:
    preds: dict[int, dict] = {}
    for match in per.matches:
        pi = int(match["pred_index"])
        if pi in preds:
            raise AnalysisError(f"{per.image}: duplicate pred_index {pi} in match details")
        preds[pi] = {
            "box": list(match.get("pred_box") or []),
            "class_id": int(match["class_id"]),
            "class_name": match.get("class_name"),
            "score": float(match["score"]),
            "diagnostic": None,
            "matched": True,
        }
    for fp in per.false_positives:
        pi = int(fp["pred_index"])
        if pi in preds:
            raise AnalysisError(f"{per.image}: pred_index {pi} both matched and unmatched")
        preds[pi] = {
            "box": list(fp["box"]),
            "class_id": int(fp["class_id"]),
            "class_name": fp.get("class_name"),
            "score": float(fp["score"]),
            "diagnostic": fp.get("diagnostic"),
            "matched": False,
        }
    return preds


def _duplicate_predictions(
    gts: dict[int, dict],
    preds: dict[int, dict],
    matched: dict[int, int],
    iou_threshold: float,
) -> dict[int, int]:
    """Map duplicate prediction index -> GT index (same class, IoU >= threshold).

    The evaluator's matched prediction (or the highest-confidence prediction for
    an unmatched GT) is the primary detection; extras are duplicates. Matched
    true positives are never reclassified as duplicates.
    """
    duplicate: dict[int, int] = {}
    matched_preds = set(matched.values())
    for gi in sorted(gts):
        gt = gts[gi]
        if gt["class_id"] < 0:
            continue
        candidates = [
            pi for pi, pred in preds.items()
            if pred["class_id"] == gt["class_id"]
            and iou_xyxy(tuple(pred["box"]), tuple(gt["box"])) >= iou_threshold
        ]
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda pi: (-preds[pi]["score"], pi))
        primary = matched.get(gi)
        if primary not in candidates:
            primary = candidates[0]
        for pi in candidates:
            if pi == primary or pi in matched_preds or pi in duplicate:
                continue
            duplicate[pi] = gi
    return duplicate


def _record(
    *, category: str, reason: str, image: str, counts_as: str,
    transformation: dict | None, iou_threshold: float,
    localization_iou_threshold: float, gt: dict | None = None,
    pred: dict | None = None, gt_index: int | None = None,
    pred_index: int | None = None, iou: float | None = None,
    score: float | None = None,
) -> VerifiedFailure:
    if category not in FAILURE_CATEGORIES:
        raise ValueError(f"unknown failure category: {category}")
    return VerifiedFailure(
        category=category, reason=reason, image=image, counts_as=counts_as,
        transformation=transformation,
        gt_index=gt_index, pred_index=pred_index,
        gt_class_id=int(gt["class_id"]) if gt else None,
        gt_class_name=gt.get("class_name") if gt else None,
        pred_class_id=int(pred["class_id"]) if pred else None,
        pred_class_name=pred.get("class_name") if pred else None,
        gt_box=list(gt["box"]) if gt else None,
        pred_box=list(pred["box"]) if pred else None,
        score=float(score) if score is not None else (
            float(pred["score"]) if pred else None
        ),
        iou=float(iou) if iou is not None else None,
        iou_threshold=float(iou_threshold),
        localization_iou_threshold=float(localization_iou_threshold),
    )


def analyze_report(
    report: EvaluationReport,
    *,
    transformation: dict | None = None,
    localization_iou_threshold: float = LOCALIZATION_IOU_THRESH,
) -> FailureAnalysis:
    """Derive verified failure records from one evaluation report.

    Baseline and perturbed reports are analyzed the same way (there is no
    baseline-specific rule). Raises :class:`AnalysisError` if per-image details
    are incomplete, rather than silently misclassifying detections.
    """
    loc_thr = float(localization_iou_threshold)
    if not (0.0 < loc_thr <= 1.0):
        raise ValueError("localization_iou_threshold must be in (0, 1]")
    if transformation is not None and "name" not in transformation:
        raise ValueError("transformation must contain a 'name'")

    failures: list[VerifiedFailure] = []
    per_sample: list[SampleFailureSummary] = []
    warnings: list[str] = []
    if loc_thr < report.iou_threshold:
        warnings.append(
            "localization_iou_threshold is below the matching IoU threshold; "
            "matched localization errors cannot be reported"
        )

    for per in report.per_image:
        gts = _image_ground_truth(per)
        preds = _image_predictions(per)
        if len(gts) != per.num_gt or len(preds) != per.num_pred:
            raise AnalysisError(
                f"{per.image}: evaluator detail count mismatch "
                f"(gt {len(gts)}/{per.num_gt}, pred {len(preds)}/{per.num_pred})"
            )

        matched = {int(m["gt_index"]): int(m["pred_index"]) for m in per.matches}
        image_failures: list[VerifiedFailure] = []
        count = {c: 0 for c in FAILURE_CATEGORIES}

        def add(record: VerifiedFailure) -> None:
            image_failures.append(record)
            count[record.category] += 1

        # Localization aspects of true positives.
        for match in per.matches:
            iou = float(match["iou"])
            if iou < loc_thr:
                gi, pi = int(match["gt_index"]), int(match["pred_index"])
                add(_record(
                    category="localization_error",
                    reason=(
                        "matched detection IoU is below the localization threshold"
                    ),
                    image=per.image, counts_as="tp", transformation=transformation,
                    iou_threshold=report.iou_threshold,
                    localization_iou_threshold=loc_thr,
                    gt=gts[gi], pred=preds[pi], gt_index=gi, pred_index=pi,
                    iou=iou,
                ))

        # Unmatched ground truth: verified misses.
        for gi in sorted(gts):
            gt = gts[gi]
            if gt["matched"]:
                continue
            best_iou, _best_pi = _best_iou(
                gt["box"], ((pi, p["box"]) for pi, p in preds.items())
            )
            add(_record(
                category="false_negative",
                reason=(
                    "no class-correct prediction reached the IoU threshold "
                    "(iou is best overlap with any prediction)"
                ),
                image=per.image, counts_as="fn", transformation=transformation,
                iou_threshold=report.iou_threshold,
                localization_iou_threshold=loc_thr,
                gt=gt, gt_index=gi, iou=best_iou,
            ))

        duplicates = _duplicate_predictions(gts, preds, matched, report.iou_threshold)

        for pi in sorted(preds):
            pred = preds[pi]
            if pred["matched"]:
                continue
            if pi in duplicates:
                gi = duplicates[pi]
                add(_record(
                    category="duplicate_detection",
                    reason=(
                        "additional same-class prediction covers the same GT box "
                        "as the primary detection"
                    ),
                    image=per.image, counts_as="fp", transformation=transformation,
                    iou_threshold=report.iou_threshold,
                    localization_iou_threshold=loc_thr,
                    gt=gts[gi], pred=pred, gt_index=gi, pred_index=pi,
                    iou=iou_xyxy(tuple(pred["box"]), tuple(gts[gi]["box"])),
                ))
                continue

            if pred["diagnostic"] == "wrong_class":
                candidates = [
                    (gi, gt) for gi, gt in gts.items()
                    if not gt["matched"] and gt["class_id"] != pred["class_id"]
                ]
                if candidates:
                    gi, gt = max(
                        candidates,
                        key=lambda item: iou_xyxy(tuple(pred["box"]), tuple(item[1]["box"])),
                    )
                    add(_record(
                        category="class_confusion",
                        reason=(
                            "prediction overlaps an unmatched GT box of a "
                            "different class at/above the IoU threshold"
                        ),
                        image=per.image, counts_as="fp",
                        transformation=transformation,
                        iou_threshold=report.iou_threshold,
                        localization_iou_threshold=loc_thr,
                        gt=gt, pred=pred, gt_index=gi, pred_index=pi,
                        iou=iou_xyxy(tuple(pred["box"]), tuple(gt["box"])),
                    ))
                    continue

            if pred["diagnostic"] == "poor_localization":
                candidates = [
                    (gi, gt) for gi, gt in gts.items()
                    if not gt["matched"] and gt["class_id"] == pred["class_id"]
                ]
                scored = [
                    (iou_xyxy(tuple(pred["box"]), tuple(gt["box"])), gi, gt)
                    for gi, gt in candidates
                ]
                scored = [item for item in scored if item[0] > 0]
                if scored:
                    value, gi, gt = max(scored, key=lambda item: (item[0], -item[1]))
                    add(_record(
                        category="localization_error",
                        reason=(
                            "unmatched same-class prediction overlaps a missed "
                            "GT box below the matching IoU threshold"
                        ),
                        image=per.image, counts_as="fp",
                        transformation=transformation,
                        iou_threshold=report.iou_threshold,
                        localization_iou_threshold=loc_thr,
                        gt=gt, pred=pred, gt_index=gi, pred_index=pi, iou=value,
                    ))
                    continue

            if pred["class_id"] < 0:
                add(_record(
                    category="unmatched",
                    reason="predicted class is not present in the dataset vocabulary",
                    image=per.image, counts_as="fp", transformation=transformation,
                    iou_threshold=report.iou_threshold,
                    localization_iou_threshold=loc_thr,
                    pred=pred, pred_index=pi,
                ))
                continue

            best_iou, best_gi = _best_iou(
                pred["box"], ((gi, gt["box"]) for gi, gt in gts.items())
            )
            if best_iou <= 0.0:
                add(_record(
                    category="false_positive",
                    reason="prediction has no spatial overlap with any GT box",
                    image=per.image, counts_as="fp", transformation=transformation,
                    iou_threshold=report.iou_threshold,
                    localization_iou_threshold=loc_thr,
                    pred=pred, pred_index=pi, iou=0.0,
                ))
            else:
                add(_record(
                    category="unmatched",
                    reason=(
                        "overlap cannot be attributed reliably to one documented "
                        "category (low/different-class overlap or GT already matched)"
                    ),
                    image=per.image, counts_as="fp", transformation=transformation,
                    iou_threshold=report.iou_threshold,
                    localization_iou_threshold=loc_thr,
                    gt=gts.get(best_gi) if best_gi is not None else None,
                    pred=pred,
                    gt_index=best_gi,
                    pred_index=pi,
                    iou=best_iou,
                ))

        failures.extend(image_failures)
        per_sample.append(SampleFailureSummary(
            image=per.image, num_gt=per.num_gt, num_pred=per.num_pred,
            tp=per.tp, fp=per.fp, fn=per.fn,
            precision=per.precision, recall=per.recall, f1=per.f1,
            counts=count,
        ))

    if not report.per_image:
        warnings.append("report contains no per-image results")

    order = {c: i for i, c in enumerate(FAILURE_CATEGORIES)}
    failures.sort(key=lambda f: (
        f.image, order[f.category],
        -1 if f.gt_index is None else f.gt_index,
        -1 if f.pred_index is None else f.pred_index,
        f.reason,
    ))
    per_sample.sort(key=lambda s: s.image)
    counts = {c: 0 for c in FAILURE_CATEGORIES}
    for failure in failures:
        counts[failure.category] += 1

    thresholds = {
        "iou_threshold": float(report.iou_threshold),
        "localization_iou_threshold": loc_thr,
    }
    transformation_payload = (
        TransformationConfig.from_dict(transformation).to_dict()
        if transformation is not None else None
    )
    return FailureAnalysis(
        failures=failures, counts=counts, per_sample=per_sample,
        thresholds=thresholds, transformation=transformation_payload,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Metric comparison
# ---------------------------------------------------------------------------

@dataclass
class MetricComparison:
    deltas: dict
    drops: dict
    compatible: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "deltas": dict(self.deltas),
            "drops": dict(self.drops),
            "compatible": bool(self.compatible),
            "warnings": list(self.warnings),
        }


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def compare_reports(
    baseline: EvaluationReport,
    perturbed: EvaluationReport,
    *,
    metrics: tuple[str, ...] = DEFAULT_RANK_METRICS,
) -> MetricComparison:
    """Compare finite metrics; missing/non-finite values become ``None``.

    Ranking must use one metric at a time (``drops`` are baseline - perturbed).
    Differences in IoU threshold or aggregation method mark the comparison as
    incompatible; split/confidence/sample-count differences are warnings.
    """
    supported = set(DEFAULT_RANK_METRICS)
    unknown = set(metrics) - supported
    if unknown:
        raise ValueError(f"unsupported metrics: {sorted(unknown)}")
    if "map50" not in supported:
        raise AssertionError("map50 must be supported")

    warnings: list[str] = []
    deltas: dict[str, float | None] = {}
    drops: dict[str, float | None] = {}
    for name in metrics:
        if name == "map50":
            base_value = _finite(baseline.map50)
            pert_value = _finite(perturbed.map50)
        else:
            base_value = _finite(baseline.aggregate.get(name))
            pert_value = _finite(perturbed.aggregate.get(name))
        if base_value is None or pert_value is None:
            deltas[name] = None
            drops[name] = None
            missing = [
                label for label, value in
                (("baseline", base_value), ("perturbed", pert_value))
                if value is None
            ]
            warnings.append(f"{name}: missing or non-finite value ({', '.join(missing)})")
        else:
            deltas[name] = pert_value - base_value
            drops[name] = base_value - pert_value

    compatible = True
    if baseline.iou_threshold != perturbed.iou_threshold:
        compatible = False
        warnings.append(
            "incomparable: IoU thresholds differ "
            f"({baseline.iou_threshold} vs {perturbed.iou_threshold})"
        )
    if baseline.aggregation_method != perturbed.aggregation_method:
        compatible = False
        warnings.append(
            "incomparable: aggregation methods differ "
            f"({baseline.aggregation_method!r} vs {perturbed.aggregation_method!r})"
        )
    if baseline.split != perturbed.split:
        warnings.append(
            f"splits differ ({baseline.split!r} vs {perturbed.split!r})"
        )
    if baseline.conf_threshold != perturbed.conf_threshold:
        warnings.append(
            "detector confidence thresholds differ "
            f"({baseline.conf_threshold} vs {perturbed.conf_threshold})"
        )
    if len(baseline.per_image) != len(perturbed.per_image):
        warnings.append(
            "evaluated image counts differ "
            f"({len(baseline.per_image)} vs {len(perturbed.per_image)})"
        )
    return MetricComparison(deltas=deltas, drops=drops,
                            compatible=compatible, warnings=warnings)


# ---------------------------------------------------------------------------
# Benchmark trial / run records
# ---------------------------------------------------------------------------

@dataclass
class TrialRecord:
    transformation: dict | None
    metrics: dict
    deltas: dict
    drops: dict
    per_sample: list[SampleFailureSummary]
    failures: list[VerifiedFailure]
    failure_counts: dict
    warnings: list[str] = field(default_factory=list)
    compatible: bool = True

    def to_dict(self) -> dict:
        return {
            "transformation": self.transformation,
            "metrics": dict(self.metrics),
            "deltas": dict(self.deltas),
            "drops": dict(self.drops),
            "compatible": bool(self.compatible),
            "per_sample": [s.to_dict() for s in self.per_sample],
            "failure_counts": dict(self.failure_counts),
            "failures": [f.to_dict() for f in self.failures],
            "warnings": list(self.warnings),
        }


def _metrics_payload(report: EvaluationReport, loc_thr: float) -> dict:
    return {
        "aggregate": dict(report.aggregate),
        "map50": report.map50,
        "iou_threshold": float(report.iou_threshold),
        "localization_iou_threshold": float(loc_thr),
        "conf_threshold": float(report.conf_threshold),
        "aggregation_method": report.aggregation_method,
        "split": report.split,
        "num_images": len(report.per_image),
    }


def _trial_from_report(
    report: EvaluationReport,
    *,
    transformation: dict | None,
    comparison: MetricComparison | None,
    loc_thr: float,
) -> TrialRecord:
    analysis = analyze_report(
        report, transformation=transformation,
        localization_iou_threshold=loc_thr,
    )
    warnings = list(analysis.warnings)
    if comparison is not None:
        warnings = list(comparison.warnings) + warnings
    return TrialRecord(
        transformation=transformation,
        metrics=_metrics_payload(report, loc_thr),
        deltas=dict(comparison.deltas) if comparison else {},
        drops=dict(comparison.drops) if comparison else {},
        per_sample=analysis.per_sample,
        failures=analysis.failures,
        failure_counts=analysis.counts,
        warnings=warnings,
        compatible=bool(comparison.compatible) if comparison else True,
    )


def analyze_perturbed(
    result: PerturbedResult,
    *,
    localization_iou_threshold: float = LOCALIZATION_IOU_THRESH,
) -> TrialRecord:
    """Build a trial record from a Day-5 ``PerturbedResult`` (no mutation)."""
    comparison = compare_reports(result.baseline, result.perturbed)
    return _trial_from_report(
        result.perturbed,
        transformation=TransformationConfig.from_dict(result.perturbation).to_dict(),
        comparison=comparison,
        loc_thr=float(localization_iou_threshold),
    )


def environment_metadata() -> dict:
    return {
        "failureforge": __version__,
        "python": platform.python_version(),
        "python_implementation": sys.implementation.name,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }


@dataclass
class BenchmarkRun:
    metadata: dict
    baseline: TrialRecord
    trials: list[TrialRecord]
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "mode": "benchmark",
            "metadata": dict(self.metadata),
            "baseline": self.baseline.to_dict(),
            "trials": [t.to_dict() for t in self.trials],
            "warnings": list(self.warnings),
        }


def build_benchmark_run(
    baseline: EvaluationReport,
    trials: Iterable[PerturbedResult],
    *,
    metadata: dict | None = None,
    warnings: Iterable[str] | None = None,
    localization_iou_threshold: float = LOCALIZATION_IOU_THRESH,
) -> BenchmarkRun:
    """Assemble a serializable benchmark run from Day-5 results.

    The supplied baseline is the single source of truth for every comparison;
    it is analyzed once and never modified. Trial order is preserved here; the
    report layer ranks a copy.
    """
    loc_thr = float(localization_iou_threshold)
    baseline_analysis = analyze_report(
        baseline, localization_iou_threshold=loc_thr
    )
    baseline_record = TrialRecord(
        transformation=None,
        metrics=_metrics_payload(baseline, loc_thr),
        deltas={}, drops={},
        per_sample=baseline_analysis.per_sample,
        failures=baseline_analysis.failures,
        failure_counts=baseline_analysis.counts,
        warnings=list(baseline_analysis.warnings),
    )

    trial_records: list[TrialRecord] = []
    for result in trials:
        comparison = compare_reports(baseline, result.perturbed)
        trial_records.append(_trial_from_report(
            result.perturbed,
            transformation=TransformationConfig.from_dict(result.perturbation).to_dict(),
            comparison=comparison,
            loc_thr=loc_thr,
        ))

    run_metadata = dict(baseline.metadata)
    run_metadata.update({
        "thresholds": {
            "iou_threshold": float(baseline.iou_threshold),
            "localization_iou_threshold": loc_thr,
        },
        "environment": environment_metadata(),
        "scope": (
            "observations for the evaluated model, dataset, split, and thresholds; "
            "not a universal claim about perturbation damage"
        ),
    })
    if metadata:
        run_metadata.update(metadata)

    run_warnings: list[str] = []
    for message in [*baseline_record.warnings,
                    *(w for t in trial_records for w in t.warnings),
                    *(warnings or [])]:
        if message not in run_warnings:
            run_warnings.append(message)

    return BenchmarkRun(
        metadata=run_metadata, baseline=baseline_record,
        trials=trial_records, warnings=run_warnings,
    )


# ---------------------------------------------------------------------------
# Quick Mode instability records (never verified failures)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuickInstability:
    kind: str
    image: str
    transformation: dict
    clean_pred_index: int | None = None
    perturbed_pred_index: int | None = None
    clean_class_id: int | None = None
    clean_class_name: str | None = None
    perturbed_class_id: int | None = None
    perturbed_class_name: str | None = None
    clean_box: list[float] | None = None
    perturbed_box: list[float] | None = None
    clean_score: float | None = None
    perturbed_score: float | None = None
    confidence_delta: float | None = None
    iou: float | None = None
    drift: float | None = None
    iou_threshold: float = 0.5
    confidence_epsilon: float = 0.01

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "image": self.image,
            "transformation": self.transformation,
            "verified_failure": False,
            "clean_pred_index": self.clean_pred_index,
            "perturbed_pred_index": self.perturbed_pred_index,
            "clean_class_id": self.clean_class_id,
            "clean_class_name": self.clean_class_name,
            "perturbed_class_id": self.perturbed_class_id,
            "perturbed_class_name": self.perturbed_class_name,
            "clean_box": list(self.clean_box) if self.clean_box is not None else None,
            "perturbed_box": (
                list(self.perturbed_box) if self.perturbed_box is not None else None
            ),
            "clean_score": self.clean_score,
            "perturbed_score": self.perturbed_score,
            "confidence_delta": self.confidence_delta,
            "iou": self.iou, "drift": self.drift,
            "iou_threshold": self.iou_threshold,
            "confidence_epsilon": self.confidence_epsilon,
        }


@dataclass
class QuickAnalysis:
    records: list[QuickInstability]
    counts: dict
    per_image: list[dict]
    summary: dict
    thresholds: dict
    transformation: dict
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": "quick",
            "label": "prediction_instability",
            "verified_failures": False,
            "transformation": self.transformation,
            "thresholds": dict(self.thresholds),
            "counts": dict(self.counts),
            "per_image": [dict(p) for p in self.per_image],
            "records": [r.to_dict() for r in self.records],
            "summary": dict(self.summary),
            "warnings": list(self.warnings),
        }


_QUICK_ORDER = {kind: i for i, kind in enumerate(QUICK_INSTABILITY_KINDS)}


def _class_name(names: list, class_id: int | None) -> str | None:
    if class_id is None or class_id < 0 or class_id >= len(names):
        return None
    return names[class_id]


def analyze_quick(result: QuickModeResult) -> QuickAnalysis:
    """Convert Day-5 Quick Mode output into instability-only records."""
    transformation = TransformationConfig.from_dict(result.perturbation).to_dict()
    iou_threshold = float(result.iou_threshold)
    confidence_epsilon = float(result.metadata.get("confidence_epsilon", 0.01))
    names = list((result.metadata.get("model") or {}).get("class_names") or [])

    records: list[QuickInstability] = []
    per_image: list[dict] = []
    counts = {kind: 0 for kind in QUICK_INSTABILITY_KINDS}

    for image in result.per_image:
        image_name = image["image"]
        image_counts = {kind: 0 for kind in QUICK_INSTABILITY_KINDS}

        def add(record: QuickInstability) -> None:
            records.append(record)
            counts[record.kind] += 1
            image_counts[record.kind] += 1

        for item in image.get("disappeared", []):
            add(QuickInstability(
                kind="disappeared", image=image_name, transformation=transformation,
                clean_pred_index=int(item["index"]),
                clean_class_id=int(item["class_id"]),
                clean_class_name=_class_name(names, int(item["class_id"])),
                clean_box=list(item["box"]), clean_score=float(item["score"]),
                iou_threshold=iou_threshold,
                confidence_epsilon=confidence_epsilon,
            ))
        for item in image.get("appeared", []):
            add(QuickInstability(
                kind="appeared", image=image_name, transformation=transformation,
                perturbed_pred_index=int(item["index"]),
                perturbed_class_id=int(item["class_id"]),
                perturbed_class_name=_class_name(names, int(item["class_id"])),
                perturbed_box=list(item["box"]), perturbed_score=float(item["score"]),
                iou_threshold=iou_threshold,
                confidence_epsilon=confidence_epsilon,
            ))
        for item in image.get("class_changed", []):
            add(QuickInstability(
                kind="class_changed", image=image_name,
                transformation=transformation,
                clean_pred_index=int(item["clean_index"]),
                perturbed_pred_index=int(item["perturbed_index"]),
                clean_class_id=int(item["from_class"]),
                clean_class_name=_class_name(names, int(item["from_class"])),
                perturbed_class_id=int(item["to_class"]),
                perturbed_class_name=_class_name(names, int(item["to_class"])),
                iou=float(item["iou"]), iou_threshold=iou_threshold,
                confidence_epsilon=confidence_epsilon,
            ))
        for item in image.get("confidence_changes", []):
            add(QuickInstability(
                kind="confidence_changed", image=image_name,
                transformation=transformation,
                clean_pred_index=int(item["clean_index"]),
                perturbed_pred_index=int(item["perturbed_index"]),
                clean_class_id=int(item["class_id"]),
                clean_class_name=_class_name(names, int(item["class_id"])),
                confidence_delta=float(item["delta"]), iou=float(item["iou"]),
                iou_threshold=iou_threshold,
                confidence_epsilon=confidence_epsilon,
            ))
        for item in image.get("box_drift", []):
            add(QuickInstability(
                kind="box_drift", image=image_name,
                transformation=transformation,
                clean_pred_index=int(item["clean_index"]),
                perturbed_pred_index=int(item["perturbed_index"]),
                clean_class_id=int(item["class_id"]),
                clean_class_name=_class_name(names, int(item["class_id"])),
                iou=float(item["iou"]), drift=float(item["drift"]),
                iou_threshold=iou_threshold,
                confidence_epsilon=confidence_epsilon,
            ))

        per_image.append({
            "image": image_name,
            "num_clean": int(image.get("num_clean", 0)),
            "num_perturbed": int(image.get("num_perturbed", 0)),
            "counts": image_counts,
        })

    allowed_summary = (
        "num_images", "clean_detections", "perturbed_detections", "matched",
        "disappeared", "appeared", "class_changed", "confidence_changed",
        "mean_confidence_delta", "mean_box_iou", "changed_detection_fraction",
    )
    summary = {
        key: result.summary[key]
        for key in allowed_summary if key in result.summary
    }
    summary["note"] = (
        "Prediction instability only; no ground truth is used, so these are not "
        "verified false positives/negatives and no accuracy metrics are reported."
    )

    records.sort(key=lambda r: (
        r.image, _QUICK_ORDER[r.kind],
        -1 if r.clean_pred_index is None else r.clean_pred_index,
        -1 if r.perturbed_pred_index is None else r.perturbed_pred_index,
    ))
    per_image.sort(key=lambda p: p["image"])

    warnings: list[str] = []
    if not result.per_image:
        warnings.append("Quick Mode evaluated no images")
    if summary.get("clean_detections", 0) == 0:
        warnings.append(
            "no clean detections; instability counts cannot be interpreted as "
            "detector degradation"
        )

    return QuickAnalysis(
        records=records, counts=counts, per_image=per_image, summary=summary,
        thresholds={
            "iou_threshold": iou_threshold,
            "confidence_epsilon": confidence_epsilon,
        },
        transformation=transformation, warnings=warnings,
    )


@dataclass
class QuickRun:
    metadata: dict
    transformation: dict
    analysis: QuickAnalysis
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        payload = self.analysis.to_dict()
        payload.update({
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
        })
        return payload


def build_quick_run(
    result: QuickModeResult,
    *,
    metadata: dict | None = None,
    warnings: Iterable[str] | None = None,
) -> QuickRun:
    """Assemble a serializable Quick Mode run (instability, not failures)."""
    analysis = analyze_quick(result)
    run_metadata = dict(result.metadata)
    run_metadata.update({
        "environment": environment_metadata(),
        "scope": (
            "label-free prediction instability for the evaluated model and "
            "images; not verified accuracy degradation"
        ),
    })
    if metadata:
        run_metadata.update(metadata)

    run_warnings = [
        "Quick Mode: no ground truth; records are prediction instability, not "
        "verified false positives or false negatives."
    ]
    for message in [*analysis.warnings, *(warnings or [])]:
        if message not in run_warnings:
            run_warnings.append(message)
    return QuickRun(
        metadata=run_metadata, transformation=analysis.transformation,
        analysis=analysis, warnings=run_warnings,
    )
