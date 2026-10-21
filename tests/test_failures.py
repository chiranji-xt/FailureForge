"""Verified failure categories, Quick instability separation, and run records."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from failureforge.dataset import YoloDataset
from failureforge.detector import Detections
from failureforge.evaluation import evaluate_perturbed, quick_mode
from failureforge.evaluator import evaluate_baseline
from failureforge.failures import (
    FAILURE_CATEGORIES,
    QUICK_INSTABILITY_KINDS,
    SCHEMA_VERSION,
    AnalysisError,
    analyze_perturbed,
    analyze_quick,
    analyze_report,
    build_benchmark_run,
    build_quick_run,
    compare_reports,
    environment_metadata,
)
from failureforge.metrics import LOCALIZATION_IOU_THRESH


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class MockDetector:
    def __init__(self, detections, class_names=("cat", "dog")):
        self._detections = list(detections)
        self._i = 0
        self.class_names = list(class_names)
        self.weights = "best.pt"
        self.sha256 = "a" * 64
        self.task = "detect"
        self.imgsz = 640
        self.device = "cpu"
        self.conf = 0.25

    def predict(self, image_bgr):
        out = self._detections[self._i]
        self._i += 1
        return out


def det(boxes, classes, scores):
    return Detections(
        boxes_xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.array(scores, dtype=np.float32),
        classes=np.array(classes, dtype=np.int64),
    )


EMPTY = det([], [], [])
GT_BOX = [40.0, 40.0, 60.0, 60.0]
GT_CAT = "0 0.5 0.5 0.2 0.2\n"


def make_dataset(tmp_path: Path, labels, names=("cat", "dog"), size=(100, 100),
                 labels_exist=True):
    root = tmp_path / "ds"
    imgs, lbls = root / "images" / "test", root / "labels" / "test"
    imgs.mkdir(parents=True, exist_ok=True)
    lbls.mkdir(parents=True, exist_ok=True)
    for i, body in enumerate(labels):
        Image.new("RGB", size, (120, 120, 120)).save(imgs / f"img{i}.png")
        if labels_exist and body is not None:
            (lbls / f"img{i}.txt").write_text(body, encoding="utf-8")
    return YoloDataset.from_dirs(
        imgs, lbls, list(names), split="test",
        allow_missing_labels=not labels_exist,
    )


def analyze_one(tmp_path, label, prediction, *, iou=0.5, loc=0.75,
                names=("cat", "dog"), model_names=("cat", "dog")):
    ds = make_dataset(tmp_path, [label], names=names)
    report = evaluate_baseline(
        MockDetector([prediction], class_names=model_names), ds,
        iou_threshold=iou,
    )
    return report, analyze_report(
        report, localization_iou_threshold=loc
    )


# ---------------------------------------------------------------------------
# verified categories
# ---------------------------------------------------------------------------

def test_false_negative_record(tmp_path):
    report, analysis = analyze_one(tmp_path, GT_CAT, EMPTY)
    assert analysis.counts["false_negative"] == 1
    assert sum(analysis.counts.values()) == 1
    record = analysis.failures[0]
    assert record.category == "false_negative"
    assert record.counts_as == "fn"
    assert record.image == "img0.png"
    assert record.gt_index == 0 and record.pred_index is None
    assert record.gt_class_id == 0 and record.gt_class_name == "cat"
    assert record.gt_box == GT_BOX
    assert record.iou == 0.0
    assert record.iou_threshold == 0.5
    assert analysis.per_sample[0].fn == 1


def test_background_false_positive_record(tmp_path):
    report, analysis = analyze_one(
        tmp_path, GT_CAT, det([[0, 0, 10, 10]], [0], [0.8])
    )
    assert analysis.counts["false_positive"] == 1
    record = next(f for f in analysis.failures if f.category == "false_positive")
    assert record.category == "false_positive"
    assert record.counts_as == "fp"
    assert record.pred_index == 0 and record.gt_index is None
    assert record.iou == 0.0 and record.score == pytest.approx(0.8)
    assert record.pred_box == [0.0, 0.0, 10.0, 10.0]


def test_class_confusion_is_distinct_from_ordinary_false_positive(tmp_path):
    report, analysis = analyze_one(
        tmp_path, GT_CAT, det([GT_BOX], [1], [0.9])
    )
    assert analysis.counts["class_confusion"] == 1
    assert analysis.counts["false_positive"] == 0
    # Class-aware matching leaves the GT unmatched, so it is also a verified miss.
    assert analysis.counts["false_negative"] == 1
    record = next(f for f in analysis.failures if f.category == "class_confusion")
    assert record.counts_as == "fp"
    assert record.gt_index == 0 and record.pred_index == 0
    assert record.gt_class_id == 0 and record.pred_class_id == 1
    assert record.iou == pytest.approx(1.0)


def test_localization_error_of_true_positive(tmp_path):
    prediction = det([[45, 40, 65, 60]], [0], [0.9])
    report, analysis = analyze_one(tmp_path, GT_CAT, prediction)
    assert (report.aggregate["tp"], report.aggregate["fp"], report.aggregate["fn"]) == (1, 0, 0)
    assert analysis.counts["localization_error"] == 1
    assert analysis.counts["false_positive"] == 0
    record = analysis.failures[0]
    assert record.category == "localization_error"
    assert record.counts_as == "tp"
    assert record.iou == pytest.approx(0.6)
    assert record.localization_iou_threshold == LOCALIZATION_IOU_THRESH
    assert record.gt_box == GT_BOX and record.pred_box == [45.0, 40.0, 65.0, 60.0]


def test_unmatched_localization_false_positive(tmp_path):
    prediction = det([[45, 40, 75, 60]], [0], [0.9])
    report, analysis = analyze_one(tmp_path, GT_CAT, prediction)
    assert analysis.counts["localization_error"] == 1
    assert analysis.counts["false_negative"] == 1
    assert analysis.counts["false_positive"] == 0
    fp = next(f for f in analysis.failures if f.category == "localization_error")
    assert fp.counts_as == "fp"
    assert fp.iou == pytest.approx(0.428571, abs=1e-5)


def test_duplicate_predictions_are_not_extra_false_positives(tmp_path):
    prediction = det(
        [GT_BOX, [41, 41, 61, 61]], [0, 0], [0.9, 0.8]
    )
    report, analysis = analyze_one(tmp_path, GT_CAT, prediction)
    assert (report.aggregate["tp"], report.aggregate["fp"], report.aggregate["fn"]) == (1, 1, 0)
    assert analysis.counts["duplicate_detection"] == 1
    assert analysis.counts["false_positive"] == 0
    assert analysis.counts["false_negative"] == 0
    record = analysis.failures[0]
    assert record.category == "duplicate_detection"
    assert record.counts_as == "fp"
    assert record.pred_index == 1 and record.gt_index == 0
    assert record.iou == pytest.approx(0.822, abs=1e-3)


def test_empty_ground_truth_and_empty_predictions(tmp_path):
    _report, analysis = analyze_one(tmp_path, "", EMPTY)
    assert sum(analysis.counts.values()) == 0
    assert analysis.failures == []
    assert analysis.per_sample[0].num_gt == 0

    _report, analysis = analyze_one(tmp_path, "", det([GT_BOX], [0], [0.9]))
    assert analysis.counts["false_positive"] == 1
    assert analysis.counts["false_negative"] == 0


def test_ambiguous_low_overlap_different_class(tmp_path):
    prediction = det([[45, 45, 65, 65]], [1], [0.7])
    _report, analysis = analyze_one(tmp_path, GT_CAT, prediction)
    assert analysis.counts["unmatched"] == 1
    assert analysis.counts["class_confusion"] == 0
    assert analysis.counts["localization_error"] == 0
    record = next(f for f in analysis.failures if f.category == "unmatched")
    assert record.counts_as == "fp"
    assert record.iou == pytest.approx(0.3913, abs=1e-3)


def test_ambiguous_different_class_overlap_with_matched_gt(tmp_path):
    prediction = det(
        [GT_BOX, [42, 42, 62, 62]], [0, 1], [0.9, 0.8]
    )
    _report, analysis = analyze_one(tmp_path, GT_CAT, prediction)
    assert analysis.counts["unmatched"] == 1
    assert analysis.counts["class_confusion"] == 0
    record = next(f for f in analysis.failures if f.category == "unmatched")
    assert record.pred_class_id == 1
    assert record.gt_index == 0


def test_unmapped_prediction_class_is_unmatched(tmp_path):
    label = GT_CAT
    ds = make_dataset(tmp_path, [label], names=("cat", "dog"))
    detector = MockDetector([det([GT_BOX], [2], [0.9])],
                            class_names=("cat", "dog", "zebra"))
    report = evaluate_baseline(detector, ds)
    analysis = analyze_report(report)
    assert analysis.counts["unmatched"] == 1
    record = next(f for f in analysis.failures if f.category == "unmatched")
    assert record.pred_class_id == -1
    assert "vocabulary" in record.reason


def test_class_confusion_at_exact_iou_threshold(tmp_path):
    label = "0 0.05 0.05 0.1 0.1\n"
    prediction = det([[0, 0, 10, 5]], [1], [0.9])
    _report, analysis = analyze_one(tmp_path, label, prediction, iou=0.5)
    assert analysis.counts["class_confusion"] == 1
    record = next(f for f in analysis.failures if f.category == "class_confusion")
    assert record.iou == pytest.approx(0.5)


def test_localization_threshold_is_configurable_and_strict(tmp_path):
    prediction = det([[45, 40, 65, 60]], [0], [0.9])
    ds = make_dataset(tmp_path, [GT_CAT])
    report = evaluate_baseline(MockDetector([prediction]), ds)
    assert analyze_report(report, localization_iou_threshold=0.65).counts[
        "localization_error"] == 1
    assert analyze_report(report, localization_iou_threshold=0.60).counts[
        "localization_error"] == 0
    below = analyze_report(report, localization_iou_threshold=0.4)
    assert below.counts["localization_error"] == 0
    assert any("below the matching IoU" in warning for warning in below.warnings)
    with pytest.raises(ValueError):
        analyze_report(report, localization_iou_threshold=0.0)


def test_incomplete_details_raise_instead_of_misclassifying(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    report = evaluate_baseline(MockDetector([EMPTY]), ds)
    broken = dataclasses.replace(
        report,
        per_image=[dataclasses.replace(report.per_image[0], missed=[])],
    )
    with pytest.raises(AnalysisError):
        analyze_report(broken)


def test_analysis_does_not_mutate_report(tmp_path):
    prediction = det([GT_BOX, [42, 42, 62, 62]], [0, 1], [0.9, 0.8])
    ds = make_dataset(tmp_path, [GT_CAT])
    report = evaluate_baseline(MockDetector([prediction]), ds)
    before = json.loads(json.dumps(report.to_dict()))
    analyze_report(report)
    assert report.to_dict() == before


def test_analysis_does_not_mutate_perturbed_or_quick_results(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT, GT_CAT])
    detector = MockDetector([
        det([GT_BOX], [0], [0.9]), EMPTY,
        det([GT_BOX], [0], [0.8]), EMPTY,
    ])
    result = evaluate_perturbed(detector, ds, "gaussian_noise", {"sigma": 15.0})
    before = json.loads(json.dumps(result.to_dict()))
    analyze_perturbed(result)
    build_benchmark_run(result.baseline, [result])
    assert result.to_dict() == before

    unlabeled = make_dataset(tmp_path / "quick", [None], labels_exist=False)
    quick_detector = MockDetector([
        det([GT_BOX], [0], [0.9]), EMPTY,
    ])
    quick_result = quick_mode(quick_detector, unlabeled, "brightness",
                              {"factor": 0.5})
    quick_before = json.loads(json.dumps(quick_result.to_dict()))
    analyze_quick(quick_result)
    build_quick_run(quick_result)
    assert quick_result.to_dict() == quick_before


def test_category_catalog_is_exactly_the_supported_set():
    assert FAILURE_CATEGORIES == (
        "false_negative", "false_positive", "class_confusion",
        "localization_error", "duplicate_detection", "unmatched",
    )
    assert QUICK_INSTABILITY_KINDS == (
        "disappeared", "appeared", "class_changed",
        "confidence_changed", "box_drift",
    )


def test_failure_analysis_payload_roundtrips_json(tmp_path):
    _report, analysis = analyze_one(
        tmp_path, GT_CAT, det([GT_BOX], [1], [0.9])
    )
    payload = analysis.to_dict()
    restored = json.loads(json.dumps(payload))
    assert restored["mode"] == "benchmark"
    assert restored["counts"]["class_confusion"] == 1
    record = next(
        f for f in restored["failures"] if f["category"] == "class_confusion"
    )
    for key in (
        "category", "reason", "image", "counts_as", "transformation",
        "gt_index", "pred_index", "gt_class_id", "gt_class_name",
        "pred_class_id", "pred_class_name", "gt_box", "pred_box", "score",
        "iou", "iou_threshold", "localization_iou_threshold",
    ):
        assert key in record


# ---------------------------------------------------------------------------
# Quick Mode separation
# ---------------------------------------------------------------------------

def test_quick_analysis_records_instability_only(tmp_path):
    ds = make_dataset(tmp_path, [None], labels_exist=False)
    detector = MockDetector([
        det([GT_BOX], [0], [0.9]),
        det([[0, 0, 10, 10]], [1], [0.6]),
    ])
    result = quick_mode(detector, ds, "brightness", {"factor": 1.0})
    analysis = analyze_quick(result)

    assert analysis.counts["disappeared"] == 1
    assert analysis.counts["appeared"] == 1
    assert set(analysis.counts) == set(QUICK_INSTABILITY_KINDS)
    assert not any(category in analysis.counts for category in FAILURE_CATEGORIES)
    for record in analysis.records:
        assert record.to_dict()["verified_failure"] is False
    payload = analysis.to_dict()
    assert payload["label"] == "prediction_instability"
    assert payload["verified_failures"] is False
    assert "precision" not in json.dumps(payload)


def test_quick_records_class_change_and_confidence_without_labels(tmp_path):
    ds = make_dataset(tmp_path, [None], labels_exist=False)
    detector = MockDetector([
        det([GT_BOX], [0], [0.9]),
        det([GT_BOX], [1], [0.4]),
    ])
    result = quick_mode(detector, ds, "brightness", {"factor": 1.0})
    analysis = analyze_quick(result)
    assert analysis.counts["class_changed"] == 1
    record = next(r for r in analysis.records if r.kind == "class_changed")
    assert record.clean_class_id == 0 and record.perturbed_class_id == 1
    assert record.clean_pred_index == record.perturbed_pred_index == 0
    assert record.to_dict()["verified_failure"] is False


def test_quick_empty_predictions_and_run_warning(tmp_path):
    ds = make_dataset(tmp_path, [None], labels_exist=False)
    detector = MockDetector([EMPTY, EMPTY])
    result = quick_mode(detector, ds, "gaussian_noise", {"sigma": 10.0})
    analysis = analyze_quick(result)
    assert analysis.records == []
    run = build_quick_run(result)
    assert any("no ground truth" in warning for warning in run.warnings)
    assert analysis.summary["num_images"] == 1


# ---------------------------------------------------------------------------
# Benchmark runs and comparisons
# ---------------------------------------------------------------------------

def test_benchmark_run_records_and_schema(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    detector = MockDetector([
        det([GT_BOX], [0], [0.9]),
        det([GT_BOX], [1], [0.9]),
    ])
    result = evaluate_perturbed(detector, ds, "brightness", {"factor": 0.5})
    trial = analyze_perturbed(result)
    assert trial.transformation["name"] == "brightness"
    assert trial.deltas["f1"] == pytest.approx(-1.0)
    assert trial.failure_counts["class_confusion"] == 1

    run = build_benchmark_run(result.baseline, [result])
    assert run.schema_version == SCHEMA_VERSION
    payload = run.to_dict()
    assert payload["mode"] == "benchmark"
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["baseline"]["transformation"] is None
    assert payload["trials"][0]["transformation"]["name"] == "brightness"
    assert "environment" in payload["metadata"]
    assert payload["metadata"]["thresholds"]["iou_threshold"] == 0.5


def test_compare_reports_handles_missing_and_incompatible_metrics(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    report = evaluate_baseline(MockDetector([det([GT_BOX], [0], [0.9])]), ds)
    missing = dataclasses.replace(
        report,
        aggregate={k: v for k, v in report.aggregate.items() if k != "precision"},
    )
    comparison = compare_reports(report, missing)
    assert comparison.deltas["precision"] is None
    assert any("precision" in warning for warning in comparison.warnings)

    incompatible = dataclasses.replace(report, iou_threshold=0.3)
    comparison = compare_reports(report, incompatible)
    assert comparison.compatible is False
    assert any("incomparable" in warning for warning in comparison.warnings)


def test_environment_metadata_is_serializable():
    env = environment_metadata()
    json.dumps(env)
    assert env["failureforge"]
    assert env["numpy"]
