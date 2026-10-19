"""Baseline evaluation tests: mocked detector + synthetic YOLO dataset."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from failureforge.dataset import YoloDataset
from failureforge.detector import Detections
from failureforge.evaluator import evaluate_baseline, evaluate_samples, metric_drop
from failureforge.metrics import iou_matrix, match_boxes


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

class MockDetector:
    """Satisfies the YOLODetector interface with scripted predictions."""

    def __init__(self, detections, class_names=("cat", "dog"), conf=0.25):
        self._detections = list(detections)
        self._i = 0
        self.class_names = list(class_names)
        self.conf = conf
        self.weights = None
        self.sha256 = None
        self.task = "detect"
        self.imgsz = 640
        self.device = "cpu"

    def predict(self, image_bgr):
        if self._i >= len(self._detections):
            raise AssertionError("MockDetector called more times than scripted")
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


def _make_dataset(tmp_path: Path, labels: list[str], names=("cat", "dog"), size=(100, 100)):
    """labels: one label-file body per image."""
    root = tmp_path / "ds"
    imgs, lbls = root / "images" / "test", root / "labels" / "test"
    imgs.mkdir(parents=True)
    lbls.mkdir(parents=True)
    for i, body in enumerate(labels):
        Image.new("RGB", size, (128, 128, 128)).save(imgs / f"img{i}.png")
        (lbls / f"img{i}.txt").write_text(body, encoding="utf-8")
    return YoloDataset.from_dirs(imgs, lbls, list(names), split="test")


# normalized 0.5,0.5,0.2,0.2 on 100x100 -> [40,40,60,60]
BOX = [40.0, 40.0, 60.0, 60.0]
GT_CAT = "0 0.5 0.5 0.2 0.2\n"


# ---------------------------------------------------------------------------
# matching primitives
# ---------------------------------------------------------------------------

def test_iou_matrix_hand_values():
    a = np.array([[0, 0, 100, 100]], dtype=float)
    b = np.array([[0, 0, 100, 50]], dtype=float)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.5)


def test_match_is_deterministic_confidence_order():
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    gt_c = np.array([0])
    # Two overlapping preds; the higher-confidence one must win.
    preds = np.array([[0, 0, 10, 10], [0, 0, 9, 9]], dtype=float)
    res = match_boxes(gt, gt_c, preds, np.array([0, 0]), np.array([0.4, 0.9]))
    assert res.tp == 1 and res.fp == 1
    assert res.tp_idx == [1]


def test_iou_threshold_boundary_matches_exactly_at_half():
    gt = np.array([[0, 0, 100, 100]], dtype=float)
    preds = np.array([[0, 0, 100, 50]], dtype=float)
    res = match_boxes(gt, np.array([0]), preds, np.array([0]), np.array([0.9]), 0.5)
    assert res.tp == 1 and res.fp == 0 and res.fn == 0


def test_no_double_counting_duplicates():
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    preds = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    res = match_boxes(gt, np.array([0]), preds, np.array([0, 0]), np.array([0.9, 0.8]))
    assert (res.tp, res.fp, res.fn) == (1, 1, 0)


# ---------------------------------------------------------------------------
# baseline evaluation
# ---------------------------------------------------------------------------

def test_perfect_predictions(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT, GT_CAT])
    report = evaluate_baseline(MockDetector([det([BOX], [0], [0.9])] * 2), ds)
    assert (report.aggregate["tp"], report.aggregate["fp"], report.aggregate["fn"]) == (2, 0, 0)
    assert report.aggregate["f1"] == 1.0
    assert report.map50 == pytest.approx(1.0)
    assert report.per_class[0].f1 == 1.0
    assert report.error_records == []


def test_all_missed(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT, GT_CAT])
    report = evaluate_baseline(MockDetector([EMPTY, EMPTY]), ds)
    assert (report.aggregate["tp"], report.aggregate["fp"], report.aggregate["fn"]) == (0, 0, 2)
    assert report.aggregate["recall"] == 0.0
    assert report.map50 == 0.0
    assert sum(1 for r in report.error_records if r["type"] == "missed") == 2


def test_false_positive(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT])
    extra = det([[0, 0, 20, 20]], [0], [0.9])  # far from GT
    report = evaluate_baseline(MockDetector([extra]), ds)
    assert (report.aggregate["tp"], report.aggregate["fp"], report.aggregate["fn"]) == (0, 1, 1)
    assert report.aggregate["precision"] == 0.0


def test_duplicate_detections_count_as_fp(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT])
    preds = det([BOX, BOX], [0, 0], [0.9, 0.8])
    report = evaluate_baseline(MockDetector([preds]), ds)
    assert (report.aggregate["tp"], report.aggregate["fp"], report.aggregate["fn"]) == (1, 1, 0)


def test_wrong_class_prediction_not_a_tp(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT])
    preds = det([BOX], [1], [0.9])  # dog where cat was labeled
    report = evaluate_baseline(MockDetector([preds]), ds)
    assert (report.aggregate["tp"], report.aggregate["fp"], report.aggregate["fn"]) == (0, 1, 1)
    assert report.per_image[0].wrong_class == 1
    assert any(r["type"] == "wrong_class" for r in report.error_records)


def test_poor_localization_diagnostic(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT])
    shifted = [[42.0, 42.0, 62.0, 62.0]]  # IoU ~0.68 with BOX
    report = evaluate_baseline(MockDetector([det(shifted, [0], [0.9])]), ds)
    assert report.aggregate["tp"] == 1  # matched at IoU >= 0.5
    assert report.per_image[0].poor_localization == 1
    assert any(r["type"] == "poor_localization" for r in report.error_records)


def test_multiple_objects_in_one_image(tmp_path):
    body = "0 0.2 0.2 0.1 0.1\n1 0.8 0.8 0.1 0.1\n"
    ds = _make_dataset(tmp_path, [body])
    preds = det([[15, 15, 25, 25], [75, 75, 85, 85]], [0, 1], [0.9, 0.8])
    report = evaluate_baseline(MockDetector([preds]), ds)
    assert (report.aggregate["tp"], report.aggregate["fp"], report.aggregate["fn"]) == (2, 0, 0)
    assert report.per_class[0].tp == 1 and report.per_class[1].tp == 1


def test_image_with_no_ground_truth(tmp_path):
    ds = _make_dataset(tmp_path, [""])
    report = evaluate_baseline(MockDetector([det([BOX], [0], [0.9])]), ds)
    assert (report.aggregate["tp"], report.aggregate["fp"], report.aggregate["fn"]) == (0, 1, 0)
    assert report.map50 == 0.0


def test_image_with_no_predictions(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT])
    report = evaluate_baseline(MockDetector([EMPTY]), ds)
    assert (report.aggregate["tp"], report.aggregate["fp"], report.aggregate["fn"]) == (0, 0, 1)
    assert report.aggregate["precision"] == 0.0


def test_precision_recall_f1_values(tmp_path):
    # two images: one TP, one FP+FN -> micro P=R=F1=0.5
    ds = _make_dataset(tmp_path, [GT_CAT, GT_CAT])
    report = evaluate_baseline(MockDetector([
        det([BOX], [0], [0.9]),
        det([[0, 0, 10, 10]], [0], [0.9]),
    ]), ds)
    assert report.aggregate["tp"] == 1
    assert report.aggregate["fp"] == 1
    assert report.aggregate["fn"] == 1
    assert report.aggregate["precision"] == pytest.approx(0.5)
    assert report.aggregate["recall"] == pytest.approx(0.5)
    assert report.aggregate["f1"] == pytest.approx(0.5)


def test_zero_denominators(tmp_path):
    ds = _make_dataset(tmp_path, [""])
    report = evaluate_baseline(MockDetector([EMPTY]), ds)
    assert report.aggregate["precision"] == 0.0
    assert report.aggregate["recall"] == 0.0
    assert report.aggregate["f1"] == 0.0


def test_map50_multi_class_and_ignores_absent_classes(tmp_path):
    body = "0 0.2 0.2 0.1 0.1\n1 0.8 0.8 0.1 0.1\n"
    ds = _make_dataset(tmp_path, [body])
    preds = det([[15, 15, 25, 25]], [0], [0.9])  # only cat detected
    report = evaluate_baseline(MockDetector([preds]), ds)
    # cat AP=1, dog AP=0 -> mean 0.5
    assert report.map50 == pytest.approx(0.5)
    assert report.per_class[1].ap50 == pytest.approx(0.0)


def test_class_mapping_reordered_model_names(tmp_path):
    # Dataset [cat, dog]; model reports [dog, cat] -> model class 1 is cat.
    ds = _make_dataset(tmp_path, [GT_CAT])
    dets = det([BOX], [1], [0.9])  # model-space class 1 == cat
    report = evaluate_baseline(
        MockDetector([dets], class_names=["dog", "cat"]), ds)
    assert report.aggregate["tp"] == 1
    assert report.metadata["class_mapping"] == {"cat": 1, "dog": 0}


def test_class_mismatch_stops_evaluation(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT])
    with pytest.raises(Exception, match="not present in model"):
        evaluate_baseline(MockDetector([EMPTY], class_names=["cat", "bird"]), ds)


def test_detector_without_class_names_rejected(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT])
    detector = MockDetector([EMPTY])
    detector.class_names = []
    with pytest.raises(ValueError, match="class_names"):
        evaluate_baseline(detector, ds)


def test_report_metadata_and_thresholds(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT])
    report = evaluate_baseline(MockDetector([det([BOX], [0], [0.9])]), ds, iou_threshold=0.6)
    assert report.iou_threshold == 0.6
    assert report.split == "test"
    assert report.conf_threshold == 0.25
    assert report.metadata["dataset"]["num_images"] == 1
    assert "micro-average" in report.aggregation_method
    d = report.to_dict()
    assert "aggregate" in d and "per_class" in d and "per_image" in d


def test_legacy_evaluate_samples_still_works(tmp_path):
    ds = _make_dataset(tmp_path, [GT_CAT])
    samples = list(ds.iter_samples())

    class LegacyStub:
        def predict(self, image):
            return [{"class_id": 0, "x_center": 0.5, "y_center": 0.5,
                     "width": 0.2, "height": 0.2, "confidence": 0.9}]

    res = evaluate_samples(samples, LegacyStub(), 0.5)
    assert res["aggregate"]["f1"] == 1.0
    assert res["map50"] == pytest.approx(1.0)
    assert res["aggregate"]["tp"] == 1
    assert metric_drop(res, res, "f1") == 0.0
