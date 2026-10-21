"""Perturbation-evaluation adapter tests (Benchmark Mode, Quick Mode, sweeps)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from failureforge.dataset import YoloDataset
from failureforge.detector import Detections
from failureforge.evaluation import (
    compare_predictions,
    derive_trial_seed,
    evaluate_perturbed,
    greedy_location_matches,
    iter_perturbed_samples,
    load_sample_bgr,
    prepare_perturbed_sample,
    quick_mode,
    run_severity_sweep,
)
from failureforge.evaluator import evaluate_baseline
from failureforge.perturbations import PerturbationError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class MockDetector:
    """Scripted YOLODetector-compatible detector (CPU only, no weights)."""

    def __init__(self, detections, class_names=("cat", "dog")):
        self._detections = list(detections)
        self._i = 0
        self.calls = 0
        self.class_names = list(class_names)
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
        self.calls += 1
        return out


def det(boxes, classes, scores):
    return Detections(
        boxes_xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.array(scores, dtype=np.float32),
        classes=np.array(classes, dtype=np.int64),
    )


EMPTY = det([], [], [])
BOX = [40.0, 40.0, 60.0, 60.0]
GT_CAT = "0 0.5 0.5 0.2 0.2\n"
GT_OFFCENTER_CAT = "0 0.2 0.5 0.2 0.2\n"


def make_dataset(tmp_path: Path, labels, names=("cat", "dog"), size=(100, 100),
                 labels_exist=True):
    root = tmp_path / "ds"
    imgs, lbls = root / "images" / "test", root / "labels" / "test"
    imgs.mkdir(parents=True)
    lbls.mkdir(parents=True)
    for i, body in enumerate(labels):
        Image.new("RGB", size, (128, 128, 128)).save(imgs / f"img{i}.png")
        if labels_exist and body is not None:
            (lbls / f"img{i}.txt").write_text(body, encoding="utf-8")
    return YoloDataset.from_dirs(
        imgs, lbls, list(names), split="test",
        allow_missing_labels=not labels_exist,
    )


# ---------------------------------------------------------------------------
# loading / preparation
# ---------------------------------------------------------------------------

def test_load_sample_bgr_order(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    Image.new("RGB", (10, 10), (10, 20, 30)).save(ds.image_paths[0])
    bgr = load_sample_bgr(ds.iter_samples().__next__())
    assert bgr.dtype == np.uint8 and bgr.shape == (10, 10, 3)
    assert bgr[0, 0].tolist() == [30, 20, 10]


def test_prepare_perturbed_sample_does_not_mutate_input(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    sample = ds.iter_samples().__next__()
    before_boxes = list(sample.boxes)
    before_width, before_height = sample.width, sample.height

    transformed, bgr, meta = prepare_perturbed_sample(
        sample, "rotation", {"angle": 90.0}, seed=0
    )

    assert list(sample.boxes) == before_boxes
    assert (sample.width, sample.height) == (before_width, before_height)
    assert transformed is not sample
    assert bgr.shape == (transformed.height, transformed.width, 3)
    assert meta["name"] == "rotation"
    assert transformed.boxes != before_boxes  # off-center box moved


def test_prepare_perturbed_sample_crop_updates_dims_and_boxes(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    sample = ds.iter_samples().__next__()

    transformed, bgr, _ = prepare_perturbed_sample(
        sample, "random_crop", {"crop_fraction": 0.5}, seed=0
    )

    assert bgr.shape[:2] == (transformed.height, transformed.width)
    assert transformed.width < sample.width
    assert transformed.height < sample.height
    assert sample.width == 100 and sample.height == 100
    for box in transformed.boxes:
        assert 0 <= box.x1 <= box.x2 <= transformed.width
        assert 0 <= box.y1 <= box.y2 <= transformed.height


def test_iter_perturbed_samples_yields_all_samples(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT, GT_CAT])
    pairs = list(iter_perturbed_samples(ds, "brightness", {"factor": 1.0}))
    assert len(pairs) == 2
    assert all(bgr.shape == (100, 100, 3) for _, bgr in pairs)


# ---------------------------------------------------------------------------
# Benchmark Mode
# ---------------------------------------------------------------------------

def test_benchmark_noop_has_zero_deltas(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT, GT_CAT])
    d = det([BOX], [0], [0.9])
    detector = MockDetector([d, d, d, d])

    result = evaluate_perturbed(detector, ds, "brightness", {"factor": 1.0})

    assert result.perturbed.aggregate["f1"] == 1.0
    assert all(v == pytest.approx(0.0) for v in result.deltas.values())
    assert result.perturbation["name"] == "brightness"
    assert result.perturbation["params"] == {"factor": 1.0}
    assert result.metadata["sample_count"] == 2
    assert result.metadata["dataset"]["split"] == "test"
    payload = result.to_dict()
    assert payload["mode"] == "benchmark"
    assert "baseline" in payload and "perturbed" in payload


def test_benchmark_reports_metric_drop(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    detector = MockDetector([det([BOX], [0], [0.9]), EMPTY])

    result = evaluate_perturbed(detector, ds, "gaussian_noise", {"sigma": 30.0})

    assert result.baseline.aggregate["f1"] == 1.0
    assert result.perturbed.aggregate["f1"] == 0.0
    assert result.deltas["f1"] == pytest.approx(-1.0)
    assert result.deltas["recall"] == pytest.approx(-1.0)
    assert result.deltas["map50"] == pytest.approx(-1.0)


def test_benchmark_uses_transformed_ground_truth(tmp_path):
    ds = make_dataset(tmp_path, [GT_OFFCENTER_CAT])
    original = det([[10.0, 40.0, 30.0, 60.0]], [0], [0.9])
    detector = MockDetector([original, original])

    result = evaluate_perturbed(detector, ds, "rotation", {"angle": 90.0})

    assert result.baseline.aggregate["tp"] == 1
    assert result.perturbed.aggregate["tp"] == 0
    per_image = result.perturbed.per_image[0]
    assert (per_image.tp, per_image.fp, per_image.fn) == (0, 1, 1)


def test_benchmark_reuses_supplied_baseline(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    d = det([BOX], [0], [0.9])
    detector = MockDetector([d, d, d])

    baseline = evaluate_baseline(detector, ds)
    result = evaluate_perturbed(
        detector, ds, "brightness", {"factor": 1.0}, baseline=baseline
    )

    assert detector.calls == 2  # baseline computed once, not twice
    assert result.baseline is baseline


# ---------------------------------------------------------------------------
# Quick Mode
# ---------------------------------------------------------------------------

def test_quick_mode_detects_class_change_without_labels(tmp_path):
    ds = make_dataset(tmp_path, [None], labels_exist=False)
    detector = MockDetector([det([BOX], [0], [0.9]), det([BOX], [1], [0.9])])

    result = quick_mode(detector, ds, "brightness", {"factor": 1.0})

    summary = result.summary
    assert summary["class_changed"] == 1
    assert summary["disappeared"] == 0
    assert summary["appeared"] == 0
    assert summary["num_images"] == 1
    assert not any(k in summary for k in ("precision", "recall", "f1", "map50"))
    payload = result.to_dict()
    assert payload["mode"] == "quick"
    assert not any(k in payload for k in ("precision", "recall", "f1", "map50"))
    assert "Instability only" in payload["note"]


def test_quick_mode_detects_disappearance_and_appearance(tmp_path):
    ds = make_dataset(tmp_path, [None], labels_exist=False)
    detector = MockDetector([det([BOX], [0], [0.9]), det([[0, 0, 10, 10]], [1], [0.5])])

    result = quick_mode(detector, ds, "gaussian_noise", {"sigma": 20.0})

    assert result.summary["disappeared"] == 1
    assert result.summary["appeared"] == 1
    assert result.summary["matched"] == 0
    assert result.per_image[0]["disappeared"][0]["score"] == pytest.approx(0.9)


def test_quick_mode_does_not_read_labels(tmp_path):
    ds = make_dataset(tmp_path, [None, None], labels_exist=False)
    detector = MockDetector(
        [det([BOX], [0], [0.9]), det([BOX], [0], [0.8]),
         det([BOX], [0], [0.9]), det([BOX], [0], [0.8])]
    )
    result = quick_mode(detector, ds, "brightness", {"factor": 1.0})
    assert result.summary["num_images"] == 2
    assert result.summary["confidence_changed"] == 2


def test_greedy_matching_is_class_agnostic_and_one_to_one():
    clean = det([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 0], [0.9, 0.8])
    pert = det([[0, 0, 10, 10]], [1], [0.7])
    matches = greedy_location_matches(clean, pert, 0.5)
    assert matches == [(0, 0, pytest.approx(1.0))]


def test_compare_predictions_empty_inputs():
    cmp = compare_predictions(EMPTY, EMPTY)
    assert cmp["matches"] == 0
    assert cmp["disappeared"] == [] and cmp["appeared"] == []


# ---------------------------------------------------------------------------
# Severity sweeps
# ---------------------------------------------------------------------------

def test_derive_trial_seed_is_deterministic_and_sensitive():
    a = derive_trial_seed(7, 0, "brightness", {"factor": 0.5})
    assert a == derive_trial_seed(7, 0, "brightness", {"factor": 0.5})
    assert a != derive_trial_seed(7, 1, "brightness", {"factor": 0.5})
    assert a != derive_trial_seed(7, 0, "brightness", {"factor": 0.6})
    assert a != derive_trial_seed(8, 0, "brightness", {"factor": 0.5})
    assert 0 <= a < 2**32


def test_sweep_is_reproducible_and_baseline_separate(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT, GT_CAT])
    configs = [{"factor": 0.8}, {"factor": 1.2}]

    def run():
        d = det([BOX], [0], [0.9])
        detector = MockDetector([d] * 6)
        return run_severity_sweep(detector, ds, "brightness", configs, seed=11)

    first, second = run(), run()
    assert [t.config["seed"] for t in first.trials] == [
        t.config["seed"] for t in second.trials
    ]
    assert [t.result.deltas for t in first.trials] == [
        t.result.deltas for t in second.trials
    ]
    assert first.baseline is not None
    payload = first.to_dict()
    assert "baseline" in payload
    assert all("baseline" not in t["result"] for t in payload["trials"])


def test_sweep_records_attributable_error(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    d = det([BOX], [0], [0.9])
    detector = MockDetector([d, d])

    sweep = run_severity_sweep(
        detector, ds, "brightness", [{"factor": 1.0}, {"factor": 999.0}]
    )

    ok, bad = sweep.trials
    assert ok.error is None and ok.result is not None
    assert bad.result is None
    assert "factor" in bad.error and "PerturbationError" in bad.error
    assert bad.index == 1


def test_sweep_strict_mode_reraises(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    detector = MockDetector([det([BOX], [0], [0.9]), det([BOX], [0], [0.9])])
    with pytest.raises(PerturbationError):
        run_severity_sweep(
            detector, ds, "brightness", [{"factor": 999.0}], strict=True
        )


def test_sweep_quick_mode_has_no_baseline(tmp_path):
    ds = make_dataset(tmp_path, [None], labels_exist=False)
    detector = MockDetector([det([BOX], [0], [0.9]), det([BOX], [0], [0.9])])

    sweep = run_severity_sweep(
        detector, ds, "brightness", [{"factor": 1.0}], mode="quick"
    )

    assert sweep.baseline is None
    assert sweep.trials[0].result.summary["num_images"] == 1
    assert sweep.to_dict()["baseline"] is None


def test_sweep_rejects_unknown_mode(tmp_path):
    ds = make_dataset(tmp_path, [GT_CAT])
    with pytest.raises(ValueError):
        run_severity_sweep(MockDetector([]), ds, "brightness", [], mode="fast")
