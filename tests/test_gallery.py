"""Failure-gallery rendering, artifact handling, and escaping tests."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from failureforge.dataset import YoloDataset
from failureforge.detector import Detections
from failureforge.evaluation import (
    PerturbedResult,
    QuickModeResult,
    quick_mode,
)
from failureforge.evaluator import evaluate_baseline
from failureforge.failures import (
    BenchmarkRun,
    TrialRecord,
    VerifiedFailure,
    build_benchmark_run,
    build_quick_run,
)
from failureforge.gallery import (
    GALLERY_SCHEMA_VERSION,
    GalleryImage,
    render_gallery,
    write_gallery,
)

GT_BOX = [80.0, 40.0, 120.0, 60.0]


class SequenceDetector:
    def __init__(self, detections, class_names=("cat", "dog")):
        self.detections = list(detections)
        self.index = 0
        self.class_names = list(class_names)

    def predict(self, image_bgr):
        value = self.detections[self.index]
        self.index += 1
        return value


def det(boxes, classes, scores):
    return Detections(
        boxes_xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.array(scores, dtype=np.float32),
        classes=np.array(classes, dtype=np.int64),
    )


EMPTY = det([], [], [])


def make_dataset(tmp_path: Path) -> YoloDataset:
    root = tmp_path / "data"
    images = root / "images" / "test"
    labels = root / "labels" / "test"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (200, 100), (80, 90, 100)).save(images / "img0.png")
    (labels / "img0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return YoloDataset.from_dirs(images, labels, ["cat", "dog"], split="test")


def make_benchmark_run(tmp_path: Path):
    dataset = make_dataset(tmp_path)
    baseline = evaluate_baseline(
        SequenceDetector([det([GT_BOX], [1], [0.9])]), dataset
    )
    perturbed = evaluate_baseline(SequenceDetector([EMPTY]), dataset)
    result = PerturbedResult(
        baseline=baseline,
        perturbed=perturbed,
        perturbation={
            "name": "brightness",
            "family": "photometric",
            "params": {"factor": 1.0},
            "seed": 7,
        },
        deltas={},
    )
    return build_benchmark_run(baseline, [result]), dataset


def manual_run(failures, *, image="x.png", warning=None) -> BenchmarkRun:
    baseline = TrialRecord(
        transformation=None, metrics={}, deltas={}, drops={}, per_sample=[],
        failures=[], failure_counts={},
    )
    trial = TrialRecord(
        transformation={
            "name": "fog", "family": "weather", "params": {"intensity": 0.4},
            "seed": 3,
        },
        metrics={}, deltas={"f1": 0.2}, drops={"f1": 0.2}, per_sample=[],
        failures=list(failures), failure_counts={},
    )
    return BenchmarkRun(
        metadata={
            "dataset": {"split": "test", "num_images": 1, "class_names": ["cat"]},
            "model": {"weights": "C:\\models\\best.pt"},
        },
        baseline=baseline, trials=[trial],
        warnings=[warning] if warning else [],
    )


def manifest(html_text: str) -> dict:
    match = re.search(
        r'<script id="gallery-data" type="application/json">(.*?)</script>',
        html_text, re.S,
    )
    assert match, "gallery manifest missing"
    return json.loads(match.group(1))


def failure(category="false_negative", *, image="img0.png", **kwargs):
    values = dict(
        category=category, reason="synthetic failure", image=image,
        counts_as="fn", gt_class_id=0, gt_class_name="cat",
        gt_box=GT_BOX, gt_index=0, iou=0.0,
    )
    values.update(kwargs)
    return VerifiedFailure(**values)


# ---------------------------------------------------------------------------
# Benchmark gallery
# ---------------------------------------------------------------------------

def test_benchmark_gallery_has_baseline_and_trial_entries(tmp_path):
    run, _dataset = make_benchmark_run(tmp_path)
    html_text = render_gallery(run)
    payload = manifest(html_text)

    assert payload["schema_version"] == GALLERY_SCHEMA_VERSION
    assert payload["mode"] == "benchmark"
    assert payload["verified_failures"] is True
    assert "FailureForge Failure Gallery" in html_text
    assert "Baseline (unperturbed)" in html_text
    assert "Trial 1" in html_text and "brightness" in html_text
    assert "factor" in html_text and "seed 7" in html_text
    assert "Class confusion" in html_text and "False negative" in html_text
    assert 'data-context="baseline"' in html_text
    assert 'data-context="perturbed"' in html_text
    assert "not a verified failure" not in html_text.lower()


def test_benchmark_gallery_copies_relative_images_and_overlays_boxes(tmp_path):
    run, _dataset = make_benchmark_run(tmp_path)
    clean = tmp_path / "clean.png"
    perturbed = tmp_path / "perturbed.png"
    Image.new("RGB", (200, 100), (10, 20, 30)).save(clean)
    Image.new("RGB", (200, 100), (40, 50, 60)).save(perturbed)

    output = tmp_path / "gallery"
    path = write_gallery(
        run, output,
        clean_images={"img0.png": clean},
        perturbed_images={"img0.png": perturbed},
    )
    html_text = path.read_text(encoding="utf-8")

    assert str(tmp_path) not in html_text
    assert 'src="images/' in html_text
    assert 'viewBox="0 0 200 100"' in html_text
    assert 'class="bbox bbox-gt"' in html_text
    assert 'class="bbox bbox-pred"' in html_text
    assert 'x="80.000"' in html_text and 'y="40.000"' in html_text
    assert 'width="40.000"' in html_text and 'height="20.000"' in html_text
    assert "Boxes are drawn over the supplied artifact." in html_text

    images = sorted((output / "images").iterdir())
    assert len(images) == 2
    for image_path in images:
        with Image.open(image_path) as image:
            assert image.size == (200, 100)
        assert image_path.name[:16].isalnum()
    assert manifest(html_text)["schema_version"] == GALLERY_SCHEMA_VERSION


def test_missing_images_fall_back_to_diagrams(tmp_path):
    run, _dataset = make_benchmark_run(tmp_path)
    path = write_gallery(run, tmp_path / "gallery")
    html_text = path.read_text(encoding="utf-8")
    assert "<img" not in html_text
    assert "Box diagram only" in html_text
    assert not (path.parent / "images").exists()


def test_missing_box_details_are_graceful(tmp_path):
    run = manual_run([failure(gt_box=None, gt_class_id=None, gt_class_name=None)])
    html_text = render_gallery(run)
    assert "No image artifact or box coordinates available" in html_text
    assert 'class="bbox' not in html_text


def test_gallery_escapes_user_controlled_content(tmp_path):
    payload_text = "</script><script>alert(1)</script>"
    run = manual_run(
        [failure(
            image=payload_text + ".png",
            gt_class_name=payload_text,
            reason="<b>bad</b>",
        )],
        warning="</script><script>alert(2)</script>",
    )
    html_text = render_gallery(run, title="<Title>")
    assert "<Title>" not in html_text
    assert "&lt;Title&gt;" in html_text
    assert "<b>bad</b>" not in html_text
    assert "&lt;b&gt;bad&lt;/b&gt;" in html_text
    assert payload_text not in html_text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_text
    # The JSON manifest must not terminate its script element.
    assert "</script><script>alert" not in html_text
    assert json.loads(re.search(
        r'<script id="gallery-data" type="application/json">(.*?)</script>',
        html_text, re.S,
    ).group(1))["warnings"]


def test_gallery_is_deterministic_and_protects_existing_files(tmp_path):
    run, _dataset = make_benchmark_run(tmp_path)
    output = tmp_path / "gallery"
    first = write_gallery(run, output).read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_gallery(run, output)
    second = write_gallery(run, output, overwrite=True).read_text(encoding="utf-8")
    assert first == second
    with pytest.raises(ValueError):
        write_gallery(run, output, basename="..\\escape", overwrite=True)


def test_max_entries_reports_truncation(tmp_path):
    run = manual_run([failure(), failure(image="other.png")])
    html_text = render_gallery(run, max_entries=1)
    assert "Showing <strong>1</strong> of <strong>2</strong>" in html_text
    with pytest.raises(ValueError):
        render_gallery(run, max_entries=0)


def test_render_gallery_accepts_prepared_relative_images():
    run = manual_run([failure()])
    html_text = render_gallery(
        run,
        perturbed_images={"img0.png": GalleryImage("images/abc.png", 200, 100)},
    )
    assert 'src="images/abc.png"' in html_text
    assert 'viewBox="0 0 200 100"' in html_text


# ---------------------------------------------------------------------------
# Quick Mode gallery
# ---------------------------------------------------------------------------

def test_quick_gallery_is_explicitly_unverified_and_keeps_boxes(tmp_path):
    dataset = make_dataset(tmp_path)
    detector = SequenceDetector([
        det([GT_BOX], [0], [0.9]),
        det([GT_BOX], [1], [0.8]),
    ])
    result = quick_mode(
        detector, dataset, "brightness", {"factor": 1.0}, seed=3
    )
    run = build_quick_run(result)
    record = run.analysis.records[0]
    assert record.kind == "class_changed"
    assert record.clean_box == GT_BOX and record.perturbed_box == GT_BOX

    html_text = render_gallery(run)
    payload = manifest(html_text)
    assert payload["mode"] == "quick"
    assert payload["verified_failures"] is False
    assert "Quick Mode" in html_text
    assert "Not a verified failure" in html_text
    assert "Prediction instability" in html_text
    assert "Clean predictions" in html_text
    assert "Perturbed predictions" in html_text
    assert "Ranked perturbations" not in html_text
    assert "bbox-clean" in html_text and "bbox-perturbed" in html_text


def test_quick_gallery_handles_missing_boxes(tmp_path):
    result = QuickModeResult(
        perturbation={
            "name": "brightness", "family": "photometric",
            "params": {"factor": 0.5}, "seed": 1,
        },
        iou_threshold=0.5,
        per_image=[{
            "image": "old.png", "num_clean": 1, "num_perturbed": 1,
            "matches": 1, "disappeared": [], "appeared": [],
            "class_changed": [{
                "clean_index": 0, "perturbed_index": 0,
                "from_class": 0, "to_class": 1, "iou": 0.9,
            }],
            "confidence_changes": [], "box_drift": [],
        }],
        summary={"num_images": 1, "clean_detections": 1},
        metadata={"model": {"class_names": ["cat", "dog"]}},
    )
    run = build_quick_run(result)
    assert run.analysis.records[0].clean_box is None
    html_text = render_gallery(run)
    assert "No image artifact or box coordinates available" in html_text
    assert "Not a verified failure" in html_text


def test_quick_does_not_add_accuracy_claims(tmp_path):
    dataset = make_dataset(tmp_path)
    detector = SequenceDetector([
        det([GT_BOX], [0], [0.9]),
        EMPTY,
    ])
    result = quick_mode(detector, dataset, "brightness", {"factor": 1.0})
    run = build_quick_run(result)
    html_text = render_gallery(run).lower()
    for forbidden in ("precision", "recall", "map50", "accuracy:"):
        assert forbidden not in html_text

