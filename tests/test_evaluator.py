"""Evaluator + dataset validation tests with synthetic fixtures."""
from pathlib import Path

import pytest
from PIL import Image

from failureforge.dataset import DatasetError, load_dataset, split_dataset, summarize
from failureforge.detector import DummyDetector
from failureforge.evaluator import evaluate_samples, metric_drop


def _make_ds(tmp_path: Path, n=3):
    imgdir = tmp_path / "images"
    lbldir = tmp_path / "labels"
    imgdir.mkdir()
    lbldir.mkdir()
    for i in range(n):
        Image.new("RGB", (64, 64), (200, 200, 200)).save(imgdir / f"img{i}.png")
        (lbldir / f"img{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    return imgdir, lbldir


def test_load_and_summarize(tmp_path):
    imgdir, lbldir = _make_ds(tmp_path)
    samples = load_dataset(imgdir, lbldir)
    assert len(samples) == 3
    s = summarize(samples)
    assert s.num_boxes == 3 and s.per_class == {0: 3}


def test_missing_images_dir(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "nope", tmp_path)


def test_malformed_row_raises(tmp_path):
    imgdir, lbldir = _make_ds(tmp_path, n=1)
    (lbldir / "img0.txt").write_text("0 0.5 999 0.1 0.1\n")
    with pytest.raises(DatasetError):
        load_dataset(imgdir, lbldir)


def test_evaluate_baseline_perfect(tmp_path):
    imgdir, lbldir = _make_ds(tmp_path, n=2)
    samples = load_dataset(imgdir, lbldir)
    det = DummyDetector([{"class_id": 0, "x_center": 0.5, "y_center": 0.5,
                          "width": 0.4, "height": 0.4, "confidence": 0.99}])
    res = evaluate_samples(samples, det, 0.5)
    assert res["aggregate"]["f1"] == 1.0
    assert res["map50"] == 1.0


def test_evaluate_perturbed_gt_used(tmp_path):
    imgdir, lbldir = _make_ds(tmp_path, n=1)
    samples = load_dataset(imgdir, lbldir)
    det = DummyDetector([])
    base = evaluate_samples(samples, det, 0.5)
    pert = evaluate_samples(samples, det, 0.5,
                            perturbation={"name": "brightness", "params": {"factor": 0.5}, "seed": 0})
    assert base["aggregate"]["fn"] == 1
    assert metric_drop(base, pert, "f1") == 0.0  # both score 0 -> no drop


def test_split_deterministic(tmp_path):
    imgdir, lbldir = _make_ds(tmp_path, n=4)
    samples = load_dataset(imgdir, lbldir)
    a1, b1 = split_dataset(samples, 0.5, seed=7)
    a2, b2 = split_dataset(samples, 0.5, seed=7)
    assert [s.image_path.name for s in a1] == [s.image_path.name for s in a2]
    assert len(a1) == 2 and len(b1) == 2
