"""Day 2 detector tests. Ultralytics is fully mocked; no weights, no network."""
import hashlib
import sys
import types

import numpy as np
import pytest

from failureforge.detector import Detections, DetectorError, YOLODetector


# --- lightweight fakes -------------------------------------------------------

class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)

    def __len__(self):
        return len(self.xyxy.numpy())


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    """Stand-in for ultralytics.YOLO. Behaviour set via class attributes."""

    task = "detect"
    names = {0: "cat", 2: "dog"}
    boxes_to_return = "two"  # 'two' | 'empty' | 'none' | 'raise'
    last_kwargs: dict | None = None

    def __init__(self, weights):
        self.weights = weights

    def predict(self, image, **kwargs):
        type(self).last_kwargs = kwargs
        mode = type(self).boxes_to_return
        if mode == "raise":
            raise RuntimeError("boom")
        if mode == "none":
            return [_FakeResult(None)]
        if mode == "empty":
            return [_FakeResult(_FakeBoxes(
                np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))))]
        return [_FakeResult(_FakeBoxes(
            [[10, 20, 30, 40], [0, 0, 5, 5]], [0.9, 0.4], [2, 0]))]


@pytest.fixture
def fake_ultra(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    _FakeYOLO.task = "detect"
    _FakeYOLO.names = {0: "cat", 2: "dog"}
    _FakeYOLO.boxes_to_return = "two"
    _FakeYOLO.last_kwargs = None
    return mod


@pytest.fixture
def weights_file(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"fake-weights-bytes")
    return p


def _bgr(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


# --- tests -------------------------------------------------------------------

def test_valid_prediction_normalization(fake_ultra, weights_file):
    det = YOLODetector(weights_file, conf=0.25, imgsz=640, device="cpu")
    out = det.predict(_bgr())
    assert isinstance(out, Detections)
    assert out.boxes_xyxy.tolist() == [[10, 20, 30, 40], [0, 0, 5, 5]]
    assert out.scores.tolist() == pytest.approx([0.9, 0.4])
    assert out.classes.tolist() == [2, 0]
    assert len(out) == 2 and not out.is_empty


def test_output_shapes_and_dtypes(fake_ultra, weights_file):
    out = YOLODetector(weights_file).predict(_bgr(32, 48))
    assert out.boxes_xyxy.shape == (2, 4) and out.boxes_xyxy.dtype == np.float32
    assert out.scores.shape == (2,) and out.scores.dtype == np.float32
    assert out.classes.shape == (2,) and out.classes.dtype == np.int64


def test_empty_detections(fake_ultra, weights_file):
    _FakeYOLO.boxes_to_return = "empty"
    out = YOLODetector(weights_file).predict(_bgr())
    assert out.is_empty and len(out) == 0
    assert out.boxes_xyxy.shape == (0, 4)
    assert out.scores.shape == (0,) and out.classes.shape == (0,)


def test_none_boxes_is_empty(fake_ultra, weights_file):
    _FakeYOLO.boxes_to_return = "none"
    assert YOLODetector(weights_file).predict(_bgr()).is_empty


def test_class_names_task_metadata_sha256(fake_ultra, weights_file):
    det = YOLODetector(weights_file)
    assert det.class_names == ["cat", "dog"]
    assert det.task == "detect"
    md = det.metadata
    assert md["task"] == "detect" and md["conf"] == 0.25
    assert md["imgsz"] == 640 and md["device"] == "cpu"
    assert md["sha256"] == hashlib.sha256(b"fake-weights-bytes").hexdigest()


def test_predict_forwards_config(fake_ultra, weights_file):
    YOLODetector(weights_file, conf=0.5, imgsz=320, device="cpu").predict(_bgr())
    assert _FakeYOLO.last_kwargs == {
        "conf": 0.5, "imgsz": 320, "device": "cpu", "verbose": False}


def test_missing_weights(fake_ultra, tmp_path):
    with pytest.raises(FileNotFoundError):
        YOLODetector(tmp_path / "nope.pt")


def test_weights_directory_rejected(fake_ultra, tmp_path):
    with pytest.raises(DetectorError, match="directory"):
        YOLODetector(tmp_path)


@pytest.mark.parametrize("bad", [
    None, "not-an-array", np.zeros((64, 64), dtype=np.uint8),
    np.zeros((64, 64, 1), dtype=np.uint8), np.zeros((0, 5, 3), dtype=np.uint8),
    np.zeros((64, 64, 3), dtype=np.float32),
])
def test_invalid_images_rejected(fake_ultra, weights_file, bad):
    with pytest.raises(DetectorError):
        YOLODetector(weights_file).predict(bad)


@pytest.mark.parametrize("task", ["segment", "pose", "obb", "classify"])
def test_unsupported_task_rejected(fake_ultra, weights_file, task):
    _FakeYOLO.task = task
    with pytest.raises(DetectorError, match="unsupported model task"):
        YOLODetector(weights_file)


def test_inference_error_wrapped(fake_ultra, weights_file):
    _FakeYOLO.boxes_to_return = "raise"
    with pytest.raises(DetectorError, match="inference failed"):
        YOLODetector(weights_file).predict(_bgr())


def test_detections_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="inconsistent"):
        Detections(boxes_xyxy=np.zeros((2, 4), np.float32),
                   scores=np.zeros((1,), np.float32),
                   classes=np.zeros((2,), np.int64))
