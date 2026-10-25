"""Dataset loading/validation tests using synthetic YOLO fixtures."""
from pathlib import Path

import pytest
from PIL import Image

from failureforge.dataset import (
    DatasetError, YoloDataset, load_dataset, split_dataset, summarize,
)


def _img(path: Path, size=(100, 100), color=(128, 128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _label(path: Path, rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")


def _make_config_dataset(tmp_path: Path, splits=("train", "val", "test"),
                         names="[cat, dog]", n=2):
    root = tmp_path / "ds"
    for split in splits:
        for i in range(n):
            _img(root / "images" / split / f"{split}{i}.png")
            _label(root / "labels" / split / f"{split}{i}.txt", "0 0.5 0.5 0.4 0.4\n")
    cfg = ["path: ."]
    for split in splits:
        cfg.append(f"{split}: images/{split}")
    cfg.append(f"names: {names}")
    (root / "data.yaml").write_text("\n".join(cfg) + "\n", encoding="utf-8")
    return root


def test_prefers_test_split(tmp_path):
    root = _make_config_dataset(tmp_path)
    ds = YoloDataset.from_config(root / "data.yaml")
    assert ds.split == "test"
    assert len(ds) == 2
    assert all("test" in p.name for p in ds.image_paths)


def test_explicit_split(tmp_path):
    root = _make_config_dataset(tmp_path)
    ds = YoloDataset.from_config(root / "data.yaml", split="val")
    assert ds.split == "val"
    assert all("val" in p.name for p in ds.image_paths)


def test_never_defaults_to_train(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("train",))
    with pytest.raises(DatasetError, match="train"):
        YoloDataset.from_config(root / "data.yaml")
    ds = YoloDataset.from_config(root / "data.yaml", split="train")
    assert ds.split == "train"


def test_normalized_to_pixel_conversion(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    _img(root / "images" / "test" / "test0.png", size=(200, 100))
    _label(root / "labels" / "test" / "test0.txt", "0 0.5 0.5 0.5 0.5\n")
    ds = YoloDataset.from_config(root / "data.yaml")
    sample = next(ds.iter_samples())
    assert sample.width == 200 and sample.height == 100
    (b,) = sample.boxes
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((50.0, 25.0, 150.0, 75.0))


def test_empty_label_file_is_valid(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    _label(root / "labels" / "test" / "test0.txt", "")
    ds = YoloDataset.from_config(root / "data.yaml")
    sample = next(ds.iter_samples())
    assert sample.boxes == []
    assert ds.validate().is_valid


def test_missing_label_is_error_by_default(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    (root / "labels" / "test" / "test0.txt").unlink()
    ds = YoloDataset.from_config(root / "data.yaml")
    result = ds.validate()
    assert not result.is_valid
    assert any("missing label" in e.message for e in result.errors)


def test_missing_label_allowed_is_warning(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    (root / "labels" / "test" / "test0.txt").unlink()
    ds = YoloDataset.from_config(root / "data.yaml", allow_missing_labels=True)
    result = ds.validate()
    assert result.is_valid
    assert any("missing label" in w.message for w in result.warnings)


def test_missing_image_dir(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        YoloDataset.from_dirs(tmp_path / "nope", tmp_path, ["cat"])


def test_malformed_row_reports_line(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    _label(root / "labels" / "test" / "test0.txt", "0 0.5 0.5\n")
    ds = YoloDataset.from_config(root / "data.yaml")
    result = ds.validate()
    (err,) = result.errors
    assert err.line == 1
    assert "5 values" in err.message


def test_non_numeric_row(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    _label(root / "labels" / "test" / "test0.txt", "0 abc 0.5 0.1 0.1\n")
    result = YoloDataset.from_config(root / "data.yaml").validate()
    assert any("non-numeric" in e.message for e in result.errors)


def test_out_of_range_coordinate(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    _label(root / "labels" / "test" / "test0.txt", "0 1.5 0.5 0.1 0.1\n")
    result = YoloDataset.from_config(root / "data.yaml").validate()
    assert any("outside normalized range" in e.message for e in result.errors)


def test_class_id_out_of_range(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    _label(root / "labels" / "test" / "test0.txt", "5 0.5 0.5 0.1 0.1\n")
    result = YoloDataset.from_config(root / "data.yaml").validate()
    assert any("out of range" in e.message for e in result.errors)


def test_negative_size(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    _label(root / "labels" / "test" / "test0.txt", "0 0.5 0.5 -0.1 0.1\n")
    result = YoloDataset.from_config(root / "data.yaml").validate()
    assert any("outside normalized range" in e.message or "must be > 0" in e.message
               for e in result.errors)


def test_corrupt_image(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    (root / "images" / "test" / "test0.png").write_bytes(b"not an image")
    result = YoloDataset.from_config(root / "data.yaml").validate()
    assert any("corrupt" in e.message or "unreadable" in e.message for e in result.errors)


def test_orphan_label(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    _label(root / "labels" / "test" / "ghost.txt", "0 0.5 0.5 0.1 0.1\n")
    result = YoloDataset.from_config(root / "data.yaml").validate()
    assert any("no matching image" in e.message for e in result.errors)


def test_class_name_mismatch(tmp_path):
    root = _make_config_dataset(tmp_path)
    ds = YoloDataset.from_config(root / "data.yaml")
    result = ds.validate(model_class_names=["cat", "bird"])
    assert not result.is_valid
    assert any("not present in model" in e.message for e in result.errors)
    with pytest.raises(DatasetError):
        ds.class_mapping(["cat", "bird"])


def test_class_mapping_by_name_not_index(tmp_path):
    root = _make_config_dataset(tmp_path)
    ds = YoloDataset.from_config(root / "data.yaml")
    mapping = ds.class_mapping(["dog", "cat", "bird"])
    assert mapping == {0: 1, 1: 0}


def test_ambiguous_model_names(tmp_path):
    root = _make_config_dataset(tmp_path)
    ds = YoloDataset.from_config(root / "data.yaml")
    with pytest.raises(DatasetError, match="ambiguous"):
        ds.class_mapping(["cat", "cat"])


def test_deterministic_ordering(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=3)
    ds = YoloDataset.from_config(root / "data.yaml")
    assert [p.name for p in ds.image_paths] == sorted(p.name for p in ds.image_paths)


def test_load_dataset_compat(tmp_path):
    imgdir, lbldir = tmp_path / "images", tmp_path / "labels"
    _img(imgdir / "a.png")
    _label(lbldir / "a.txt", "0 0.5 0.5 0.2 0.2\n")
    samples = load_dataset(imgdir, lbldir)
    assert len(samples) == 1
    assert samples[0].boxes[0].class_id == 0
    summary = summarize(samples)
    assert summary["num_boxes"] == 1


def test_split_dataset_deterministic(tmp_path):
    imgdir, lbldir = tmp_path / "images", tmp_path / "labels"
    for i in range(4):
        _img(imgdir / f"a{i}.png")
        _label(lbldir / f"a{i}.txt", "0 0.5 0.5 0.2 0.2\n")
    samples = load_dataset(imgdir, lbldir)
    a1, b1 = split_dataset(samples, 0.5, seed=7)
    a2, b2 = split_dataset(samples, 0.5, seed=7)
    assert [s.image_path.name for s in a1] == [s.image_path.name for s in a2]
    assert len(a1) == 2 and len(b1) == 2


def test_iteration_rejects_missing_label_by_default(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    (root / "labels" / "test" / "test0.txt").unlink()
    ds = YoloDataset.from_config(root / "data.yaml")
    with pytest.raises(DatasetError, match="missing label"):
        list(ds.iter_samples())


def test_iteration_allows_missing_label_when_opted_in(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=1)
    (root / "labels" / "test" / "test0.txt").unlink()
    ds = YoloDataset.from_config(root / "data.yaml", allow_missing_labels=True)
    (sample,) = list(ds.iter_samples())
    assert sample.boxes == [] and sample.label_path is None


def test_partition_is_disjoint_deterministic_and_nonempty(tmp_path):
    root = _make_config_dataset(tmp_path, splits=("test",), n=6)
    ds = YoloDataset.from_config(root / "data.yaml")
    a, b = ds.partition(0.33, seed=1)
    a2, b2 = ds.partition(0.33, seed=1)
    assert set(a.image_paths).isdisjoint(b.image_paths)
    assert len(a) + len(b) == len(ds) and len(b) == 2
    assert a.image_paths == a2.image_paths and b.image_paths == b2.image_paths
    with pytest.raises(DatasetError):
        ds.partition(1.0)
