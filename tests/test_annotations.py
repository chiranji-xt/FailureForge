"""Annotation helpers and read-only annotation-validation tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from failureforge.annotations import (
    AnnotationValidationResult,
    clip_xyxy,
    is_degenerate,
    validate_annotations,
    validate_label_file,
    validate_yolo_row,
    xyxy_to_yolo,
    yolo_to_xyxy,
)
from failureforge.dataset import YoloDataset


def make_dataset(
    tmp_path: Path,
    labels: list[str | None],
    *,
    names: tuple[str, ...] = ("cat", "dog"),
    size: tuple[int, int] = (100, 80),
    allow_missing_labels: bool = False,
) -> YoloDataset:
    root = tmp_path / "data"
    images_dir = root / "images" / "test"
    labels_dir = root / "labels" / "test"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for index, body in enumerate(labels):
        Image.new("RGB", size, (100, 120, 140)).save(images_dir / f"img{index}.png")
        if body is not None:
            (labels_dir / f"img{index}.txt").write_text(body, encoding="utf-8")
    return YoloDataset.from_dirs(
        images_dir, labels_dir, list(names), split="test",
        allow_missing_labels=allow_missing_labels,
    )


def codes(result: AnnotationValidationResult, severity: str | None = None) -> list[str]:
    return [
        issue.code for issue in result.issues
        if severity is None or issue.severity == severity
    ]


def test_roundtrip():
    b = {"class_id": 1, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3}
    x1, y1, x2, y2 = yolo_to_xyxy(b)
    assert (x1, y1, x2, y2) == pytest.approx((0.4, 0.35, 0.6, 0.65))
    assert xyxy_to_yolo(x1, y1, x2, y2, 1) == pytest.approx(
        {"class_id": 1, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3})


def test_clip():
    assert clip_xyxy(-0.1, -0.2, 1.5, 0.8) == (0.0, 0.0, 1.0, 0.8)
    assert is_degenerate(0.5, 0.5, 0.5, 0.6)
    assert not is_degenerate(0.1, 0.1, 0.5, 0.5)


def test_validate_rejects_bad_rows():
    with pytest.raises(ValueError):
        validate_yolo_row(["0", "0.5", "0.5"], 1, "a.txt")
    with pytest.raises(ValueError):
        validate_yolo_row(["-1", "0.5", "0.5", "0.1", "0.1"], 1, "a.txt")
    with pytest.raises(ValueError):
        validate_yolo_row(["0", "1.5", "0.5", "0.1", "0.1"], 1, "a.txt")
    with pytest.raises(ValueError):
        validate_yolo_row(["0", "0.5", "0.5", "-0.1", "0.1"], 1, "a.txt")
    with pytest.raises(ValueError):
        validate_yolo_row(["0", "nan", "0.5", "0.1", "0.1"], 1, "a.txt")
    with pytest.raises(ValueError):
        validate_yolo_row(["2", "0.5", "0.5", "0.1", "0.1"], 1, "a.txt",
                          class_names=["cat", "dog"])
    ok = validate_yolo_row(["2", "0.5", "0.5", "0.1", "0.1"], 1, "a.txt")
    assert ok["class_id"] == 2


def test_valid_labels_empty_file_and_boundaries_are_accepted(tmp_path):
    dataset = make_dataset(tmp_path, [
        "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 1 1\n",
        "",
    ])
    before = dataset.image_paths[0].read_bytes()
    result = validate_annotations(dataset)
    assert result.is_valid
    assert result.errors == [] and result.warnings == []
    assert result.num_images == 2
    assert result.num_label_files == 2
    assert result.num_empty_label_files == 1
    assert result.num_missing_label_files == 0
    assert result.num_boxes == 2
    assert result.per_class == {0: 1, 1: 1}
    assert dataset.image_paths[0].read_bytes() == before
    assert len(list(dataset.iter_samples())[0].boxes) == 2


def test_malformed_rows_report_file_and_line(tmp_path):
    dataset = make_dataset(tmp_path, [
        "0 0.5 0.5\n"
        "x 0.5 0.5 0.1 0.1\n"
        "0 0.5 0.5 0.1\n",
    ])
    result = validate_annotations(dataset)
    assert not result.is_valid
    assert result.num_boxes == 0
    assert codes(result, "error") == ["malformed_row", "invalid_class_id", "malformed_row"]
    assert [issue.line for issue in result.errors] == [1, 2, 3]
    assert all(issue.path and issue.path.endswith("img0.txt") for issue in result.errors)
    assert all(issue.image == "img0.png" for issue in result.errors)


def test_invalid_and_out_of_range_class_ids(tmp_path):
    dataset = make_dataset(tmp_path, ["2 0.5 0.5 0.1 0.1\n-1 0.5 0.5 0.1 0.1\n"])
    result = validate_annotations(dataset)
    assert codes(result) == ["class_id_out_of_range", "invalid_class_id"]
    assert all(issue.line in (1, 2) for issue in result.errors)


def test_non_finite_coordinates_are_errors(tmp_path):
    dataset = make_dataset(tmp_path, [
        "0 nan 0.5 0.1 0.1\n"
        "0 0.5 inf 0.1 0.1\n"
        "0 0.5 0.5 -inf 0.1\n",
    ])
    result = validate_annotations(dataset)
    assert not result.is_valid
    assert codes(result) == ["non_finite_coordinate"] * 3
    assert [issue.line for issue in result.errors] == [1, 2, 3]


def test_invalid_sizes_and_out_of_range_coordinates(tmp_path):
    dataset = make_dataset(tmp_path, [
        "0 0.5 0.5 0 0.1\n"
        "0 0.5 0.5 -0.1 0.1\n"
        "0 1.5 0.5 0.1 0.1\n"
        "0 0.5 0.5 1.2 0.1\n",
    ])
    result = validate_annotations(dataset)
    assert codes(result) == [
        "invalid_box_size", "invalid_box_size",
        "coordinate_out_of_range", "coordinate_out_of_range",
    ]
    assert not result.is_valid
    assert result.num_boxes == 0


def test_box_outside_canvas_is_a_warning_not_an_error(tmp_path):
    dataset = make_dataset(tmp_path, [
        "0 0.02 0.5 0.1 0.1\n"
        "1 0.5 0.98 0.1 0.1\n",
    ])
    result = validate_annotations(dataset)
    assert result.is_valid
    outside = [issue for issue in result.warnings
               if issue.code == "box_outside_canvas"]
    assert len(outside) == 2
    assert all("100x80" in issue.message for issue in outside)
    assert result.num_boxes == 2


def test_missing_label_is_error_unless_allowed(tmp_path):
    strict = make_dataset(tmp_path / "strict", [None])
    result = validate_annotations(strict)
    assert not result.is_valid
    assert codes(result) == ["missing_label"]
    assert result.num_missing_label_files == 1
    assert result.num_label_files == 0

    allowed = make_dataset(
        tmp_path / "allowed", [None], allow_missing_labels=True
    )
    allowed_result = validate_annotations(allowed)
    assert allowed_result.is_valid
    assert codes(allowed_result, "warning") == ["missing_label"]
    assert allowed_result.num_missing_label_files == 1


def test_empty_label_file_is_distinct_from_missing(tmp_path):
    dataset = make_dataset(tmp_path, [None, "", "  \n\n"])
    result = validate_annotations(dataset)
    assert codes(result) == ["missing_label"]
    assert result.num_missing_label_files == 1
    assert result.num_empty_label_files == 2
    assert result.num_label_files == 2
    assert result.num_boxes == 0


def test_overlapping_annotations_are_warnings(tmp_path):
    dataset = make_dataset(tmp_path, [
        "0 0.5 0.5 0.2 0.2\n"
        "0 0.5 0.5 0.2 0.2\n",
    ])
    result = validate_annotations(dataset)
    assert result.is_valid
    assert codes(result, "warning") == ["overlapping_annotations"]
    assert result.warnings[0].line == 2
    assert "possible duplicate" in result.warnings[0].message


def test_orphan_label_file_is_an_error(tmp_path):
    dataset = make_dataset(tmp_path, ["0 0.5 0.5 0.1 0.1\n"])
    (dataset.labels_dir / "orphan.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
    )
    result = validate_annotations(dataset)
    assert not result.is_valid
    assert codes(result) == ["orphan_label"]
    assert result.errors[0].path.endswith("orphan.txt")


def test_validate_label_file_accepts_image_size_for_warnings(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.99 0.5 0.1 0.1\n", encoding="utf-8")
    issues = validate_label_file(
        label, ["cat"], image_name="a.png", image_size=(200, 100)
    )
    assert [issue.code for issue in issues] == ["box_outside_canvas"]
    assert issues[0].severity == "warning"
    assert "200x100" in issues[0].message


def test_validation_result_serializes_and_does_not_mutate_source(tmp_path):
    dataset = make_dataset(tmp_path, ["0 0.5 0.5 0.1 0.1\n"])
    result = validate_annotations(dataset)
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["split"] == "test"
    assert payload["num_boxes"] == 1
    assert payload["per_class"] == {"0": 1}
    assert payload["issues"] == []
    samples = list(dataset.iter_samples())
    assert len(samples[0].boxes) == 1
    assert samples[0].boxes[0].class_id == 0
