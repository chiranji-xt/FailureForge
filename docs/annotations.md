# Annotation validation (Day 7)

`failureforge.annotations` provides box helpers plus a **read-only** validator
for YOLO label files. It uses the existing dataset loader contract and never
repairs, rewrites, or deletes annotations.

```python
from failureforge.annotations import validate_annotations, validate_label_file

result = validate_annotations(dataset)       # one split
print(result.is_valid)                       # False if any error exists
for issue in result.issues:
    print(issue.severity, issue.code, issue.path, issue.line, issue.message)
result.raise_if_invalid()                    # raises DatasetError on errors

issues = validate_label_file(                # one file, optional image context
    "labels/val/img0.txt", ["cat", "dog"],
    image_name="img0.png", image_size=(640, 480),
)
```

`validate_annotations(...)` also returns `num_images`, `num_label_files`,
`num_empty_label_files`, `num_missing_label_files`, `num_boxes`, `per_class`,
and a JSON-safe `to_dict()`. Results are sorted by severity, image/path, line,
and code so output is deterministic.

## Label semantics

One non-empty line is `class_id x_center y_center width height`.

- All four geometry values are normalized to `[0, 1]`.
- `class_id` must be a non-negative integer and, when class names are known,
  less than `len(class_names)`.
- Width and height must be finite and greater than zero.
- Blank lines are ignored. An empty label file is valid and means **zero
  ground-truth objects**.
- Missing label files follow the dataset's `allow_missing_labels` setting.

## Errors

| Code | Meaning |
|---|---|
| `missing_label` | No label file for an image. Error by default; warning when `allow_missing_labels=True`. |
| `unreadable_label` | The label file could not be read. |
| `unreadable_image` | The image could not be opened, so dimensions/matching cannot be checked. |
| `orphan_label` | A `.txt` label file has no matching image stem. |
| `malformed_row` | A non-empty row does not have exactly five values. |
| `invalid_class_id` | Class ID is not an integer or is negative. |
| `class_id_out_of_range` | Class ID is outside the configured class-name range. |
| `invalid_coordinate` | A geometry value is not numeric. |
| `non_finite_coordinate` | A geometry value is NaN or infinite. |
| `invalid_box_size` | Width or height is zero or negative. |
| `coordinate_out_of_range` | Center/width/height is outside normalized `[0, 1]`. |

Errors make `is_valid` false. The validator continues checking later rows and
files, so one pass reports all independent problems it can see.

## Warnings

| Code | Meaning |
|---|---|
| `box_outside_canvas` | Center/width/height are individually valid, but the derived box extends outside the image canvas. This is allowed by the loader but is likely a dimension/annotation issue. Includes image dimensions when known. |
| `overlapping_annotations` | Two annotations overlap at IoU >= the configured threshold (default **0.85**). Possible duplicate; **not** automatically invalid. |
| `missing_label` | Only when `allow_missing_labels=True`; treated as an empty file. |

Missing files and explicitly empty files remain distinguishable:
`num_missing_label_files` versus `num_empty_label_files`. Neither is silently
converted into the other.

## Properties and limitations

- The validator never mutates dataset samples, boxes, or label files.
- No model weights, inference, or network access are used.
- YOLO labels do not carry explicit image dimensions, so dimension checks use
  the actual image size; a box can only be flagged as outside the normalized
  canvas (and the message includes pixel dimensions when available).
- Overlap warnings are advisory. Different-class overlaps may be legitimate
  (crowds/occlusion), so there is no automatic rejection.
- It validates annotation **format/geometry**, not semantic label correctness.
- There is no new CLI command; call the Python API directly (CLI remains Day 13).
