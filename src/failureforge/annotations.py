"""Annotation helpers and dataset annotation validation.

The parser/validator here is read-only. It follows the existing dataset loader
contract: one YOLO row is ``class_id x_center y_center width height`` with all
geometry normalized to ``[0, 1]``. Only non-negative class IDs below
``len(class_names)`` are accepted (when class names are known).

Severity policy
---------------
- **error**: malformed rows, non-numeric/non-finite values, invalid class IDs,
  non-positive sizes, coordinates outside ``[0, 1]``, unreadable/orphan label
  files, and missing labels (unless the dataset allows them).
- **warning**: a normalized row extends outside the image canvas (allowed by
  the loader but likely a dimension/annotation issue), or two annotations have
  an unusually high IoU (possible duplicates). Warnings never reject a dataset.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .dataset import DatasetError, YoloDataset

DEFAULT_OVERLAP_IOU = 0.85


def yolo_to_xyxy(box: dict) -> tuple[float, float, float, float]:
    x1 = box["x_center"] - box["width"] / 2
    y1 = box["y_center"] - box["height"] / 2
    x2 = box["x_center"] + box["width"] / 2
    y2 = box["y_center"] + box["height"] / 2
    return (x1, y1, x2, y2)


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, class_id: int) -> dict:
    return {
        "class_id": int(class_id),
        "x_center": (x1 + x2) / 2,
        "y_center": (y1 + y2) / 2,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def clip_xyxy(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))


def is_degenerate(x1: float, y1: float, x2: float, y2: float, min_size: float = 1e-4) -> bool:
    return (x2 - x1) < min_size or (y2 - y1) < min_size


def box_area_xyxy(x1: float, y1: float, x2: float, y2: float) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def validate_yolo_row(
    parts: list[str],
    line_no: int,
    path: str,
    *,
    class_names: list[str] | None = None,
) -> dict:
    """Validate one YOLO label row; raises ``ValueError`` on the first problem."""
    if len(parts) != 5:
        raise ValueError(f"{path}:{line_no}: expected 5 values, got {len(parts)}: {' '.join(parts)}")
    try:
        cid = int(parts[0])
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        raise ValueError(f"{path}:{line_no}: non-numeric value in row: {' '.join(parts)}")
    if cid < 0:
        raise ValueError(f"{path}:{line_no}: class_id must be >= 0, got {cid}")
    if class_names is not None and cid >= len(class_names):
        raise ValueError(
            f"{path}:{line_no}: class_id {cid} out of range for {len(class_names)} classes"
        )
    for name, value in zip(("x_center", "y_center", "width", "height"), vals):
        if not math.isfinite(value):
            raise ValueError(f"{path}:{line_no}: {name} is not finite: {value}")
    xc, yc, w, h = vals
    if w <= 0 or h <= 0:
        raise ValueError(f"{path}:{line_no}: width/height must be > 0")
    for name, value in zip(("x_center", "y_center", "width", "height"), vals):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{path}:{line_no}: {name}={value} out of [0,1]")
    return {"class_id": cid, "x_center": xc, "y_center": yc, "width": w, "height": h}


# ---------------------------------------------------------------------------
# Validation records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnnotationIssue:
    """One annotation problem, tied to an image/label file and line where known."""

    severity: str
    code: str
    message: str
    image: str | None = None
    path: str | None = None
    line: int | None = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity, "code": self.code, "message": self.message,
            "image": self.image, "path": self.path, "line": self.line,
        }

    def __str__(self) -> str:
        loc = self.path or self.image or "<dataset>"
        if self.line is not None:
            loc = f"{loc}:{self.line}"
        return f"[{self.severity}] {loc}: {self.message}"


@dataclass
class AnnotationValidationResult:
    """Read-only annotation validation outcome for one dataset split."""

    issues: list[AnnotationIssue]
    num_images: int
    num_label_files: int
    num_empty_label_files: int
    num_missing_label_files: int
    num_boxes: int
    per_class: dict[int, int]
    class_names: list[str]
    split: str

    @property
    def errors(self) -> list[AnnotationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[AnnotationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            details = "\n".join(f"  {issue}" for issue in self.errors)
            raise DatasetError(
                f"annotation validation failed with {len(self.errors)} error(s):\n{details}"
            )

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "num_images": self.num_images,
            "num_label_files": self.num_label_files,
            "num_empty_label_files": self.num_empty_label_files,
            "num_missing_label_files": self.num_missing_label_files,
            "num_boxes": self.num_boxes,
            "per_class": {str(k): v for k, v in sorted(self.per_class.items())},
            "class_names": list(self.class_names),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _issue(severity: str, code: str, message: str, *, image: str | None = None,
           path: Path | str | None = None, line: int | None = None) -> AnnotationIssue:
    return AnnotationIssue(
        severity=severity, code=code, message=message, image=image,
        path=str(path) if path is not None else None, line=line,
    )


def _iou_xyxy(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def validate_label_file(
    label_path: str | Path,
    class_names: list[str] | None = None,
    *,
    image_name: str | None = None,
    image_size: tuple[int, int] | None = None,
    overlap_iou_threshold: float = DEFAULT_OVERLAP_IOU,
) -> list[AnnotationIssue]:
    """Validate one label file without modifying it.

    ``image_size`` is ``(width, height)`` in pixels; when supplied, normalized
    boxes extending past the image canvas produce warnings. Malformed rows are
    errors; later rows are still checked.
    """
    if not (0.0 < float(overlap_iou_threshold) <= 1.0):
        raise ValueError("overlap_iou_threshold must be in (0, 1]")
    path = Path(label_path)
    image = image_name or path.name
    names = list(class_names) if class_names is not None else []
    issues: list[AnnotationIssue] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [_issue("error", "unreadable_label", f"cannot read label file: {exc}",
                       image=image, path=path)]

    parsed: list[dict] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(_issue(
                "error", "malformed_row",
                f"expected 5 values (class xc yc w h), got {len(parts)}",
                image=image, path=path, line=line_no,
            ))
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            issues.append(_issue(
                "error", "invalid_class_id",
                f"class ID is not an integer: {parts[0]!r}",
                image=image, path=path, line=line_no,
            ))
            continue
        if cid < 0:
            issues.append(_issue(
                "error", "invalid_class_id", f"class ID must be >= 0, got {cid}",
                image=image, path=path, line=line_no,
            ))
            continue
        if names and cid >= len(names):
            issues.append(_issue(
                "error", "class_id_out_of_range",
                f"class ID {cid} is outside the configured range 0..{len(names) - 1}",
                image=image, path=path, line=line_no,
            ))
            continue
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append(_issue(
                "error", "invalid_coordinate",
                f"geometry must be numeric: {' '.join(parts[1:])}",
                image=image, path=path, line=line_no,
            ))
            continue
        if not all(math.isfinite(value) for value in values):
            issues.append(_issue(
                "error", "non_finite_coordinate",
                f"geometry must be finite: {' '.join(parts[1:])}",
                image=image, path=path, line=line_no,
            ))
            continue
        xc, yc, w, h = values
        if w <= 0 or h <= 0:
            issues.append(_issue(
                "error", "invalid_box_size",
                f"width and height must be > 0, got {w:g}x{h:g}",
                image=image, path=path, line=line_no,
            ))
            continue
        for name, value in (("x_center", xc), ("y_center", yc),
                            ("width", w), ("height", h)):
            if not (0.0 <= value <= 1.0):
                issues.append(_issue(
                    "error", "coordinate_out_of_range",
                    f"{name}={value:g} is outside the normalized range [0, 1]",
                    image=image, path=path, line=line_no,
                ))
                break
        else:
            parsed.append({
                "line": line_no, "class_id": cid,
                "box": (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2),
            })
            x1, y1, x2, y2 = parsed[-1]["box"]
            if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                detail = (
                    f" for image dimensions {image_size[0]}x{image_size[1]}"
                    if image_size is not None else ""
                )
                issues.append(_issue(
                    "warning", "box_outside_canvas",
                    f"normalized box ({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}) "
                    f"extends outside the image canvas [0, 1]{detail}",
                    image=image, path=path, line=line_no,
                ))

    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            first, second = parsed[i], parsed[j]
            value = _iou_xyxy(first["box"], second["box"])
            if value >= overlap_iou_threshold:
                issues.append(_issue(
                    "warning", "overlapping_annotations",
                    f"annotations on lines {first['line']} and {second['line']} "
                    f"overlap at IoU {value:.3f} (possible duplicate); "
                    "not marked invalid automatically",
                    image=image, path=path, line=second["line"],
                ))
    return issues


def validate_annotations(
    dataset: YoloDataset,
    *,
    overlap_iou_threshold: float = DEFAULT_OVERLAP_IOU,
) -> AnnotationValidationResult:
    """Validate a dataset split's labels without mutating dataset state."""
    if not (0.0 < float(overlap_iou_threshold) <= 1.0):
        raise ValueError("overlap_iou_threshold must be in (0, 1]")
    image_paths = list(dataset.image_paths)
    labels_dir = Path(dataset.labels_dir)
    class_names = list(dataset.class_names)
    issues: list[AnnotationIssue] = []
    if not class_names:
        issues.append(_issue("error", "missing_class_names",
                             "dataset has no configured class names"))

    per_class: dict[int, int] = {}
    num_boxes = 0
    num_label_files = 0
    num_empty = 0
    num_missing = 0
    image_stems = {path.stem for path in image_paths}

    for image_path in image_paths:
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            num_missing += 1
            if dataset.allow_missing_labels:
                issues.append(_issue(
                    "warning", "missing_label",
                    "missing label file (treated as an empty annotation file)",
                    image=image_path.name, path=label_path,
                ))
            else:
                issues.append(_issue(
                    "error", "missing_label",
                    "missing label file (use allow_missing_labels=True only if "
                    "this image truly has no objects)",
                    image=image_path.name, path=label_path,
                ))
            continue

        num_label_files += 1
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            width = height = None
            issues.append(_issue(
                "error", "unreadable_image", f"cannot open image: {exc}",
                image=image_path.name, path=image_path,
            ))

        try:
            label_text = label_path.read_text(encoding="utf-8")
        except OSError as exc:
            issues.append(_issue(
                "error", "unreadable_label", f"cannot read label file: {exc}",
                image=image_path.name, path=label_path,
            ))
            continue
        if not label_text.strip():
            num_empty += 1

        file_issues = validate_label_file(
            label_path, class_names, image_name=image_path.name,
            image_size=(width, height) if width and height else None,
            overlap_iou_threshold=overlap_iou_threshold,
        )
        issues.extend(file_issues)
        for line_no, raw in enumerate(label_text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                validate_yolo_row(
                    line.split(), line_no, str(label_path),
                    class_names=class_names,
                )
            except ValueError:
                continue
            cid = int(line.split()[0])
            per_class[cid] = per_class.get(cid, 0) + 1
            num_boxes += 1

    for label_file in sorted(labels_dir.glob("*.txt")):
        if label_file.stem not in image_stems:
            issues.append(_issue(
                "error", "orphan_label",
                "label file has no matching image",
                path=label_file,
            ))

    order = {"error": 0, "warning": 1}
    issues.sort(key=lambda issue: (
        order.get(issue.severity, 2), issue.image or "",
        issue.path or "", -1 if issue.line is None else issue.line,
        issue.code, issue.message,
    ))
    return AnnotationValidationResult(
        issues=issues, num_images=len(image_paths),
        num_label_files=num_label_files, num_empty_label_files=num_empty,
        num_missing_label_files=num_missing, num_boxes=num_boxes,
        per_class=per_class, class_names=class_names, split=dataset.split,
    )
