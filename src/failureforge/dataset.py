"""YOLO-format dataset loading, validation, and class-name mapping.

Supports the standard Ultralytics layout::

    dataset/
    ├── data.yaml          # path, train/val/test, names
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/

Path resolution follows YOLO conventions: ``path`` (optional) is the dataset
root resolved relative to ``data.yaml``; split entries are resolved relative to
that root. Labels are found by replacing the ``images`` path component with
``labels`` and swapping the image extension for ``.txt``.

Ground truth is exposed in **pixel** ``[x1, y1, x2, y2]`` coordinates, using
each image's real dimensions. Missing label files are a validation *error* by
default (never silently empty ground truth); pass ``allow_missing_labels=True``
to treat them as zero-object images with a warning.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_KEYS = ("train", "val", "test")


class DatasetError(Exception):
    """Raised when a dataset cannot be loaded or is fundamentally invalid."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundTruthBox:
    """One labeled object in pixel coordinates."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Sample:
    """A single image and its ground-truth boxes (pixel coordinates)."""

    image_path: Path
    label_path: Path | None
    width: int
    height: int
    boxes: list[GroundTruthBox] = field(default_factory=list)


@dataclass
class ValidationIssue:
    """A single validation problem tied to a file (and line where possible)."""

    severity: str  # "error" | "warning"
    message: str
    path: str | None = None
    line: int | None = None

    def __str__(self) -> str:
        loc = self.path or "<dataset>"
        if self.line is not None:
            loc = f"{loc}:{self.line}"
        return f"[{self.severity}] {loc}: {self.message}"


@dataclass
class ValidationResult:
    """Outcome of :meth:`YoloDataset.validate`."""

    issues: list[ValidationIssue]
    num_images: int
    num_boxes: int
    per_class: dict[int, int]
    class_names: list[str]
    split: str

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            details = "\n".join(f"  {e}" for e in self.errors)
            raise DatasetError(
                f"dataset validation failed with {len(self.errors)} error(s):\n{details}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_names(names) -> list[str]:
    if names is None:
        return []
    if isinstance(names, dict):
        try:
            return [str(names[k]) for k in sorted(names, key=int)]
        except (TypeError, ValueError):
            return [str(v) for _, v in sorted(names.items())]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    raise DatasetError(f"class names must be a list or mapping, got {type(names).__name__}")


def _resolve_image_dir(root: Path, value) -> Path:
    """Resolve a split entry (dir or .txt list) to an image directory."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise DatasetError("split entry is an empty list")
        value = value[0]
    raw = Path(str(value))
    path = raw if raw.is_absolute() else (root / raw)
    if path.is_dir():
        return path
    if path.is_file() and path.suffix.lower() == ".txt":
        # YOLO allows a text file listing image paths; not supported for
        # directory-level streaming — ask the user for a directory instead.
        raise DatasetError(
            f"split '{path}' is an image-list file; this loader requires an "
            "image directory. Point the split at a directory instead."
        )
    raise DatasetError(f"split path does not exist or is not a directory: {path}")


def _labels_dir_for(image_dir: Path) -> Path:
    """Derive the labels directory from an images directory."""
    parts = list(image_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    return image_dir.parent / "labels" / image_dir.name


def _label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / (image_path.stem + ".txt")


def _parse_yolo_line(parts: list[str], line_no: int, path: Path,
                     class_names: list[str]) -> GroundTruthBox:
    """Parse/validate one label row into a *normalized* xyxy box.

    Coordinates are scaled to pixels later, once the image size is known.
    Raises ``ValueError`` with file:line context on malformed rows.
    """
    if len(parts) != 5:
        raise ValueError(f"expected 5 values (class xc yc w h), got {len(parts)}")
    try:
        cid = int(parts[0])
        xc, yc, w, h = (float(v) for v in parts[1:])
    except ValueError:
        raise ValueError("non-numeric value in label row")
    if cid < 0:
        raise ValueError(f"class_id must be >= 0, got {cid}")
    if class_names and cid >= len(class_names):
        raise ValueError(f"class_id {cid} out of range for {len(class_names)} classes")
    for name, v in (("x_center", xc), ("y_center", yc), ("width", w), ("height", h)):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name}={v} outside normalized range [0, 1]")
    if w <= 0 or h <= 0:
        raise ValueError(f"width/height must be > 0, got {w}x{h}")
    return GroundTruthBox(
        class_id=cid,
        x1=xc - w / 2,
        y1=yc - h / 2,
        x2=xc + w / 2,
        y2=yc + h / 2,
    )


def _scale_to_pixels(boxes: list[GroundTruthBox], width: int, height: int) -> list[GroundTruthBox]:
    """Convert normalized xyxy boxes to pixel coordinates for one image."""
    return [GroundTruthBox(b.class_id, b.x1 * width, b.y1 * height,
                           b.x2 * width, b.y2 * height) for b in boxes]


def _read_label_file(label_path: Path, class_names: list[str]) -> tuple[list[GroundTruthBox], list[ValidationIssue]]:
    boxes: list[GroundTruthBox] = []
    issues: list[ValidationIssue] = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError as e:
        issues.append(ValidationIssue("error", f"cannot read label file: {e}", str(label_path)))
        return boxes, issues
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            boxes.append(_parse_yolo_line(line.split(), line_no, label_path, class_names))
        except ValueError as e:
            issues.append(ValidationIssue("error", str(e), str(label_path), line_no))
    return boxes, issues


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class YoloDataset:
    """Streaming YOLO detection dataset for one split.

    Construct with :meth:`from_config` (``data.yaml``) or :meth:`from_dirs`.
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels_dir: Path,
        class_names: list[str],
        *,
        split: str,
        root: Path | None = None,
        allow_missing_labels: bool = False,
    ) -> None:
        if not image_paths:
            raise DatasetError(f"no images found for split '{split}'")
        self.image_paths = sorted(image_paths, key=lambda p: str(p))
        self.labels_dir = labels_dir
        self.class_names = list(class_names)
        self.split = split
        self.root = root
        self.allow_missing_labels = allow_missing_labels

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        split: str | None = None,
        *,
        allow_missing_labels: bool = False,
    ) -> "YoloDataset":
        """Load a split from a YOLO ``data.yaml``.

        When ``split`` is omitted, ``test`` is preferred, then ``val``.
        The loader never silently falls back to ``train``; if only ``train``
        exists the caller must pass ``split="train"`` explicitly.
        """
        config_path = Path(config_path)
        if not config_path.is_file():
            raise DatasetError(f"dataset config not found: {config_path}")
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            raise DatasetError(f"cannot read dataset config {config_path}: {e}")
        if not isinstance(cfg, dict):
            raise DatasetError(f"dataset config {config_path} must be a mapping")

        root = config_path.parent
        if cfg.get("path"):
            root = (root / str(cfg["path"])).resolve()

        available = [k for k in SPLIT_KEYS if cfg.get(k)]
        if not available:
            raise DatasetError(f"{config_path} defines none of train/val/test splits")

        if split is None:
            if "test" in available:
                split = "test"
            elif "val" in available:
                split = "val"
            else:
                raise DatasetError(
                    f"{config_path} only defines a 'train' split; refusing to "
                    "evaluate on training data implicitly. Pass split='train' "
                    "explicitly if that is intended."
                )
        if split not in cfg:
            raise DatasetError(
                f"split '{split}' not defined in {config_path}; available: {available}"
            )

        image_dir = _resolve_image_dir(root, cfg[split])
        labels_dir = _labels_dir_for(image_dir)
        if not labels_dir.is_dir():
            raise DatasetError(
                f"labels directory not found for split '{split}': {labels_dir}"
            )
        class_names = _normalize_names(cfg.get("names"))
        if not class_names:
            raise DatasetError(f"{config_path} is missing class 'names'")

        images = [p for p in image_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        return cls(images, labels_dir, class_names, split=split, root=root,
                   allow_missing_labels=allow_missing_labels)

    @classmethod
    def from_dirs(
        cls,
        images_dir: str | Path,
        labels_dir: str | Path,
        class_names: list[str],
        *,
        split: str = "custom",
        allow_missing_labels: bool = False,
    ) -> "YoloDataset":
        """Load a split from explicit image/label directories."""
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        if not images_dir.is_dir():
            raise DatasetError(f"images directory not found: {images_dir}")
        if not labels_dir.is_dir():
            raise DatasetError(f"labels directory not found: {labels_dir}")
        names = _normalize_names(class_names)
        if not names:
            raise DatasetError("class_names must be provided")
        images = [p for p in images_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        return cls(images, labels_dir, names, split=split,
                   allow_missing_labels=allow_missing_labels)

    # -- basic access -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    @property
    def metadata(self) -> dict:
        return {
            "root": str(self.root) if self.root else None,
            "split": self.split,
            "num_images": len(self.image_paths),
            "class_names": list(self.class_names),
            "labels_dir": str(self.labels_dir),
        }

    def partition(
        self, holdout_fraction: float, seed: int = 0
    ) -> "tuple[YoloDataset, YoloDataset]":
        """Deterministically split into disjoint ``(discovery, holdout)`` datasets.

        Used to keep search tuning and confirmation on different images. Both
        parts must be non-empty, so at least two images are required.
        """
        if not 0.0 < holdout_fraction < 1.0:
            raise DatasetError("holdout_fraction must be in (0, 1)")
        if len(self.image_paths) < 2:
            raise DatasetError("need at least 2 images to partition a dataset")
        order = sorted(range(len(self.image_paths)))
        random.Random(seed).shuffle(order)
        n_hold = min(max(1, round(len(order) * holdout_fraction)), len(order) - 1)
        hold = {i for i in order[:n_hold]}

        def build(indices: list[int], tag: str) -> "YoloDataset":
            return YoloDataset(
                [self.image_paths[i] for i in indices], self.labels_dir,
                self.class_names, split=f"{self.split}:{tag}", root=self.root,
                allow_missing_labels=self.allow_missing_labels,
            )

        keep = [i for i in range(len(self.image_paths)) if i not in hold]
        return build(keep, "discovery"), build(sorted(hold), "holdout")

    def class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def iter_samples(self) -> Iterator[Sample]:
        """Yield samples one at a time (images are not cached)."""
        for image_path in self.image_paths:
            label_path = _label_path_for(image_path, self.labels_dir)
            if label_path.exists():
                boxes, issues = _read_label_file(label_path, self.class_names)
                errors = [i for i in issues if i.severity == "error"]
                if errors:
                    raise DatasetError("; ".join(str(e) for e in errors))
            elif self.allow_missing_labels:
                boxes = []
            else:
                raise DatasetError(
                    f"missing label file for {image_path.name}: {label_path} "
                    "(an empty file is a valid zero-object label; pass "
                    "allow_missing_labels=True to treat missing files as empty)"
                )
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception as e:  # corrupt/unreadable image
                raise DatasetError(f"cannot open image {image_path}: {e}")
            yield Sample(
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
                width=width,
                height=height,
                boxes=_scale_to_pixels(boxes, width, height),
            )

    # -- validation ---------------------------------------------------------

    def validate(self, model_class_names: list[str] | None = None) -> ValidationResult:
        """Validate images, labels, coordinates, class IDs, and class mapping.

        Returns a :class:`ValidationResult`; never silently drops data.
        """
        issues: list[ValidationIssue] = []
        per_class: dict[int, int] = {}
        num_boxes = 0
        image_stems = {p.stem for p in self.image_paths}

        if not self.class_names:
            issues.append(ValidationIssue("error", "dataset has no class names"))

        for image_path in self.image_paths:
            label_path = _label_path_for(image_path, self.labels_dir)
            if not label_path.exists():
                msg = f"missing label file: {label_path}"
                if self.allow_missing_labels:
                    issues.append(ValidationIssue("warning", msg + " (treated as empty)", str(image_path)))
                else:
                    issues.append(ValidationIssue("error", msg, str(image_path)))
                continue
            try:
                with Image.open(image_path) as im:
                    im.verify()
            except Exception as e:
                issues.append(ValidationIssue("error", f"corrupt/unreadable image: {e}", str(image_path)))
                continue
            boxes, label_issues = _read_label_file(label_path, self.class_names)
            issues.extend(label_issues)
            for b in boxes:
                per_class[b.class_id] = per_class.get(b.class_id, 0) + 1
            num_boxes += len(boxes)

        # Orphan label files (labels with no matching image).
        for label_file in sorted(self.labels_dir.glob("*.txt")):
            if label_file.stem not in image_stems:
                issues.append(ValidationIssue(
                    "error", "label file has no matching image", str(label_file)))

        # Class-name consistency against the model.
        if model_class_names is not None:
            model_names = [str(n) for n in model_class_names]
            if not model_names:
                issues.append(ValidationIssue(
                    "error", "model exposes no class names; cannot verify class mapping"))
            else:
                try:
                    self.class_mapping(model_names)
                except DatasetError as e:
                    issues.append(ValidationIssue("error", str(e)))

        return ValidationResult(
            issues=issues,
            num_images=len(self.image_paths),
            num_boxes=num_boxes,
            per_class=per_class,
            class_names=list(self.class_names),
            split=self.split,
        )

    def class_mapping(self, model_class_names: list[str]) -> dict[int, int]:
        """Map dataset class IDs to model class IDs by **name**.

        Raises :class:`DatasetError` if any dataset class is absent from the
        model or if model names are ambiguous (duplicates).
        """
        model_names = [str(n) for n in model_class_names]
        if not model_names:
            raise DatasetError("model exposes no class names; cannot build class mapping")
        if len(set(model_names)) != len(model_names):
            raise DatasetError(f"model class names are ambiguous (duplicates): {model_names}")
        mapping: dict[int, int] = {}
        missing: list[str] = []
        for dataset_id, name in enumerate(self.class_names):
            hits = [model_id for model_id, m in enumerate(model_names) if m == name]
            if len(hits) != 1:
                missing.append(name)
            else:
                mapping[dataset_id] = hits[0]
        if missing:
            raise DatasetError(
                f"dataset classes not present in model: {missing}; "
                f"dataset={self.class_names}, model={model_names}"
            )
        return mapping


# ---------------------------------------------------------------------------
# Backwards-compatible helper used by earlier preview code
# ---------------------------------------------------------------------------

def load_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    *,
    allow_missing_labels: bool = True,
    split: str | None = None,
) -> list[Sample]:
    """Load a flat dataset from explicit directories.

    Kept as a convenience wrapper around :class:`YoloDataset`. Class names are
    unknown here, so class-ID range is only checked for non-negativity.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not images_dir.is_dir():
        raise DatasetError(f"images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise DatasetError(f"labels directory not found: {labels_dir}")
    image_paths = sorted(
        (p for p in images_dir.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: str(p),
    )
    if not image_paths:
        raise DatasetError(f"no images found in {images_dir}")

    samples: list[Sample] = []
    for image_path in image_paths:
        label_path = labels_dir / (image_path.stem + ".txt")
        boxes: list[GroundTruthBox] = []
        if label_path.exists():
            text = label_path.read_text(encoding="utf-8")
            for line_no, raw in enumerate(text.splitlines(), start=1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise DatasetError(
                        f"{label_path}:{line_no}: expected 5 values, got {len(parts)}")
                try:
                    cid = int(parts[0])
                    xc, yc, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    raise DatasetError(
                        f"{label_path}:{line_no}: non-numeric value in row")
                if cid < 0:
                    raise DatasetError(
                        f"{label_path}:{line_no}: class_id must be >= 0")
                for name, v in (("x_center", xc), ("y_center", yc),
                                ("width", w), ("height", h)):
                    if not (0.0 <= v <= 1.0):
                        raise DatasetError(
                            f"{label_path}:{line_no}: {name}={v} out of [0,1]")
                if w <= 0 or h <= 0:
                    raise DatasetError(
                        f"{label_path}:{line_no}: width/height must be > 0")
                boxes.append(GroundTruthBox(
                    cid, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
        elif not allow_missing_labels:
            raise DatasetError(f"missing label for image {image_path.name}")
        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception as e:
            raise DatasetError(f"cannot open image {image_path}: {e}")
        samples.append(Sample(
            image_path, label_path if label_path.exists() else None,
            width, height, _scale_to_pixels(boxes, width, height)))
    return samples


def summarize(samples: list[Sample]) -> dict:
    per_class: dict[int, int] = {}
    for s in samples:
        for b in s.boxes:
            per_class[b.class_id] = per_class.get(b.class_id, 0) + 1
    return {
        "num_images": len(samples),
        "num_boxes": sum(len(s.boxes) for s in samples),
        "per_class": per_class,
        "image_sizes": [(s.width, s.height) for s in samples],
    }


def split_dataset(samples: list[Sample], ratio: float, seed: int = 0) -> tuple[list[Sample], list[Sample]]:
    """Deterministic discovery/confirmation split."""
    import random

    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    if ratio >= 1:
        n_a = len(samples)
    elif ratio <= 0:
        n_a = 0
    else:
        n_a = max(1, int(len(samples) * ratio)) if samples else 0
    return [samples[i] for i in idx[:n_a]], [samples[i] for i in idx[n_a:]]
