"""YOLO-format dataset loading and validation.

Never silently discards malformed data: every problem raises a
DatasetError with a clear message. Missing label file => image with
zero ground-truth boxes (valid for background images) only when
`allow_missing_labels` is True; otherwise raises.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .annotations import validate_yolo_row

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetError(Exception):
    pass


@dataclass
class Sample:
    image_path: Path
    label_path: Path | None
    width: int
    height: int
    boxes: list[dict] = field(default_factory=list)


@dataclass
class DatasetSummary:
    num_images: int
    num_boxes: int
    per_class: dict[int, int]
    image_sizes: list[tuple[int, int]]


SPLIT_PRIORITY = ["test", "val", "train"]


def _determine_split(img_name: str) -> str:
    """Determine the split for an image based on its filename prefix.

    Looks for '_train', '_val', or '_test' suffix before the image extension.
    Falls back to 'train' if no marker is found.
    """
    stem = Path(img_name).stem
    for split in SPLIT_PRIORITY:
        if f"_{split}" in stem:
            return split
    return "train"


def _class_names_from_boxes(boxes: list[dict]) -> set[int]:
    """Extract the set of class IDs present in a list of YOLO boxes."""
    return {b["class_id"] for b in boxes if isinstance(b, dict) and "class_id" in b}


def load_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    *,
    allow_missing_labels: bool = True,
    split: str | None = None,
) -> list[Sample]:
    """Load a YOLO-format dataset.

    Parameters:
        images_dir: path to the images subdirectory.
        labels_dir: path to the labels subdirectory.
        allow_missing_labels: if True, images without label files are included
            with zero ground-truth boxes; if False, raises DatasetError.
        split: optional split name ("train", "val", "test"). If not specified,
            the split is inferred from image filenames using the ``_train``,
            ``_val``, ``_test`` marker convention. If inference cannot determine
            the split, "train" is used.

    Returns:
        A list of :class:`Sample` objects, ordered deterministically.

    Raises:
        DatasetError: if images_dir or labels_dir are not directories,
            if an image cannot be opened, or if a label row is malformed.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not images_dir.is_dir():
        raise DatasetError(f"images_dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise DatasetError(f"labels_dir not found: {labels_dir}")

    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_files:
        raise DatasetError(f"no images found in {images_dir}")

    # Determine which images belong to the requested split.
    if split is None:
        split = _determine_split(image_files[0].name)
    elif split not in {"train", "val", "test"}:
        raise DatasetError(f"unknown split '{split}'; must be one of train, val, test")

    samples: list[Sample] = []
    for img_path in image_files:
        # Skip images that do not belong to the requested split.
        if _determine_split(img_path.name) != split:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        boxes: list[dict] = []
        if not label_path.exists():
            if not allow_missing_labels:
                raise DatasetError(
                    f"missing label for image {img_path.name}: {label_path} not found"
                )
            # allow_missing_labels=True => image with zero GT boxes
        else:
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except OSError as e:
                raise DatasetError(f"cannot read {label_path}: {e}")
            for i, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    boxes.append(validate_yolo_row(parts, i, str(label_path)))
                except ValueError as e:
                    raise DatasetError(str(e))
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            raise DatasetError(f"cannot open image {img_path}: {e}")
        samples.append(
            Sample(
                image_path=img_path,
                label_path=label_path if label_path.exists() else None,
                width=w,
                height=h,
                boxes=boxes,
            )
        )
    return samples


def summarize(samples: list[Sample]) -> DatasetSummary:
    per_class: dict[int, int] = {}
    for s in samples:
        for b in s.boxes:
            per_class[b["class_id"]] = per_class.get(b["class_id"], 0) + 1
    return DatasetSummary(
        num_images=len(samples),
        num_boxes=sum(len(s.boxes) for s in samples),
        per_class=per_class,
        image_sizes=[(s.width, s.height) for s in samples],
    )


def split_dataset(samples: list[Sample], ratio: float, seed: int = 0) -> tuple[list[Sample], list[Sample]]:
    """Deterministic split: first ``ratio`` fraction -> part A (discovery).

    The split shuffles indices using ``random.Random(seed)`` and divides
    them so that the first ``ratio`` fraction goes to part A and the
    remainder to part B.
    """
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_a = max(1 if samples else 0, int(len(samples) * ratio)) if 0 < ratio < 1 else len(samples) if ratio >= 1 else 0
    a = [samples[i] for i in idx[:n_a]]
    b = [samples[i] for i in idx[n_a:]]
    return a, b