"""Baseline evaluation: ground-truth matching and detection metrics.

The public entry point is :func:`evaluate_baseline`, which connects a
:class:`~failureforge.dataset.YoloDataset` and a detector exposing
``class_names`` plus ``predict(bgr_ndarray) -> Detections`` (as implemented by
``failureforge.detector.YOLODetector``).

Evaluation uses **original, unmodified images**. No perturbations or search
logic lives here; robustness testing belongs to later stages.

Conventions
-----------
- Boxes are pixel ``[x1, y1, x2, y2]``.
- Matching: class-aware, one-to-one, greedy by descending confidence, IoU ≥
  ``iou_threshold`` (default 0.50). See :mod:`failureforge.metrics`.
- Corpus precision/recall/F1 are micro-averaged from summed TP/FP/FN counts.
- mAP50 is pooled per-class all-point AP at IoU 0.50, averaged over classes
  that occur in the ground truth.
- Zero denominators yield ``0.0`` for precision/recall/F1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from . import metrics as M
from .dataset import GroundTruthBox, Sample, YoloDataset

AGGREGATION_METHOD = "micro-average over summed TP/FP/FN across images"


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class PerImageResult:
    """Metrics and error details for one image."""

    image: str
    num_gt: int
    num_pred: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    wrong_class: int = 0
    poor_localization: int = 0
    matches: list[dict] = field(default_factory=list)
    missed: list[dict] = field(default_factory=list)
    false_positives: list[dict] = field(default_factory=list)
    iou_threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "image": self.image, "num_gt": self.num_gt, "num_pred": self.num_pred,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "wrong_class": self.wrong_class,
            "poor_localization": self.poor_localization,
            "matches": self.matches, "missed": self.missed,
            "false_positives": self.false_positives,
            "iou_threshold": self.iou_threshold,
        }


@dataclass
class ClassMetrics:
    """Per-class counts and scores."""

    class_id: int
    name: str
    gt_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    ap50: float | None

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id, "name": self.name,
            "gt_count": self.gt_count, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "ap50": self.ap50,
        }


@dataclass
class EvaluationReport:
    """Structured result of a baseline evaluation."""

    aggregate: dict
    per_class: dict[int, ClassMetrics]
    per_image: list[PerImageResult]
    map50: float
    error_records: list[dict]
    metadata: dict
    conf_threshold: float
    iou_threshold: float
    split: str
    aggregation_method: str = AGGREGATION_METHOD

    def to_dict(self) -> dict:
        return {
            "aggregate": self.aggregate,
            "per_class": {str(k): v.to_dict() for k, v in self.per_class.items()},
            "per_image": [p.to_dict() for p in self.per_image],
            "map50": self.map50,
            "error_records": self.error_records,
            "metadata": self.metadata,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "split": self.split,
            "aggregation_method": self.aggregation_method,
        }

    def __getitem__(self, key: str) -> Any:
        if key in {"aggregate", "per_class", "per_image", "map50",
                   "error_records", "metadata", "conf_threshold",
                   "iou_threshold", "split", "aggregation_method"}:
            return getattr(self, key)
        return self.aggregate.get(key)


# ---------------------------------------------------------------------------
# Adaptation helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _detections_to_arrays(preds: Any, width: int | None = None,
                          height: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert detector output to (boxes_xyxy, classes, scores) arrays.

    ``width``/``height`` are required to scale legacy *normalized* dict
    predictions into pixel coordinates; they are ignored for ``Detections``,
    which are already in pixels.
    """
    if hasattr(preds, "boxes_xyxy"):  # Detections
        boxes = np.asarray(preds.boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        classes = np.asarray(preds.classes, dtype=np.int64).reshape(-1)
        scores = np.asarray(preds.scores, dtype=np.float64).reshape(-1)
        return boxes, classes, scores
    if isinstance(preds, list):  # legacy normalized dicts
        if not preds:
            return np.zeros((0, 4)), np.zeros((0,), np.int64), np.zeros((0,))
        sx = width if width is not None else 1.0
        sy = height if height is not None else 1.0
        boxes = np.array([[
            (p["x_center"] - p["width"] / 2) * sx,
            (p["y_center"] - p["height"] / 2) * sy,
            (p["x_center"] + p["width"] / 2) * sx,
            (p["y_center"] + p["height"] / 2) * sy,
        ] for p in preds], dtype=np.float64)
        classes = np.array([int(p["class_id"]) for p in preds], dtype=np.int64)
        scores = np.array([float(p.get("confidence", 1.0)) for p in preds], dtype=np.float64)
        return boxes, classes, scores
    raise TypeError(f"unsupported detector output type: {type(preds).__name__}")


def _gt_arrays(boxes: list[GroundTruthBox]) -> tuple[np.ndarray, np.ndarray]:
    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,), np.int64)
    return (np.array([b.as_tuple() for b in boxes], dtype=np.float64),
            np.array([b.class_id for b in boxes], dtype=np.int64))


def _map_classes(classes: np.ndarray, model_to_dataset: dict[int, int]) -> np.ndarray:
    return np.array([model_to_dataset.get(int(c), -1) for c in classes], dtype=np.int64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_baseline(
    detector: Any,
    dataset: YoloDataset,
    *,
    iou_threshold: float = 0.5,
) -> EvaluationReport:
    """Evaluate ``detector`` on ``dataset`` using original images.

    Raises:
        ValueError: if the detector exposes no class names or the dataset/model
            class vocabularies cannot be reconciled. Evaluation never proceeds
            with an unverified class mapping.
    """
    model_names = list(getattr(detector, "class_names", []) or [])
    if not model_names:
        raise ValueError(
            "detector exposes no class_names; a verified class mapping is "
            "required before evaluation"
        )
    dataset_to_model = dataset.class_mapping(model_names)  # validates names
    model_to_dataset = {m: d for d, m in dataset_to_model.items()}

    conf_threshold = float(getattr(detector, "conf", 0.0))

    per_image: list[PerImageResult] = []
    image_inputs: list[M.ImageDetections] = []
    per_class_counts: dict[int, dict[str, int]] = {}
    error_records: list[dict] = []

    def _counts(cid: int) -> dict[str, int]:
        return per_class_counts.setdefault(cid, {"gt": 0, "tp": 0, "fp": 0, "fn": 0})

    for sample in dataset.iter_samples():
        with Image.open(sample.image_path) as im:
            bgr = _pil_to_bgr(im)
        preds = detector.predict(bgr)
        pred_boxes, pred_classes, pred_scores = _detections_to_arrays(preds)
        pred_classes = _map_classes(pred_classes, model_to_dataset)

        gt_boxes, gt_classes = _gt_arrays(sample.boxes)
        res = M.match_boxes(gt_boxes, gt_classes, pred_boxes, pred_classes,
                            pred_scores, iou_threshold)

        for b in sample.boxes:
            _counts(b.class_id)["gt"] += 1
        for _pi, gi, _iou in res.matches:
            _counts(int(gt_classes[gi]))["tp"] += 1
        for pi in res.fp_idx:
            cid = int(pred_classes[pi])
            _counts(cid)["fp"] += 1
        for gi in res.fn_idx:
            _counts(int(gt_classes[gi]))["fn"] += 1

        matches_detail = [{
            "gt_index": gi, "pred_index": pi, "iou": round(v, 6),
            "class_id": int(gt_classes[gi]),
            "class_name": dataset.class_name(int(gt_classes[gi])),
            "score": float(pred_scores[pi]),
        } for pi, gi, v in res.matches]

        missed = [{
            "gt_index": gi, "class_id": int(gt_classes[gi]),
            "class_name": dataset.class_name(int(gt_classes[gi])),
            "box": [round(float(x), 3) for x in gt_boxes[gi]],
        } for gi in res.fn_idx]

        false_positives = []
        for pi in res.fp_idx:
            cid = int(pred_classes[pi])
            diagnostic = None
            if cid >= 0:
                overlap_other = any(
                    int(gt_classes[gi]) != cid
                    and M.iou_xyxy(tuple(pred_boxes[pi]), tuple(gt_boxes[gi])) >= iou_threshold
                    for gi in res.fn_idx
                )
                if overlap_other:
                    diagnostic = "wrong_class"
                else:
                    best_same = max(
                        (M.iou_xyxy(tuple(pred_boxes[pi]), tuple(gt_boxes[gi]))
                         for gi in res.fn_idx if int(gt_classes[gi]) == cid),
                        default=0.0,
                    )
                    if 0.0 < best_same < iou_threshold:
                        diagnostic = "poor_localization"
            false_positives.append({
                "pred_index": pi, "class_id": cid,
                "class_name": dataset.class_name(cid) if cid >= 0 else "<not_in_dataset>",
                "score": float(pred_scores[pi]),
                "box": [round(float(x), 3) for x in pred_boxes[pi]],
                "diagnostic": diagnostic,
            })

        precision, recall, f1 = M.prf1(res.tp, res.fp, res.fn)
        per_image.append(PerImageResult(
            image=sample.image_path.name,
            num_gt=len(gt_boxes), num_pred=len(pred_boxes),
            tp=res.tp, fp=res.fp, fn=res.fn,
            precision=precision, recall=recall, f1=f1,
            wrong_class=res.wrong_class,
            poor_localization=res.poor_localization,
            matches=matches_detail, missed=missed,
            false_positives=false_positives,
            iou_threshold=iou_threshold,
        ))

        for rec in missed:
            error_records.append({"image": sample.image_path.name,
                                  "type": "missed", **rec})
        for rec in false_positives:
            error_records.append({"image": sample.image_path.name,
                                  "type": rec["diagnostic"] or "false_positive",
                                  **rec})
        for rec in matches_detail:
            if rec["iou"] < M.LOCALIZATION_IOU_THRESH:
                error_records.append({"image": sample.image_path.name,
                                      "type": "poor_localization", **rec})

        image_inputs.append(M.ImageDetections(
            gt_boxes=gt_boxes, gt_classes=gt_classes,
            pred_boxes=pred_boxes, pred_classes=pred_classes,
            pred_scores=pred_scores,
        ))

    aggregate = M.aggregate_counts([
        {"tp": p.tp, "fp": p.fp, "fn": p.fn} for p in per_image
    ])
    map_result = M.compute_map50(image_inputs, iou_threshold)
    ap_per_class = map_result["ap_per_class"]

    per_class: dict[int, ClassMetrics] = {}
    for cid in sorted(per_class_counts):
        c = per_class_counts[cid]
        precision, recall, f1 = M.prf1(c["tp"], c["fp"], c["fn"])
        ap = ap_per_class.get(cid)
        per_class[cid] = ClassMetrics(
            class_id=cid, name=dataset.class_name(cid),
            gt_count=c["gt"], tp=c["tp"], fp=c["fp"], fn=c["fn"],
            precision=precision, recall=recall, f1=f1,
            ap50=float(ap) if ap is not None and ap == ap else None,
        )

    metadata = {
        "dataset": dataset.metadata,
        "model": {
            "weights": getattr(detector, "weights", None),
            "sha256": getattr(detector, "sha256", None),
            "task": getattr(detector, "task", "detect"),
            "imgsz": getattr(detector, "imgsz", None),
            "device": getattr(detector, "device", None),
            "class_names": model_names,
        },
        "class_mapping": {dataset.class_name(d): m for d, m in dataset_to_model.items()},
        "aggregation": AGGREGATION_METHOD,
        "map50_convention": "per-class all-point AP at IoU 0.50, pooled by confidence",
    }

    return EvaluationReport(
        aggregate=aggregate,
        per_class=per_class,
        per_image=per_image,
        map50=float(map_result["map50"]),
        error_records=error_records,
        metadata=metadata,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        split=dataset.split,
    )


# ---------------------------------------------------------------------------
# Legacy compatibility wrappers (used by earlier preview modules)
# ---------------------------------------------------------------------------

class EvaluationResults(dict):
    """Dict-like legacy result wrapper."""

    def __init__(self, *, per_image=None, aggregate=None, map50=0.0,
                 ap_per_class=None, wrong_class_total=0,
                 localization_error_total=0, iou_threshold=0.5,
                 perturbation=None, seed=0) -> None:
        super().__init__()
        self.per_image = per_image or []
        self.aggregate = aggregate or {
            "tp": 0, "fp": 0, "fn": 0, "precision": 0.0,
            "recall": 0.0, "f1": 0.0, "num_images": 0,
        }
        self.map50 = map50
        self.ap_per_class = ap_per_class or {}
        self.wrong_class_total = wrong_class_total
        self.localization_error_total = localization_error_total
        self.iou_threshold = iou_threshold
        self.perturbation = perturbation
        self.seed = seed
        for key in ("aggregate", "map50", "ap_per_class", "wrong_class_total",
                    "localization_error_total", "per_image"):
            self[key] = getattr(self, key)

    def __getitem__(self, key: str) -> Any:
        if key in self.keys():
            return super().__getitem__(key)
        return self.aggregate.get(key)


def _normalized_boxes_from_pixel(boxes: list[GroundTruthBox],
                                 width: int, height: int) -> list[dict]:
    return [{
        "class_id": b.class_id,
        "x_center": ((b.x1 + b.x2) / 2) / width,
        "y_center": ((b.y1 + b.y2) / 2) / height,
        "width": (b.x2 - b.x1) / width,
        "height": (b.y2 - b.y1) / height,
    } for b in boxes]


def _pixel_from_normalized(boxes: list[dict], width: int, height: int) -> list[GroundTruthBox]:
    return [GroundTruthBox(
        class_id=int(b["class_id"]),
        x1=(b["x_center"] - b["width"] / 2) * width,
        y1=(b["y_center"] - b["height"] / 2) * height,
        x2=(b["x_center"] + b["width"] / 2) * width,
        y2=(b["y_center"] + b["height"] / 2) * height,
    ) for b in boxes]


def evaluate_samples(samples: list[Sample], detector: Any,
                     iou_threshold: float = 0.5,
                     perturbation: dict | None = None,
                     seed: int = 0) -> EvaluationResults:
    """Legacy evaluator for a list of samples with optional perturbation.

    Kept for compatibility with earlier preview modules. New code should use
    :func:`evaluate_baseline`.
    """
    per_image: list[PerImageResult] = []
    image_inputs: list[M.ImageDetections] = []

    for sample in samples:
        with Image.open(sample.image_path) as im:
            image = im.copy()

        gt_boxes_list = list(sample.boxes)
        if perturbation is not None:
            from .transforms import apply_perturbation

            normalized = _normalized_boxes_from_pixel(
                gt_boxes_list, sample.width, sample.height)
            image, normalized, _info = apply_perturbation(
                image, normalized, perturbation["name"],
                perturbation.get("params", {}),
                seed=perturbation.get("seed", seed))
            gt_boxes_list = _pixel_from_normalized(
                normalized, sample.width, sample.height)

        preds = detector.predict(image)
        pred_boxes, pred_classes, pred_scores = _detections_to_arrays(
            preds, sample.width, sample.height)
        gt_boxes, gt_classes = _gt_arrays(gt_boxes_list)

        res = M.match_boxes(gt_boxes, gt_classes, pred_boxes, pred_classes,
                            pred_scores, iou_threshold)
        precision, recall, f1 = M.prf1(res.tp, res.fp, res.fn)
        per_image.append(PerImageResult(
            image=sample.image_path.name,
            num_gt=len(gt_boxes), num_pred=len(pred_boxes),
            tp=res.tp, fp=res.fp, fn=res.fn,
            precision=precision, recall=recall, f1=f1,
            wrong_class=res.wrong_class,
            poor_localization=res.poor_localization,
            matches=[{"gt_index": gi, "pred_index": pi, "iou": v}
                     for pi, gi, v in res.matches],
            iou_threshold=iou_threshold,
        ))
        image_inputs.append(M.ImageDetections(
            gt_boxes=gt_boxes, gt_classes=gt_classes,
            pred_boxes=pred_boxes, pred_classes=pred_classes,
            pred_scores=pred_scores,
        ))

    aggregate = M.aggregate_counts([
        {"tp": p.tp, "fp": p.fp, "fn": p.fn} for p in per_image
    ])
    map_result = M.compute_map50(image_inputs, iou_threshold)
    return EvaluationResults(
        per_image=per_image,
        aggregate=aggregate,
        map50=float(map_result["map50"]),
        ap_per_class=map_result["ap_per_class"],
        wrong_class_total=sum(p.wrong_class for p in per_image),
        localization_error_total=sum(p.poor_localization for p in per_image),
        iou_threshold=iou_threshold,
        perturbation=perturbation,
        seed=seed,
    )


def metric_drop(baseline: dict, perturbed: dict, metric: str = "f1") -> float:
    """Deterioration = baseline - perturbed (higher = more damaging)."""
    return float(baseline[metric] - perturbed[metric])
