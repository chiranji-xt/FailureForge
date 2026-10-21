"""Perturbation-evaluation integration: Benchmark Mode, Quick Mode, sweeps.

This is a thin layer over the Day 3 evaluator and the Day 4 perturbation
engine. It never re-parses datasets, re-implements matching/metrics, or changes
the baseline path; instead it feeds pre-loaded ``(sample, bgr)`` pairs into the
evaluator's shared stream and reuses the existing perturbation registry.

Modes
-----
- **Benchmark Mode** (:func:`evaluate_perturbed`): labeled data. Evaluates the
  original dataset, then the same samples with transformed ground truth, and
  reports metric deltas (perturbed - baseline). Metrics come from
  :func:`failureforge.evaluator.evaluate_baseline` (P/R/F1, mAP50).
- **Quick Mode** (:func:`quick_mode`): **no labels required**. Compares clean
  vs perturbed *predictions* per image and reports instability (disappeared,
  newly appearing, class changes, confidence changes, box drift). It does **not**
  compute accuracy metrics, because without ground truth correctness is unknown.

Box conventions: dataset samples already provide **pixel** ``GroundTruthBox``
objects (converted from normalized YOLO rows at load time using real image
dimensions). Transformed boxes returned by the engine are used directly for
geometric/cropping transforms; non-geometric transforms leave them unchanged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

import numpy as np

from .dataset import Sample, YoloDataset
from .detector import Detections
from .evaluator import (
    EvaluationReport,
    _evaluate_stream,
    _load_bgr,
    evaluate_baseline,
)
from .metrics import iou_matrix
from .perturbations import apply, get, validate_params

DEFAULT_IOU = 0.5
DEFAULT_CONF_EPS = 0.01


# ---------------------------------------------------------------------------
# Loading / preparation
# ---------------------------------------------------------------------------

def load_sample_bgr(sample: Sample) -> np.ndarray:
    """Load a sample's image as BGR ``uint8`` ``(H, W, 3)`` (never mutated)."""
    return _load_bgr(sample.image_path)


def prepare_perturbed_sample(
    sample: Sample, name: str, params: dict[str, Any] | None = None, seed: int = 0
) -> tuple[Sample, np.ndarray, dict]:
    """Return ``(transformed_sample, bgr, metadata)`` for one sample.

    The returned ``Sample`` is a new object (``dataclasses.replace``) carrying
    transformed boxes and the transformed image's dimensions; the input sample
    and its boxes are not modified.
    """
    bgr = load_sample_bgr(sample)
    result = apply(name, bgr, params, boxes=sample.boxes, seed=seed)
    height, width = result.image.shape[:2]
    transformed = replace(
        sample, width=int(width), height=int(height), boxes=list(result.boxes or [])
    )
    return transformed, result.image, result.metadata


def iter_perturbed_samples(
    dataset: YoloDataset, name: str, params: dict[str, Any] | None = None, seed: int = 0
) -> Iterator[tuple[Sample, np.ndarray]]:
    """Yield ``(transformed_sample, bgr)`` pairs for a whole dataset."""
    for sample in dataset.iter_samples():
        transformed, bgr, _meta = prepare_perturbed_sample(sample, name, params, seed)
        yield transformed, bgr


# ---------------------------------------------------------------------------
# Benchmark Mode
# ---------------------------------------------------------------------------

@dataclass
class PerturbedResult:
    """Baseline + perturbed evaluation for one transformation configuration."""

    baseline: EvaluationReport
    perturbed: EvaluationReport
    perturbation: dict
    deltas: dict[str, float]
    metadata: dict = field(default_factory=dict)

    def to_dict(self, *, include_baseline: bool = True) -> dict:
        out = {
            "mode": "benchmark",
            "perturbation": self.perturbation,
            "perturbed": self.perturbed.to_dict(),
            "deltas": self.deltas,
            "metadata": self.metadata,
        }
        if include_baseline:
            out["baseline"] = self.baseline.to_dict()
        return out


def _metric_deltas(baseline: EvaluationReport, perturbed: EvaluationReport) -> dict:
    keys = ("tp", "fp", "fn", "precision", "recall", "f1")
    deltas = {
        k: float(perturbed.aggregate.get(k, 0.0) - baseline.aggregate.get(k, 0.0))
        for k in keys
    }
    deltas["map50"] = float(perturbed.map50 - baseline.map50)
    return deltas


def evaluate_perturbed(
    detector: Any,
    dataset: YoloDataset,
    name: str,
    params: dict[str, Any] | None = None,
    *,
    seed: int = 0,
    iou_threshold: float = DEFAULT_IOU,
    baseline: EvaluationReport | None = None,
) -> PerturbedResult:
    """Benchmark Mode: evaluate clean vs perturbed predictions against GT.

    ``baseline`` may be supplied to reuse a previously computed clean report
    (e.g. across sweep trials) so every trial shares identical baseline,
    class mapping, and matching rules.
    """
    transform = get(name)
    normalized = validate_params(name, params)
    base = baseline if baseline is not None else evaluate_baseline(
        detector, dataset, iou_threshold=iou_threshold
    )
    stream = iter_perturbed_samples(dataset, transform.name, normalized, seed)
    perturbed = _evaluate_stream(
        detector, dataset, stream, iou_threshold=iou_threshold
    )
    perturbation = {
        "name": transform.name,
        "family": transform.family,
        "params": normalized,
        "seed": int(seed),
    }
    metadata = {
        "mode": "benchmark",
        "iou_threshold": float(iou_threshold),
        "sample_count": len(base.per_image),
        "dataset": base.metadata.get("dataset"),
        "model": base.metadata.get("model"),
        "class_mapping": base.metadata.get("class_mapping"),
    }
    return PerturbedResult(
        baseline=base,
        perturbed=perturbed,
        perturbation=perturbation,
        deltas=_metric_deltas(base, perturbed),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Quick Mode (no labels)
# ---------------------------------------------------------------------------

def _box_record(boxes, classes, scores, idx: int) -> dict:
    return {
        "index": int(idx),
        "class_id": int(classes[idx]),
        "score": float(scores[idx]),
        "box": [float(x) for x in boxes[idx]],
    }


def greedy_location_matches(
    clean: Detections, perturbed: Detections, iou_threshold: float = DEFAULT_IOU
) -> list[tuple[int, int, float]]:
    """Class-agnostic one-to-one IoU matching (greedy, deterministic).

    Matches by location so a class change at the same box is reported as a
    class change rather than a separate disappearance + appearance.
    """
    n, m = len(clean), len(perturbed)
    if n == 0 or m == 0:
        return []
    iou = iou_matrix(clean.boxes_xyxy, perturbed.boxes_xyxy)
    candidates = [
        (float(iou[i, j]), float(clean.scores[i]), i, j)
        for i in range(n) for j in range(m)
        if iou[i, j] >= iou_threshold
    ]
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
    used_clean: set[int] = set()
    used_pert: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for value, _score, i, j in candidates:
        if i in used_clean or j in used_pert:
            continue
        used_clean.add(i)
        used_pert.add(j)
        matches.append((i, j, value))
    return matches


def compare_predictions(
    clean: Detections,
    perturbed: Detections,
    iou_threshold: float = DEFAULT_IOU,
) -> dict:
    """Compare clean vs perturbed predictions for one image (location matching)."""
    matches = greedy_location_matches(clean, perturbed, iou_threshold)
    matched_clean = {i for i, _, _ in matches}
    matched_pert = {j for _, j, _ in matches}

    disappeared = [
        _box_record(clean.boxes_xyxy, clean.classes, clean.scores, i)
        for i in range(len(clean)) if i not in matched_clean
    ]
    appeared = [
        _box_record(perturbed.boxes_xyxy, perturbed.classes, perturbed.scores, j)
        for j in range(len(perturbed)) if j not in matched_pert
    ]

    class_changed: list[dict] = []
    confidence_changes: list[dict] = []
    box_drift: list[dict] = []
    for i, j, iou in matches:
        from_class = int(clean.classes[i])
        to_class = int(perturbed.classes[j])
        if from_class != to_class:
            class_changed.append({
                "clean_index": i, "perturbed_index": j,
                "from_class": from_class, "to_class": to_class, "iou": iou,
                "clean_box": [float(x) for x in clean.boxes_xyxy[i]],
                "perturbed_box": [float(x) for x in perturbed.boxes_xyxy[j]],
                "clean_score": float(clean.scores[i]),
                "perturbed_score": float(perturbed.scores[j]),
            })
            continue
        confidence_changes.append({
            "clean_index": i, "perturbed_index": j, "class_id": from_class,
            "delta": float(perturbed.scores[j] - clean.scores[i]), "iou": iou,
            "clean_box": [float(x) for x in clean.boxes_xyxy[i]],
            "perturbed_box": [float(x) for x in perturbed.boxes_xyxy[j]],
            "clean_score": float(clean.scores[i]),
            "perturbed_score": float(perturbed.scores[j]),
        })
        box_drift.append({
            "clean_index": i, "perturbed_index": j, "class_id": from_class,
            "iou": iou, "drift": float(1.0 - iou),
            "clean_box": [float(x) for x in clean.boxes_xyxy[i]],
            "perturbed_box": [float(x) for x in perturbed.boxes_xyxy[j]],
            "clean_score": float(clean.scores[i]),
            "perturbed_score": float(perturbed.scores[j]),
        })

    return {
        "matches": len(matches),
        "disappeared": disappeared,
        "appeared": appeared,
        "class_changed": class_changed,
        "confidence_changes": confidence_changes,
        "box_drift": box_drift,
    }


@dataclass
class QuickModeResult:
    """Prediction-instability report for one transformation configuration.

    Contains **no** precision/recall/F1/mAP: without labels correctness cannot
    be determined, so only instability indicators are reported.
    """

    perturbation: dict
    iou_threshold: float
    per_image: list[dict]
    summary: dict
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": "quick",
            "perturbation": self.perturbation,
            "iou_threshold": self.iou_threshold,
            "per_image": self.per_image,
            "summary": self.summary,
            "metadata": self.metadata,
            "note": "Instability only; no ground-truth accuracy metrics are reported.",
        }


def quick_mode(
    detector: Any,
    dataset: YoloDataset,
    name: str,
    params: dict[str, Any] | None = None,
    *,
    seed: int = 0,
    iou_threshold: float = DEFAULT_IOU,
    confidence_epsilon: float = DEFAULT_CONF_EPS,
) -> QuickModeResult:
    """Quick Mode: clean vs perturbed predictions, **without reading labels**."""
    transform = get(name)
    normalized = validate_params(name, params)

    per_image: list[dict] = []
    totals = {k: 0 for k in
              ("clean_detections", "perturbed_detections", "matched", "disappeared",
               "appeared", "class_changed", "confidence_changed")}
    confidence_deltas: list[float] = []
    match_ious: list[float] = []

    for image_path in dataset.image_paths:
        clean_bgr = _load_bgr(image_path)
        clean = detector.predict(clean_bgr)
        transformed_bgr = apply(transform.name, clean_bgr, normalized, boxes=None,
                                seed=seed).image
        perturbed = detector.predict(transformed_bgr)

        cmp = compare_predictions(clean, perturbed, iou_threshold)
        changed_conf = [c for c in cmp["confidence_changes"]
                        if abs(c["delta"]) > confidence_epsilon]
        confidence_deltas.extend(c["delta"] for c in cmp["confidence_changes"])
        match_ious.extend(d["iou"] for d in cmp["box_drift"])

        per_image.append({
            "image": image_path.name,
            "num_clean": len(clean),
            "num_perturbed": len(perturbed),
            **cmp,
            "confidence_changed": len(changed_conf),
        })
        totals["clean_detections"] += len(clean)
        totals["perturbed_detections"] += len(perturbed)
        totals["matched"] += cmp["matches"]
        totals["disappeared"] += len(cmp["disappeared"])
        totals["appeared"] += len(cmp["appeared"])
        totals["class_changed"] += len(cmp["class_changed"])
        totals["confidence_changed"] += len(changed_conf)

    summary = {
        "num_images": len(per_image),
        **totals,
        "mean_confidence_delta": (
            float(np.mean(confidence_deltas)) if confidence_deltas else 0.0
        ),
        "mean_box_iou": float(np.mean(match_ious)) if match_ious else 0.0,
        "changed_detection_fraction": (
            (totals["disappeared"] + totals["appeared"] + totals["class_changed"])
            / totals["clean_detections"] if totals["clean_detections"] else 0.0
        ),
        "note": "Instability only; no ground-truth accuracy metrics are reported.",
    }

    perturbation = {
        "name": transform.name,
        "family": transform.family,
        "params": normalized,
        "seed": int(seed),
    }
    metadata = {
        "mode": "quick",
        "iou_threshold": float(iou_threshold),
        "confidence_epsilon": float(confidence_epsilon),
        "dataset": dataset.metadata,
        "model": {
            "weights": getattr(detector, "weights", None),
            "sha256": getattr(detector, "sha256", None),
            "class_names": list(getattr(detector, "class_names", []) or []),
        },
    }
    return QuickModeResult(
        perturbation=perturbation,
        iou_threshold=float(iou_threshold),
        per_image=per_image,
        summary=summary,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Severity sweeps (foundation for later search stages)
# ---------------------------------------------------------------------------

def derive_trial_seed(
    base_seed: int, index: int, name: str, params: dict[str, Any]
) -> int:
    """Deterministic, platform-stable per-trial seed."""
    payload = f"{int(base_seed)}|{int(index)}|{name}|{json.dumps(params, sort_keys=True, default=str)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


@dataclass
class SweepTrial:
    index: int
    config: dict
    result: Any | None
    error: str | None = None

    def to_dict(self) -> dict:
        if self.result is None:
            payload = None
        elif isinstance(self.result, PerturbedResult):
            payload = self.result.to_dict(include_baseline=False)
        else:
            payload = self.result.to_dict()
        return {
            "index": self.index,
            "config": self.config,
            "result": payload,
            "error": self.error,
        }


@dataclass
class SweepResult:
    mode: str
    name: str
    base_seed: int
    trials: list[SweepTrial]
    baseline: EvaluationReport | None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "name": self.name,
            "base_seed": self.base_seed,
            "baseline": self.baseline.to_dict() if self.baseline is not None else None,
            "trials": [t.to_dict() for t in self.trials],
            "metadata": self.metadata,
        }


def run_severity_sweep(
    detector: Any,
    dataset: YoloDataset,
    name: str,
    configs: list[dict[str, Any]],
    *,
    seed: int = 0,
    iou_threshold: float = DEFAULT_IOU,
    mode: str = "benchmark",
    confidence_epsilon: float = DEFAULT_CONF_EPS,
    strict: bool = False,
) -> SweepResult:
    """Evaluate explicit parameter configurations of one transformation.

    ``configs`` are caller-chosen parameter dicts (no combinatorial expansion).
    Per-trial seeds are derived deterministically from ``seed`` + index + name +
    params, so any single trial is independently reproducible. A failing trial
    is recorded with its error (or re-raised when ``strict=True``) without
    corrupting other trials.
    """
    if mode not in ("benchmark", "quick"):
        raise ValueError(f"mode must be 'benchmark' or 'quick', got {mode!r}")
    transform = get(name)
    baseline = (
        evaluate_baseline(detector, dataset, iou_threshold=iou_threshold)
        if mode == "benchmark" else None
    )

    trials: list[SweepTrial] = []
    for index, config in enumerate(configs):
        params = dict(config or {})
        trial_seed = derive_trial_seed(seed, index, transform.name, params)
        trial_config = {"name": transform.name, "params": params, "seed": trial_seed}
        try:
            if mode == "benchmark":
                result = evaluate_perturbed(
                    detector, dataset, transform.name, params, seed=trial_seed,
                    iou_threshold=iou_threshold, baseline=baseline,
                )
            else:
                result = quick_mode(
                    detector, dataset, transform.name, params, seed=trial_seed,
                    iou_threshold=iou_threshold,
                    confidence_epsilon=confidence_epsilon,
                )
            trials.append(SweepTrial(index=index, config=trial_config, result=result))
        except Exception as exc:  # attributable per-trial failure
            if strict:
                raise
            trials.append(SweepTrial(
                index=index, config=trial_config, result=None,
                error=f"{type(exc).__name__}: {exc}",
            ))

    metadata = {
        "mode": mode,
        "transform_family": transform.family,
        "dataset": dataset.metadata if mode == "benchmark" else None,
        "model": {
            "weights": getattr(detector, "weights", None),
            "sha256": getattr(detector, "sha256", None),
        },
        "iou_threshold": float(iou_threshold),
        "base_seed": int(seed),
    }
    return SweepResult(
        mode=mode, name=transform.name, base_seed=int(seed),
        trials=trials, baseline=baseline, metadata=metadata,
    )
