"""Evaluation engine: baseline + perturbed evaluation.

Evaluates a detector on a set of samples and returns structured results
including per-image metrics, aggregate corpus metrics, mAP50, and error
records.  Does not store raw Ultralytics result objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from . import metrics as M


# ---------------------------------------------------------------------------
# Error–category tracking (diagnostic only; does not alter TP/FP/FN)
# ---------------------------------------------------------------------------

#: A matched pair whose IoU is below a "localized" threshold but above the
#: matching threshold is flagged as a localization-error candidate.
LOCALIZATION_IOU_THRESH = 0.75


@dataclass
class PerImageResult:
    """Metrics + error records for a single image."""
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
    localization_error: int = 0
    # Per-class breakdown (optional; may be empty if not implemented).
    ap_per_class: dict[int, float] | None = None

class EvaluationResults(dict):
    """Structured results from a baseline (or perturbed) evaluation run.

    Behaves like a dict with keys ``"aggregate"``, ``"map50"``,
    ``"ap_per_class"``, ``"wrong_class_total"``, ``"localization_error_total"``,
    and ``"per_image"``.  Additional keys (e.g. ``"f1"``) are delegated to
    the ``aggregate`` dict.
    """

    def __init__(
        self,
        *,
        per_image: list[PerImageResult] | None = None,
        aggregate: dict[str, int | float] | None = None,
        map50: float = 0.0,
        ap_per_class: dict[int, float] | None = None,
        wrong_class_total: int = 0,
        localization_error_total: int = 0,
        iou_threshold: float = 0.5,
        perturbation: dict | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.per_image = per_image or []
        self.aggregate = aggregate or {"tp": 0, "fp": 0, "fn": 0,
                                        "precision": 0.0, "recall": 0.0,
                                        "f1": 0.0, "num_images": 0}
        self.map50 = map50
        self.ap_per_class = ap_per_class or {}
        self.wrong_class_total = wrong_class_total
        self.localization_error_total = localization_error_total
        self.iou_threshold = iou_threshold
        self.perturbation = perturbation
        self.seed = seed
        # Populate the dict view for backward‑compatible indexing.
        self["aggregate"] = self.aggregate
        self["map50"] = self.map50
        self["ap_per_class"] = self.ap_per_class
        self["wrong_class_total"] = self.wrong_class_total
        self["localization_error_total"] = self.localization_error_total
        self["per_image"] = self.per_image

    def __getitem__(self, key: str) -> Any:
        if key in {
            "aggregate", "map50", "ap_per_class",
            "wrong_class_total", "localization_error_total", "per_image",
        }:
            return super().__getitem__(key)
        # Delegated access: res["f1"] → aggregate["f1"]
        return self.aggregate.get(key)


def _categorize_errors(
    matches: list[tuple[int, int, float]],
    fn: list[int],
    fp_idx: list[int],
    preds: list[dict],
    gt: list[dict],
    iou_thresh: float = 0.5,
    loc_iou_thresh: float = LOCALIZATION_IOU_THRESH,
) -> tuple[int, int]:
    """Return (wrong_class_count, localization_error_count) for one image.

    Diagnostic rules (do not alter TP/FP/FN):

    - *wrong_class*: a prediction in fp_idx has IoU >= iou_thresh with at least
      one unmatched GT of a *different* class.
    - *localization_error*: a matched prediction has IoU >= loc_iou_thresh with
      its matched GT (these are already TP; the flag is informative about
      localisation quality beyond the matching threshold).
    """
    wrong_class = 0
    # Build sets for quick lookup
    matched_gt_ids = {gt_idx for gt_idx in range(len(gt))}
    # Collect unmatched GT class ids
    unmatched_gt_classes: dict[int, list[int]] = {}  # class_id -> [gt_idx]
    for gi in fn:
        g = gt[gi]
        cid = g["class_id"]
        unmatched_gt_classes.setdefault(cid, []).append(gi)

    # Check each FP for wrong-class
    for pi in fp_idx:
        p = preds[pi]
        pb = M._xyxy_of(p)
        has_different_class_gt = False
        for cid, g_indices in unmatched_gt_classes.items():
            if cid == p["class_id"]:
                continue
            for gi in g_indices:
                g = gt[gi]
                iou = M.iou_xyxy(pb, M._xyxy_of(g))
                if iou >= iou_thresh:
                    has_different_class_gt = True
                    break
            if has_different_class_gt:
                break
        if has_different_class_gt:
            wrong_class += 1

    # Check each matched pair for localization error
    loc_error = 0
    for pi, gi, iou in matches:
        if iou >= loc_iou_thresh:
            loc_error += 1

    return wrong_class, loc_error


def evaluate_samples(
    samples,
    detector,
    iou_threshold: float = 0.5,
    perturbation: dict | None = None,
    seed: int = 0,
) -> EvaluationResults:
    """Evaluate detector on *original, unmodified* images.

    Returns an :class:`EvaluationResults` containing per-image metrics,
    aggregate corpus scores, mAP50, and diagnostic error counts.

    Ground-truth used is always the **unperturbed** GT — perturbations
    are only supported through the ``perturbation`` parameter for future
    compatibility; the baseline evaluation always uses original images.
    """
    per_image: list[PerImageResult] = []
    gt_all: list[list[dict]] = []
    pred_all: list[list[dict]] = []

    for s in samples:
        # Run the detector on the original image.
        with (
            Image.open(s.image_path).convert("RGB") as im
        ):
            img = im.copy()
        gt = [dict(b) for b in s.boxes]

        preds = detector.predict(img)  # returns Detections or list[dict] stub

        # Normalise predictions into the dict format expected by metrics.
        if hasattr(preds, "boxes_xyxy"):
            # YOLODetector Detections → list of dicts
            n = len(preds)
            preds_list: list[dict] = []
            for i in range(n):
                preds_list.append(
                    {
                        "class_id": int(preds.classes[i]),
                        "confidence": float(preds.scores[i]),
                        "x_center": preds.classes[i],  # placeholder; not used if confidence dominates
                        "width": 0.0,
                        "height": 0.0,
                    }
                )
            # Actually, Detections has boxes_xyxy, scores, classes – rebuild properly
            preds_list = []
            for i in range(n):
                preds_list.append(
                    {
                        "class_id": int(preds.classes[i]),
                        "confidence": float(preds.scores[i]),
                    }
                )
        elif isinstance(preds, list):
            preds_list = preds
        else:
            preds_list = [preds] if preds else []

        # Match predictions to ground truth.
        m = M.evaluate_image(gt, preds_list, iou_threshold)

        # Diagnostic error categorisation.
        wrong_class, localization_error = _categorize_errors(
            m.get("matches", []),
            m.get("fn_idx", []),
            m.get("fp_idx", []),
            preds_list,
            gt,
            iou_thresh=iou_threshold,
        )

        # Build per-image result – note: wrong_class and localization_error are
        # *diagnostic* and do not change the TP/FP/FN base.
        pir = PerImageResult(
            image=str(s.image_path.name),
            num_gt=len(gt),
            num_pred=len(preds_list),
            tp=m["tp"],
            fp=m["fp"],
            fn=m["fn"],
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            wrong_class=wrong_class,
            localization_error=localization_error,
            ap_per_class=m.get("ap_per_class"),
        )
        per_image.append(pir)
        gt_all.append(gt)
        pred_all.append(preds_list)

    # Aggregate corpus metrics.
    # Convert PerImageResult to dict format that M.aggregate expects.
    results_as_dicts = [
        {
            "tp": p.tp, "fp": p.fp, "fn": p.fn,
            "precision": p.precision, "recall": p.recall, "f1": p.f1,
        }
        for p in per_image
    ]
    agg = M.aggregate(results_as_dicts)  # {tp, fp, fn, precision, recall, f1, num_images}
    cmap = M.corpus_map50(gt_all, pred_all, iou_threshold)

    return EvaluationResults(
        per_image=per_image,
        aggregate=agg,
        map50=cmap["map50"],
        ap_per_class=cmap["ap_per_class"],
        wrong_class_total=sum(p.wrong_class for p in per_image),
        localization_error_total=sum(p.localization_error for p in per_image),
        iou_threshold=iou_threshold,
        perturbation=perturbation,
        seed=seed,
    )


def metric_drop(baseline: dict, perturbed: dict, metric: str = "f1") -> float:
    """Deterioration = baseline - perturbed (higher = more damaging)."""
    return float(baseline[metric] - perturbed[metric])