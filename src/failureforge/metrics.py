"""Detection metrics: IoU matching, TP/FP/FN, per-class scores, and mAP50.

Canonical (equation) boxes for the modern API are pixel ``[x1, y1, x2, y2]``
NumPy arrays. The legacy helpers near the bottom operate on normalized YOLO
dicts for backwards compatibility with earlier preview code.

Matching policy
---------------
For each image, predictions are processed in descending confidence order
(ties broken by original index, so the result is deterministic). Each
prediction is matched to the *unmatched* ground-truth box of the **same class**
with the highest IoU ≥ ``iou_threshold`` (ties → lowest GT index). A GT box and
a prediction can each participate in at most one match. Unmatched predictions
are false positives; unmatched GT boxes are false negatives. Wrong-class
predictions therefore never count as true positives.

Diagnostics
-----------
``wrong_class`` and ``poor_localization`` are **diagnostic counts only** and
never added to TP/FP/FN:

- ``wrong_class``: an unmatched prediction overlaps an unmatched GT box of a
  different class with IoU ≥ threshold.
- ``poor_localization``: a matched pair whose IoU is in
  ``[iou_threshold, localization_threshold)``, or an unmatched prediction whose
  best same-class IoU is in ``(0, iou_threshold)``.

Metric conventions
------------------
- Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 = harmonic mean.
  All return ``0.0`` when their denominator is zero (documented convention).
- Corpus precision/recall/F1 are **micro-averaged**: counts are summed over all
  images first, then the ratio is computed. Per-image F1 values are never
  averaged and presented as the corpus score.
- mAP50 is computed from confidence-ranked detections with all-point
  interpolated average precision at IoU 0.50, computed per class over the whole
  corpus and averaged over classes that have ground-truth objects. It is not a
  threshold score and is not interchangeable with F1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LOCALIZATION_IOU_THRESH = 0.75


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(A, 4)`` and ``(B, 4)`` pixel xyxy boxes."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def iou_xyxy(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> float:
    """IoU between two pixel xyxy boxes (legacy scalar helper)."""
    return float(iou_matrix(np.array([a]), np.array([b]))[0, 0])


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """One image's matching outcome."""

    tp: int
    fp: int
    fn: int
    tp_idx: list[int] = field(default_factory=list)
    fp_idx: list[int] = field(default_factory=list)
    fn_idx: list[int] = field(default_factory=list)
    matches: list[tuple[int, int, float]] = field(default_factory=list)
    wrong_class: int = 0
    poor_localization: int = 0


def match_boxes(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float = 0.5,
    localization_threshold: float = LOCALIZATION_IOU_THRESH,
) -> MatchResult:
    """Deterministic class-aware one-to-one matching. See module docstring."""
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    gt_classes = np.asarray(gt_classes, dtype=np.int64).reshape(-1)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    pred_classes = np.asarray(pred_classes, dtype=np.int64).reshape(-1)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)

    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    if len(gt_classes) != n_gt or len(pred_classes) != n_pred or len(pred_scores) != n_pred:
        raise ValueError("inconsistent box/class/score array lengths")

    ious = iou_matrix(pred_boxes, gt_boxes)  # (P, G)
    order = sorted(range(n_pred), key=lambda i: (-pred_scores[i], i))

    gt_used = np.zeros(n_gt, dtype=bool)
    tp_idx: list[int] = []
    fp_idx: list[int] = []
    matches: list[tuple[int, int, float]] = []

    for pi in order:
        best_gi = -1
        best_iou = 0.0
        for gi in range(n_gt):
            if gt_used[gi] or gt_classes[gi] != pred_classes[pi]:
                continue
            v = float(ious[pi, gi])
            if v > best_iou or (v == best_iou and v > 0 and (best_gi < 0 or gi < best_gi)):
                best_gi, best_iou = gi, v
        if best_gi >= 0 and best_iou >= iou_threshold:
            gt_used[best_gi] = True
            tp_idx.append(pi)
            matches.append((pi, best_gi, best_iou))
        else:
            fp_idx.append(pi)

    fn_idx = [gi for gi in range(n_gt) if not gt_used[gi]]

    # Diagnostics (never alter TP/FP/FN).
    wrong_class = 0
    for pi in fp_idx:
        for gi in fn_idx:
            if gt_classes[gi] != pred_classes[pi] and ious[pi, gi] >= iou_threshold:
                wrong_class += 1
                break

    poor_localization = 0
    for _pi, _gi, v in matches:
        if iou_threshold <= v < localization_threshold:
            poor_localization += 1
    for pi in fp_idx:
        same = [float(ious[pi, gi]) for gi in fn_idx
                if gt_classes[gi] == pred_classes[pi]]
        best_same = max(same, default=0.0)
        if 0.0 < best_same < iou_threshold:
            poor_localization += 1

    return MatchResult(
        tp=len(tp_idx), fp=len(fp_idx), fn=len(fn_idx),
        tp_idx=tp_idx, fp_idx=fp_idx, fn_idx=fn_idx,
        matches=matches, wrong_class=wrong_class,
        poor_localization=poor_localization,
    )


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision/recall/F1 with zero-denominator → 0.0."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def aggregate_counts(counts: list[dict]) -> dict:
    """Micro-average TP/FP/FN across images, then compute P/R/F1."""
    tp = sum(c["tp"] for c in counts)
    fp = sum(c["fp"] for c in counts)
    fn = sum(c["fn"] for c in counts)
    precision, recall, f1 = prf1(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "num_images": len(counts),
    }


def average_precision(tp_flags: list[bool], gt_count: int) -> float:
    """All-point interpolated AP from confidence-ordered TP flags."""
    if gt_count <= 0:
        return float("nan")
    if not tp_flags:
        return 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    tp_cum = 0
    for i, is_tp in enumerate(tp_flags, start=1):
        if is_tp:
            tp_cum += 1
        precisions.append(tp_cum / i)
        recalls.append(tp_cum / gt_count)
    mrec = [0.0, *recalls, recalls[-1]]
    mpre = [0.0, *precisions, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] != mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


@dataclass
class ImageDetections:
    """Input to corpus mAP: one image's GT and predictions (pixel xyxy)."""

    gt_boxes: np.ndarray
    gt_classes: np.ndarray
    pred_boxes: np.ndarray
    pred_classes: np.ndarray
    pred_scores: np.ndarray


def compute_map50(images: list[ImageDetections], iou_threshold: float = 0.5) -> dict:
    """Corpus mAP@50 with pooled, confidence-ranked per-class AP.

    Returns ``{"map50": float, "ap_per_class": {class_id: ap}}``. Classes with
    no ground-truth objects are excluded from the mean.
    """
    classes = sorted({int(c) for im in images for c in np.asarray(im.gt_classes).reshape(-1)})
    ap_per_class: dict[int, float] = {}
    for cid in classes:
        gt_count = 0
        pooled: list[tuple[float, int, bool]] = []  # (score, image_idx, is_tp)
        for img_idx, im in enumerate(images):
            gt_boxes = np.asarray(im.gt_boxes, dtype=np.float64).reshape(-1, 4)
            gt_cls = np.asarray(im.gt_classes, dtype=np.int64).reshape(-1)
            pr_boxes = np.asarray(im.pred_boxes, dtype=np.float64).reshape(-1, 4)
            pr_cls = np.asarray(im.pred_classes, dtype=np.int64).reshape(-1)
            pr_scores = np.asarray(im.pred_scores, dtype=np.float64).reshape(-1)

            gt_sel = gt_cls == cid
            pr_sel = pr_cls == cid
            g_boxes, p_boxes = gt_boxes[gt_sel], pr_boxes[pr_sel]
            p_scores = pr_scores[pr_sel]
            gt_count += len(g_boxes)
            if len(p_boxes) == 0:
                continue
            res = match_boxes(
                g_boxes, np.full(len(g_boxes), cid, dtype=np.int64),
                p_boxes, np.full(len(p_boxes), cid, dtype=np.int64),
                p_scores, iou_threshold,
            )
            tp_set = set(res.tp_idx)
            for pi, score in enumerate(p_scores):
                pooled.append((float(score), img_idx, pi in tp_set))
        # Confidence-ranked, deterministic tie-break.
        pooled.sort(key=lambda t: (-t[0], t[1]))
        flags = [t[2] for t in pooled]
        ap_per_class[cid] = average_precision(flags, gt_count)
    valid = [v for v in ap_per_class.values() if v == v]
    return {
        "map50": float(sum(valid) / len(valid)) if valid else 0.0,
        "ap_per_class": ap_per_class,
    }


# ---------------------------------------------------------------------------
# Legacy normalized-dict helpers (kept for earlier preview modules/tests)
# ---------------------------------------------------------------------------

def _xyxy_of(box: dict) -> tuple[float, float, float, float]:
    return (
        box["x_center"] - box["width"] / 2,
        box["y_center"] - box["height"] / 2,
        box["x_center"] + box["width"] / 2,
        box["y_center"] + box["height"] / 2,
    )


def match_predictions(gt: list[dict], preds: list[dict], iou_threshold: float = 0.5) -> dict:
    """Legacy normalized-dict matching (see module-level policy)."""
    order = sorted(range(len(preds)), key=lambda i: preds[i].get("confidence", 1.0), reverse=True)
    matched_gt: set[int] = set()
    tp, fp, matches = [], [], []
    for pi in order:
        p = preds[pi]
        best, best_iou = -1, 0.0
        pb = _xyxy_of(p)
        for gi, g in enumerate(gt):
            if gi in matched_gt or g["class_id"] != p["class_id"]:
                continue
            v = iou_xyxy(pb, _xyxy_of(g))
            if v >= iou_threshold and v > best_iou:
                best, best_iou = gi, v
        if best >= 0:
            matched_gt.add(best)
            tp.append(pi)
            matches.append((pi, best, best_iou))
        else:
            fp.append(pi)
    fn = [i for i in range(len(gt)) if i not in matched_gt]
    wrong_class = 0
    for pi in fp:
        pb = _xyxy_of(preds[pi])
        for gi in fn:
            if iou_xyxy(pb, _xyxy_of(gt[gi])) >= iou_threshold:
                wrong_class += 1
                break
    return {"tp": len(tp), "fp": len(fp), "fn": len(fn),
            "tp_idx": tp, "fp_idx": fp, "fn_idx": fn,
            "matches": matches, "wrong_class": wrong_class}


def evaluate_image(gt: list[dict], preds: list[dict], iou_threshold: float = 0.5) -> dict:
    m = match_predictions(gt, preds, iou_threshold)
    p, r, f = prf1(m["tp"], m["fp"], m["fn"])
    return {**m, "precision": p, "recall": r, "f1": f}


def aggregate(results: list[dict]) -> dict:
    return aggregate_counts(results)


def corpus_map50(gt_list: list[list[dict]], pred_list: list[list[dict]],
                 iou_threshold: float = 0.5) -> dict:
    """Legacy normalized-dict corpus mAP50."""
    images: list[ImageDetections] = []
    for gt, preds in zip(gt_list, pred_list):
        def xyxy(b: dict) -> list[float]:
            x1, y1, x2, y2 = _xyxy_of(b)
            return [x1, y1, x2, y2]
        images.append(ImageDetections(
            gt_boxes=np.array([xyxy(b) for b in gt], dtype=np.float64).reshape(-1, 4),
            gt_classes=np.array([b["class_id"] for b in gt], dtype=np.int64),
            pred_boxes=np.array([xyxy(b) for b in preds], dtype=np.float64).reshape(-1, 4),
            pred_classes=np.array([b["class_id"] for b in preds], dtype=np.int64),
            pred_scores=np.array([b.get("confidence", 1.0) for b in preds], dtype=np.float64),
        ))
    return compute_map50(images, iou_threshold)
