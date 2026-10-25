"""Cross-check the vectorised metrics against an independent pure-Python reference.

The reference follows the documented conventions (confidence-ordered greedy
matching to the best unmatched same-class GT, pooled all-point AP at IoU 0.5)
but shares no code with ``failureforge.metrics``.
"""
import random

import numpy as np
import pytest

from failureforge.metrics import ImageDetections, compute_map50, match_boxes


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ref_image(gt, preds, thr):
    """gt: [(cls, box)], preds: [(cls, score, box)] -> (tp flags per pred idx)."""
    used = [False] * len(gt)
    is_tp = {}
    for pi in sorted(range(len(preds)), key=lambda i: (-preds[i][1], i)):
        pc, _, pb = preds[pi]
        best, best_iou = None, 0.0
        for gi, (gc, gb) in enumerate(gt):
            if used[gi] or gc != pc:
                continue
            v = _iou(pb, gb)
            if v > best_iou:
                best, best_iou = gi, v
        if best is not None and best_iou >= thr:
            used[best] = True
            is_tp[pi] = True
        else:
            is_tp[pi] = False
    return is_tp


def _ref_map(scenes, thr=0.5):
    classes = sorted({c for gt, _ in scenes for c, _ in gt})
    aps = []
    for c in classes:
        n_gt = sum(1 for gt, _ in scenes for g in gt if g[0] == c)
        rows = []
        for si, (gt, preds) in enumerate(scenes):
            flags = _ref_image(gt, preds, thr)
            for pi, (pc, ps, _) in enumerate(preds):
                if pc == c:
                    rows.append((-ps, si, pi, flags[pi]))
        rows.sort(key=lambda r: r[:3])
        tp = fp = 0
        pts = []
        for r in rows:
            tp, fp = tp + r[3], fp + (not r[3])
            pts.append((tp / n_gt, tp / (tp + fp)))
        # all-point interpolation: area under the monotone precision envelope
        ap, prev_r = 0.0, 0.0
        for i, (rec, _) in enumerate(pts):
            env = max(p for _, p in pts[i:])
            ap += (rec - prev_r) * env
            prev_r = rec
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0


def _scene(rng, n_classes):
    def box():
        x, y = rng.uniform(0, 80), rng.uniform(0, 80)
        return (x, y, x + rng.uniform(5, 40), y + rng.uniform(5, 40))
    gt = [(rng.randrange(n_classes), box()) for _ in range(rng.randint(0, 5))]
    preds = []
    for c, b in gt:
        if rng.random() < 0.8:  # jittered true detection
            j = [v + rng.uniform(-6, 6) for v in b]
            j[2], j[3] = max(j[2], j[0] + 1), max(j[3], j[1] + 1)
            preds.append((c if rng.random() < 0.9 else rng.randrange(n_classes),
                          round(rng.random(), 2), tuple(j)))
    for _ in range(rng.randint(0, 3)):
        preds.append((rng.randrange(n_classes), round(rng.random(), 2), box()))
    return gt, preds


@pytest.mark.parametrize("seed", range(40))
def test_map50_and_counts_match_independent_reference(seed):
    rng = random.Random(seed)
    scenes = [_scene(rng, 3) for _ in range(rng.randint(1, 8))]
    images = []
    tp = fp = fn = 0
    for gt, preds in scenes:
        g_b = np.array([b for _, b in gt], dtype=float).reshape(-1, 4)
        g_c = np.array([c for c, _ in gt], dtype=int)
        p_b = np.array([b for _, _, b in preds], dtype=float).reshape(-1, 4)
        p_c = np.array([c for c, _, _ in preds], dtype=int)
        p_s = np.array([s for _, s, _ in preds], dtype=float)
        images.append(ImageDetections(g_b, g_c, p_b, p_c, p_s))
        res = match_boxes(g_b, g_c, p_b, p_c, p_s, 0.5)
        flags = _ref_image(gt, preds, 0.5)
        ref_tp = sum(flags.values())
        assert (res.tp, res.fp) == (ref_tp, len(preds) - ref_tp)
        assert res.fn == len(gt) - ref_tp
        tp, fp, fn = tp + res.tp, fp + res.fp, fn + res.fn
    got = compute_map50(images)["map50"]
    if any(gt for gt, _ in scenes):
        assert got == pytest.approx(_ref_map(scenes), abs=1e-12)
    else:
        assert got == 0.0
