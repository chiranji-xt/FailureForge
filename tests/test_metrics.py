"""Hand-verified matching/metric tests."""
from failureforge.metrics import (
    aggregate, corpus_map50, evaluate_image, iou_xyxy, prf1,
)


def _b(cid=0, xc=0.5, yc=0.5, w=0.2, h=0.2, conf=0.9):
    return {"class_id": cid, "x_center": xc, "y_center": yc,
            "width": w, "height": h, "confidence": conf}


def test_iou():
    assert iou_xyxy((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert iou_xyxy((0, 0, 1, 1), (1, 1, 2, 2)) == 0.0
    assert abs(iou_xyxy((0, 0, 1, 1), (0.5, 0.5, 1.5, 1.5)) - 1 / 7) < 1e-9


def test_perfect_match():
    gt = [_b()]
    r = evaluate_image(gt, [_b(conf=0.9)])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    assert r["f1"] == 1.0


def test_no_double_counting():
    gt = [_b()]
    r = evaluate_image(gt, [_b(conf=0.9), _b(conf=0.8)])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)


def test_class_aware():
    gt = [_b(cid=0)]
    r = evaluate_image(gt, [_b(cid=1, conf=0.9)])
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)
    assert r["wrong_class"] == 1


def test_empty_cases():
    r = evaluate_image([], [])
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 0)
    assert r["precision"] == 0.0 and r["recall"] == 0.0
    r2 = evaluate_image([], [_b()])
    assert (r2["tp"], r2["fp"]) == (0, 1)


def test_prf1_and_aggregate():
    p, r, f = prf1(2, 1, 1)
    assert abs(p - 2 / 3) < 1e-9 and abs(r - 2 / 3) < 1e-9
    agg = aggregate([{"tp": 1, "fp": 0, "fn": 0}, {"tp": 0, "fp": 1, "fn": 1}])
    assert (agg["tp"], agg["fp"], agg["fn"]) == (1, 1, 1)


def test_map50_perfect_is_one():
    gt = [[_b(cid=0)], [_b(cid=1)]]
    pr = [[_b(cid=0)], [_b(cid=1)]]
    m = corpus_map50(gt, pr)
    assert abs(m["map50"] - 1.0) < 1e-9


def test_map50_no_predictions_is_zero():
    gt = [[_b(cid=0)]]
    m = corpus_map50(gt, [[{"class_id": 0, "x_center": 0.0, "y_center": 0.0,
                             "width": 0.01, "height": 0.01, "confidence": 0.9}]])
    assert m["map50"] == 0.0
