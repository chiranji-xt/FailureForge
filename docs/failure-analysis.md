# Failure analysis & reporting (Day 6)

This layer explains **what kind of errors** a benchmark evaluation contains and
turns Day-5 evaluation results into reproducible JSON and Markdown reports. It
does not re-evaluate the detector, change TP/FP/FN, or infer verified failures
from unlabeled data.

## Benchmark Mode vs Quick Mode

| | Benchmark Mode | Quick Mode |
|---|---|---|
| Ground truth | Required | Not used |
| Matching | Class-aware GT/prediction matching (existing evaluator) | Class-agnostic location matching between clean and perturbed predictions |
| Outputs | Verified failure categories + metric deltas | Prediction-instability records only |
| May claim accuracy? | Yes, for the evaluated data only | **No** — never precision/recall/F1/mAP |
| Builder | `analyze_report`, `analyze_perturbed`, `build_benchmark_run` | `analyze_quick`, `build_quick_run` |

Quick Mode records are explicitly labelled `prediction_instability` and
`verified_failures: false`. They are not false positives or false negatives.

## Matching and IoU rules (Benchmark Mode)

Inherited unchanged from `failureforge.metrics` / `FailureReport`:

- Boxes are pixel `[x1, y1, x2, y2]`.
- Predictions are processed by descending confidence (ties by index).
- A prediction matches the unmatched GT box of the **same class** with the
  highest IoU ≥ `iou_threshold` (default **0.50**, ties → lowest GT index).
- One-to-one: a prediction and a GT box each participate in at most one match.
- Unmatched predictions are FPs; unmatched GT boxes are FNs. Wrong-class
  predictions never count as TPs.
- `localization_iou_threshold` defaults to **0.75** and is only diagnostic.

Quick Mode uses a **class-agnostic, one-to-one greedy IoU match** (default
0.50), so a same-location class flip is reported as `class_changed` rather than
one disappearance plus one appearance.

## Verified failure categories

| Category | Definition | `counts_as` |
|---|---|---|
| `false_negative` | A GT box with no class-correct prediction at IoU ≥ threshold. `iou` is the best overlap with any prediction (diagnostic only). | `fn` |
| `false_positive` | A prediction with no spatial overlap with any GT box. | `fp` |
| `class_confusion` | An unmatched prediction overlapping an unmatched GT box of a different class at IoU ≥ threshold (evaluator `wrong_class` diagnostic). | `fp` |
| `localization_error` | (a) A matched prediction with IoU in `[iou_threshold, localization_iou_threshold)`; it remains a TP. (b) An unmatched same-class prediction overlapping a missed GT below the matching threshold (evaluator `poor_localization` diagnostic). | `tp` / `fp` |
| `duplicate_detection` | An extra same-class prediction covering a GT at IoU ≥ threshold when another prediction is already the primary detection (the evaluator's match, else highest confidence). Extras are classified as duplicates instead of plain FPs. | `fp` |
| `unmatched` | Cannot be attributed reliably: low/different-class overlap, different-class overlap with a GT already matched to another prediction, or a predicted class absent from the dataset vocabulary. | `fp` |

Rules that prevent double counting:

- Each unmatched GT produces at most one `false_negative`.
- Each unmatched prediction produces at most one record, with priority
  **duplicate > class confusion > localization > background false positive >
  unmatched**.
- A matched `localization_error` is an aspect of a TP and never changes metrics.
- Category counts explain TP/FP/FN; they are not added to them.

## Quick Mode instability

Kinds: `disappeared`, `appeared`, `class_changed`, `confidence_changed`,
`box_drift`. Each record keeps image, clean/perturbed prediction indices,
classes, boxes, scores, IoU/drift, confidence delta, transformation
(name/family/params/seed), and thresholds. Empty predictions are handled by
ordinary unmatched sets; no labels are read. Matched-change records now include
clean/perturbed boxes and scores when the Day-5 comparison provides them.

## Structured schema (v1)

`schema_version = "failureforge.failure-analysis.v1"`.

Benchmark run (`BenchmarkRun.to_dict()`):

```json
{
  "schema_version": "failureforge.failure-analysis.v1",
  "mode": "benchmark",
  "metadata": {
    "dataset": {"split": "val", "num_images": 100, "class_names": ["cat", "dog"]},
    "model": {"weights": "best.pt", "sha256": "…", "task": "detect", "device": "cpu"},
    "thresholds": {"iou_threshold": 0.5, "localization_iou_threshold": 0.75},
    "aggregation": "micro-average over summed TP/FP/FN across images",
    "environment": {"failureforge": "0.1.0", "python": "3.14.3", "numpy": "…", "pillow": "…"},
    "scope": "observations for the evaluated model/dataset/thresholds; not universal"
  },
  "baseline": { "transformation": null, "metrics": {}, "deltas": {}, "drops": {},
                "per_sample": [], "failure_counts": {}, "failures": [], "warnings": [] },
  "trials": [
    { "transformation": {"name": "gaussian_noise", "family": "noise",
                         "params": {"sigma": 25}, "seed": 0},
      "metrics": {"aggregate": {}, "map50": 0.61, "iou_threshold": 0.5,
                  "conf_threshold": 0.25, "aggregation_method": "…",
                  "split": "val", "num_images": 100},
      "deltas": {"precision": -0.02, "recall": -0.08, "f1": -0.05, "map50": -0.07},
      "drops": {"precision": 0.02, "recall": 0.08, "f1": 0.05, "map50": 0.07},
      "compatible": true,
      "per_sample": [], "failure_counts": {}, "failures": [], "warnings": [] }
  ],
  "warnings": []
}
```

A verified failure record contains: `category`, `reason`, `image`, `counts_as`,
`transformation`, `gt_index`, `pred_index`, `gt_class_id`/`gt_class_name`,
`pred_class_id`/`pred_class_name`, `gt_box`, `pred_box`, `score`, `iou`,
`iou_threshold`, `localization_iou_threshold`.

Quick run (`QuickRun.to_dict()`) contains `mode: "quick"`,
`label: "prediction_instability"`, `verified_failures: false`, transformation,
thresholds, `counts`, `per_image`, `records`, instability `summary`, `warnings`,
and `metadata`.

JSON policy:

- Always valid JSON with `schema_version`.
- NumPy scalars/arrays converted to native values; non-finite floats → `null`.
- Absolute paths are reduced to basenames unless
  `include_absolute_paths=True` is passed.
- Ranked trials, groups, severity points, and records are sorted
  deterministically; no timestamps are generated.

## Reporting API

```python
from failureforge.failures import build_benchmark_run, build_quick_run
from failureforge.reporting import (
    dumps_report, render_markdown, write_json_report,
    write_markdown_report, write_report,
)

run = build_benchmark_run(baseline_report, perturbed_results)  # shared baseline
text = render_markdown(run, rank_metric="f1")                  # no visualization deps
paths = write_report(run, "results/report", formats=("json", "md"))

quick_run = build_quick_run(quick_result)
write_report(quick_run, "results/quick", basename="quick")
```

Ranking uses **one** metric at a time (`precision`, `recall`, `f1`, or `map50`)
and ranks `drop = baseline - perturbed` (positive = degradation on this
evaluation). Missing/non-finite values and incompatible comparisons (different
IoU threshold or aggregation method) are **not ranked** and are surfaced as
warnings. Raw instability counts are never ranked.

Markdown includes: scope/reproducibility, baseline metrics, ranked perturbations,
family/operation groups, severity ladders (only when trials vary exactly one
numeric parameter), failure-category counts (baseline, trial sum, top-ranked
trials), warnings, and limitations. Quick Mode reports instability summary and
per-image counts instead.

## What is not a verified failure
- Quick Mode records (no labels).
- A metric drop by itself: it is an observation for the evaluated model,
  dataset, split, seeds, and thresholds.
- Any claim that one transform is universally most damaging.
- Categories derived from incomplete evaluator details: analysis raises
  `AnalysisError` instead of guessing.
- Pixel-space box approximations for rotated objects (axis-aligned boxes).

## Limitations

- Single-seed results are noisy; repeat runs.
- Duplicate detection is defined by same-class IoU ≥ threshold; different-class
  overlapping extras are reported as class confusion or unmatched.
- Localization of matched predictions is a diagnostic aspect of a TP.
- Severity ladders are parameter ladders, not invented severity labels.
- No CLI command is exposed yet (Day 13); use the Python functions above.

## Gallery and annotation checks

Day 7 adds a portable HTML gallery (`failureforge.gallery`, see
`docs/gallery.md`) over these run records, plus read-only annotation validation
(`failureforge.annotations`, see `docs/annotations.md`). Neither changes the JSON
or Markdown schemas.
