# Search objective (Day 8)

`failureforge.objective` defines the deterministic scalar objective that future
robustness search algorithms will maximize. It consumes existing evaluation
reports and never loads a model, reads images, or re-runs inference.

## What the score means

The score is **signed performance degradation** for one higher-is-better metric:

```text
degradation = baseline metric - perturbed metric
```

- `0.0` means the metric did not change.
- A positive value means the perturbed evaluation was worse.
- A negative value means the perturbed evaluation improved.
- A larger value always means a worse robustness outcome, so future searches
  maximize the objective.

The result exposes the metric, direction, both report values, compatibility,
validity, warnings, and the degradation. Negative degradation is preserved
rather than clamped to zero.

## API

```python
from failureforge.objective import ObjectiveConfig, evaluate_objective

config = ObjectiveConfig(metric="f1")  # default is also f1
result = evaluate_objective(
    baseline_report,       # EvaluationReport from the clean evaluation
    perturbed_report,      # EvaluationReport evaluated against transformed GT
    config=config,
)

if result.valid:
    score = result.degradation   # float; maximize this
else:
    print(result.warnings)       # no score for this trial

payload = result.to_dict()
```

`ObjectiveResult` fields:

| Field | Meaning |
|---|---|
| `metric` | Selected metric |
| `direction` | `"higher_is_better"` for every currently supported metric |
| `baseline_value` | Finite baseline value, or `None` |
| `perturbed_value` | Finite perturbed value, or `None` |
| `degradation` | Signed `baseline - perturbed`, or `None` when invalid |
| `compatible` | Whether the reports passed compatibility checks |
| `valid` | Whether a degradation score may be used for ranking |
| `warnings` | Compatibility and missing/non-finite explanations |

Supported metrics are exactly the existing ranking metrics: `precision`,
`recall`, `f1`, and `map50`. The objective uses `failures.compare_reports`, so
it does not define competing metric, matching, or mAP semantics. Only one metric
is selected at a time; there are no weights or multi-objective combinations.

## Invalid inputs

`ObjectiveConfig` rejects unsupported metric names with `ValueError`. Report
problems do not become fabricated scores:

- Missing or non-finite metric values produce `valid=False` and
  `degradation=None`, with a warning identifying the missing/non-finite side.
- Different IoU thresholds or aggregation methods mark the reports
  incompatible, set `valid=False`, and suppress the degradation score.
- Split, detector confidence-threshold, and evaluated-image-count differences
  are warnings from `compare_reports`; the comparison remains usable unless the
  core compatibility checks fail.
- A valid `0.0` is distinct from an invalid result (`valid=True,
  degradation=0.0` versus `valid=False, degradation=None`).

The objective never substitutes zero for missing data.

## Relationship to failures and Quick Mode

This objective is **performance degradation**, an accuracy-metric difference.
It is deliberately separate from:

- **failure counts** — categorized detection errors from `failures.py`; these
  explain a result but do not replace its metric degradation;
- **Quick Mode instability** — prediction changes without labels; it has no
  accuracy metric and therefore cannot produce this objective.

No failure count, instability count, or arbitrary weight is folded into the
score.

## Future search consumption

Days 9-10 search algorithms will evaluate a candidate perturbation, call
`evaluate_objective` with the shared baseline and the trial's perturbed report,
and maximize `degradation` only when `valid` is true. Invalid trials must be
recorded as invalid, not ranked. Search-space sampling, budgets, random search,
and Optuna/TPE are **not implemented in Day 8**.

## Limitations

- One higher-is-better metric at a time; no aggregate or weighted objective.
- Signed scores can reward a perturbation only as much as the selected metric
  changes; they do not measure failure type or severity.
- Single-report comparisons do not estimate seed variance or confidence
  intervals.
- Metric values and compatibility are only as trustworthy as the input reports,
  their transformed ground truth, and the evaluator's established matching
  rules.
- The objective does not validate annotations, run inference, choose a search
  space, or select trials.
