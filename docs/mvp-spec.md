# FailureForge — MVP specification

## Research question

> Under an equal evaluation budget, does Optuna's Tree-structured Parzen
> Estimator (TPE) identify more damaging image perturbation configurations
> than random search?

No superiority claim is made a priori; the experiment decides.

## Inputs

- A labeled object-detection dataset in YOLO format (images + normalized
  `class_id, x_center, y_center, width, height` label files).
- An existing YOLO detector weights file (e.g. `best.pt`), used as-is.
  FailureForge evaluates the detector; it is not a training platform.

## Planned pipeline

Labeled dataset → baseline evaluation → controlled image perturbations →
transformed ground-truth annotations → detector predictions → IoU-based
prediction matching and metrics → failure records → random search vs TPE →
independent confirmation evaluation → JSON/CSV/Markdown reports + gallery.

## Supported perturbations

The `failureforge.perturbations` engine implements **12 families** (see
`docs/perturbations.md` for full parameters): noise (Gaussian, salt-and-pepper,
speckle); brightness/exposure; blur (Gaussian, motion, defocus); weather
(fog, haze, rain, snow); colour distortion (hue, saturation, contrast,
temperature); occlusion (cutout, random patches, partial obstruction);
geometric (rotation, scaling, translation, perspective); image quality
(JPEG, downscale, pixelate); shadows/lighting; lens effects (glare, flare,
vignetting); cropping/framing; and configurable `combined` compositions.

Transforms operate on BGR `uint8` arrays. Non-geometric transforms leave ground
truth unchanged; geometric and cropping transforms remap, clip, and (per the
documented rule) drop degenerate boxes. Every transform is seeded and
reproducible, with documented experimental parameter ranges.

`failureforge.transforms.PARAM_RANGES` exposes a legacy `{name: {param: (low,
high)}}` search-space view for the original five transforms; it is derived from
the engine specs.

## Evaluation approach (Days 3–5, implemented)

- IoU-based one-to-one, class-aware prediction/ground-truth matching.
- TP/FP/FN, precision, recall, F1; per-image and aggregated corpus results.
- Corpus mAP50 (pooled, standard) — never a mean of per-image scores.
- Baseline (unperturbed) result always recorded first.

`failureforge.evaluation` connects perturbations to that evaluator:

- **Benchmark Mode** reuses one baseline report, evaluates perturbed images
  against transformed ground truth, and reports `perturbed − baseline` deltas
  for TP/FP/FN/P/R/F1/mAP50.
- **Quick Mode** requires **no labels** and reports only prediction instability
  (disappeared / appeared / class-changed / confidence-changed / box drift).
  It intentionally exposes no precision/recall/F1/mAP, because correctness is
  unknown without ground truth.
- **Severity sweeps** evaluate explicit parameter lists with deterministic
  per-trial seeds and record failures per trial instead of aborting.

Deltas are observations for a fixed detector/dataset/seed, not proof of a
real-world failure. The search objective (deterioration `baseline − perturbed`,
higher = more damaging) remains planned for Days 8–10 so neither search method is
rewarded for invalid annotations or bugs.

## Failure analysis & reporting (Day 6, implemented)

- **Verified Benchmark categories** from GT/prediction matching:
  `false_negative`, `false_positive`, `class_confusion`, `localization_error`,
  `duplicate_detection`, `unmatched`. Each record keeps sample/image, GT and
  prediction indices, classes, boxes, score, IoU, thresholds, and its metric
  role (`tp`/`fp`/`fn`). Categories never alter TP/FP/FN.
- Matching stays class-aware/one-to-one at IoU ≥ 0.50 with a diagnostic
  localization threshold of 0.75. Duplicate detection additionally requires
  same-class IoU ≥ 0.50 and is assigned only to extra predictions.
- **Quick Mode** records (`disappeared`, `appeared`, `class_changed`,
  `confidence_changed`, `box_drift`) are labelled prediction instability and
  carry no accuracy metrics; they are never mixed with verified failures.
- Schema-versioned runs preserve dataset/model identifiers, model SHA-256 when
  available, detector configuration, thresholds, transformation
  name/family/params/seed, environment metadata, and per-sample results.
- Ranking uses one supported metric at a time (`precision`, `recall`, `f1`,
  `map50`) with `drop = baseline − perturbed`. Missing/non-finite values and
  incompatible comparisons are not ranked and are reported as warnings.
- Reports: JSON (NumPy-safe, non-finite → `null`, absolute paths redacted by
  default) and dependency-free Markdown (baseline, ranked trials,
  family/operation groups, severity ladders when supported, failure counts,
  instability summary, warnings, limitations). See `docs/failure-analysis.md`.

## Search comparison (planned)

- Random search and Optuna TPE share the search space, trial budget,
  objective, model, inference settings, and discovery data.
- A held-out confirmation split is never used during optimization; winning
  configurations are re-evaluated there.
- Multiple seeds supported; trial counts, timing, and variability recorded.

## Outputs

- Machine-readable JSON + CSV (metrics, trials, failure records).
- Human-readable Markdown report (baseline, discovery vs confirmation,
  failure categories, runtime, reproducibility, limitations).
- Visual failure gallery (ground truth vs predictions).

Implemented so far: schema-versioned JSON and Markdown for Benchmark/Quick runs
(`failureforge.reporting`); CSV trial export and the gallery remain planned.

## Acceptance criteria (MVP)

- Editable install works; smoke test passes.
- Dataset validation accepts a small synthetic YOLO dataset and rejects
  malformed rows with clear errors.
- Each perturbation returns a valid image and valid transformed annotations.
- Perturbed evaluation reuses the baseline evaluator over transformed images/GT;
  Quick Mode never reports accuracy metrics without labels.
- Metrics pass hand-verified unit tests, including empty-prediction edge cases.
- Every verified failure category, duplicate predictions, empty GT/predictions,
  and ambiguous cases have synthetic tests; Quick instability records remain
  separate from verified failures.
- JSON reports are valid, schema-versioned, handle NumPy/non-finite values, and
  redact absolute paths by default; Markdown output is deterministic.
- Both searches run against a deterministic test objective and record trials.
- Report generation works from saved results.
- CI installs the project and runs the core test suite.

## Out of scope

- Model training, web dashboards, databases, distributed workers,
  authentication, cloud infrastructure.
- Claiming to find every possible real-world failure.
