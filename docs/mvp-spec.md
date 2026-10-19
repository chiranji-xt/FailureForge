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

## Evaluation approach (planned)

- IoU-based one-to-one, class-aware prediction/ground-truth matching.
- TP/FP/FN, precision, recall, F1; per-image and aggregated corpus results.
- Corpus mAP50 (pooled, standard) — never a mean of per-image scores.
- Baseline (unperturbed) result always recorded first.
- Objective: deterioration `baseline_metric − perturbed_metric` (higher = more
  damaging), so neither method is rewarded for invalid annotations or bugs.

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

## Acceptance criteria (MVP)

- Editable install works; smoke test passes.
- Dataset validation accepts a small synthetic YOLO dataset and rejects
  malformed rows with clear errors.
- Each perturbation returns a valid image and valid transformed annotations.
- Metrics pass hand-verified unit tests, including empty-prediction edge cases.
- Both searches run against a deterministic test objective and record trials.
- Report generation works from saved results.
- CI installs the project and runs the core test suite.

## Out of scope

- Model training, web dashboards, databases, distributed workers,
  authentication, cloud infrastructure.
- Claiming to find every possible real-world failure.
