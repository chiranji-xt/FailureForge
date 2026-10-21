# FailureForge — architecture (planned)

## Planned data flow

```text
Labeled dataset → baseline evaluation → image perturbations →
annotation updates → detector predictions → prediction matching & metrics →
failure records → random search vs TPE → independent confirmation evaluation →
reports (JSON/CSV/Markdown) + failure gallery
```

## Planned module boundaries (`src/failureforge/`)

| Module | Responsibility |
|---|---|
| `config.py` | YAML configuration loading with defaults |
| `dataset.py` | Strict YOLO dataset loading/validation; never silently drops rows |
| `detector.py` | Detector interface + Ultralytics adapter; model loaded once; CPU-capable |
| `perturbations/` | Seeded perturbation engine: 12 families over BGR `uint8` arrays with pixel-space box remapping |
| `transforms.py` | Backward-compatible shim over `perturbations/` (PIL + normalized boxes) |
| `annotations.py` | Box geometry helpers (YOLO ↔ xyxy, clip, validation) |
| `metrics.py` | Greedy confidence-sorted class-aware IoU matching; corpus mAP50 |
| `evaluator.py` | Baseline evaluation over a `(sample, bgr)` stream; unchanged public `evaluate_baseline` |
| `evaluation.py` | Perturbation→evaluation adapter: Benchmark Mode (deltas vs transformed GT), label-free Quick Mode (instability), severity sweeps |
| `failures.py` | Verified failure categories + Quick instability records + schema-versioned benchmark/quick runs; gallery planned Day 7 |
| `search/random_search.py` | Seeded uniform sampling over the shared search space |
| `search/tpe_search.py` | Optuna TPESampler over the same space/objective |
| `reporting.py` | Schema-versioned JSON (NumPy/non-finite/path-safe) + dependency-free Markdown with ranking, groups, ladders, warnings |
| `cli.py` | `validate-data`, `evaluate`, `search`, `report` commands |

`perturbations/` submodules group the 12 families: `photometric.py` (noise,
exposure, colour), `blur.py`, `weather.py`, `occlusion.py`, `geometry.py`,
`quality.py`, `lighting.py`, `lens.py`, `cropping.py`, `combined.py`, with shared
core/registry in `base.py` and box maths in `boxops.py`. See
`docs/perturbations.md` for the full parameter reference.

## Implemented vs planned

- **Implemented (Days 1–6):** project foundation, `detector.py` (inference-only
  `YOLODetector`), `dataset.py`/`metrics.py`/`evaluator.py` (baseline
  evaluation), the `perturbations/` engine (12 families) with a `transforms.py`
  compatibility shim, `evaluation.py` (Benchmark/Quick Mode + severity sweeps),
  and `failures.py`/`reporting.py` (verified failure analysis, instability
  records, JSON/Markdown reports).
- **Planned (Days 7–15):** failure gallery, search comparison, objective
  definition, remaining reporting, and CLI integration, implemented and verified
  one stage at a time per `progress.md`.

The evaluation adapter does not reimplement evaluation: it builds a stream of
`(transformed_sample, bgr)` pairs and passes it to the evaluator's shared
`_evaluate_stream`. Quick Mode uses the same prediction dataclasses but
location-matches clean vs perturbed outputs without reading labels.

Failure analysis consumes only serialized evaluator details (`matches`,
`missed`, `false_positives`). It verifies that per-image detail counts match
`num_gt`/`num_pred` and raises `AnalysisError` rather than guessing. Category
records never alter TP/FP/FN; a matched localization error is an aspect of a
true positive. Reporting consumes run records only and never mutates them.
See `docs/failure-analysis.md` for categories, thresholds, schema, and
limitations.

## Dependency boundaries

- Core: `numpy`, `Pillow`, `PyYAML` (minimal, CPU-only, Windows-friendly).
- Optional: `optuna` (TPE search), `ultralytics` (real YOLO inference).
- Core tests must never require weights, downloads, or GPUs.
