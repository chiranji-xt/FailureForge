# FailureForge — architecture (implemented Days 1–7; Days 8–15 planned)

## Implemented data flow

```text
Labeled dataset → baseline evaluation → image perturbations →
annotation updates → detector predictions → prediction matching & metrics →
failure records → random search vs TPE → independent confirmation evaluation →
reports (JSON/CSV/Markdown) + failure gallery
```

## Module boundaries (`src/failureforge/`)

| Module | Responsibility |
|---|---|
| `config.py` | YAML configuration loading with defaults |
| `dataset.py` | Strict YOLO dataset loading/validation; never silently drops rows |
| `detector.py` | Detector interface + Ultralytics adapter; model loaded once; CPU-capable |
| `perturbations/` | Seeded perturbation engine: 12 families over BGR `uint8` arrays with pixel-space box remapping |
| `transforms.py` | Backward-compatible shim over `perturbations/` (PIL + normalized boxes) |
| `annotations.py` | Read-only box helpers + annotation validation (errors/warnings, file:line context, missing vs empty labels) |
| `metrics.py` | Greedy confidence-sorted class-aware IoU matching; corpus mAP50 |
| `evaluator.py` | Baseline evaluation over a `(sample, bgr)` stream; unchanged public `evaluate_baseline` |
| `evaluation.py` | Perturbation→evaluation adapter: Benchmark Mode (deltas vs transformed GT), label-free Quick Mode (instability), severity sweeps |
| `failures.py` | Verified failure categories + Quick instability records + schema-versioned benchmark/quick runs |
| `search/random_search.py` | Seeded uniform sampling over the shared search space |
| `search/tpe_search.py` | Optuna TPESampler over the same space/objective |
| `reporting.py` | Schema-versioned JSON (NumPy/non-finite/path-safe) + dependency-free Markdown with ranking, groups, ladders, warnings |
| `gallery.py` | Offline HTML failure gallery: escaped entries, optional hashed image copies/box overlays, no-claim fallbacks |
| `cli.py` | `validate-data`, `evaluate`, `search`, `report` commands |

`perturbations/` submodules group the 12 families: `photometric.py` (noise,
exposure, colour), `blur.py`, `weather.py`, `occlusion.py`, `geometry.py`,
`quality.py`, `lighting.py`, `lens.py`, `cropping.py`, `combined.py`, with shared
core/registry in `base.py` and box maths in `boxops.py`. See
`docs/perturbations.md` for the full parameter reference.

## Implemented vs planned

- **Implemented (Days 1–7):** project foundation, `detector.py` (inference-only
  `YOLODetector`), `dataset.py`/`metrics.py`/`evaluator.py` (baseline
  evaluation), the `perturbations/` engine (12 families) with a `transforms.py`
  compatibility shim, `evaluation.py` (Benchmark/Quick Mode + severity sweeps),
  `failures.py`/`reporting.py` (verified failure analysis, instability records,
  JSON/Markdown reports), `gallery.py` (offline HTML failure gallery), and
  `annotations.py` (read-only annotation validation).
- **Planned (Days 8–15):** search objective/algorithms, independent
  confirmation, remaining exports, and CLI integration, implemented and verified
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

The gallery consumes run records only. It never re-runs inference or analysis;
image artifacts are optional caller-supplied files copied under content-hashed
relative names. Without an artifact it renders an explicitly labelled box
diagram instead of claiming an image overlay. Annotation validation reads label
files and image dimensions without modifying dataset state. See
`docs/gallery.md` and `docs/annotations.md`.

## Dependency boundaries

- Core: `numpy`, `Pillow`, `PyYAML` (minimal, CPU-only, Windows-friendly).
- Optional: `optuna` (TPE search), `ultralytics` (real YOLO inference).
- Core tests must never require weights, downloads, or GPUs.
