# FailureForge — progress log

## Day 1 — Project foundation (2026-10-16): DONE

Objective: clean, GitHub-ready foundation; no Day 2+ implementation.

- [x] `pyproject.toml`: `src/` layout, name `failureforge`, version `0.1.0`,
      editable install, pytest config, minimal deps + optional extras.
- [x] `src/failureforge/__init__.py`: package docstring + `__version__`.
- [x] `tests/test_project.py`: smoke test (import + version). **2 passed.**
- [x] `README.md`: problem, workflow, planned features, install/usage,
      testing, status, limitations.
- [x] `docs/mvp-spec.md`, `docs/architecture.md`, `docs/research-notes.md`:
      scope, planned pipeline, research design, biases.
- [x] `LICENSE` (MIT), `.gitignore`, `data/.gitkeep`, `results/.gitkeep`.
- [x] Editable install verified; `git diff`/`status` inspected; Day-1 commit
      created (see below).

Verification: `pip install -e .` → success; `pytest tests/test_project.py -v`
→ 2 passed. Full-suite run (informational; covers later-day modules present
in the working tree) → all pass.

Note: the working tree also contains functional preview modules for later
days (dataset, transforms, metrics, search, CLI, …) from exploratory work.
They are intentionally **not** part of the Day-1 commit (left untracked) and
will each be reviewed, tested, and verified on their scheduled day. No
benchmark results are claimed.

## Day 2 — YOLO detector wrapper (2026-10-16): DONE

- [x] `src/failureforge/detector.py`: `YOLODetector` (Ultralytics, inference
      only) + `Detections` (`boxes_xyxy`/`scores`/`classes`, pixel coords,
      validated shapes/dtypes, valid empties) + `DetectorError`.
- [x] Weights validation (missing → `FileNotFoundError`, directory/load
      failure → `DetectorError`), BGR `uint8` image validation, detection-only
      task gate (segment/pose/obb/classify rejected), SHA-256 fingerprint,
      metadata, lazy ultralytics import (core import never requires it).
- [x] `tests/test_detector.py`: 20 tests, fully mocked (no weights/network).
      **20 passed** + 2 smoke = 22 passed.
- [x] `pyproject.toml`: no change needed (`yolo` extra already declared Day 1).
- [x] Full test suite: **49 passed** across all modules.

## Day 3 — Dataset loading, validation & baseline evaluation (2026-10-17): DONE

- [x] `src/failureforge/dataset.py`: rewritten `YoloDataset` loader with real
      `data.yaml` support (`from_config`), split-directory support
      (`from_dirs`), streaming `iter_samples()`, and a programmatic API
      (`metadata`, `class_names`, `class_name`, `class_mapping`, `len`,
      `image_paths`). Split preference is **test → val → explicit** (never a
      silent train default; train-only datasets require `split="train"`).
      Labels directory is derived by replacing the `images` path component with
      `labels`. Coordinate convention: normalized rows are converted to **pixel**
      `[x1,y1,x2,y2]` using the real image size.
- [x] Validation (`validate()`): missing/corrupt/unreadable images, missing
      labels (error; `allow_missing_labels=True` downgrades to warning),
      malformed/non-numeric rows (with file:line context), out-of-range
      normalized coordinates, non-positive width/height, out-of-range class IDs,
      orphan labels, and **class-name consistency** against the detector’s
      names (`validate(model_class_names=...)`). Issues are returned as data
      (`ValidationIssue`/`ValidationResult`), never silently dropped.
- [x] Class mapping is **name-based**, not index-based (`class_mapping`):
      reordered or extra model classes map correctly; missing/duplicate names
      raise `DatasetError`.
- [x] `src/failureforge/metrics.py`: array-based `iou_matrix`, class-aware
      one-to-one greedy `match_boxes` (by descending confidence; `MatchResult`
      exposes tp/fp/fn indices, matches, `wrong_class`, `poor_localization`
      diagnostics — diagnostics never enter TP/FP/FN), `prf1` (zero-denominator
      → 0.0), `aggregate_counts` (micro-average), `average_precision` (all-point
      interpolation), `compute_map50` (pooled per-class AP @ IoU 0.50). Legacy
      normalized-dict helpers retained for earlier modules.
- [x] `src/failureforge/evaluator.py`: `evaluate_baseline(detector, dataset,
      iou_threshold=0.5)` on original unmodified images; `PerImageResult`,
      `ClassMetrics` (per-class AP50), `EvaluationReport` with `to_dict()`,
      `aggregate`, `per_class`, `per_image`, `error_records`, `metadata`
      (dataset + model fingerprint + class mapping). Fixes a prior adapter bug
      that dropped box geometry. Requires the detector to expose `class_names`
      (raises `ValueError` otherwise). Legacy `evaluate_samples`,
      `EvaluationResults`, `metric_drop` retained.
- [x] `tests/test_dataset.py`: 19 tests (split preference, explicit split,
      no-train-default, pixel conversion, empty/missing labels, malformed/
      non-numeric/out-of-range rows/coords, corrupt image, orphan label,
      name-based + ambiguous mapping, deterministic order, compat load/split).
- [x] `tests/test_evaluator.py`: rewritten — 21 tests (matching primitives,
      IoU boundary, determinism, no double-counting, perfect/missed/FP/duplicate/
      wrong-class/poor-localization/multi-object/empty-GT/empty-pred, P/R/F1
      values, zero denominators, multi-class mAP50, reordered class mapping,
      mismatch rejection, metadata/thresholds, legacy `evaluate_samples`).
- [x] Full test suite: **85 passed**. End-to-end smoke on `examples/`
      demo data (validate → `evaluate_baseline`): valid, F1 = 1.0, mAP50 = 1.0.
- [x] `pyproject.toml`: no change needed. No results are claimed beyond the
      synthetic smoke check; no fabricated benchmarks.


## Day 4 — Robustness perturbation engine (2026-10-20): DONE

Implemented a modular, seeded perturbation engine covering **all 12 required
families** (39 transforms) over BGR `uint8` images, with correct pixel-space
ground-truth remapping and a backward-compatible `transforms.py` shim.

- [x] New package `src/failureforge/perturbations/`:
  - `base.py`: registry (`register`/`get`/`available`/`families`/`param_specs`),
    `ParamSpec` validation, `PerturbationResult(image, boxes, metadata)`,
    `apply(name, image, params, boxes, seed)`, BGR↔PIL helpers, `MIN_BOX_SIDE_PX`.
  - `boxops.py`: affine/homography maths; `map_boxes` maps corners, rebuilds
    clipped AABBs, drops degenerate boxes and records `removed_boxes`.
  - Family modules: `photometric.py` (noise 1, exposure 2, colour 5), `blur.py`
    (3), `weather.py` (4), `occlusion.py` (6), `geometry.py` (7), `quality.py`
    (8), `lighting.py` (9), `lens.py` (10), `cropping.py` (11), `combined.py` (12).
- [x] Contract: BGR `uint8` in/out, input never mutated, deterministic per seed,
  useful `PerturbationError` on bad params/images, metadata records
  name/family/params/seed/geometry + diagnostics. `combined` runs ordered steps
  with deterministic child seeds and records `order`/`components`.
- [x] `src/failureforge/transforms.py` rewritten as a compatibility shim:
  re-exports the engine and keeps `apply_perturbation` (PIL + normalized boxes)
  and `PARAM_RANGES` (derived from engine specs) so `evaluator.evaluate_samples`
  (perturbation path), `failures.build_gallery`, and the `search/` previews keep
  working — single source of truth, no duplicated logic.
- [x] `tests/test_transforms.py` rewritten: 376 tests (parametrized contract
  tests across all transforms + per-family positive/limit/edge tests + geometry
  label correctness + combined order/reproducibility + legacy wrapper).
- [x] Full suite: **455 passed** (was 85). No pre-existing failures existed.
- [x] Integration smoke: `evaluate_samples(samples, detector, iou,
  perturbation=...)` on the demo dataset for brightness/rotation/occlusion →
  valid results (no detector/evaluator interface changes).
- [x] `pyproject.toml`: **no new dependencies** (NumPy + Pillow only).
- [x] Docs: `docs/perturbations.md` (full parameter reference + limitations);
  updated `architecture.md` and `mvp-spec.md`.

Known issues / notes:

- Weather, shadows, glare, and lens effects are **approximations**, not physical
  simulations; occlusion coverage uses a rectangle-intersection approximation;
  `motion_blur` is average-of-shifts, not a convolution.
- Preview modules `cli.py`/`config.py`, `failures.py`, `reporting.py`, and
  `search/` remain **untracked** and are scheduled for Days 9–14. `cli.py` still
  targets the pre-Day-3 API (e.g. `load_detector`, `summarize`), so it is not yet
  runnable; this is pre-existing and out of scope for Day 4.
- Random search and Optuna TPE are **not** implemented yet (Days 9–10).
- No benchmark results are claimed.

## 15-day roadmap (planned milestones, not completed features)

- Day 1: Project foundation, scope, research plan, GitHub setup. ✅
- Day 2: YOLO detector wrapper (inference only). ✅
- Day 3: Dataset loading, validation & baseline evaluation. ✅
- Day 4: Robustness perturbation engine (all 12 families) + tests. ✅
- Day 5: Perturbation sweeps / severity ladders + annotation-correctness checks.
- Day 6: Evaluation engine and metrics (revisited with the perturbation layer).
- Day 7: Failure records and visual gallery.
- Day 8: Perturbation sweeps and objective definition.
- Day 9: Random search.
- Day 10: Optuna TPE search.
- Day 11: Multi-seed comparison and confirmation evaluation.
- Day 12: Failure categorization, deduplication, reports.
- Day 13: CLI and integration.
- Day 14: Documentation and reproducible demo.
- Day 15: Fresh-environment verification and release.

## Next recommended task

Connect the engine to the evaluator for **Quick Mode vs Benchmark Mode** and
build severity sweeps + the objective definition (Days 5–8):

1. Add a thin perturbation→evaluation adapter for `YoloDataset` samples
   (BGR conversion, pixel `GroundTruthBox` in/out) so `evaluate_baseline` can run
   on perturbed images; keep the baseline evaluator untouched.
2. Define Quick Mode (no labels: report disappeared/new/class/confidence/box-drift)
   and Benchmark Mode (metrics vs transformed GT), documenting what each may claim.
3. Curate the searchable parameter ranges/specs to feed Days 9–10.