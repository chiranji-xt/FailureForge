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

## Day 5 — Perturbation→evaluation adapter, Benchmark/Quick Mode, sweeps (2026-10-21): DONE

Objective: connect the perturbation engine to the existing evaluator and add
severity sweeps, without duplicating dataset, box, or metric logic.

- [x] `evaluator.py` refactor (behavior-preserving): the evaluation loop moved
      into private `_evaluate_stream(detector, dataset, stream, iou_threshold)`
      consuming `(sample, bgr)` pairs; `_load_bgr`/`_default_stream` load original
      images from disk. `evaluate_baseline` keeps its exact public signature and
      returns identical reports (42 evaluator/dataset tests still pass).
- [x] New `src/failureforge/evaluation.py` (thin adapter, no new dependencies):
  - `load_sample_bgr`, `prepare_perturbed_sample`, `iter_perturbed_samples`:
    load BGR, apply a registered transform with pixel GT boxes, return a new
    `Sample` via `dataclasses.replace` (transformed boxes + real image dims).
    Input samples/boxes/images are never mutated.
  - **Benchmark Mode** (`evaluate_perturbed`): baseline evaluated once (or
    supplied), perturbed pass evaluated by the shared stream, report
    `PerturbedResult(baseline, perturbed, perturbation, deltas, metadata)` with
    `deltas = perturbed − baseline` for TP/FP/FN/P/R/F1/mAP50. `to_dict` can omit
    the baseline to avoid duplication in sweeps.
  - **Quick Mode** (`quick_mode`): iterates `dataset.image_paths` only (never
    reads labels), compares clean vs perturbed predictions with deterministic,
    class-agnostic, one-to-one greedy IoU matching, and reports disappeared /
    appeared / class-changed / confidence-changed / box drift. It deliberately
    reports **no** accuracy metrics (no GT), and its summary/to_dict contain no
    precision/recall/F1/mAP keys.
  - **Severity sweeps** (`run_severity_sweep`): explicit `configs` list (no
    combinatorial expansion), `mode="benchmark"|"quick"`; per-trial seed derived
    via SHA-256 of `base_seed|index|name|params` (deterministic, stable); one
    shared baseline per sweep; a failing trial records
    `PerturbationError: ...` with its index/config (`strict=True` re-raises).
- [x] `tests/test_evaluation.py`: 19 tests covering BGR order, no-mutation,
      transformed dims/boxes (crop bounds, rotation remap), benchmark deltas on
      transformed GT, baseline reuse (no recompute), Quick Mode instability with
      no label files present, class-agnostic matching, sweep reproducibility,
      seed sensitivity, baseline separation, attributable errors, strict mode,
      and unknown-mode rejection.
- [x] Full suite: **474 passed** (was 455), CPU-only, no weights/network.
- [x] Docs: updated `README.md` (Day 5 status + usage), `architecture.md`,
      `mvp-spec.md`, and this log.

Known issues / notes:

- Benchmark deltas are **observations** for a fixed detector/dataset/seed, not
  proof of real-world failure; single-seed sweeps are noisy.
- Quick Mode matches by location, so a class flip at the same box is reported as
  `class_changed` (not as a disappearance + appearance). It uses all dataset
  images and the same seed for every image in a run.
- Sweeps only cover caller-supplied configurations; automatic random/TPE search
  is still Days 9–10.
- `annotations.py` + `tests/test_annotations.py` remain an untracked preview and
  are not wired into this adapter; annotation-correctness checks are moved to
  Day 6.
- `cli.py` remains stale (pre-Day-3 `load_detector`/`summarize` API), so no CLI
  commands are exposed. Reporting/failure records remain Days 7–13.
- No benchmark results are claimed.

## 15-day roadmap (planned milestones, not completed features)

- Day 1: Project foundation, scope, research plan, GitHub setup. ✅
- Day 2: YOLO detector wrapper (inference only). ✅
- Day 3: Dataset loading, validation & baseline evaluation. ✅
- Day 4: Robustness perturbation engine (all 12 families) + tests. ✅
- Day 5: Perturbation→evaluation adapter, Benchmark/Quick Mode, severity sweeps. ✅
- Day 6: Evaluation engine and metrics (revisited with the perturbation layer;
  annotation-correctness checks).
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

Revisit metrics/matching under the perturbation layer and define the objective
(Days 6–8):

1. Verify annotation correctness end-to-end (wire the untracked `annotations.py`
   preview or fold its checks into dataset/transform tests) and document the
   geometric/cropping assumptions.
2. Define the search objective (`baseline − perturbed`, higher = more damaging)
   and the failure-record schema so search results are attributable to
   configurations, seeds, and images.
3. Keep the CLI disabled (Day 13) and continue reusing
   `evaluate_baseline`/`_evaluate_stream` rather than reimplementing evaluation.