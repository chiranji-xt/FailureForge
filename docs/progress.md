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
- `annotations.py` + `tests/test_annotations.py` moved out of preview in Day 7:
  the module now provides read-only dataset/label validation and is tracked.
- `cli.py` remains stale (pre-Day-3 `load_detector`/`summarize` API), so no CLI
  commands are exposed. CSV export and CLI integration remain later days.
- No benchmark results are claimed.

## Day 6 — Failure analysis, categorization & reporting (2026-10-21): DONE

Objective: explain verified benchmark errors and prediction instability on top of
Day 5, without rewriting the detector, dataset, perturbation engine, or
evaluation logic.

- [x] Inspection: clean tree (only known untracked previews), HEAD `2d0cbf8`,
      full suite **474 passed** before changes. Found `failures.py`/`reporting.py`
      stale (normalized YOLO dicts + old search bundle) and replaced them.
- [x] Evaluator additive change: matched-pair details now include `gt_box` and
      `pred_box` (rounded). No matching, metric, or report-calculation logic
      changed; existing tests stayed green.
- [x] Rewrote `src/failureforge/failures.py`:
  - Six verified categories: `false_negative`, `false_positive`,
    `class_confusion`, `localization_error`, `duplicate_detection`,
    `unmatched`. Each record keeps image, GT/pred indices, classes, boxes,
    score, IoU, thresholds, and metric role (`counts_as`).
  - Reuses evaluator diagnostics and matching. Defaults: match IoU 0.50,
    localization threshold 0.75. Duplicates are extra same-class predictions
    covering a GT already claimed by the primary detection.
  - One record per unmatched GT/prediction with documented priority
    duplicate > class confusion > localization > background FP > unmatched;
    matched localization errors remain TP aspects, so metrics never change.
  - Handles empty labels/predictions, unmapped prediction classes, low-overlap
    ambiguity, and GT already matched by another class. Raises `AnalysisError`
    when per-image details are incomplete instead of misclassifying.
  - Quick Mode records (`disappeared`, `appeared`, `class_changed`,
    `confidence_changed`, `box_drift`) are built in a separate structure labelled
    `prediction_instability` / `verified_failures: false`; no accuracy metrics.
  - Schema-versioned run records (`BenchmarkRun`, `QuickRun`), per-trial metric
    comparison with missing/non-finite → `None`, compatibility checks
    (IoU/aggregation), and environment metadata.
- [x] Rewrote `src/failureforge/reporting.py`:
  - `json_safe` handles NumPy scalars/arrays, sets, `Path`, and non-finite
    floats (→ `null`); JSON always carries `schema_version`; absolute paths are
    reduced to basenames unless `include_absolute_paths=True`.
  - Ranking uses exactly one metric (`precision`/`recall`/`f1`/`map50`) with
    `drop = baseline − perturbed`; missing and incompatible trials are unranked.
  - Markdown: scope/reproducibility, baseline metrics, ranked trials,
    family/operation groups, severity ladders (single numeric parameter only),
    failure counts (baseline, trial sum, top-ranked trials), warnings, and
    limitations. Quick reports show instability counts/per-image only.
  - Writers: `write_json_report`, `write_markdown_report`, `write_report`.
- [x] New `tests/test_failures.py` (23) and `tests/test_reporting.py` (14):
      every category, exact IoU boundary, configurable localization threshold,
      duplicates, empty GT/predictions, ambiguity, incomplete details,
      Quick separation/no labels, no-mutation, NumPy/non-finite JSON, path
      redaction, ranking/ties/missing/incompatible, groups, ladders, stable
      Markdown, writers.
- [x] Full suite: **511 passed** (was 474), CPU-only, no weights/network.
- [x] Docs: new `docs/failure-analysis.md` (categories, rules, schema, JSON
      policy, API, limitations); updated README, `architecture.md`,
      `mvp-spec.md`, and this log.

Known issues / notes:

- CSV failure/trial export is **not** implemented; the modular CSV writer from
  the stale preview was intentionally removed. The gallery arrived in Day 7.
- Annotation-correctness checks were completed in Day 7 (see below).
- `cli.py` remains stale and is not wired; no CLI commands are exposed.
- Class confusion relies on the evaluator's `wrong_class` diagnostic; duplicate
  detection is same-class only (different-class extras become class confusion or
  unmatched). Localization of matched detections is diagnostic and remains TP.
- Quick Mode has no ground truth; its counts are never accuracy degradation.
- Dataset identity is split/counts/class names (plus redacted root); no dataset
  content hash is computed.
- Rankings use one metric at a time and single-seed results are noisy; no
  universal damage claim is made.
- No benchmark results are claimed.

## Day 7 — Failure gallery & annotation correctness (2026-10-22): DONE

Objective: make verified failures and Quick instability inspectable offline, and
close the outstanding annotation-correctness checks without rebuilding the
evaluator, dataset loader, or perturbation engine.

- [x] Inspection: branch `main`, HEAD `46c7a6f`, remote in sync; suite **511
      passed** before changes. Reviewed evaluator/dataset/evaluation/failures/
      reporting plus the untracked `annotations.py` preview; replaced its
      single-row helper-only scope with a read-only validator while keeping the
      geometry helpers.
- [x] Completed `src/failureforge/annotations.py`:
  - `validate_annotations(dataset)` and `validate_label_file(...)` return
    deterministic `AnnotationIssue` records with severity, code, image/path,
    and line where available. Errors vs warnings are explicit; no repair or
    mutation.
  - Errors: malformed rows, non-numeric/non-finite geometry, negative/invalid
    class IDs, out-of-range class IDs, non-positive width/height, normalized
    coordinates outside `[0, 1]`, unreadable/orphan files, and missing labels
    (following `allow_missing_labels`).
  - Warnings: derived boxes extending outside the image canvas (with image
    dimensions when known) and overlapping annotations at IoU >= 0.85. Missing
    files are counted separately from explicitly empty files; empty files are
    valid zero-object images.
  - Counts/`per_class`/JSON-safe result record; unreadable images are reported
    without stopping label checks.
- [x] Additive Quick Mode detail: matched clean/perturbed records now carry
      boxes and scores for class changes, confidence changes, and box drift.
      `analyze_quick` propagates them when present and remains compatible with
      older records that only contain indices/classes.
- [x] Added `src/failureforge/gallery.py`:
  - Single offline HTML file (inline CSS/JS) from `BenchmarkRun`/`QuickRun`;
    verified Benchmark entries and Quick instability are labelled separately.
  - Shows sample/image, baseline-vs-trial context, category/kind, reason,
    metric role, classes/indices, boxes, score/IoU/thresholds, transformation
    name/family/params/seed, and trial rank.
  - Optional `clean_images`/`perturbed_images`: only referenced files are copied
    into `images/` under content-hashed relative names; no source paths in HTML.
    Boxes are SVG overlays at the artifact's real pixel dimensions.
  - No artifact -> explicitly labelled box diagram; no boxes -> graceful
    message. Out-of-bounds coordinates versus the artifact produce a warning.
  - HTML-escaped text/attributes, script-safe embedded JSON, deterministic
    ordering, `max_entries`, local filters, and `FileExistsError` unless
    `overwrite=True`.
  - Schema marker: `failureforge.failure-gallery.v1`.
- [x] New/expanded tests: `tests/test_annotations.py` (15) and
      `tests/test_gallery.py` (11) cover valid/empty/missing labels, all error and
      warning classes, boundaries, non-finite values, overlap warnings,
      no-mutation, Benchmark and Quick galleries, missing images/boxes, HTML
      escaping, relative paths/artifact dimensions, deterministic output,
      overwrite protection, truncation, and Quick accuracy-claim separation.
- [x] Full suite: **534 passed** (was 511), CPU-only, no weights/network.
- [x] Docs: new `docs/gallery.md` and `docs/annotations.md`; updated README,
      `docs/architecture.md`, `docs/mvp-spec.md`, `docs/failure-analysis.md`,
      and this log.

Known issues / notes:

- Gallery image maps are caller-supplied and trusted to match the coordinate
  space (clean for baseline/Quick-clean, transformed for trials/Quick-perturbed).
  Dimension mismatches are only heuristically detectable.
- Gallery shows run records only (default cap 500); it is not a full image
  browser. CSV export and a gallery CLI remain unimplemented.
- Annotation validation checks format/geometry/class vocabulary, not semantic
  label correctness. Overlap warnings are advisory (different-class overlaps
  may be legitimate).
- `cli.py` remains stale and is not wired.
- No benchmark results are claimed.

## Day 8 — Deterministic search objective (2026-10-22): DONE

Objective: define, implement, test, and document the scalar robustness score
future search algorithms will maximize, without implementing search itself.

- [x] Inspection: branch `main`, HEAD `1690589`, remote in sync; full suite
      **534 passed** before changes. Reviewed evaluator report fields, Benchmark
      deltas, `compare_reports`, reporting rankings, tests, and docs.
- [x] Added `src/failureforge/objective.py`:
  - `ObjectiveConfig(metric="f1")` supports exactly `precision`, `recall`, `f1`,
    and `map50`; invalid names raise `ValueError`. No weights or multi-objective
    composition.
  - `evaluate_objective(baseline, perturbed, config=...)` returns an
    `ObjectiveResult` with metric, explicit `higher_is_better` direction,
    baseline/perturbed values, signed `degradation`, `compatible`, `valid`, and
    warnings.
  - Score is `baseline − perturbed`; zero = unchanged, positive = worse,
    negative = improvement. Larger is always worse, so search maximizes it.
  - Delegates metric extraction/compatibility warnings to the existing
    `failures.compare_reports`; matching and metric semantics are unchanged.
  - Missing/non-finite metrics and incompatible IoU/aggregation settings produce
    `valid=False` and `degradation=None`, never a zero score. Valid zero remains
    distinct. Warning-only split/confidence/image-count differences preserve
    warnings and existing compatibility behavior.
  - `to_dict()` is deterministic and JSON-safe; computation needs no inference.
- [x] Added `tests/test_objective.py` (19 tests): default/explicit metric,
      equal/positive/negative degradation, all four metrics, selected-metric
      isolation, invalid config, missing aggregate/mAP metrics, NaN/inf,
      incompatible IoU/aggregation, combined missing+incompatible, warning-only
      differences, invalid-vs-valid-zero, deterministic JSON output.
- [x] Full suite: **553 passed** (was 534), CPU-only, no weights/network.
- [x] Docs: new `docs/objective.md`; updated README, `docs/architecture.md`,
      `docs/mvp-spec.md`, and this log.

Known issues / notes:

- The objective supports one higher-is-better metric at a time; no multi-objective
  score, seed aggregation, variance estimate, or confidence interval.
- Split/confidence/image-count differences remain warnings rather than hard
  incompatibilities, matching `compare_reports`; only IoU/aggregation differences
  force `compatible=False`.
- Failure counts and Quick Mode instability are intentionally not part of the
  objective.
- Random Search, Optuna/TPE, confirmation evaluation, CSV export, and CLI
  integration remain unimplemented.
- No benchmark results are claimed.

## Days 9–15 — Search, confirmation, CLI, demo, release audit (2026-10-25): DONE

Objective: finish the remaining roadmap after verifying Days 1–8 against the
code. Full requirement table and defect list: `docs/implementation-audit.md`.

- [x] Audit: HEAD `d53efb6`, suite **553 passed** before changes. The untracked
      `cli.py`, `config.py`, `search/` previews were stale (pre-Day-3 API,
      `-inf` for failed trials) and were rewritten.
- [x] Defect D1 fixed: `YoloDataset.iter_samples` silently treated missing label
      files as empty (inflating false positives); it now raises unless
      `allow_missing_labels=True`. Added `YoloDataset.partition` (deterministic
      disjoint discovery/holdout split).
- [x] Day 9 `search/` package: `SearchSpace` (from registry `ParamSpec`s, odd
      `motion_blur` kernels, narrowing-only overrides), `TrialRunner` (shared
      baseline, `evaluate_perturbed` + `evaluate_objective`), seeded
      `run_random_search`, trial states `ok/invalid/error` (never ranked, never
      zero), ranking with `unique=True` dedup.
- [x] Day 10 optional `run_tpe_search` (Optuna ask/tell, invalid → FAIL,
      `OptionalDependencyError` when absent).
- [x] Day 11 `confirm_trials`: top-k distinct configs re-scored on the holdout
      with fresh seeds (mean/std/min/max, `degrades_in_all_seeds`).
- [x] Day 12 search JSON/Markdown/CSV (`write_search_report`; CSV formula-safe);
      confirmed candidates flow through Day-6 analysis into `confirmation.*`.
- [x] Day 13 CLI (`list-perturbations`, `validate-data`, `evaluate`, `perturb`,
      `search`) + strict YAML config (`configs/default.yaml`); injectable
      detector factory for tests.
- [x] Day 14 `failureforge.demo` (`make_demo_dataset`, rule-based
      `ColorBlobDetector`), `examples/make_demo_data.py`, README quick start
      verified end to end.
- [x] Follow-up improvements (same day): `compare` command / `compare_methods`
      (equal-budget random-vs-TPE over seeds), `write_failures_csv` +
      `perturb_failures.csv`/`confirmation_failures.csv`, ruff config with all
      pyflakes findings fixed, `.gitattributes`, CI lint step. Suite: **608 passed**.
- [x] Improvement pass (2026-10-26, see `PROJECT_IMPROVEMENT_REPORT.md`): metrics
      validated against an independent reference (no bug); bit-identical speedups
      for rain/glare/lens_flare/motion_blur (search -13% on a synthetic benchmark);
      `blur` alias removed from the default space; search warnings. Suite: **668 passed**.
- [x] Day 15 verification: full suite, wheel build (all subpackages present),
      docs reconciled, CI updated (matrix + wheel build).
- [x] Tests: new `test_search.py` (22), `test_cli.py` (24), 3 dataset tests,
      1 detector test; `test_metrics.py` now tracked. Final suite: **599 passed**
      (`python -m pytest -q`, Python 3.14.3, ~6 s, CPU-only).

Known issues / notes:

- No real-YOLO smoke test: `ultralytics` and weights are unavailable here
  (never downloaded). The CLI's real-detector path is exercised only through
  `YOLODetector` unit tests with a mocked model.
- Only Python 3.14 was available locally; the 3.9/3.11/3.12 CI matrix is
  unrun. Optuna was tested at 5.0.0 only.
- Deferred: regenerating reports from saved JSON (audit R21).
- No benchmark results are claimed; demo numbers come from a rule-based detector.

## 15-day roadmap (planned milestones, not completed features)

- Day 1: Project foundation, scope, research plan, GitHub setup. ✅
- Day 2: YOLO detector wrapper (inference only). ✅
- Day 3: Dataset loading, validation & baseline evaluation. ✅
- Day 4: Robustness perturbation engine (all 12 families) + tests. ✅
- Day 5: Perturbation→evaluation adapter, Benchmark/Quick Mode, severity sweeps. ✅
- Day 6: Failure analysis, verified categories, Quick instability records,
  schema-versioned JSON/Markdown reporting. ✅
- Day 7: Failure gallery + annotation-correctness checks. ✅
- Day 8: Deterministic search objective over existing evaluation metrics. ✅
- Day 9: Random search. ✅
- Day 10: Optuna TPE search (optional dependency). ✅
- Day 11: Multi-seed confirmation on a held-out split. ✅
- Day 12: Search reports (JSON/Markdown/CSV), deduplication, confirmed-failure reports. ✅
- Day 13: CLI and integration. ✅
- Day 14: Documentation and reproducible demo. ✅
- Day 15: Verification and release audit. ✅ (see `implementation-audit.md` for deferred items)

## Next recommended task

Deferred items (see `docs/implementation-audit.md`): report regeneration from saved
JSON and a real-YOLO smoke test once local weights and `ultralytics` are available.
