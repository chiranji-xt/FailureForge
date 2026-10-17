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

## Day 3 — Dataset loading, validation & baseline evaluation (2026-10-16): DONE

- [x] `src/failureforge/dataset.py`: YOLO-format loader with split-awareness
      (`_train`/`_val`/`_test` filename markers), class-name consistency check,
      validation (missing/corrupt files, malformed rows, out-of-range coords,
      orphan labels, empty GT images), deterministic ordering.
- [x] `src/failureforge/evaluator.py`: `evaluate_samples()` on original
      unmodified images, `PerImageResult` + `EvaluationResults` dataclasses,
      IoU ≥ 0.50 class-aware one-to-one matching, TP/FP/FN, precision/recall/F1,
      corpus mAP50 (pooled AP per class, then mean), diagnostic wrong-class and
      localization-error counts, zero‑denominator safe aggregation.
- [x] `tests/test_evaluator.py`: 6 tests (baseline perfect, missed, FP, duplicate,
      wrong-class, localization, multiple objects, empty GT/pred, precision/recall/F1,
      zero-denominators, aggregate across images, IoU boundary, deterministic matching).
- [x] `tests/test_detector.py`: 20 tests, fully mocked (no weights/network).
- [x] `tests/test_project.py`: 2 tests (import + version).
- [x] Full test suite: **49 passed** (all modules).
- [x] `pyproject.toml`: no change needed.

## 15-day roadmap (planned milestones, not completed features)

- Day 1: Project foundation, scope, research plan, GitHub setup. ✅
- Day 2: YOLO detector wrapper (inference only). ✅
- Day 3: Dataset loading, validation & baseline evaluation. ✅
- Day 4: Brightness and blur transformations.
- Day 5: Rotation, scaling, occlusion, annotation correctness.
- Day 6: Evaluation engine and metrics (revisited with new dataset layer).
- Day 7: Failure records and visual gallery.
- Day 8: Perturbation sweeps and objective definition.
- Day 9: Random search.
- Day 10: Optuna TPE search.
- Day 11: Multi-seed comparison and confirmation evaluation.
- Day 12: Failure categorization, deduplication, reports.
- Day 13: CLI and integration.
- Day 14: Documentation and reproducible demo.
- Day 15: Fresh-environment verification and release.

## Next (Day 4)

Brightness and blur transformations with deterministic seeded randomness and
correct annotation handling (geometry unchanged, clip-to-boundary policy).