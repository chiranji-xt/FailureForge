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
- [x] Known tree state: untracked Day-6 preview `tests/test_evaluator.py`
      imports the old `DummyDetector` stub and fails collection; ignored for
      now (`--ignore`), everything else in tree passes (43 passed). Day 6
      rebuilds evaluator + its doubles/tests properly.

## 15-day roadmap (planned milestones, not completed features)

- Day 1: Project foundation, scope, research plan, GitHub setup. ✅
- Day 2: YOLO detector wrapper (inference only). ✅
- Day 3: Dataset loader, YOLO annotation validation, baseline evaluation.
  (Roadmap reordered: detector first so evaluation days can build on it.)
- Day 4: Brightness and blur transformations.
- Day 5: Rotation, scaling, occlusion, annotation correctness.
- Day 6: Evaluation engine and metrics.
- Day 7: Failure records and visual gallery.
- Day 8: Perturbation sweeps and objective definition.
- Day 9: Random search.
- Day 10: Optuna TPE search.
- Day 11: Multi-seed comparison and confirmation evaluation.
- Day 12: Failure categorization, deduplication, reports.
- Day 13: CLI and integration.
- Day 14: Documentation and reproducible demo.
- Day 15: Fresh-environment verification and release.

## Next (Day 3)

Implement + verify `dataset.py` (strict YOLO loading/validation, summary
stats, deterministic split, tests on synthetic fixtures) plus the baseline
evaluation harness that runs `YOLODetector` over the labeled set.
