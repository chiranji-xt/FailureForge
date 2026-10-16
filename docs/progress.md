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

## 15-day roadmap (planned milestones, not completed features)

- Day 1: Project foundation, scope, research plan, GitHub setup. ✅
- Day 2: Dataset loader and YOLO annotation validation.
- Day 3: Detector interface and baseline evaluation.
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

## Next (Day 2)

Implement + verify `dataset.py`: strict YOLO loading/validation with clear
errors, summary statistics, deterministic split helper, and unit tests using
temporary synthetic fixtures.
