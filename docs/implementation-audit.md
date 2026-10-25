# FailureForge — implementation audit

Audit date: 2026-10-25. Starting point: branch `main`, HEAD `d53efb6`
("feat: add robustness search objective"), in sync with `origin/main`.

## Method

- Inspected tracked source, untracked previews, docs, and git history; compared
  the 15-day roadmap in `docs/progress.md` with the code.
- Ran `python -m pytest -q` **before** changes: **553 passed** (Python 3.14.3,
  Windows, CPU-only; no weights, network, or GPU).
- Historical claims in `progress.md` (Days 1–8, "511 tests", etc.) were treated
  as claims; the code and tests below were checked directly. Days 7 and 8 are
  genuinely implemented and committed (`1690589`, `d53efb6`).
- Untracked at start: `cli.py`, `config.py`, `search/`, `tests/test_search.py`,
  `tests/test_metrics.py`, `configs/`, `examples/`, `.github/`. `cli.py`,
  `config.py`, and `search/` were **stale previews** written against the
  pre-Day-3 API (`load_detector`, `summarize`, `evaluate_samples`, normalized
  dicts, `-inf` for failed trials). They were rewritten, not reused.

## Defects found and fixed

| # | Defect | Evidence | Fix |
|---|---|---|---|
| D1 | `YoloDataset.iter_samples` silently treated a **missing** label file as an empty label even with `allow_missing_labels=False`, so `evaluate_baseline` would count every detection on such an image as a false positive. `validate()` flagged it, but nothing forced callers to validate. | reproduced by test `test_iteration_rejects_missing_label_by_default` (failed before fix) | `iter_samples` raises `DatasetError` unless `allow_missing_labels=True`; empty files remain valid zero-object labels. Compatible: valid datasets are unaffected; Quick Mode never reads labels. |
| D2 | Stale preview search sampled `motion_blur.kernel_size` as a float/even value; the engine rejects even kernels (1 of 12 random trials errored). | found by running the new search end to end | search space samples odd kernels (`ODD_PARAMS`); 3000-sample test asserts every sample validates. |
| D3 | Stale preview search recorded failed trials as score `-inf` (could be ranked/best) and TPE returned `-inf` to Optuna. | code review of `search/*.py` | new search records `error`/`invalid` states, never ranks them, tells Optuna FAIL. |
| D4 | No test that `YOLODetector.predict` leaves the caller's image unmodified. | gap in `test_detector.py` | test added (passes; no behavior change needed). |

## Requirement audit

Status key: **C** complete and verified · **F** implemented but needs fixes ·
**P** partially implemented · **N** not implemented · **B** blocked by
dependency/environment · **D** deferred with justification.
"Verified" means automated tests pass in this audit run (598+ tests, final count
in `progress.md`).

| ID | Requirement | Source files | Verification / tests | Status | Notes / action |
|---|---|---|---|---|---|
| R01 | Package, packaging, editable install, CI | `pyproject.toml`, `.github/workflows/tests.yml` | `tests/test_project.py`; wheel built and inspected (all subpackages present) | C | CI updated to also build the wheel. CI itself has not run on GitHub from this machine. |
| R02 | Detector wrapper: BGR input validation, shaped boxes/scores/classes, class names, empty predictions, unsupported tasks, no input mutation | `detector.py` | `test_detector.py` (21, mocked Ultralytics) | C | D4 test added. |
| R03 | Real YOLO smoke test with local weights | `detector.py` | — | B | `ultralytics` and weights not installed/available; weights are never downloaded. Wrapper is tested via a mocked model only. |
| R04 | Dataset config parsing, split preference (never implicit train), path resolution, class mapping by name | `dataset.py` | `test_dataset.py` | C | |
| R05 | Missing vs empty labels, malformed rows, invalid class ids, bad/non-finite/out-of-range coords, zero-size boxes, missing/corrupt images | `dataset.py`, `annotations.py` | `test_dataset.py`, `test_annotations.py` | C (after D1) | Errors vs warnings distinguished; nothing auto-repaired. |
| R06 | Perturbation engine: 12 families/39 transforms, seeded, validated, no mutation, geometry-aware boxes, removed-box metadata | `perturbations/` | `test_transforms.py` (376) | C | Weather/lens/shadow effects are approximations (documented). |
| R07 | Baseline evaluation: IoU, confidence-ordered one-to-one class-aware matching, TP/FP/FN, P/R/F1, mAP@0.5, per-class/image, empty cases | `metrics.py`, `evaluator.py` | `test_evaluator.py`, `test_metrics.py` | C | `test_metrics.py` (previously untracked) covers legacy helpers; now tracked. |
| R08 | Perturbed evaluation: Benchmark Mode, Quick Mode (label-free), severity sweeps, deterministic seeds, attributable errors | `evaluation.py` | `test_evaluation.py` | C | Instability kept separate from accuracy. |
| R09 | Failure analysis: FN, FP, class confusion, localization, duplicate, unmatched, Quick instability; counts agree with report | `failures.py` | `test_failures.py` | C | |
| R10 | JSON/Markdown reporting: valid JSON, NumPy/non-finite safe, path redaction, ranking excludes invalid trials but lists them | `reporting.py` | `test_reporting.py` | C | |
| R11 | Offline HTML gallery with escaping, relative artifacts, missing-artifact fallbacks | `gallery.py` | `test_gallery.py`; CLI `--gallery` test | C | Not a full image browser (capped, run-record based). |
| R12 | Search objective: deterministic, higher = worse, compatibility, invalid ≠ zero | `objective.py` | `test_objective.py` | C | Day 8 verified genuinely present. |
| R13 | Search space + parameter validation | `search/space.py` | `test_search.py` | C | D2. |
| R14 | Random Search: seeded, budget, trial tracking, invalid handling, ranking, best | `search/random_search.py`, `runner.py`, `trials.py` | `test_search.py` | C | |
| R15 | Optuna TPE (optional dependency) | `search/tpe_search.py` | `test_search.py` (ran with Optuna 5.0.0; skipped if absent); missing-dependency error test | C | Core works without Optuna. Other Optuna versions not tested. |
| R16 | Held-out multi-seed confirmation (Day 11) | `search/confirm.py`, `YoloDataset.partition` | `test_search.py`, `test_dataset.py`, CLI e2e | C | No pass/fail threshold invented. |
| R17 | Failure categorization of search winners + deduplication (Day 12) | `search/trials.py` (`unique`), CLI `confirmation.*` | `test_search.py`, `test_cli.py` | C | Dedup is of near-identical *configurations* (4-decimal rounding); failure records are categorized by the Day-6 analysis of confirmed candidates. |
| R18 | Search reports: JSON, Markdown, CSV | `search/report.py` | `test_search.py` | C | CSV is formula-injection safe. |
| R19 | CSV export of per-image failure records | — | — | D | Not required by acceptance criteria beyond trial CSV; JSON carries the full records. |
| R20 | CLI: install/configure/baseline/perturb/inspect/search/report/gallery | `cli.py`, `config.py` | `test_cli.py` (24) incl. synthetic end-to-end | C | Real-YOLO path is covered only by the factory seam (R03). Stale `report` subcommand removed (reports are written by `perturb`/`search`). |
| R21 | Regenerating reports from saved JSON | — | — | D | No consumer needs it; reports are produced at run time. Removed from docs rather than shipped half-working. |
| R22 | Automated random-vs-TPE comparison report | — | — | D | Original research question; run both methods with the same seed/budget/space and compare `search.json` files. A comparison harness needs multi-seed statistics beyond the MVP. |
| R23 | Weight-free demo | `demo.py`, `examples/make_demo_data.py` | `test_cli.py`, `test_search.py` | C | `ColorBlobDetector` is a rule-based stand-in, not a model. |
| R24 | Docs agree with implementation | README, `docs/*` | manual review in this audit | C | |
| R25 | Python ≥ 3.9 support (`requires-python`) | all | — | B | Only Python 3.14 is installed locally; CI matrix (3.9/3.11/3.12) added but unrun. |
| R26 | Lint/type-check configuration | — | `ruff` run ad hoc only | D | Project has no lint/type config. Pre-existing minor findings (unused imports/variables) were not changed to keep the diff focused. |

## Final verification

See `docs/progress.md` (Days 9–15) for the final test count and commands.
