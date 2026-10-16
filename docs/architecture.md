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
| `transforms.py` | 5 seeded perturbations with documented ranges |
| `annotations.py` | Box geometry helpers (YOLO ↔ xyxy, clip, validation) |
| `metrics.py` | Greedy confidence-sorted class-aware IoU matching; corpus mAP50 |
| `evaluator.py` | Baseline vs perturbed evaluation on transformed GT |
| `failures.py` | Failure records (CSV/JSON) + visual gallery |
| `search/random_search.py` | Seeded uniform sampling over the shared search space |
| `search/tpe_search.py` | Optuna TPESampler over the same space/objective |
| `reporting.py` | Markdown report distinguishing observation from conclusion |
| `cli.py` | `validate-data`, `evaluate`, `search`, `report` commands |

## Implemented vs planned (Day 1 status)

- **Implemented (Day 1):** project foundation — packaging (`pyproject.toml`,
  `src/` layout, version `0.1.0`), license, `.gitignore`, smoke test,
  this documentation set, `configs/default.yaml` placeholder values.
- **Planned (Days 2–15):** all functional modules above, to be implemented,
  tested, and verified one stage at a time per the roadmap in `progress.md`.

## Dependency boundaries

- Core: `numpy`, `Pillow`, `PyYAML` (minimal, CPU-only, Windows-friendly).
- Optional: `optuna` (TPE search), `ultralytics` (real YOLO inference).
- Core tests must never require weights, downloads, or GPUs.
