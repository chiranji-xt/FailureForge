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

- **Implemented (Days 1–2):** project foundation plus `detector.py`
  (`YOLODetector` inference-only wrapper, `Detections`, task gate, SHA-256
  fingerprint) with mocked unit tests.
- **Planned (Days 3–15):** all other functional modules above, to be
  implemented, tested, and verified one stage at a time per `progress.md`.

## Dependency boundaries

- Core: `numpy`, `Pillow`, `PyYAML` (minimal, CPU-only, Windows-friendly).
- Optional: `optuna` (TPE search), `ultralytics` (real YOLO inference).
- Core tests must never require weights, downloads, or GPUs.
