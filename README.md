# FailureForge

Reproducible robustness testing for YOLO object detectors. Evaluates a detector
under controlled perturbations and compares **random search vs Optuna TPE** at
finding the most damaging configurations — decided by experiment, not assumption.

## Development status (Day 5)

- ✅ Day 1: project foundation (packaging, docs, smoke test).
- ✅ Day 2: **YOLO detector wrapper** (`YOLODetector`, inference only) —
  implemented, tested (20 mocked tests, no weights/network required), verified.
- ✅ Day 3: **Dataset loading, validation & baseline evaluation** — `YoloDataset`
  with real `data.yaml`/split-directory support, name-based class mapping, strict
  validation (malformed rows, bad coords, missing/orphan/corrupt files, class
  mismatch) and pixel-coordinate conversion; `evaluate_baseline` reports
  IoU≥0.50 matching, TP/FP/FN, micro P/R/F1, per-class AP and mAP50 plus
  wrong-class / poor-localization diagnostics.
- ✅ Day 4: **Robustness perturbation engine** — `failureforge.perturbations`
  implements **12 families / 39 transforms** over BGR `uint8` images (noise,
  exposure, blur, weather, colour, occlusion, geometric, quality, shadows,
  lens, cropping, `combined`), seeded and reproducible, with correct
  pixel-space ground-truth remapping for geometric/cropping ops. `transforms.py`
  is retained as a compatibility shim. Fully tested — **455 tests pass** across
  all modules, no weights/network required. See `docs/perturbations.md`.
- ✅ Day 5: **Perturbation→evaluation adapter** — `failureforge.evaluation`
  adds **Benchmark Mode** (metric deltas vs transformed GT, baseline computed
  once), **Quick Mode** (label-free prediction instability: disappeared,
  appeared, class/confidence changes, box drift — never accuracy metrics), and
  deterministic **severity sweeps** with attributable per-trial errors.
  `evaluator.py` was refactored behind an unchanged `evaluate_baseline`.
  Full suite: **474 tests pass**, CPU-only, no weights/network.
- ⏳ Days 6–15 per `docs/progress.md`: failure records/gallery, search
  algorithms, reporting, and CLI are **planned, not yet verified**. The
  quick-start and CLI sections below describe the planned workflow.

## Perturbation evaluation (Day 5)

```python
from failureforge.dataset import YoloDataset
from failureforge.evaluation import evaluate_perturbed, quick_mode, run_severity_sweep

ds = YoloDataset.from_config("data/data.yaml", split="val")

# Benchmark Mode: labeled data, deltas are perturbed - baseline
res = evaluate_perturbed(det, ds, "gaussian_noise", {"sigma": 25}, seed=0)
print(res.deltas["f1"], res.deltas["map50"])
print(res.to_dict())

# Quick Mode: no label files needed; instability only (no accuracy claims)
q = quick_mode(det, ds, "brightness", {"factor": 0.4})
print(q.summary["disappeared"], q.summary["class_changed"])

# Explicit severity ladder; reusable baseline; per-trial reproducible seeds
sweep = run_severity_sweep(
    det, ds, "gaussian_noise",
    [{"sigma": s} for s in (5, 10, 20, 30)], seed=0,
)
```

Transforms are applied with `sample.boxes` (pixel `GroundTruthBox` objects);
geometric/cropping transforms therefore evaluate against correctly remapped
ground truth. A failed sweep trial is recorded with its index/config/error
instead of aborting the sweep (`strict=True` re-raises). Deltas are observations
for one detector/dataset/seed, not proof of real-world failure.

## Perturbation engine (Day 4)

```python
import numpy as np
from failureforge.perturbations import apply, available, families
from failureforge.dataset import GroundTruthBox

img = np.zeros((64, 64, 3), dtype=np.uint8)          # BGR uint8 (H, W, 3)
boxes = [GroundTruthBox(0, 10, 10, 40, 40)]          # pixel [x1,y1,x2,y2]

res = apply("rotation", img, {"angle": 30}, boxes=boxes, seed=0)
res.image          # new BGR uint8 array (input never mutated)
res.boxes          # re-mapped, clipped GroundTruthBox list
res.metadata       # name, family, params, seed, geometry_changed, diagnostics

print(len(available()), families())                  # 39, {family: [...]}
```

Non-geometric transforms leave ground truth unchanged; geometric and cropping
transforms remap all four corners, clip to the canvas, and drop degenerate boxes
(recorded in `metadata["removed_boxes"]`). Full parameter reference:
`docs/perturbations.md`.

## YOLO detector (Day 2, inference only)

Loads your existing detection checkpoint (e.g. `best.pt`) — never trains,
never downloads. Optional dependency: `pip install -e ".[yolo]"`.

```python
import cv2
from failureforge.detector import YOLODetector

det = YOLODetector("best.pt", conf=0.25, imgsz=640, device="cpu")
print(det.class_names, det.task, det.sha256[:12])

img_bgr = cv2.imread("frame.png")          # OpenCV-style BGR uint8 (H, W, 3)
dets = det.predict(img_bgr)                # Detections: boxes_xyxy (N,4) float32
print(dets.boxes_xyxy, dets.scores, dets.classes)
print(det.metadata)                        # weights, config, task, fingerprint
```

Only the `detect` task is supported; segmentation/pose/OBB checkpoints are
rejected with an error. Tests: `pytest tests/test_detector.py tests/test_project.py -v`.

## Installation (Windows PowerShell)

```powershell
cd D:\FailureForge
pip install -e .
# optional extras:
pip install -e ".[optuna]"   # TPE search
pip install -e ".[yolo]"      # Ultralytics YOLO inference
pip install -e ".[dev]"       # tests + optuna
```

## Planned quick-start (not yet working; CLI is Day 13)

```powershell
# 1. create a tiny demo dataset
python examples\make_demo_data.py --out data
# 2. validate
failureforge validate-data --images data\images --labels data\labels
# 3. baseline evaluate (DummyDetector via no --weights)
failureforge evaluate --images data\images --labels data\labels --output results\baseline.json
# 4. search (random + TPE) with confirmation split
failureforge search --images data\images --labels data\labels --method both --budget 10 --output-dir results
# 5. regenerate report
failureforge report --input-dir results --output results\report.md
```

## Planned CLI (Day 13; not implemented)

- `failureforge validate-data --images ... --labels ...`
- `failureforge evaluate --images ... --labels ... [--weights best.pt] [--iou 0.5]`
- `failureforge search --method {random,tpe,both} --budget N --seed S [--weights best.pt]`
- `failureforge report --input-dir results --output results/report.md`
- Every command supports `--help`.

## Testing

```powershell
pytest -q
```

The suite is CPU-only and uses synthetic fixtures and mocked detectors (no
weights, downloads, or network). Current status: **474 passed**.

## Limitations

- Axis-aligned boxes approximate rotation conservatively.
- Evaluates a given detector; does not enumerate all real-world failures.
- Single-seed comparisons are noisy — run multiple seeds.
- No fabricated benchmarks are included; run the pipeline to get real numbers.
