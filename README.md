# FailureForge

Reproducible robustness testing for YOLO object detectors. Evaluates a detector
under controlled perturbations and compares **random search vs Optuna TPE** at
finding the most damaging configurations — decided by experiment, not assumption.

## Development status (Day 2)

- ✅ Day 1: project foundation (packaging, docs, smoke test).
- ✅ Day 2: **YOLO detector wrapper** (`YOLODetector`, inference only) —
  implemented, tested (20 mocked tests, no weights/network required), verified.
- ⏳ Days 3–15 per `docs/progress.md`: dataset loading, evaluation, and search
  are **planned, not yet verified**. The quick-start and CLI sections below
  describe the planned workflow and will fail until their scheduled day.

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

## Planned quick-start (available once Days 2–14 land; not working on Day 1)

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

## Planned CLI (Day 13; not implemented on Day 1)

- `failureforge validate-data --images ... --labels ...`
- `failureforge evaluate --images ... --labels ... [--weights best.pt] [--iou 0.5]`
- `failureforge search --method {random,tpe,both} --budget N --seed S [--weights best.pt]`
- `failureforge report --input-dir results --output results/report.md`
- Every command supports `--help`.

## Testing (Day 1)

```powershell
pytest tests/test_project.py -v
```

Smoke test only: verifies the package imports and exposes its version.
The full module test suite arrives with its modules on Days 2–11 and runs
CPU-only with synthetic fixtures (no weights or downloads).

## Limitations

- Axis-aligned boxes approximate rotation conservatively.
- Evaluates a given detector; does not enumerate all real-world failures.
- Single-seed comparisons are noisy — run multiple seeds.
- No fabricated benchmarks are included; run the pipeline to get real numbers.
