# FailureForge — project improvement report

Date: 2026-10-26. Starting point: `main` at `144d802`, 608 tests passing.
Environment: Windows 11, Python 3.14.3, NumPy 2.4.3, CPU only, no YOLO weights.

## 1. Executive summary

FailureForge measures how a YOLO detector degrades under image perturbations and
searches for the most damaging configurations. This pass looked for evidence-backed
problems in correctness, evaluation validity, and speed. Findings:

- **No metric bug.** The matcher/AP code agrees with an independently written
  reference on 40 random scenes (exact counts, mAP to 1e-12).
- **Real efficiency problems** in three perturbations (`rain`, `lens_flare`/`glare`,
  `motion_blur`), fixed with **bit-identical** output (verified against the original
  algorithms).
- **Two search-validity gaps** fixed: a legacy duplicate transform (`blur`) that
  inflated the sampling weight of Gaussian blur, and silence when a search is
  uninformative (baseline metric 0 or no trial degrades the metric).

A real-model evaluation was not possible (no weights); every number below comes
from the synthetic rule-based `ColorBlobDetector`, so timings measure the
perturbation/search pipeline, not YOLO inference.

## 2. Original problems (verified)

| # | Problem | Evidence |
|---|---|---|
| P1 | `rain` used a Python per-pixel loop | 104 ms/image at 640x640, intensity 1.0 |
| P2 | `lens_flare`/`glare` built five/one full-size `sqrt` masks although the mask is zero outside a small circle | 110 ms / 28 ms |
| P3 | `motion_blur` accumulated in float64 | 450 ms at kernel 31 (PIL warps are 296 of that) |
| P4 | `blur` is the legacy alias of `gaussian_blur` (same output, range 0-5 inside 0-10) yet sat in the default search space | code inspection of `blur.py` |
| P5 | A search on a detector with baseline F1 = 0 (wrong weights/classes/labels) returned "valid" all-zero scores with no warning | reasoned from `evaluate_objective` semantics; reproduced in test |
| P6 | Initial benchmark under `tracemalloc` overstated rain cost (11 s/trial) — a measurement artefact, discarded | re-measured without tracing |

Checked and found **correct**: IoU/matching/AP (independent reference), RNG draw
order contract, per-trial seed derivation, objective direction and invalid handling.

## 3. Root-cause analysis

- P1-P3: algorithmic cost, not I/O. Profile of a 20-trial search: perturbation
  `apply` was ~50% of wall time, detector stub ~35%, image decode ~10%.
- P4: `available()` lists registered names, so any alias is sampled as a separate
  operation.
- P5: degradation is absolute (`baseline - perturbed`); with baseline 0 it cannot
  be positive, but nothing told the user.

## 4. Implemented changes

| File | Change |
|---|---|
| `perturbations/weather.py` | rain: vectorised rendering; RNG draws unchanged. A pixel hit once gains +170 (capped), hit >=2 times is 255 — identical to sequential capping |
| `perturbations/lens.py` | masks computed only in the circle's bounding window (`_circle_window`, `_add_spot`) |
| `perturbations/blur.py` | motion blur accumulates exact `uint32` sums instead of float64 |
| `search/space.py` | `blur` excluded from the default space (still selectable via `include`) |
| `search/runner.py`, `trials.py`, `report.py`, `cli.py` | `SearchResult.warnings` (baseline 0 / very low / no degrading trial / no valid trial), shown in JSON, Markdown, and CLI stderr |
| `tests/test_metrics_reference.py` | 40 scenes vs independent reference |
| `tests/test_perturbation_equivalence.py` | 17 tests: new rain/glare/lens_flare/motion_blur equal the original algorithms bit-for-bit |
| `tests/test_search.py` | 3 tests for the alias and warnings |

Public APIs and data formats are unchanged. Seeded perturbation output is identical
to before. The default random-search *sample stream* changes (one fewer
transform name), so seeds from before this change pick different configurations.

## 5. Experimental results

Conditions: 20 synthetic 640x640 images, `ColorBlobDetector`, random search,
budget 30, seed 0, `blur` excluded in both runs so the sample stream is identical.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Search wall time, 3 runs (s) | 22.62 / 23.87 / 24.75 (mean 23.75) | 20.30 / 20.00 / 21.51 (mean 20.60) | -13% |
| Trial scores (hash of 30 scores) | 4033208005107677544 | 4033208005107677544 | identical |
| `rain` default params (ms/img) | 39.1 | 15.4 | -61% |
| `rain` intensity 1.0 (ms/img) | 104 | 20 | -81% |
| `lens_flare` default (ms/img) | 101.0 | 4.7 | -95% |
| `glare` default (ms/img) | 21.2 | 5.1 | -76% |
| `motion_blur` k=31 (ms/img) | 450 | 389 | -14% |
| `motion_blur` default k=9 (ms/img) | 130.2 | 136.0 | within noise |
| Tests passing | 608 | 668 | +60 tests |
| Metric vs reference | not checked | 40/40 scenes exact | — |

Timings are best-of-3 on a shared desktop; differences under ~10% (e.g. default
motion blur) are noise. With a real YOLO model, inference time will dominate and
the end-to-end gain will be smaller than 13%.

## 6. Tests performed

- `python -m pytest -q` -> 668 passed (Python 3.14.3).
- `ruff check src tests` -> clean (pyflakes + syntax rules).
- Equivalence checks: 180 random (image, params, seed) cases compared old vs new
  implementations during development (0 mismatches), now frozen as tests.
- Not run: real YOLO weights, Python 3.9/3.11/3.12, GitHub Actions.

## 7. Improvements not yet implemented

| Item | Priority | Why deferred |
|---|---|---|
| Cache decoded images across trials (~10% of search time) | Medium | Memory trade-off for large datasets; needs a size cap design |
| Faster `motion_blur` (PIL warps dominate) | Low | Exact-equivalence is hard; 389 ms only at k=31 |
| `snow` vectorisation | Low | RNG draws are interleaved per flake; batching would change seeded output |
| Relative degradation (drop / baseline) as an optional score | Medium | Changes objective semantics; needs a requirement |
| Real-YOLO smoke test and calibration/threshold study | High | Needs weights and `ultralytics` |
| Report regeneration from saved JSON | Low | No consumer |

## 8. Known limitations

- Performance numbers use a stub detector; YOLO inference cost is unmeasured.
- Axis-aligned remapping of rotated boxes inflates ground truth for non-square
  objects, so rotation/perspective scores partly reflect label geometry.
- Search scores are optimistic for the selected configurations; use confirmation.
- Python 3.9 compatibility and CI remain unverified.

## 9. How to run

```powershell
pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python examples\demo_search.py --out demo_out
```

## 10. Recommended next steps

1. Run `failureforge evaluate` / `search` with real weights and check the new
   warnings (baseline sanity).
2. Add the image cache with a memory cap and re-measure on real inference.
3. Decide whether a relative-degradation objective is wanted.
