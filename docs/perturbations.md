# FailureForge — perturbation reference

The engine lives in `failureforge.perturbations`. It operates on **OpenCV-style
BGR `uint8` arrays** shaped `(H, W, 3)` (the same format
`YOLODetector.predict` consumes) and represents ground truth as
`failureforge.dataset.GroundTruthBox` objects in **pixel** `[x1, y1, x2, y2]`.

```python
import numpy as np
from failureforge.perturbations import apply, available, families
from failureforge.dataset import GroundTruthBox

img = np.zeros((64, 64, 3), dtype=np.uint8)
boxes = [GroundTruthBox(0, 10, 10, 40, 40)]

res = apply("rotation", img, {"angle": 30}, boxes=boxes, seed=0)
res.image    # new BGR uint8 array (input never mutated)
res.boxes    # re-mapped, clipped GroundTruthBox list
res.metadata # {"name", "family", "params", "seed", "geometry_changed", ...}
```

## Contract

All transforms: accept a valid BGR `uint8` image; return a valid BGR `uint8`
image; never mutate the input; validate parameters with clear errors; produce
identical output for identical `(image, params, seed)`; and record name, family,
params, seed, and geometry flag in `metadata`. A **falsey seed is still a seed**
(default `seed=0`); stochastic transforms are reproducible.

- `apply(name, image, params=None, boxes=None, seed=0) -> PerturbationResult`
- `available() -> list[str]`, `families() -> dict[str, list[str]]`
- `validate_params(name, params) -> dict` (fills defaults; raises
  `PerturbationError` on unknown/missing/out-of-range values)
- `param_specs() -> dict[str, dict[str, ParamSpec]]`

Parameters marked **required** have no default and must be supplied; the rest
default to the value shown. Ranges are **configurable experimental ranges**, not
universal standards.

## Geometry / label policy

Non-geometric transforms leave box geometry untouched. Geometric and cropping
transforms map all four box corners with the same matrix/point map used for the
image, rebuild the axis-aligned box, clip to the canvas, and **drop** boxes whose
side falls below `MIN_BOX_SIDE_PX = 1.0` (recorded in `metadata["removed_boxes"]`
with a reason). Class IDs follow their object. Cropping changes the canvas size;
`translation` keeps the canvas (black fill); `shifted_framing` keeps the canvas
and wraps exposed edges.

## Families and parameters

### 1. Noise
| Name | Params (range, default) |
|---|---|
| `gaussian_noise` | `sigma` (0–50, 15) additive std-dev |
| `salt_pepper_noise` | `amount` (0–0.2, 0.02) affected fraction; `salt_ratio` (0–1, 0.5) |
| `speckle_noise` | `sigma` (0–0.5, 0.1) multiplicative strength |

### 2. Brightness / exposure
| Name | Params |
|---|---|
| `darken` | `factor` (0–1, 0.5) gain ≤ 1 |
| `brighten` | `factor` (1–3, 1.5) gain ≥ 1, then clip |
| `overexposure` | `strength` (0–1, 0.5) saturating highlight gain |
| `brightness` | `factor` (0.3–2.0, 1.0) generic gain (legacy name) |

### 3. Blur
| Name | Params |
|---|---|
| `gaussian_blur` | `radius` (0–10, 2) Gaussian sigma |
| `defocus_blur` | `radius` (0–10, 3) box (disk-like) radius |
| `motion_blur` | `kernel_size` (odd 3–31, 9), `angle` (0–180, 0) degrees |
| `blur` | `radius` (0–5, 2) Gaussian (legacy name) |

### 4. Weather / visibility (approximations)
| Name | Params |
|---|---|
| `fog` | `intensity` (0–1, 0.4) blend toward white |
| `haze` | `intensity` (0–1, 0.3) blend + contrast compression |
| `rain` | `intensity` (0–1, 0.4) streak density |
| `snow` | `intensity` (0–1, 0.4) flake density |

### 5. Colour distortion
| Name | Params |
|---|---|
| `hue_shift` | `degrees` (−180–180, 30) via HSV |
| `saturation` | `factor` (0–2, 1.5); 0 = grayscale |
| `contrast` | `factor` (0–2, 1.5) |
| `color_temperature` | `warmth` (−1 cool … +1 warm, 0.3) |

### 6. Occlusion
| Name | Params |
|---|---|
| `occlusion` (alias `cutout`) | `ratio` (0–0.6, 0.2) side/min-side; `remove_fully_occluded` (False); `full_threshold` (0–1, 0.85) |
| `random_patch` | `ratio` (0–0.6, 0.15), `num_patches` (1–10, 4) |
| `partial_obstruction` | `ratio` (0–0.6, 0.2), `side` (top/bottom/left/right, bottom) |

### 7. Geometric (boxes transformed)
| Name | Params |
|---|---|
| `rotation` | `angle` (−90–90, 15) CCW; canvas kept |
| `scaling` | `scale` (0.5–2.0, 1.2) zoom about centre |
| `translation` | `dx` (−0.5–0.5, 0.1), `dy` (−0.5–0.5, 0) fraction of size |
| `perspective` | `jitter` (0–0.15, 0.05) corner jitter fraction |

### 8. Image quality
| Name | Params |
|---|---|
| `jpeg_compression` | `quality` (1–100, 25) |
| `downscale` | `scale` (0.1–0.9, 0.4) down then up |
| `pixelate` | `pixel_size` (2–32, 8) block side |

### 9. Shadows / lighting (approximations)
| Name | Params |
|---|---|
| `cast_shadow` | `intensity` (0–1, 0.5), `coverage` (0.05–1, 0.4), `angle` (0–180, 45) |
| `uneven_illumination` | `strength` (0–1, 0.4), `angle` (0–180, 0) |
| `local_darkening` | `intensity` (0–1, 0.5), `radius` (0.05–1, 0.3) |

### 10. Lens effects (approximations)
| Name | Params |
|---|---|
| `glare` | `intensity` (0–1, 0.6), `radius` (0.05–1, 0.25) |
| `lens_flare` | `intensity` (0–1, 0.4) |
| `vignetting` | `strength` (0–1, 0.5) |

### 11. Cropping / framing (boxes transformed)
| Name | Params |
|---|---|
| `random_crop` | `crop_fraction` (0.1–0.9, 0.7) kept fraction of both sides |
| `edge_clip` | `amount` (0–0.3, 0.1), `side` (top/bottom/left/right, bottom) |
| `shifted_framing` | `dx` (−0.5–0.5, 0.1), `dy` (−0.5–0.5, 0) |

### 12. Combined
| Name | Params |
|---|---|
| `combined` | `steps` (required): ordered list of `{"name", "params"}` |

Steps run **in order** (order affects the result). Each step gets a deterministic
child seed drawn from the request RNG, so a composition is reproducible from
`(steps, seed)`. Example compositions: `gaussian_noise` + `gaussian_blur`;
`darken` + `gaussian_noise`; `rain` + `contrast`; `gaussian_blur` +
`jpeg_compression`. Logic is reused from the individual families — nothing is
reimplemented.

## Search space

`failureforge.transforms.PARAM_RANGES` exposes the legacy
`{name: {param: (low, high)}}` view for the five original transforms, derived
from the engine specs. Days 9–10 (random vs TPE search) are **not** implemented
here.

## Reproducibility & metadata

Every result carries `metadata` with `name`, `family`, `params`, `seed`,
`geometry_changed`, `input_shape`, `output_shape`, plus transform-specific
diagnostics (`occluders`, `removed_boxes`, `order`/`components` for combined).
For experiment records, combine this with the detector `sha256`, dataset
identifier, and metrics — later roadmap stages.

## Limitations

- Weather, shadows, glare, and lens effects are **approximations**, not
  physically accurate simulations; they must not be interpreted as real-world
  failure probabilities.
- Occlusion coverage fraction uses a rectangle-intersection approximation.
- `motion_blur` is an average-of-shifts streak, not a true convolution kernel.
- `scaling`/`rotation` keep the original canvas (no expansion); content moved
  off-canvas is clipped.
- Parameter ranges are experimental defaults, not standards.
