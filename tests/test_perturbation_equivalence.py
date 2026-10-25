"""Optimised rain / glare / lens_flare must equal the original reference output.

The references below are the original (slow) per-pixel / full-mask algorithms.
Seeded output is a public contract, so the fast versions must be bit-identical.
"""
import numpy as np
import pytest

from failureforge.perturbations import apply


def _ref_rain(img, intensity, seed):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = img.astype(np.float32) * (1.0 - 0.2 * intensity)
    n = int(intensity * min(w, h))
    max_len = max(6, min(w, h) // 4)
    for _ in range(n):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        length = int(rng.integers(5, max_len + 1))
        slant = max(1, length // 3)
        for t in range(length):
            x, y = x0 + (t * slant) // length, y0 + t
            if 0 <= x < w and 0 <= y < h:
                out[y, x] = np.minimum(255.0, out[y, x] + 170.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _ref_mask(h, w, cx, cy, r):
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(1.0 - dist / max(1e-6, r), 0.0, 1.0).astype(np.float32)


def _ref_glare(img, intensity, radius, seed):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
    mask = _ref_mask(h, w, cx, cy, max(1.0, radius * min(w, h)))
    out = img.astype(np.float32) + intensity * 255.0 * mask[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _ref_flare(img, intensity, seed):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    lx, ly = float(rng.uniform(0, w)), float(rng.uniform(0, h))
    for i in range(5):
        f = (i + 1) / 6
        cx, cy = lx + (w * 0.5 - lx) * (2 * f), ly + (h * 0.5 - ly) * (2 * f)
        r = (0.05 + 0.1 * (1 - f)) * min(w, h)
        out = out + intensity * 120.0 * (1 - f) * _ref_mask(h, w, cx, cy, max(1.0, r))[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("seed", range(12))
def test_fast_paths_match_reference(seed):
    rng = np.random.default_rng(seed)
    h, w = (int(v) for v in rng.integers(8, 160, 2))
    img = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    inten = float(rng.uniform(0, 1))
    rad = float(rng.uniform(0.05, 1))
    assert np.array_equal(apply("rain", img, {"intensity": inten}, seed=seed).image,
                          _ref_rain(img, inten, seed))
    assert np.array_equal(apply("glare", img, {"intensity": inten, "radius": rad}, seed=seed).image,
                          _ref_glare(img, inten, rad, seed))
    assert np.array_equal(apply("lens_flare", img, {"intensity": inten}, seed=seed).image,
                          _ref_flare(img, inten, seed))


def test_rain_full_intensity_overlaps_saturate():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    out = apply("rain", img, {"intensity": 1.0}, seed=3).image
    assert np.array_equal(out, _ref_rain(img, 1.0, 3)) and out.max() == 255


def _ref_motion_blur(img, k, angle):
    from PIL import Image

    from failureforge.perturbations.base import bgr_to_pil, pil_to_bgr
    from failureforge.perturbations.boxops import translate_forward, warp_affine

    pil = bgr_to_pil(img)
    w, h = pil.size
    t = np.deg2rad(angle)
    dx, dy = float(np.cos(t)), float(np.sin(t))
    acc = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(k):
        off = i - k // 2
        acc += np.asarray(warp_affine(pil, translate_forward(off * dx, off * dy), (w, h)),
                          dtype=np.float64)
    acc /= float(k)
    return pil_to_bgr(Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8), mode="RGB"))


@pytest.mark.parametrize("k,angle", [(3, 0.0), (9, 33.3), (31, 90.0), (15, 171.0)])
def test_motion_blur_matches_float_reference(k, angle):
    img = np.random.default_rng(k).integers(0, 256, (48, 64, 3)).astype(np.uint8)
    out = apply("motion_blur", img, {"kernel_size": k, "angle": angle}, seed=0).image
    assert np.array_equal(out, _ref_motion_blur(img, k, angle))
