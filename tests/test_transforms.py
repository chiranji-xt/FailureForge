"""Perturbation-engine tests (deterministic, no weights/network).

Covers the engine contract (shape/dtype/no-mutation/determinism/validation),
every implemented family, geometric label correctness, and the legacy
``failureforge.transforms`` compatibility wrapper.
"""
import numpy as np
import pytest
from PIL import Image

from failureforge.dataset import GroundTruthBox
from failureforge.perturbations import (
    PerturbationError, apply, available, families, param_specs, validate_params,
)
from failureforge.transforms import PARAM_RANGES, apply_perturbation

ALL = available()
NON_COMBINED = [n for n in ALL if n != "combined"]


def _img(h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _defaults(name):
    return {p: s.default for p, s in param_specs()[name].items() if s.default is not None}


# ---------------------------------------------------------------------------
# Contract / general
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NON_COMBINED)
def test_output_shape_dtype_range(name):
    out = apply(name, _img(), _defaults(name), seed=0).image
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.min() >= 0 and out.max() <= 255


@pytest.mark.parametrize("name", NON_COMBINED)
def test_input_not_mutated(name):
    img = _img()
    before = img.copy()
    apply(name, img, _defaults(name), seed=0)
    assert np.array_equal(img, before)


@pytest.mark.parametrize("name", NON_COMBINED)
def test_deterministic_same_seed(name):
    img = _img()
    a = apply(name, img, _defaults(name), seed=7).image
    b = apply(name, img, _defaults(name), seed=7).image
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", NON_COMBINED)
@pytest.mark.parametrize("size", [(1, 1), (2, 3), (3, 2), (5, 5)])
def test_small_images_safe(name, size):
    out = apply(name, _img(size[0], size[1]), _defaults(name), seed=0).image
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3


@pytest.mark.parametrize("name", NON_COMBINED)
def test_empty_boxes_supported(name):
    result = apply(name, _img(), _defaults(name), boxes=[], seed=0)
    assert result.boxes == []


@pytest.mark.parametrize("name", NON_COMBINED)
def test_metadata_round_trip(name):
    params = _defaults(name)
    meta = apply(name, _img(), params, seed=3).metadata
    assert meta["name"] == name
    assert meta["seed"] == 3
    assert meta["family"] in families()
    assert meta["params"] == validate_params(name, params)


def test_unknown_transform_raises():
    with pytest.raises(PerturbationError):
        apply("does_not_exist", _img())


def test_invalid_params_raise():
    with pytest.raises(PerturbationError):
        validate_params("gaussian_noise", {"nope": 1})
    with pytest.raises(PerturbationError):
        validate_params("gaussian_noise", {"sigma": 999})
    with pytest.raises(PerturbationError):
        validate_params("partial_obstruction", {"side": "middle"})
    with pytest.raises(PerturbationError):
        validate_params("combined", {})  # required 'steps'


def test_invalid_image_raises():
    with pytest.raises(PerturbationError):
        apply("gaussian_noise", np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(PerturbationError):
        apply("gaussian_noise", np.zeros((4, 4, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# Family 1 - noise
# ---------------------------------------------------------------------------

def test_gaussian_noise_zero_is_identity_and_nonzero_changes():
    img = _img()
    assert np.array_equal(apply("gaussian_noise", img, {"sigma": 0}, seed=0).image, img)
    assert not np.array_equal(apply("gaussian_noise", img, {"sigma": 25}, seed=0).image, img)


def test_salt_pepper_zero_identity_and_spikes():
    img = _img()
    assert np.array_equal(apply("salt_pepper_noise", img, {"amount": 0}, seed=0).image, img)
    out = apply("salt_pepper_noise", img, {"amount": 0.1}, seed=0).image
    assert (out == 255).any() and (out == 0).any()


def test_speckle_zero_identity():
    img = _img()
    assert np.array_equal(apply("speckle_noise", img, {"sigma": 0}, seed=0).image, img)


def test_noise_differs_across_seeds():
    img = _img()
    a = apply("gaussian_noise", img, {"sigma": 25}, seed=1).image
    b = apply("gaussian_noise", img, {"sigma": 25}, seed=2).image
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Family 2 - exposure
# ---------------------------------------------------------------------------

def test_exposure_identity_and_direction():
    img = _img()
    assert np.array_equal(apply("darken", img, {"factor": 1.0}, seed=0).image, img)
    assert np.array_equal(apply("brighten", img, {"factor": 1.0}, seed=0).image, img)
    assert np.array_equal(apply("overexposure", img, {"strength": 0.0}, seed=0).image, img)
    assert apply("darken", img, {"factor": 0.5}, seed=0).image.mean() < img.mean()
    assert apply("brighten", img, {"factor": 2.0}, seed=0).image.mean() > img.mean()


# ---------------------------------------------------------------------------
# Family 3 - blur
# ---------------------------------------------------------------------------

def test_blur_identity_and_smoothing():
    img = _img()
    assert np.array_equal(apply("gaussian_blur", img, {"radius": 0}, seed=0).image, img)
    assert np.array_equal(apply("defocus_blur", img, {"radius": 0}, seed=0).image, img)
    blurred = apply("gaussian_blur", img, {"radius": 3}, seed=0).image
    assert blurred.std() <= img.std()


def test_motion_blur_kernel_validation_and_effect():
    with pytest.raises(PerturbationError):
        apply("motion_blur", _img(), {"kernel_size": 4, "angle": 0}, seed=0)
    img = _img()
    out = apply("motion_blur", img, {"kernel_size": 5, "angle": 30}, seed=0).image
    assert out.shape == img.shape and not np.array_equal(out, img)


# ---------------------------------------------------------------------------
# Family 4 - weather
# ---------------------------------------------------------------------------

def test_weather_zero_is_identity():
    img = _img()
    for name in ("fog", "haze", "rain", "snow"):
        assert np.array_equal(apply(name, img, {"intensity": 0.0}, seed=0).image, img), name


def test_fog_brightens_and_rain_reproducible():
    img = _img()
    assert apply("fog", img, {"intensity": 0.5}, seed=0).image.mean() > img.mean()
    a = apply("rain", img, {"intensity": 0.5}, seed=11).image
    b = apply("rain", img, {"intensity": 0.5}, seed=11).image
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Family 5 - colour
# ---------------------------------------------------------------------------

def test_colour_identity_values():
    img = _img()
    assert np.array_equal(apply("hue_shift", img, {"degrees": 0}, seed=0).image, img)
    assert np.array_equal(apply("saturation", img, {"factor": 1.0}, seed=0).image, img)
    assert np.array_equal(apply("contrast", img, {"factor": 1.0}, seed=0).image, img)
    assert np.array_equal(apply("color_temperature", img, {"warmth": 0.0}, seed=0).image, img)


def test_saturation_zero_is_grey():
    out = apply("saturation", _img(), {"factor": 0.0}, seed=0).image.astype(int)
    assert np.abs(out[:, :, 0] - out[:, :, 1]).max() <= 1
    assert np.abs(out[:, :, 1] - out[:, :, 2]).max() <= 1


# ---------------------------------------------------------------------------
# Family 6 - occlusion
# ---------------------------------------------------------------------------

def test_occlusion_zero_noop_and_nonzero_blackens():
    img = _img()
    r0 = apply("occlusion", img, {"ratio": 0.0}, seed=0)
    assert np.array_equal(r0.image, img) and r0.metadata["occluders"] == []
    out = apply("occlusion", img, {"ratio": 0.3}, seed=0).image
    assert (out == 0).sum() > 0


def test_occlusion_box_policy_removal():
    boxes = [GroundTruthBox(0, 0, 0, 10, 10)]
    keep = apply("occlusion", _img(), {"ratio": 0.5}, boxes=boxes, seed=0)
    assert [b.class_id for b in keep.boxes] == [0]  # default: geometry unchanged
    removed = apply("occlusion", _img(), {"ratio": 0.5, "remove_fully_occluded": True,
                                          "full_threshold": 0.0}, boxes=boxes, seed=0)
    assert removed.boxes == []
    assert removed.metadata["removed_boxes"]


def test_random_patch_and_partial_obstruction_run():
    img = _img()
    rp = apply("random_patch", img, {"ratio": 0.2, "num_patches": 3}, seed=0)
    assert len(rp.metadata["occluders"]) == 3
    po = apply("partial_obstruction", img, {"ratio": 0.3, "side": "top"}, seed=0)
    assert po.image[: int(round(0.3 * min(img.shape[:2])))].max() == 0


# ---------------------------------------------------------------------------
# Family 7 - geometry
# ---------------------------------------------------------------------------

def _bounds_ok(boxes, w, h):
    for b in boxes:
        assert 0 <= b.x1 < b.x2 <= w + 1e-6
        assert 0 <= b.y1 < b.y2 <= h + 1e-6


def test_rotation_identity_and_90_swaps_extent():
    img = _img(100, 100)
    boxes = [GroundTruthBox(0, 30, 40, 70, 60)]  # 40 x 20 at centre
    ident = apply("rotation", img, {"angle": 0}, boxes=boxes, seed=0)
    b = ident.boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((30, 40, 70, 60))
    r90 = apply("rotation", img, {"angle": 90}, boxes=boxes, seed=0).boxes[0]
    assert (r90.x2 - r90.x1) == pytest.approx(20, abs=1)
    assert (r90.y2 - r90.y1) == pytest.approx(40, abs=1)
    assert r90.class_id == 0
    _bounds_ok(apply("rotation", img, {"angle": 45}, boxes=boxes, seed=0).boxes, 100, 100)


def test_scaling_identity_and_growth_clip():
    img = _img(100, 100)
    boxes = [GroundTruthBox(0, 30, 30, 70, 70)]
    ident = apply("scaling", img, {"scale": 1.0}, boxes=boxes, seed=0).boxes[0]
    assert (ident.x1, ident.x2) == pytest.approx((30, 70))
    big = apply("scaling", img, {"scale": 2.0}, boxes=boxes, seed=0).boxes[0]
    assert (big.x1, big.y1) == pytest.approx((10.5, 10.5), abs=1e-6)
    assert (big.x2, big.y2) == pytest.approx((90.5, 90.5), abs=1e-6)
    _bounds_ok([big], 100, 100)


def test_translation_shifts_and_clips():
    img = _img(100, 100)
    boxes = [GroundTruthBox(0, 30, 40, 70, 60)]
    out = apply("translation", img, {"dx": 0.25, "dy": 0.0}, boxes=boxes, seed=0).boxes[0]
    assert (out.x1, out.x2) == pytest.approx((55, 95))
    clipped = apply("translation", img, {"dx": 0.5, "dy": 0.0}, boxes=boxes, seed=0).boxes[0]
    assert clipped.x2 <= 100 and clipped.x1 == pytest.approx(80)


def test_perspective_identity_and_jitter():
    img = _img(100, 100)
    boxes = [GroundTruthBox(0, 30, 30, 70, 70)]
    ident = apply("perspective", img, {"jitter": 0.0}, boxes=boxes, seed=0).boxes[0]
    assert (ident.x1, ident.y1, ident.x2, ident.y2) == pytest.approx((30, 30, 70, 70))
    out = apply("perspective", img, {"jitter": 0.1}, boxes=boxes, seed=0)
    _bounds_ok(out.boxes, 100, 100)
    assert out.boxes[0].class_id == 0


def test_geometry_preserves_class_and_empty_boxes():
    boxes = [GroundTruthBox(2, 10, 10, 40, 40), GroundTruthBox(5, 50, 50, 90, 90)]
    out = apply("rotation", _img(100, 100), {"angle": 20}, boxes=boxes, seed=0).boxes
    assert {b.class_id for b in out} == {2, 5}
    assert apply("rotation", _img(), {"angle": 20}, boxes=[], seed=0).boxes == []


# ---------------------------------------------------------------------------
# Family 8 - quality
# ---------------------------------------------------------------------------

def test_quality_effects_change_image():
    img = _img()
    for name, params in (("jpeg_compression", {"quality": 5}),
                         ("downscale", {"scale": 0.25}),
                         ("pixelate", {"pixel_size": 8})):
        out = apply(name, img, params, seed=0).image
        assert out.shape == img.shape and not np.array_equal(out, img), name


# ---------------------------------------------------------------------------
# Family 9 - lighting
# ---------------------------------------------------------------------------

def test_lighting_zero_identity_and_darkens():
    img = _img()
    for name, p in (("cast_shadow", {"intensity": 0.0, "coverage": 0.5, "angle": 45.0}),
                    ("uneven_illumination", {"strength": 0.0, "angle": 0.0}),
                    ("local_darkening", {"intensity": 0.0, "radius": 0.3})):
        assert np.array_equal(apply(name, img, p, seed=0).image, img), name
    assert apply("cast_shadow", img, {"intensity": 0.7, "coverage": 0.5, "angle": 45.0},
                 seed=0).image.mean() < img.mean()


# ---------------------------------------------------------------------------
# Family 10 - lens
# ---------------------------------------------------------------------------

def test_lens_effects():
    img = _img()
    assert np.array_equal(apply("glare", img, {"intensity": 0.0, "radius": 0.3}, seed=0).image, img)
    assert np.array_equal(apply("lens_flare", img, {"intensity": 0.0}, seed=0).image, img)
    assert np.array_equal(apply("vignetting", img, {"strength": 0.0}, seed=0).image, img)
    assert apply("glare", img, {"intensity": 0.8, "radius": 0.4}, seed=0).image.mean() > img.mean()
    assert apply("vignetting", img, {"strength": 0.8}, seed=0).image.mean() < img.mean()


# ---------------------------------------------------------------------------
# Family 11 - cropping / framing
# ---------------------------------------------------------------------------

def test_random_crop_changes_size_and_keeps_in_bounds():
    img = _img(100, 100)
    boxes = [GroundTruthBox(0, 30, 30, 70, 70)]
    out = apply("random_crop", img, {"crop_fraction": 0.6}, boxes=boxes, seed=0)
    h, w = out.image.shape[:2]
    assert (w, h) == (60, 60)
    _bounds_ok(out.boxes, w, h)


def test_edge_clip_removes_offscreen_box():
    img = _img(100, 100)
    boxes = [GroundTruthBox(0, 40, 80, 60, 95)]  # near the bottom
    out = apply("edge_clip", img, {"amount": 0.3, "side": "bottom"}, boxes=boxes, seed=0)
    assert out.boxes == []
    assert out.metadata["removed_boxes"]


def test_shifted_framing_identity_and_shift():
    img = _img(100, 100)
    boxes = [GroundTruthBox(0, 30, 40, 70, 60)]
    ident = apply("shifted_framing", img, {"dx": 0.0, "dy": 0.0}, boxes=boxes, seed=0)
    assert ident.image.shape == img.shape
    assert (ident.boxes[0].x1, ident.boxes[0].x2) == pytest.approx((30, 70))
    moved = apply("shifted_framing", img, {"dx": 0.2, "dy": 0.0}, boxes=boxes, seed=0)
    assert (moved.boxes[0].x1, moved.boxes[0].x2) == pytest.approx((50, 90))


# ---------------------------------------------------------------------------
# Family 12 - combined
# ---------------------------------------------------------------------------

def test_combined_order_and_reproducibility():
    img = _img()
    steps = [{"name": "darken", "params": {"factor": 0.5}},
             {"name": "gaussian_noise", "params": {"sigma": 10}}]
    a = apply("combined", img, {"steps": steps}, seed=4)
    b = apply("combined", img, {"steps": steps}, seed=4)
    assert np.array_equal(a.image, b.image)
    assert a.metadata["order"] == ["darken", "gaussian_noise"]
    assert [c["name"] for c in a.metadata["components"]] == ["darken", "gaussian_noise"]


def test_combined_reuses_geometry_and_remaps_boxes():
    img = _img(100, 100)
    boxes = [GroundTruthBox(0, 30, 30, 70, 70)]
    steps = [{"name": "rotation", "params": {"angle": 30}},
             {"name": "gaussian_noise", "params": {"sigma": 5}}]
    out = apply("combined", img, {"steps": steps}, boxes=boxes, seed=0)
    assert out.metadata["geometry_changed"] is True
    _bounds_ok(out.boxes, 100, 100)


def test_combined_invalid_step_raises():
    with pytest.raises(PerturbationError):
        apply("combined", _img(), {"steps": [{"bad": 1}]}, seed=0)
    with pytest.raises(PerturbationError):
        apply("combined", _img(), {"steps": []}, seed=0)


# ---------------------------------------------------------------------------
# Legacy compatibility (failureforge.transforms)
# ---------------------------------------------------------------------------

def test_legacy_param_ranges_keys():
    assert set(PARAM_RANGES) == {"brightness", "blur", "rotation", "scaling", "occlusion"}


def test_legacy_apply_perturbation_contract():
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    box = [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.4, "height": 0.4}]
    out, boxes, info = apply_perturbation(img, box, "brightness", {"factor": 1.5}, seed=0)
    assert isinstance(out, Image.Image) and out.size == (100, 100)
    assert boxes[0]["x_center"] == pytest.approx(0.5)
    assert info["geometry_changed"] is False

    out2, boxes2, info2 = apply_perturbation(
        img, box, "occlusion", {"ratio": 0.6, "full_threshold": 0.5}, seed=0)
    assert "cutout_xyxy" in info2 and "removed" in info2


def test_legacy_unknown_perturbation_raises():
    img = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError):
        apply_perturbation(img, [], "nope", {}, seed=0)
