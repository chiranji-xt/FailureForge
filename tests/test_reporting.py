"""Report schema, JSON safety, ranking/grouping, and Markdown rendering."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from failureforge.evaluation import PerturbedResult, QuickModeResult
from failureforge.evaluator import EvaluationReport
from failureforge.failures import (
    SCHEMA_VERSION,
    build_benchmark_run,
    build_quick_run,
    environment_metadata,
)
from failureforge.reporting import (
    dumps_report,
    group_by_family,
    group_by_operation,
    json_safe,
    ranked_trials,
    render_markdown,
    run_payload,
    severity_ladders,
    write_report,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_report(
    aggregate=None,
    *,
    map50=1.0,
    iou=0.5,
    split="val",
    conf=0.25,
    metadata=None,
):
    agg = {
        "tp": 1, "fp": 0, "fn": 0,
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "num_images": 0,
    }
    if aggregate is not None:
        agg = dict(aggregate)
    meta = {
        "dataset": {
            "root": "D:\\datasets\\val", "split": split, "num_images": 0,
            "class_names": ["cat"], "labels_dir": "D:\\datasets\\labels\\val",
        },
        "model": {
            "weights": "C:\\models\\best.pt", "sha256": "a" * 64,
            "task": "detect", "device": "cpu", "class_names": ["cat"],
        },
        "aggregation": "micro-average over summed TP/FP/FN across images",
    }
    if metadata:
        meta.update(metadata)
    return EvaluationReport(
        aggregate=agg, per_class={}, per_image=[], map50=map50,
        error_records=[], metadata=meta, conf_threshold=conf,
        iou_threshold=iou, split=split,
    )


def make_trial(name, family, params, seed, *, f1=1.0, map50=1.0, iou=0.5,
               aggregate=None, baseline=None):
    base = baseline or make_report()
    if aggregate is None:
        aggregate = {
            "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0,
            "f1": f1, "num_images": 0,
        }
    perturbed = make_report(aggregate, map50=map50, iou=iou)
    return PerturbedResult(
        baseline=base, perturbed=perturbed,
        perturbation={"name": name, "family": family, "params": params, "seed": seed},
        deltas={},
    )


def make_run(trials, **kwargs):
    baseline = kwargs.pop("baseline", make_report())
    return build_benchmark_run(baseline, trials, **kwargs)


# ---------------------------------------------------------------------------
# JSON policy
# ---------------------------------------------------------------------------

def test_json_safe_handles_numpy_and_non_finite():
    assert json_safe(np.int64(3)) == 3
    assert json_safe(np.float32(1.5)) == 1.5
    assert json_safe(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    assert json_safe(np.float64("nan")) is None
    assert json_safe(float("inf")) is None
    assert json_safe(float("-inf")) is None
    assert json_safe({np.int64(1): np.float32(2.5)}) == {"1": 2.5}


def test_json_valid_schema_version_and_non_finite_metrics():
    run = make_run([make_trial("gaussian_noise", "noise", {"sigma": 10}, 0,
                                map50=float("nan"))],
                   metadata={
                       "numpy_scalar": np.float32(1.5),
                       "numpy_array": np.array([1, 2, 3]),
                       "numpy_nan": np.float32("nan"),
                   })
    text = dumps_report(run)
    payload = json.loads(text)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["mode"] == "benchmark"
    assert payload["metadata"]["numpy_scalar"] == 1.5
    assert payload["metadata"]["numpy_array"] == [1, 2, 3]
    assert payload["metadata"]["numpy_nan"] is None
    assert payload["trials"][0]["metrics"]["map50"] is None
    assert "NaN" not in text and "Infinity" not in text


def test_absolute_paths_redacted_unless_requested():
    run = make_run([make_trial("fog", "weather", {}, 0)])
    default = dumps_report(run)
    assert "C:\\\\models" not in default
    assert "D:\\\\datasets" not in default
    assert "best.pt" in default
    explicit = dumps_report(run, include_absolute_paths=True)
    assert "C:\\\\models\\\\best.pt" in explicit
    assert "D:\\\\datasets\\\\val" in explicit


def test_run_payload_includes_environment_without_timestamp():
    payload = run_payload(make_run([]))
    assert "environment" in payload["metadata"]
    assert payload["metadata"]["environment"]["failureforge"]
    assert "generated_at" not in payload


def test_report_does_not_mutate_run():
    run = make_run([make_trial("fog", "weather", {"density": 0.5}, 3)])
    before = json.loads(json.dumps(run.to_dict()))
    dumps_report(run)
    render_markdown(run)
    assert run.to_dict() == before


# ---------------------------------------------------------------------------
# Ranking, grouping, ladders
# ---------------------------------------------------------------------------

def test_ranked_trials_use_one_metric_and_sort_missing_last():
    run = make_run([
        make_trial("fog", "weather", {"density": 0.3}, 0, f1=0.9),
        make_trial("gaussian_noise", "noise", {"sigma": 10}, 0, f1=0.7),
        make_trial("downscale", "quality", {"scale": 0.5}, 0, aggregate={
            "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0,
            "num_images": 0,  # f1 intentionally missing
        }),
    ])
    ranked = ranked_trials(run, "f1")
    assert [(e["trial"].transformation["name"], e["rank"]) for e in ranked] == [
        ("gaussian_noise", 1), ("fog", 2), ("downscale", None),
    ]
    assert ranked[0]["drop"] == pytest.approx(0.3)
    assert ranked[-1]["drop"] is None
    with pytest.raises(ValueError):
        ranked_trials(run, "f2")


def test_ranking_is_deterministic_on_ties():
    run = make_run([
        make_trial("b_op", "fam", {"x": 1}, 0, f1=0.8),
        make_trial("a_op", "fam", {"x": 1}, 0, f1=0.8),
    ])
    names = [e["trial"].transformation["name"] for e in ranked_trials(run)]
    assert names == ["a_op", "b_op"]
    assert ranked_trials(run) == ranked_trials(run)


def test_grouping_by_family_and_operation():
    run = make_run([
        make_trial("fog", "weather", {"density": 0.2}, 0, f1=0.9),
        make_trial("rain", "weather", {"density": 0.4}, 0, f1=0.8),
        make_trial("gaussian_noise", "noise", {"sigma": 5}, 0, f1=0.95),
    ])
    families = group_by_family(run)
    assert [g["key"] for g in families] == ["noise", "weather"]
    assert families[0]["trials"] == 1
    assert families[1]["trials"] == 2
    assert families[1]["max_drop"] == pytest.approx(0.2)
    operations = group_by_operation(run)
    assert [g["key"] for g in operations] == ["fog", "gaussian_noise", "rain"]


def test_severity_ladders_require_exactly_one_numeric_parameter():
    run = make_run([
        make_trial("gaussian_noise", "noise", {"sigma": 20}, 0, f1=0.6),
        make_trial("gaussian_noise", "noise", {"sigma": 5}, 0, f1=0.9),
        make_trial("gaussian_noise", "noise", {"sigma": 10}, 0, f1=0.75),
        make_trial("combined", "combined", {"steps": "x", "seed": 1}, 0),
    ])
    ladders = severity_ladders(run)
    assert len(ladders) == 1
    ladder = ladders[0]
    assert ladder["name"] == "gaussian_noise"
    assert ladder["param"] == "sigma"
    assert [p["value"] for p in ladder["points"]] == [5.0, 10.0, 20.0]
    assert [p["drop"] for p in ladder["points"]] == pytest.approx([0.1, 0.25, 0.4])


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def test_markdown_benchmark_sections_and_stable_output():
    run = make_run([
        make_trial("fog", "weather", {"density": 0.5}, 7, f1=0.8),
        make_trial("gaussian_noise", "noise", {"sigma": 10}, 7, f1=0.6),
        make_trial("gaussian_noise", "noise", {"sigma": 20}, 7, f1=0.4),
    ])
    text = render_markdown(run, rank_metric="f1")
    assert text == render_markdown(run, rank_metric="f1")
    assert "# FailureForge Report" in text
    assert "Benchmark (verified against ground truth)" in text
    assert "## Baseline (unperturbed)" in text
    assert "## Ranked perturbations" in text
    assert "gaussian_noise" in text and "fog" in text
    assert "### By transformation family" in text
    assert "### By operation" in text
    assert "## Severity ladders" in text
    assert "## Failure categories (Benchmark Mode)" in text
    assert "False negative" in text and "Unmatched/ambiguous" in text
    assert "not proof of universal" in text
    # Ranked order: gaussian_noise (0.4 drop) before fog (0.2 drop).
    assert text.index("gaussian_noise") < text.index("fog")


def test_markdown_warns_missing_and_incompatible_metrics():
    baseline = make_report()
    missing = make_trial("fog", "weather", {"density": 0.5}, 0, aggregate={
        "tp": 1, "fp": 0, "fn": 0, "recall": 1.0, "f1": 1.0, "num_images": 0,
    })
    incompatible = make_trial("rain", "weather", {"density": 0.5}, 0, iou=0.3)
    run = make_run([missing, incompatible], baseline=baseline)
    text = render_markdown(run)
    assert "missing or non-finite" in text
    assert "incomparable" in text
    assert "marked incompatible" in text

    payload = run_payload(run)
    missing_trial = next(
        t for t in payload["trials"]
        if t["transformation"]["name"] == "fog"
    )
    assert missing_trial["deltas"]["precision"] is None
    assert missing_trial["drops"]["precision"] is None
    rain = next(
        t for t in payload["trials"]
        if t["transformation"]["name"] == "rain"
    )
    assert rain["compatible"] is False
    rain_entry = next(
        entry for entry in ranked_trials(run)
        if entry["trial"].transformation["name"] == "rain"
    )
    assert rain_entry["rank"] is None


def test_quick_markdown_and_payload_never_report_accuracy():
    result = QuickModeResult(
        perturbation={
            "name": "brightness", "family": "photometric",
            "params": {"factor": 0.4}, "seed": 2,
        },
        iou_threshold=0.5,
        per_image=[{
            "image": "img0.png", "num_clean": 1, "num_perturbed": 0,
            "matches": 0,
            "disappeared": [{
                "index": 0, "class_id": 0, "score": 0.9, "box": [40, 40, 60, 60],
            }],
            "appeared": [], "class_changed": [],
            "confidence_changes": [], "box_drift": [],
        }],
        summary={
            "num_images": 1, "clean_detections": 1, "perturbed_detections": 0,
            "matched": 0, "disappeared": 1, "appeared": 0, "class_changed": 0,
            "confidence_changed": 0, "mean_confidence_delta": 0.0,
            "mean_box_iou": 0.0, "changed_detection_fraction": 1.0,
        },
        metadata={
            "model": {"weights": None, "sha256": None, "class_names": ["cat"]},
            "dataset": {"split": "unlabeled", "num_images": 1,
                        "class_names": ["cat"], "root": "D:\\unlabeled"},
            "confidence_epsilon": 0.01,
        },
    )
    run = build_quick_run(result)
    text = render_markdown(run)
    assert "Quick (prediction instability; no ground truth)" in text
    assert "Disappeared" in text
    assert "Ranked perturbations" not in text
    assert "not verified" in text
    assert "never reports precision, recall, F1, or mAP" in text

    payload = run_payload(run)
    assert payload["mode"] == "quick"
    assert payload["label"] == "prediction_instability"
    assert payload["verified_failures"] is False
    blob = json.dumps(payload).lower()
    for forbidden in ("precision", "recall", "map50", "false_positive",
                      "false_negative"):
        assert forbidden not in blob
    assert "not verified" in " ".join(payload["warnings"]).lower()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def test_write_report_writes_deterministic_files(tmp_path):
    run = make_run([make_trial("fog", "weather", {"density": 0.3}, 1, f1=0.9)])
    paths = write_report(run, tmp_path, basename="demo")
    assert set(paths) == {"json", "md"}
    first_json = paths["json"].read_text(encoding="utf-8")
    first_md = paths["md"].read_text(encoding="utf-8")
    json.loads(first_json)
    assert first_md.startswith("# FailureForge Report")

    write_report(run, tmp_path, basename="demo")
    assert paths["json"].read_text(encoding="utf-8") == first_json
    assert paths["md"].read_text(encoding="utf-8") == first_md

    with pytest.raises(ValueError):
        write_report(run, tmp_path, formats=("xml",))


def test_environment_metadata_is_finite_json():
    env = environment_metadata()
    assert json.loads(json.dumps(env)) == env
    assert all(not isinstance(v, float) or math.isfinite(v) for v in env.values())
