"""Search space, Random Search, TPE, confirmation, and search reporting tests.

All tests use ``ColorBlobDetector`` (a real deterministic rule-based detector)
on synthetic images: no weights, GPU, or network.
"""
import csv
import json
import math
import random

import pytest

from failureforge.demo import ColorBlobDetector, make_demo_dataset
from failureforge.dataset import YoloDataset
from failureforge.objective import ObjectiveConfig
from failureforge.perturbations import PerturbationError, validate_params
from failureforge.search import (
    OptionalDependencyError,
    SearchSpace,
    SearchSpaceError,
    SearchTrial,
    confirm_trials,
    optuna_available,
    rank_trials,
    render_search_markdown,
    run_random_search,
    run_tpe_search,
    search_payload,
    write_search_report,
    write_trials_csv,
)
from failureforge.search.trials import STATUS_ERROR, STATUS_INVALID, STATUS_OK, config_key


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return YoloDataset.from_config(make_demo_dataset(tmp_path_factory.mktemp("demo"), 8))


@pytest.fixture(scope="module")
def det():
    return ColorBlobDetector()


# --------------------------------------------------------------- search space

def test_default_space_excludes_combined_and_samples_validly():
    space = SearchSpace.default()
    assert "combined" not in space.names and len(space.names) >= 30
    rng = random.Random(0)
    for _ in range(1500):
        name, params = space.sample(rng)
        validate_params(name, params)  # never raises (e.g. odd motion_blur kernel)


def test_space_sampling_is_deterministic():
    a = [SearchSpace.default().sample(random.Random(5)) for _ in range(3)]
    b = [SearchSpace.default().sample(random.Random(5)) for _ in range(3)]
    assert a == b


def test_space_include_overrides_and_errors():
    space = SearchSpace.default(include=["gaussian_blur"],
                                overrides={"gaussian_blur": {"radius": (1.0, 2.0)}})
    assert space.names == ["gaussian_blur"]
    for _ in range(50):
        assert 1.0 <= space.sample(random.Random(_))[1]["radius"] <= 2.0
    with pytest.raises(PerturbationError):
        SearchSpace.default(include=["nope"])
    with pytest.raises(SearchSpaceError):
        SearchSpace.default(include=["gaussian_blur"],
                            overrides={"gaussian_blur": {"radius": (0.0, 99.0)}})
    with pytest.raises(SearchSpaceError):
        SearchSpace.default(include=["combined"])
    with pytest.raises(SearchSpaceError):
        SearchSpace.default(include=["gaussian_blur"],
                            overrides={"rotation": {"angle": (0, 1)}})


# ------------------------------------------------------------- random search

def test_random_search_reproducible_and_bounded(det, ds):
    r1 = run_random_search(det, ds, budget=8, seed=3)
    r2 = run_random_search(det, ds, budget=8, seed=3)
    r3 = run_random_search(det, ds, budget=8, seed=4)
    key = lambda r: [(t.name, t.params, t.seed, t.score, t.status) for t in r.trials]  # noqa: E731
    assert key(r1) == key(r2)
    assert key(r1) != key(r3)
    assert len(r1.trials) == 8
    assert all(t.valid for t in r1.trials)


def test_scores_use_objective_convention(det, ds):
    r = run_random_search(det, ds, budget=10, seed=0,
                          space=SearchSpace.default(include=["darken"]))
    for t in r.ranked():
        assert t.score == pytest.approx(t.baseline_value - t.perturbed_value)
    # baseline is perfect, so degradation can never be negative here
    assert all(t.score >= 0 for t in r.trials)
    scores = [t.score for t in r.ranked()]
    assert scores == sorted(scores, reverse=True)
    assert r.best is r.ranked()[0]


def test_strong_perturbation_scores_worse_than_mild(det, ds):
    strong = run_random_search(
        det, ds, budget=3, seed=0,
        space=SearchSpace.default(include=["darken"],
                                  overrides={"darken": {"factor": (0.0, 0.05)}}))
    mild = run_random_search(
        det, ds, budget=3, seed=0,
        space=SearchSpace.default(include=["darken"],
                                  overrides={"darken": {"factor": (0.95, 1.0)}}))
    assert strong.best.score > mild.best.score


@pytest.mark.parametrize("budget", [0, -1, True, 2.5])
def test_invalid_budget_rejected(det, ds, budget):
    with pytest.raises(ValueError):
        run_random_search(det, ds, budget=budget)


def test_errored_trials_recorded_not_zero(ds):
    class Exploding(ColorBlobDetector):
        calls = 0

        def predict(self, image_bgr):
            Exploding.calls += 1
            if Exploding.calls > len(ds):  # baseline fine, every trial image fails
                raise RuntimeError("boom")
            return super().predict(image_bgr)

    r = run_random_search(Exploding(), ds, budget=3, seed=0)
    assert [t.status for t in r.trials] == [STATUS_ERROR] * 3
    assert all(t.score is None and "boom" in t.error for t in r.trials)
    assert r.best is None and r.ranked() == []
    assert r.counts() == {"total": 3, "valid": 0, "invalid": 0, "error": 3}


def test_invalid_objective_not_ranked_as_zero(det, ds):
    # map50 can be missing (nan) when nothing is detected on baseline; emulate via
    # a detector that never detects: baseline map50 is a finite 0.0, so instead
    # check the ranking primitive directly with an invalid trial.
    ok = SearchTrial(0, "a", "f", {}, 1, STATUS_OK, score=0.0)
    bad = SearchTrial(1, "b", "f", {}, 2, STATUS_INVALID, score=None)
    err = SearchTrial(2, "c", "f", {}, 3, STATUS_ERROR)
    nan = SearchTrial(3, "d", "f", {}, 4, STATUS_OK, score=float("nan"))
    assert rank_trials([bad, err, nan, ok]) == [ok]


def test_rank_ties_and_unique():
    t = lambda i, s, p: SearchTrial(i, "x", "f", {"v": p}, i, STATUS_OK, score=s)  # noqa: E731
    trials = [t(2, 0.5, 1.00001), t(0, 0.5, 2.0), t(1, 0.5, 1.00002), t(3, 0.1, 3.0)]
    assert [x.index for x in rank_trials(trials)] == [0, 1, 2, 3]
    assert [x.index for x in rank_trials(trials, unique=True)] == [0, 1, 3]
    assert config_key("x", {"v": 1.00001}) == config_key("x", {"v": 1.00002})


def test_metric_selection(det, ds):
    r = run_random_search(det, ds, budget=2, seed=0,
                          objective=ObjectiveConfig("map50"),
                          space=SearchSpace.default(include=["darken"]))
    assert r.objective.metric == "map50"
    assert r.trials[0].baseline_value == pytest.approx(r.baseline_metrics["map50"])


# ------------------------------------------------------------------------ TPE

@pytest.mark.skipif(not optuna_available(), reason="optuna not installed (optional extra)")
def test_tpe_reproducible_and_valid(det, ds):
    space = SearchSpace.default(include=["gaussian_blur", "darken", "motion_blur"])
    a = run_tpe_search(det, ds, budget=8, seed=1, space=space)
    b = run_tpe_search(det, ds, budget=8, seed=1, space=space)
    key = lambda r: [(t.name, t.params, t.score) for t in r.trials]  # noqa: E731
    assert key(a) == key(b) and len(a.trials) == 8
    assert a.method == "tpe" and a.metadata["sampler"] == "optuna.TPESampler"
    for t in a.trials:
        validate_params(t.name, t.params)


def test_tpe_without_optuna_fails_clearly(monkeypatch, det, ds):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "optuna" or name.startswith("optuna."):
            raise ImportError("no optuna")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(OptionalDependencyError, match="failureforge\\[optuna\\]"):
        run_tpe_search(det, ds, budget=1)


# -------------------------------------------------------------- confirmation

def test_confirmation_multi_seed_stats(det, ds):
    disc, hold = ds.partition(0.4, seed=1)
    r = run_random_search(det, disc, budget=6, seed=0,
                          space=SearchSpace.default(include=["gaussian_noise", "darken"]))
    c = confirm_trials(det, hold, r.trials, top_k=2, n_seeds=3, seed=7)
    assert len(c.records) == 2
    assert set(disc.image_paths).isdisjoint(hold.image_paths)
    for rec in c.records:
        assert len(rec.scores) == 3 and rec.n_valid == 3
        assert rec.min <= rec.mean <= rec.max
        assert rec.std is None or rec.std >= 0
        assert math.isfinite(rec.mean)
    # reproducible
    c2 = confirm_trials(det, hold, r.trials, top_k=2, n_seeds=3, seed=7)
    assert [x.scores for x in c.records] == [x.scores for x in c2.records]
    assert c.baseline.split.endswith("holdout")


def test_confirmation_with_no_valid_trials_is_empty(det, ds):
    err = SearchTrial(0, "darken", "exposure", {"factor": 0.1}, 1, STATUS_ERROR, error="x")
    c = confirm_trials(det, ds, [err], top_k=3, n_seeds=2)
    assert c.records == [] and c.results == []


def test_confirmation_arg_validation(det, ds):
    with pytest.raises(ValueError):
        confirm_trials(det, ds, [], top_k=0)


# ------------------------------------------------------------------ reporting

def test_search_report_files(tmp_path, det, ds):
    disc, hold = ds.partition(0.4, seed=1)
    r = run_random_search(det, disc, budget=4, seed=0)
    c = confirm_trials(det, hold, r.trials, top_k=2, n_seeds=2)
    paths = write_search_report(r, tmp_path / "out", c)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "failureforge.search-report.v1"
    assert payload["search"]["objective"]["metric"] == "f1"
    assert payload["search"]["counts"]["total"] == 4
    assert payload["confirmation"]["records"]
    md = paths["md"].read_text(encoding="utf-8")
    assert "## Best trial" in md and "## Confirmation (held-out)" in md and "Limitations" in md
    rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8")))
    assert len(rows) == 4 and {"index", "status", "score", "params"} <= set(rows[0])


def test_report_marks_invalid_trials_and_stays_valid_json(tmp_path, det, ds):
    r = run_random_search(det, ds, budget=2, seed=0)
    r.trials.append(SearchTrial(9, "=cmd|x", "f", {"a": float("nan")}, 1, STATUS_ERROR,
                                error="=HYPERLINK(1)"))
    md = render_search_markdown(r)
    assert "Invalid or errored trials" in md and "trial 9" in md
    json.dumps(search_payload(r), allow_nan=False)  # NaN/inf safe
    path = write_trials_csv(r, tmp_path / "t.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[-1]["error"].startswith("'=") and rows[-1]["transform"].startswith("'=")
    assert rows[-1]["score"] == ""


def test_report_no_valid_trials(det, ds):
    r = run_random_search(det, ds, budget=1, seed=0)
    r.trials[0].status, r.trials[0].score = STATUS_INVALID, None
    assert "nothing is ranked" in render_search_markdown(r)
