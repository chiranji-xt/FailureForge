"""CLI + synthetic end-to-end tests (no weights, GPU, or network)."""
import json

import pytest

from failureforge.cli import main
from failureforge.config import ConfigError, load_config
from failureforge.demo import ColorBlobDetector, make_demo_dataset


def factory(_settings):
    return ColorBlobDetector()


@pytest.fixture()
def data(tmp_path):
    return str(make_demo_dataset(tmp_path / "demo", 8))


def run(argv, **kw):
    return main(argv, detector_factory=factory, **kw)


def test_help_and_subcommand_required(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "search" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([])


def test_list_perturbations(capsys):
    assert run(["list-perturbations"]) == 0
    out = capsys.readouterr().out
    assert "gaussian_blur" in out and "rotation" in out


def test_validate_ok_and_invalid(data, tmp_path, capsys):
    assert run(["validate-data", "--data", data]) == 0
    assert "0 error(s)" in capsys.readouterr().out
    labels = tmp_path / "demo" / "labels" / "test"
    (labels / "demo0.txt").write_text("7 0.5 0.5 0.2 0.2\n")  # bad class id
    assert run(["validate-data", "--data", data]) == 1
    cap = capsys.readouterr()
    assert "ERROR" in cap.out and "NOT valid" in cap.err


def test_missing_label_is_reported_and_blocks_evaluation(data, tmp_path, capsys):
    (tmp_path / "demo" / "labels" / "test" / "demo1.txt").unlink()
    assert run(["validate-data", "--data", data]) == 1
    assert run(["evaluate", "--data", data, "--out-dir", str(tmp_path / "o")]) == 2
    assert "missing label" in capsys.readouterr().err
    assert run(["evaluate", "--data", data, "--allow-missing-labels",
                "--out-dir", str(tmp_path / "o")]) == 0


def test_evaluate_writes_valid_json(data, tmp_path, capsys):
    out = tmp_path / "base.json"
    assert run(["evaluate", "--data", data, "--output", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["aggregate"]["f1"] == 1.0
    assert "f1=1.0000" in capsys.readouterr().out


def test_perturb_benchmark_with_gallery(data, tmp_path, capsys):
    out = tmp_path / "res"
    code = run(["perturb", "--data", data, "--transform", "darken", "--param", "factor=0.1",
                "--seed", "3", "--gallery", "--out-dir", str(out)])
    assert code == 0
    assert {"perturb.json", "perturb.md", "failure_gallery.html"} <= {p.name for p in out.iterdir()}
    report = json.loads((out / "perturb.json").read_text())
    assert report["mode"] == "benchmark"
    assert report["trials"][0]["transformation"]["params"]["factor"] == 0.1
    html = (out / "failure_gallery.html").read_text()
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "delta f1=-" in capsys.readouterr().out


def test_perturb_quick_mode_has_no_accuracy_metrics(data, tmp_path):
    out = tmp_path / "q"
    assert run(["perturb", "--data", data, "--transform", "darken", "--param", "factor=0.1",
                "--mode", "quick", "--out-dir", str(out)]) == 0
    assert json.loads((out / "perturb.json").read_text())["mode"] == "quick"


@pytest.mark.parametrize("argv, message", [
    (["perturb", "--transform", "darken"], "--data"),
    (["perturb", "--data", "{d}", "--transform", "nope"], "unknown transform"),
    (["perturb", "--data", "{d}", "--transform", "darken", "--param", "factor"], "KEY=VALUE"),
    (["perturb", "--data", "{d}", "--transform", "darken", "--param", "factor=9"], "above maximum"),
    (["perturb", "--data", "{d}", "--transform", "darken", "--param", "bogus=1"], "unknown parameter"),
    (["evaluate", "--data", "{d}", "--iou", "0"], "--iou"),
    (["search", "--data", "{d}", "--budget", "0"], "--budget"),
    (["search", "--data", "{d}", "--holdout-fraction", "1"], "holdout"),
    (["search", "--data", "{d}", "--transforms", "nope"], "unknown transform"),
    (["evaluate", "--data", "/does/not/exist.yaml"], "not found"),
])
def test_argument_errors_are_clean(argv, message, data, capsys):
    argv = [a.replace("{d}", data) for a in argv]
    assert run(argv) == 2
    assert message in capsys.readouterr().err


def test_weights_required_for_real_detector(data, capsys):
    assert main(["evaluate", "--data", data]) == 2
    assert "--weights" in capsys.readouterr().err


def test_search_end_to_end_random(data, tmp_path, capsys):
    out = tmp_path / "s"
    argv = ["search", "--data", data, "--method", "random", "--budget", "6", "--seed", "1",
            "--transforms", "darken,gaussian_noise", "--confirm-top", "2",
            "--confirm-seeds", "2", "--gallery", "--out-dir", str(out)]
    assert run(argv) == 0
    names = {p.name for p in out.iterdir()}
    assert {"search.json", "search.md", "search_trials.csv", "confirmation.json",
            "failure_gallery.html"} <= names
    first = json.loads((out / "search.json").read_text())
    assert first["search"]["counts"]["total"] == 6 and first["confirmation"]["records"]
    # reproducible: same command -> same trials/scores
    out2 = tmp_path / "s2"
    assert run(argv[:-1] + [str(out2)]) == 0
    second = json.loads((out2 / "search.json").read_text())
    strip = lambda p: [(t["transformation"], t["score"]) for t in p["search"]["trials"]]  # noqa: E731
    assert strip(first) == strip(second)
    assert "best:" in capsys.readouterr().out


def test_search_without_holdout_skips_confirmation(data, tmp_path, capsys):
    out = tmp_path / "n"
    assert run(["search", "--data", data, "--budget", "2", "--holdout-fraction", "0",
                "--transforms", "darken", "--out-dir", str(out)]) == 0
    assert json.loads((out / "search.json").read_text())["confirmation"] is None
    assert "confirmation skipped" in capsys.readouterr().out


def test_search_tpe_graceful(data, tmp_path, capsys):
    pytest.importorskip("optuna", reason="optional extra")
    assert run(["search", "--data", data, "--method", "tpe", "--budget", "4",
                "--transforms", "darken,gaussian_blur", "--out-dir", str(tmp_path / "t")]) == 0


def test_config_file_and_cli_override(data, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"dataset:\n  data: {data!r}\nsearch:\n  budget: 3\n  transforms: [darken]\n"
                   f"  holdout_fraction: 0\noutput:\n  dir: {str(tmp_path / 'cfgout')!r}\n")
    assert run(["search", "--config", str(cfg)]) == 0
    assert json.loads((tmp_path / "cfgout" / "search.json").read_text())["search"]["counts"]["total"] == 3
    assert run(["search", "--config", str(cfg), "--budget", "2"]) == 0
    assert json.loads((tmp_path / "cfgout" / "search.json").read_text())["search"]["counts"]["total"] == 2


def test_config_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("search:\n  budgett: 3\n")
    with pytest.raises(ConfigError, match="budgett"):
        load_config(bad)
    bad.write_text("nonsense: {}\n")
    with pytest.raises(ConfigError, match="unknown config section"):
        load_config(bad)
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    assert load_config(None) == {}


def test_shipped_default_config_is_valid():
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg["search"]["metric"] == "f1"
