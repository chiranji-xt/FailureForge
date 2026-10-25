"""FailureForge command line interface.

Commands: ``list-perturbations``, ``validate-data``, ``evaluate``, ``perturb``,
``search``. Real inference needs ``pip install "failureforge[yolo]"`` and an
existing detection checkpoint; nothing is downloaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .annotations import validate_annotations
from .config import ConfigError, load_config, pick
from .dataset import DatasetError, YoloDataset
from .detector import DetectorError, YOLODetector
from .evaluation import evaluate_perturbed, prepare_perturbed_sample, quick_mode
from .evaluator import evaluate_baseline
from .failures import build_benchmark_run, build_quick_run
from .gallery import write_gallery
from .objective import SUPPORTED_METRICS, ObjectiveConfig
from .perturbations import PerturbationError, available, get
from .perturbations.base import bgr_to_pil
from .reporting import json_safe, write_failures_csv, write_report
from .search import (
    OptionalDependencyError,
    SearchSpace,
    SearchSpaceError,
    compare_methods,
    confirm_trials,
    run_random_search,
    run_tpe_search,
    write_comparison,
    write_search_report,
)

DetectorFactory = Callable[[dict], Any]

EXIT_OK, EXIT_INVALID, EXIT_ERROR = 0, 1, 2


class CliError(Exception):
    """User-facing error (bad arguments or unusable inputs)."""


def _default_detector(settings: dict) -> Any:
    if not settings["weights"]:
        raise CliError("--weights (or detector.weights in --config) is required")
    return YOLODetector(
        settings["weights"], conf=settings["conf"],
        imgsz=settings["imgsz"], device=settings["device"],
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _settings(args: argparse.Namespace) -> tuple[dict, dict]:
    cfg = load_config(args.config)
    g = lambda cli, sec, key, default=None: pick(cli, cfg, sec, key, default)  # noqa: E731
    s = {
        "data": g(getattr(args, "data", None), "dataset", "data"),
        "split": g(getattr(args, "split", None), "dataset", "split"),
        "allow_missing_labels": bool(
            getattr(args, "allow_missing_labels", False)
            or cfg.get("dataset", {}).get("allow_missing_labels", False)
        ),
        "weights": g(getattr(args, "weights", None), "detector", "weights"),
        "conf": float(g(getattr(args, "conf", None), "detector", "conf", 0.25)),
        "imgsz": int(g(getattr(args, "imgsz", None), "detector", "imgsz", 640)),
        "device": str(g(getattr(args, "device", None), "detector", "device", "cpu")),
        "iou": float(g(getattr(args, "iou", None), "evaluation", "iou_threshold", 0.5)),
        "out": str(g(getattr(args, "out_dir", None), "output", "dir", "results")),
    }
    if not 0.0 < s["iou"] <= 1.0:
        raise CliError(f"--iou must be in (0, 1], got {s['iou']}")
    return s, cfg


def _load_dataset(s: dict) -> YoloDataset:
    if not s["data"]:
        raise CliError("--data (or dataset.data in --config) is required")
    return YoloDataset.from_config(
        s["data"], s["split"], allow_missing_labels=s["allow_missing_labels"]
    )


def _parse_value(text: str) -> Any:
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_params(items: Sequence[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise CliError(f"--param expects KEY=VALUE, got {item!r}")
        if key in params:
            raise CliError(f"--param {key} given more than once")
        params[key] = _parse_value(value)
    return params


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _perturbed_image_files(
    dataset: YoloDataset, name: str, params: dict, seed: int, out: Path
) -> dict[str, Path]:
    """Re-apply the (deterministic) transform and save PNGs for the gallery."""
    folder = out / "perturbed_images"
    folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for sample in dataset.iter_samples():
        _, bgr, _ = prepare_perturbed_sample(sample, name, params, seed)
        target = folder / f"{sample.image_path.stem}.png"
        bgr_to_pil(bgr).save(target)
        files[sample.image_path.name] = target
    return files


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace, factory: DetectorFactory) -> int:
    for name in available():
        t = get(name)
        spec = ", ".join(
            f"{p}[{s.low}..{s.high}]" if s.low is not None else p
            for p, s in sorted(t.params.items())
        )
        print(f"{name:22s} {t.family:10s} {'geometry ' if t.geometry else '         '}{spec}")
    return EXIT_OK


def cmd_validate(args: argparse.Namespace, factory: DetectorFactory) -> int:
    s, _ = _settings(args)
    ds = _load_dataset(s)
    structural = ds.validate()
    annotations = validate_annotations(ds)
    errors = [str(i) for i in structural.errors] + [str(i) for i in annotations.errors]
    warnings = [str(i) for i in structural.warnings] + [str(i) for i in annotations.warnings]
    print(f"split={ds.split} images={len(ds)} classes={ds.class_names}")
    for line in errors:
        print(f"ERROR   {line}")
    for line in warnings:
        print(f"WARNING {line}")
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    if errors:
        print("dataset is NOT valid for evaluation", file=sys.stderr)
        return EXIT_INVALID
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace, factory: DetectorFactory) -> int:
    s, _ = _settings(args)
    ds = _load_dataset(s)
    report = evaluate_baseline(factory(s), ds, iou_threshold=s["iou"])
    out = Path(args.output) if args.output else Path(s["out"]) / "baseline.json"
    _write_json(out, report.to_dict())
    a = report.aggregate
    print(
        f"baseline on {len(ds)} images: precision={a['precision']:.4f} "
        f"recall={a['recall']:.4f} f1={a['f1']:.4f} map50={report.map50:.4f}"
    )
    print(f"saved {out}")
    return EXIT_OK


def cmd_perturb(args: argparse.Namespace, factory: DetectorFactory) -> int:
    s, _ = _settings(args)
    ds = _load_dataset(s)
    detector = factory(s)
    params = _parse_params(args.param)
    out = Path(s["out"])
    if args.mode == "quick":
        result = quick_mode(detector, ds, args.transform, params, seed=args.seed,
                            iou_threshold=s["iou"])
        run = build_quick_run(result)
        print(f"quick mode (label-free instability): {result.summary}")
    else:
        result = evaluate_perturbed(detector, ds, args.transform, params,
                                    seed=args.seed, iou_threshold=s["iou"])
        run = build_benchmark_run(result.baseline, [result])
        d = result.deltas
        print(f"{result.perturbation['name']} {result.perturbation['params']}: "
              f"delta f1={d['f1']:+.4f} recall={d['recall']:+.4f} "
              f"precision={d['precision']:+.4f} map50={d['map50']:+.4f}")
    paths = write_report(run, out, basename="perturb")
    if args.mode == "benchmark":
        paths["csv"] = write_failures_csv(run, out / "perturb_failures.csv")
    for path in paths.values():
        print(f"saved {path}")
    if args.gallery:
        applied = result.perturbation
        clean = {p.name: p for p in ds.image_paths}
        pert = _perturbed_image_files(ds, applied["name"], applied["params"],
                                      applied["seed"], out)
        print(f"saved {write_gallery(run, out, clean_images=clean, perturbed_images=pert, overwrite=True)}")
    return EXIT_OK


def cmd_search(args: argparse.Namespace, factory: DetectorFactory) -> int:
    s, cfg = _settings(args)
    g = lambda cli, key, default: pick(cli, cfg, "search", key, default)  # noqa: E731
    method = g(args.method, "method", "random")
    if method not in ("random", "tpe"):
        raise CliError(f"method must be 'random' or 'tpe', got {method!r}")
    metric = g(args.metric, "metric", "f1")
    if metric not in SUPPORTED_METRICS:
        raise CliError(f"metric must be one of {SUPPORTED_METRICS}, got {metric!r}")
    budget, seed = int(g(args.budget, "budget", 20)), int(g(args.seed, "seed", 0))
    holdout = float(g(args.holdout_fraction, "holdout_fraction", 0.3))
    top_k, n_seeds = int(g(args.confirm_top, "confirm_top", 3)), int(g(args.confirm_seeds, "confirm_seeds", 3))
    names = g(args.transforms, "transforms", None)
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    if budget < 1:
        raise CliError("--budget must be >= 1")
    if not 0.0 <= holdout < 1.0:
        raise CliError("--holdout-fraction must be in [0, 1)")

    ds = _load_dataset(s)
    space = SearchSpace.default(include=names)
    detector = factory(s)
    objective = ObjectiveConfig(metric=metric)

    discovery, held = ds, None
    if holdout > 0:
        if len(ds) < 2:
            raise CliError("confirmation needs >= 2 images; use --holdout-fraction 0 to skip it")
        discovery, held = ds.partition(holdout, g(args.partition_seed, "partition_seed", 123))
    else:
        print("note: confirmation skipped (holdout_fraction=0); scores are discovery-split only")

    runner = run_random_search if method == "random" else run_tpe_search
    result = runner(detector, discovery, budget=budget, seed=seed, space=space,
                    objective=objective, iou_threshold=s["iou"])
    confirmation = None
    if held is not None:
        confirmation = confirm_trials(
            detector, held, result.trials, top_k=top_k, n_seeds=n_seeds,
            seed=int(g(args.confirm_seed, "confirm_seed", 1000)),
            objective=objective, iou_threshold=s["iou"],
        )
    out = Path(s["out"])
    paths = write_search_report(result, out, confirmation)
    c = result.counts()
    best = result.best
    print(f"{method}: {c['valid']}/{c['total']} valid trials "
          f"({c['invalid']} invalid, {c['error']} errored)")
    if best is not None:
        print(f"best: {best.name} {best.params} {metric} degradation={best.score:.4f}")
    else:
        print("no valid trial; nothing ranked")
    for path in paths.values():
        print(f"saved {path}")
    if confirmation is not None and confirmation.results:
        run = build_benchmark_run(confirmation.baseline, confirmation.results)
        conf_paths = write_report(run, out, basename="confirmation")
        conf_paths["csv"] = write_failures_csv(run, out / "confirmation_failures.csv")
        for path in conf_paths.values():
            print(f"saved {path}")
        if args.gallery:
            clean = {p.name: p for p in held.image_paths}
            first = confirmation.results[0].perturbation
            pert = _perturbed_image_files(held, first["name"], first["params"], first["seed"], out)
            print("note: gallery images are shown for the first confirmed candidate only")
            print(f"saved {write_gallery(run, out, clean_images=clean, perturbed_images=pert, overwrite=True)}")
    return EXIT_OK if best is not None else EXIT_INVALID


def cmd_compare(args: argparse.Namespace, factory: DetectorFactory) -> int:
    s, cfg = _settings(args)
    g = lambda cli, key, default: pick(cli, cfg, "search", key, default)  # noqa: E731
    metric = g(args.metric, "metric", "f1")
    if metric not in SUPPORTED_METRICS:
        raise CliError(f"metric must be one of {SUPPORTED_METRICS}, got {metric!r}")
    n_seeds = int(args.runs)
    if n_seeds < 1:
        raise CliError("--runs must be >= 1")
    base_seed = int(g(args.seed, "seed", 0))
    names = g(args.transforms, "transforms", None)
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    result = compare_methods(
        factory(s), _load_dataset(s), methods=methods,
        seeds=range(base_seed, base_seed + n_seeds),
        budget=int(g(args.budget, "budget", 20)),
        space=SearchSpace.default(include=names),
        objective=ObjectiveConfig(metric=metric), iou_threshold=s["iou"],
    )
    for m in result.methods:
        st = m.stats
        mean = "n/a" if st["mean"] is None else f"{st['mean']:.4f}"
        print(f"{m.method}: mean best {metric} degradation={mean} over {st['n']}/{len(m.seeds)} runs")
    for path in write_comparison(result, s["out"]).values():
        print(f"saved {path}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# parser / entry point
# ---------------------------------------------------------------------------

def _common(p: argparse.ArgumentParser, *, detector: bool = True) -> None:
    p.add_argument("--config", help="YAML run config (see configs/default.yaml)")
    p.add_argument("--data", help="YOLO data.yaml")
    p.add_argument("--split", help="split name (default: test, then val; never train)")
    p.add_argument("--allow-missing-labels", action="store_true",
                   help="treat missing label files as empty (otherwise an error)")
    if detector:
        p.add_argument("--weights", help="existing YOLO detection checkpoint (.pt)")
        p.add_argument("--conf", type=float, help="confidence threshold (default 0.25)")
        p.add_argument("--imgsz", type=int, help="inference size (default 640)")
        p.add_argument("--device", help="e.g. cpu, cuda:0 (default cpu)")
        p.add_argument("--iou", type=float, help="matching IoU threshold (default 0.5)")
        p.add_argument("--out-dir", help="output directory (default results)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="failureforge",
        description="Local-first robustness testing for YOLO object detectors.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-perturbations", help="list available perturbations and ranges"
                   ).set_defaults(func=cmd_list)

    v = sub.add_parser("validate-data", help="validate dataset structure and annotations")
    _common(v, detector=False)
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("evaluate", help="baseline evaluation on clean images")
    _common(e)
    e.add_argument("--output", help="JSON path (default <out-dir>/baseline.json)")
    e.set_defaults(func=cmd_evaluate)

    t = sub.add_parser("perturb", help="evaluate one perturbation; write report (+gallery)")
    _common(t)
    t.add_argument("--transform", required=True, help="see list-perturbations")
    t.add_argument("--param", action="append", metavar="KEY=VALUE",
                   help="transform parameter; repeatable")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--mode", choices=["benchmark", "quick"], default="benchmark",
                   help="benchmark needs labels; quick is label-free instability only")
    t.add_argument("--gallery", action="store_true", help="also write an HTML failure gallery")
    t.set_defaults(func=cmd_perturb)

    s = sub.add_parser("search", help="search for the most damaging perturbation, then confirm")
    _common(s)
    s.add_argument("--method", choices=["random", "tpe"], help="default random")
    s.add_argument("--budget", type=int, help="number of trials (default 20)")
    s.add_argument("--seed", type=int, help="search seed (default 0)")
    s.add_argument("--metric", choices=list(SUPPORTED_METRICS), help="default f1")
    s.add_argument("--transforms", help="comma-separated subset (default: all searchable)")
    s.add_argument("--holdout-fraction", type=float,
                   help="fraction of images kept for confirmation, 0 to skip (default 0.3)")
    s.add_argument("--partition-seed", type=int)
    s.add_argument("--confirm-top", type=int, help="distinct candidates to confirm (default 3)")
    s.add_argument("--confirm-seeds", type=int, help="fresh seeds per candidate (default 3)")
    s.add_argument("--confirm-seed", type=int)
    s.add_argument("--gallery", action="store_true", help="also write an HTML gallery")
    s.set_defaults(func=cmd_search)

    c = sub.add_parser("compare", help="equal-budget comparison of search methods over several seeds")
    _common(c)
    c.add_argument("--methods", default="random,tpe", help="comma-separated: random,tpe")
    c.add_argument("--runs", type=int, default=3, help="seeds per method (default 3)")
    c.add_argument("--budget", type=int, help="trials per run (default 20)")
    c.add_argument("--seed", type=int, help="first seed (default 0)")
    c.add_argument("--metric", choices=list(SUPPORTED_METRICS), help="default f1")
    c.add_argument("--transforms", help="comma-separated subset")
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: Sequence[str] | None = None, *,
         detector_factory: DetectorFactory | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args, detector_factory or _default_detector))
    except (CliError, ConfigError, DatasetError, DetectorError, PerturbationError,
            SearchSpaceError, OptionalDependencyError, FileNotFoundError,
            ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
