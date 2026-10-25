"""Schema-versioned JSON and dependency-free Markdown reporting.

Reports consume the records produced by :mod:`failureforge.failures`. They do
not evaluate models, mutate results, or turn Quick Mode instability counts into
accuracy claims.

Ranking policy
--------------
One metric at a time (default F1). ``drop = baseline - perturbed`` so a positive
drop means degradation *on this evaluated model/dataset/split/thresholds*.
Missing or non-finite values are not ranked. Raw instability counts are never
ranked as accuracy degradation.

JSON policy
-----------
Schema version is always present. NumPy scalars/arrays are converted to native
Python values; non-finite floats become ``null``; absolute paths are reduced to
their basename unless ``include_absolute_paths=True``. Deterministic ordering is
preserved by sorting ranked trials, groups, and record lists.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .failures import (
    DEFAULT_RANK_METRICS,
    FAILURE_CATEGORIES,
    QUICK_INSTABILITY_KINDS,
    SCHEMA_VERSION,
    BenchmarkRun,
    QuickRun,
    TrialRecord,
)

_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")
_FAILURE_LABELS = {
    "false_negative": "False negative",
    "false_positive": "False positive",
    "class_confusion": "Class confusion",
    "localization_error": "Localization error",
    "duplicate_detection": "Duplicate detection",
    "unmatched": "Unmatched/ambiguous",
}
_KIND_LABELS = {
    "disappeared": "Disappeared",
    "appeared": "Appeared",
    "class_changed": "Class changed",
    "confidence_changed": "Confidence changed",
    "box_drift": "Box drift",
}


# ---------------------------------------------------------------------------
# JSON sanitisation
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats into JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _redact_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_paths(item) for item in value]
    if isinstance(value, str) and _ABSOLUTE_PATH.match(value):
        return Path(value).name or value
    return value


def run_payload(run: BenchmarkRun | QuickRun, *, include_absolute_paths: bool = False) -> dict:
    """Return a JSON-safe run payload with a guaranteed schema version."""
    payload = json_safe(run.to_dict())
    payload.setdefault("schema_version", SCHEMA_VERSION)
    if not include_absolute_paths:
        payload = _redact_paths(payload)
    return payload


def dumps_report(run: BenchmarkRun | QuickRun, *, include_absolute_paths: bool = False) -> str:
    """Serialize a run to valid JSON text (non-finite floats become ``null``)."""
    payload = run_payload(run, include_absolute_paths=include_absolute_paths)
    return json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


# ---------------------------------------------------------------------------
# Ranking and grouping
# ---------------------------------------------------------------------------

def _check_rank_metric(rank_metric: str) -> str:
    if rank_metric not in DEFAULT_RANK_METRICS:
        raise ValueError(
            f"rank_metric must be one of {DEFAULT_RANK_METRICS}, got {rank_metric!r}"
        )
    return rank_metric


def _params_key(trial: TrialRecord) -> str:
    params = (trial.transformation or {}).get("params") or {}
    return json.dumps(params, sort_keys=True, default=str)


def ranked_trials(run: BenchmarkRun, rank_metric: str = "f1") -> list[dict]:
    """Deterministically rank finite drops; unranked trials follow in stable order."""
    _check_rank_metric(rank_metric)
    entries = []
    for index, trial in enumerate(run.trials):
        value = trial.drops.get(rank_metric)
        finite = (
            trial.compatible
            and value is not None
            and math.isfinite(float(value))
        )
        entries.append({
            "index": index,
            "trial": trial,
            "drop": float(value) if value is not None else None,
            "finite": finite,
        })
    entries.sort(key=lambda entry: (
        0 if entry["finite"] else 1,
        -entry["drop"] if entry["finite"] else 0.0,
        (entry["trial"].transformation or {}).get("name", ""),
        _params_key(entry["trial"]),
        int((entry["trial"].transformation or {}).get("seed", 0)),
        entry["index"],
    ))
    rank = 0
    for entry in entries:
        if entry["finite"]:
            rank += 1
            entry["rank"] = rank
        else:
            entry["rank"] = None
    return entries


def group_by_family(run: BenchmarkRun, rank_metric: str = "f1") -> list[dict]:
    return _group(run, rank_metric, lambda trial: (
        (trial.transformation or {}).get("family") or "unknown"
    ))


def group_by_operation(run: BenchmarkRun, rank_metric: str = "f1") -> list[dict]:
    return _group(run, rank_metric, lambda trial: (
        (trial.transformation or {}).get("name") or "unknown"
    ))


def _group(run: BenchmarkRun, rank_metric: str, key) -> list[dict]:
    _check_rank_metric(rank_metric)
    groups: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for trial in run.trials:
        name = key(trial)
        counts[name] = counts.get(name, 0) + 1
        if not trial.compatible:
            continue
        value = trial.drops.get(rank_metric)
        if value is not None and math.isfinite(float(value)):
            groups.setdefault(name, []).append(float(value))
    output = []
    for name in sorted(counts):
        values = groups.get(name, [])
        output.append({
            "key": name,
            "trials": counts[name],
            "ranked": len(values),
            "mean_drop": sum(values) / len(values) if values else None,
            "max_drop": max(values) if values else None,
            "min_drop": min(values) if values else None,
        })
    return output


def severity_ladders(run: BenchmarkRun, rank_metric: str = "f1") -> list[dict]:
    """Parameter ladders when an operation's trials vary exactly one numeric param.

    No severity labels are invented: the ladder is the tested parameter value.
    """
    _check_rank_metric(rank_metric)
    by_name: dict[str, list[tuple[int, TrialRecord]]] = {}
    for index, trial in enumerate(run.trials):
        name = (trial.transformation or {}).get("name")
        if not name or not trial.compatible:
            continue
        by_name.setdefault(name, []).append((index, trial))

    ladders: list[dict] = []
    for name in sorted(by_name):
        group = by_name[name]
        keys: set[str] = set()
        for _index, trial in group:
            keys.update(((trial.transformation or {}).get("params") or {}).keys())
        if len(keys) != 1:
            continue
        param = next(iter(keys))
        points = []
        numeric = True
        for index, trial in group:
            params = (trial.transformation or {}).get("params") or {}
            if param not in params or isinstance(params[param], bool):
                numeric = False
                break
            try:
                value = float(params[param])
            except (TypeError, ValueError):
                numeric = False
                break
            drop = trial.drops.get(rank_metric)
            points.append({
                "value": value,
                "drop": float(drop) if drop is not None else None,
                "seed": int((trial.transformation or {}).get("seed", 0)),
                "index": index,
            })
        if not numeric or len(points) < 2:
            continue
        points.sort(key=lambda point: (point["value"], point["seed"], point["index"]))
        ladders.append({"name": name, "param": param, "points": points})
    return ladders


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _fmt_signed(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{digits}f}"


def _basename(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if "\\" in text or "/" in text:
        return Path(text).name or text
    return text


def _trial_label(trial: TrialRecord) -> str:
    transformation = trial.transformation or {}
    params = transformation.get("params") or {}
    return f"{transformation.get('name', 'n/a')} {json.dumps(params, sort_keys=True, default=str)}"


def _failure_total(trial: TrialRecord) -> int:
    return int(sum(trial.failure_counts.values()))


def _scope_lines(run: BenchmarkRun | QuickRun) -> list[str]:
    meta = run.metadata or {}
    dataset = meta.get("dataset") or {}
    model = meta.get("model") or {}
    thresholds = meta.get("thresholds") or {}
    environment = meta.get("environment") or {}
    lines = []
    lines.append(
        f"- Dataset split: `{dataset.get('split', 'n/a')}`; "
        f"images: {dataset.get('num_images', 'n/a')}; "
        f"classes: {', '.join(dataset.get('class_names') or []) or 'n/a'}"
    )
    weights = _basename(model.get("weights")) or "n/a"
    sha = model.get("sha256")
    sha_text = f"`{str(sha)[:12]}`" if sha else "n/a"
    lines.append(
        f"- Model: `{weights}`; task: `{model.get('task', 'n/a')}`; "
        f"sha256: {sha_text}; device: `{model.get('device', 'n/a')}`"
    )
    lines.append(
        f"- Matching: IoU >= {thresholds.get('iou_threshold', 'n/a')}; "
        f"localization threshold: {thresholds.get('localization_iou_threshold', 'n/a')}; "
        f"aggregation: {meta.get('aggregation', 'n/a')}"
    )
    lines.append(
        f"- Environment: FailureForge {environment.get('failureforge', 'n/a')}, "
        f"Python {environment.get('python', 'n/a')}, "
        f"NumPy {environment.get('numpy', 'n/a')}, "
        f"Pillow {environment.get('pillow', 'n/a')}, "
        f"{environment.get('platform', 'n/a')} {environment.get('platform_release', '')}".rstrip()
    )
    lines.append(f"- Schema: `{run.schema_version}`")
    if meta.get("scope"):
        lines.append(f"- Scope: {meta['scope']}")
    return lines


def _warnings_section(run, *, unranked: int, incompatible: int,
                      rank_metric: str) -> list[str]:
    lines = ["## Warnings", ""]
    warnings = list(run.warnings)
    if unranked:
        warnings.append(
            f"{unranked} trial(s) could not be ranked for `{rank_metric}` "
            "(missing/non-finite metric or incompatible comparison)."
        )
    if incompatible:
        warnings.append(
            f"{incompatible} trial(s) are marked incompatible with the baseline "
            "(different IoU threshold or aggregation method)."
        )
    if warnings:
        lines.extend(f"- {message}" for message in warnings)
    else:
        lines.append("No warnings recorded.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_markdown(
    run: BenchmarkRun | QuickRun,
    *,
    title: str = "FailureForge Report",
    rank_metric: str = "f1",
) -> str:
    """Render a deterministic, dependency-free Markdown report."""
    _check_rank_metric(rank_metric)
    lines: list[str] = [f"# {title}", ""]
    if isinstance(run, BenchmarkRun):
        lines.extend(_render_benchmark(run, rank_metric))
    elif isinstance(run, QuickRun):
        lines.extend(_render_quick(run))
    else:
        raise TypeError(f"unsupported run type: {type(run).__name__}")
    return "\n".join(lines).rstrip() + "\n"


def _render_benchmark(run: BenchmarkRun, rank_metric: str) -> list[str]:
    lines: list[str] = [
        f"Mode: **Benchmark (verified against ground truth)**; ranking metric: `{rank_metric}`.",
        "",
        "## Scope and reproducibility",
        "",
        *_scope_lines(run),
        "",
        "## Baseline (unperturbed)",
        "",
    ]
    baseline = run.baseline.metrics
    aggregate = baseline.get("aggregate") or {}
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key, label in (("tp", "TP"), ("fp", "FP"), ("fn", "FN")):
        lines.append(f"| {label} | {aggregate.get(key, 'n/a')} |")
    for key in ("precision", "recall", "f1"):
        lines.append(f"| {key.title()} | {_fmt(aggregate.get(key))} |")
    lines.append(f"| mAP50 | {_fmt(baseline.get('map50'))} |")
    lines.append(f"| Images | {baseline.get('num_images', 'n/a')} |")
    lines.append("")

    ranked = ranked_trials(run, rank_metric)
    finite = [entry for entry in ranked if entry["finite"]]
    unranked = [entry for entry in ranked if not entry["finite"]]
    incompatible = sum(1 for trial in run.trials if not trial.compatible)

    lines.extend([
        "## Ranked perturbations",
        "",
        f"Drop = baseline - perturbed for `{rank_metric}`; a positive value means "
        "degradation on this evaluated model/dataset/thresholds. Rankings are "
        "ordinal (ties broken deterministically) and are not a universal claim.",
        "",
    ])
    if not run.trials:
        lines.extend(["No perturbation trials were evaluated.", ""])
    else:
        lines.append(
            f"| Rank | Family | Operation | Params | Seed | {rank_metric} drop "
            "| mAP50 drop | Verified failures |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for entry in ranked:
            trial = entry["trial"]
            transformation = trial.transformation or {}
            params = transformation.get("params") or {}
            lines.append(
                f"| {entry['rank'] if entry['rank'] is not None else '—'} "
                f"| {transformation.get('family') or 'unknown'} "
                f"| {transformation.get('name', 'n/a')} "
                f"| `{json.dumps(params, sort_keys=True, default=str)}` "
                f"| {transformation.get('seed', 0)} "
                f"| {_fmt_signed(entry['drop'])} "
                f"| {_fmt_signed(trial.drops.get('map50'))} "
                f"| {_failure_total(trial)} |"
            )
        lines.append("")

    lines.extend([
        "## Grouped results",
        "",
        f"Mean/maximum `{rank_metric}` drop by transformation family and operation "
        "(only finite drops are averaged).",
        "",
    ])
    for heading, groups in (
        ("### By transformation family", group_by_family(run, rank_metric)),
        ("### By operation", group_by_operation(run, rank_metric)),
    ):
        lines.append(heading)
        lines.append("")
        lines.append(
            f"| {heading.split('By ')[-1].title()} | Trials | Ranked | Mean drop "
            "| Max drop | Min drop |"
        )
        lines.append("|---|---|---|---|---|---|")
        for group in groups:
            lines.append(
                f"| {group['key']} | {group['trials']} | {group['ranked']} "
                f"| {_fmt_signed(group['mean_drop'])} "
                f"| {_fmt_signed(group['max_drop'])} "
                f"| {_fmt_signed(group['min_drop'])} |"
            )
        lines.append("")

    ladders = severity_ladders(run, rank_metric)
    if ladders:
        lines.extend(["## Severity ladders", ""])
        lines.append(
            "Shown only when an operation's tested configurations vary exactly one "
            "numeric parameter; values are the tested parameter values."
        )
        lines.append("")
        for ladder in ladders:
            points = ", ".join(
                f"{point['value']:g} → {_fmt_signed(point['drop'])}"
                for point in ladder["points"]
            )
            lines.append(f"- **{ladder['name']}** (`{ladder['param']}`): {points}")
        lines.append("")

    lines.extend([
        "## Failure categories (Benchmark Mode)",
        "",
        "Verified categories derived from GT/prediction matching; they explain "
        "TP/FP/FN and never change them. A matched localization error is an "
        "aspect of a true positive.",
        "",
        "| Category | Baseline | All trials (sum) |",
        "|---|---|---|",
    ])
    trial_totals = {category: 0 for category in FAILURE_CATEGORIES}
    for trial in run.trials:
        for category in FAILURE_CATEGORIES:
            trial_totals[category] += int(trial.failure_counts.get(category, 0))
    for category in FAILURE_CATEGORIES:
        lines.append(
            f"| {_FAILURE_LABELS[category]} "
            f"| {int(run.baseline.failure_counts.get(category, 0))} "
            f"| {trial_totals[category]} |"
        )
    lines.append("")

    top = finite[:10]
    if top:
        lines.extend([
            "### Failure counts for the top-ranked trials",
            "",
            "| Rank | Operation | Params | "
            + " | ".join(_FAILURE_LABELS[c] for c in FAILURE_CATEGORIES) + " |",
            "|---|---|---|" + "---|" * len(FAILURE_CATEGORIES),
        ])
        for entry in top:
            trial = entry["trial"]
            transformation = trial.transformation or {}
            params = transformation.get("params") or {}
            counts = " | ".join(
                str(int(trial.failure_counts.get(category, 0)))
                for category in FAILURE_CATEGORIES
            )
            lines.append(
                f"| {entry['rank']} | {transformation.get('name', 'n/a')} "
                f"| `{json.dumps(params, sort_keys=True, default=str)}` | {counts} |"
            )
        lines.append("")

    lines.extend(_warnings_section(
        run, unranked=len(unranked), incompatible=incompatible,
        rank_metric=rank_metric,
    ))
    lines.extend([
        "## Interpretation and limitations",
        "",
        "- Results are observations for the evaluated model, dataset, split, seeds, "
        "and thresholds; they are not proof of universal or real-world failure rates.",
        "- Single-seed comparisons are noisy; run multiple seeds for stability.",
        "- Geometric/cropping transforms use axis-aligned pixel boxes and can be "
        "conservative for rotated objects.",
        "- Failure categories are deterministic rules over matched detections; "
        "unmatched/ambiguous records are not forced into another category.",
        "",
    ])
    return lines


def _render_quick(run: QuickRun) -> list[str]:
    analysis = run.analysis
    lines: list[str] = [
        "Mode: **Quick (prediction instability; no ground truth)**.",
        "",
        "## Scope and reproducibility",
        "",
        *_scope_lines(run),
        "",
        "## Instability summary",
        "",
        "These are prediction changes between clean and perturbed images. They are "
        "**not** verified false positives/negatives and are **not** accuracy metrics.",
        "",
        "| Instability | Count |",
        "|---|---|",
    ]
    for kind in QUICK_INSTABILITY_KINDS:
        lines.append(f"| {_KIND_LABELS[kind]} | {int(analysis.counts.get(kind, 0))} |")
    summary = analysis.summary
    lines.append(
        f"| Mean confidence delta | {_fmt_signed(summary.get('mean_confidence_delta'))} |"
    )
    lines.append(f"| Mean matched box IoU | {_fmt(summary.get('mean_box_iou'))} |")
    lines.append(
        f"| Changed detection fraction | {_fmt(summary.get('changed_detection_fraction'))} |"
    )
    lines.append("")

    lines.extend([
        "## Per-image instability",
        "",
        "| Image | Clean detections | Perturbed detections | "
        + " | ".join(_KIND_LABELS[kind] for kind in QUICK_INSTABILITY_KINDS) + " |",
        "|---|---|---|" + "---|" * len(QUICK_INSTABILITY_KINDS),
    ])
    for image in analysis.per_image:
        counts = " | ".join(
            str(int(image["counts"].get(kind, 0)))
            for kind in QUICK_INSTABILITY_KINDS
        )
        lines.append(
            f"| {image['image']} | {image['num_clean']} "
            f"| {image['num_perturbed']} | {counts} |"
        )
    lines.append("")
    if not analysis.per_image:
        lines.extend(["No images were evaluated.", ""])

    warnings = list(run.warnings)
    lines.extend(["## Warnings", ""])
    if warnings:
        lines.extend(f"- {message}" for message in warnings)
    else:
        lines.append("No warnings recorded.")
    lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "- Quick Mode never reports precision, recall, F1, or mAP: without labels, "
        "correctness is unknown.",
        "- Location-matched changes are treated as class changes where classes differ; "
        "unmatched clean detections are `disappeared`, unmatched perturbed "
        "detections are `appeared`.",
        "- Results describe this model and image set only; they do not establish "
        "accuracy degradation.",
        "",
    ])
    return lines


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_json_report(
    run: BenchmarkRun | QuickRun,
    output_path: str | Path,
    *,
    include_absolute_paths: bool = False,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        dumps_report(run, include_absolute_paths=include_absolute_paths),
        encoding="utf-8",
    )
    return path


def write_markdown_report(
    run: BenchmarkRun | QuickRun,
    output_path: str | Path,
    *,
    title: str = "FailureForge Report",
    rank_metric: str = "f1",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown(run, title=title, rank_metric=rank_metric),
        encoding="utf-8",
    )
    return path


def write_report(
    run: BenchmarkRun | QuickRun,
    output_dir: str | Path,
    *,
    basename: str = "failureforge_report",
    formats: tuple[str, ...] = ("json", "md"),
    include_absolute_paths: bool = False,
    title: str = "FailureForge Report",
    rank_metric: str = "f1",
) -> dict[str, Path]:
    """Write one or both report formats; returns ``{format: path}``."""
    unknown = set(formats) - {"json", "md"}
    if unknown:
        raise ValueError(f"unsupported report formats: {sorted(unknown)}")
    directory = Path(output_dir)
    paths: dict[str, Path] = {}
    if "json" in formats:
        paths["json"] = write_json_report(
            run, directory / f"{basename}.json",
            include_absolute_paths=include_absolute_paths,
        )
    if "md" in formats:
        paths["md"] = write_markdown_report(
            run, directory / f"{basename}.md",
            title=title, rank_metric=rank_metric,
        )
    return paths


# ---------------------------------------------------------------------------
# CSV failure export
# ---------------------------------------------------------------------------

def csv_safe(value: Any) -> str:
    """Text for a CSV cell with spreadsheet-formula injection neutralized."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


FAILURE_CSV_COLUMNS = (
    "trial", "transform", "params", "seed", "category", "counts_as", "image",
    "gt_class", "pred_class", "gt_box", "pred_box", "score", "iou", "reason",
)


def write_failures_csv(run: BenchmarkRun, path: str | Path) -> Path:
    """One row per verified failure record (baseline = trial ``baseline``).

    Benchmark runs only: Quick Mode instability is not a verified failure and is
    never written here.
    """
    if not isinstance(run, BenchmarkRun):
        raise ValueError("failure CSV is only available for benchmark runs")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = [("baseline", run.baseline)] + [(str(i), t) for i, t in enumerate(run.trials)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(FAILURE_CSV_COLUMNS)
        for label, record in groups:
            tr = record.transformation or {}
            for f in record.failures:
                writer.writerow([
                    label, csv_safe(tr.get("name")),
                    csv_safe(json.dumps(json_safe(tr.get("params")), sort_keys=True) if tr else ""),
                    tr.get("seed", ""), csv_safe(f.category), f.counts_as, csv_safe(f.image),
                    csv_safe(f.gt_class_name), csv_safe(f.pred_class_name),
                    json.dumps(json_safe(f.gt_box)) if f.gt_box else "",
                    json.dumps(json_safe(f.pred_box)) if f.pred_box else "",
                    "" if f.score is None else repr(float(f.score)),
                    "" if f.iou is None else repr(float(f.iou)),
                    csv_safe(f.reason),
                ])
    return path
