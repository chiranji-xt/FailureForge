"""JSON, Markdown, and CSV output for search results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..reporting import csv_safe, json_safe
from .confirm import ConfirmationResult
from .trials import SearchResult

SCHEMA_VERSION = "failureforge.search-report.v1"

CSV_COLUMNS = (
    "index", "status", "transform", "family", "seed", "score", "metric",
    "baseline_value", "perturbed_value", "params", "error",
)

_LIMITATIONS = (
    "Scores are observations of one metric for one detector, dataset split, and "
    "seed stream; they are not universal claims about a perturbation.",
    "Larger score = larger drop of the chosen higher-is-better metric "
    "(baseline - perturbed); negative = improvement. Failure counts and Quick "
    "Mode instability are not part of the score.",
    "Invalid/errored trials are listed but never ranked and never scored as zero.",
    "Search scores are optimistic for the discovery split (selection effect); "
    "use the confirmation section for held-out evidence.",
)


def search_payload(
    result: SearchResult, confirmation: ConfirmationResult | None = None
) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "search": result.to_dict(),
        "confirmation": confirmation.to_dict() if confirmation else None,
        "limitations": list(_LIMITATIONS),
    }
    return json_safe(payload)


def write_trials_csv(result: SearchResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for t in result.trials:
            writer.writerow([
                t.index, t.status, csv_safe(t.name), csv_safe(t.family), t.seed,
                "" if t.score is None else repr(t.score), result.objective.metric,
                "" if t.baseline_value is None else repr(t.baseline_value),
                "" if t.perturbed_value is None else repr(t.perturbed_value),
                csv_safe(json.dumps(t.params, sort_keys=True)), csv_safe(t.error),
            ])
    return path


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_search_markdown(
    result: SearchResult,
    confirmation: ConfirmationResult | None = None,
    *,
    title: str = "FailureForge Search Report",
    top: int = 10,
) -> str:
    c = result.counts()
    metric = result.objective.metric
    lines = [
        f"# {title}", "",
        "## Setup", "",
        f"- Method: `{result.method}` (seed {result.seed}, budget {result.budget})",
        f"- Objective: `{metric}` degradation = baseline - perturbed "
        "(larger is worse)",
        f"- IoU threshold: {result.iou_threshold}",
        f"- Split: `{result.metadata.get('split')}` "
        f"({result.metadata.get('num_images')} images)",
        f"- Baseline {metric}: {_fmt(result.baseline_metrics.get(metric))}",
        f"- Trials: {c['total']} total, {c['valid']} valid, "
        f"{c['invalid']} invalid, {c['error']} errored", "",
    ]
    best = result.best
    lines += ["## Best trial", ""]
    if best is None:
        lines.append("No valid trial: nothing is ranked.")
    else:
        lines.append(
            f"`{best.name}` {json.dumps(best.params, sort_keys=True)} "
            f"(trial {best.index}, seed {best.seed}): score {_fmt(best.score)}"
        )
    ranked = result.ranked(unique=True)[:top]
    lines += ["", f"## Top {len(ranked)} distinct trials", ""]
    if ranked:
        lines += ["| rank | trial | transform | params | score |", "|---|---|---|---|---|"]
        for rank, t in enumerate(ranked, 1):
            lines.append(
                f"| {rank} | {t.index} | {t.name} | "
                f"`{json.dumps(t.params, sort_keys=True)}` | {_fmt(t.score)} |"
            )
    bad = [t for t in result.trials if not t.valid]
    lines += ["", "## Invalid or errored trials", ""]
    if not bad:
        lines.append("None.")
    for t in bad:
        reason = t.error or "; ".join(t.warnings) or "invalid objective"
        lines.append(f"- trial {t.index} `{t.name}` ({t.status}): {reason}")
    if confirmation is not None:
        lines += [
            "", "## Confirmation (held-out)", "",
            f"Split `{confirmation.baseline.split}`, "
            f"{len(confirmation.baseline.per_image)} images, "
            f"{confirmation.metadata['n_seeds']} fresh seeds per candidate.", "",
            "| transform | params | search score | mean | std | min | max | valid/total | all seeds degrade |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in confirmation.records:
            lines.append(
                f"| {r.name} | `{json.dumps(r.params, sort_keys=True)}` | "
                f"{_fmt(r.search_score)} | {_fmt(r.mean)} | {_fmt(r.std)} | "
                f"{_fmt(r.min)} | {_fmt(r.max)} | {r.n_valid}/{len(r.scores)} | "
                f"{'yes' if r.degrades_in_all_seeds else 'no'} |"
            )
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in _LIMITATIONS] + [""]
    return "\n".join(lines)


def write_search_report(
    result: SearchResult,
    output_dir: str | Path,
    confirmation: ConfirmationResult | None = None,
    *,
    basename: str = "search",
) -> dict[str, Path]:
    """Write ``<basename>.json``, ``.md`` and ``_trials.csv``; returns their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{basename}.json"
    json_path.write_text(
        json.dumps(search_payload(result, confirmation), indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path = out / f"{basename}.md"
    md_path.write_text(render_search_markdown(result, confirmation), encoding="utf-8")
    csv_path = write_trials_csv(result, out / f"{basename}_trials.csv")
    return {"json": json_path, "md": md_path, "csv": csv_path}
