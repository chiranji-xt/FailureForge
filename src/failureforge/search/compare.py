"""Equal-budget comparison of search methods over several seeds.

Answers "does TPE find more damaging configurations than Random Search for this
detector and dataset?" with descriptive statistics only. Every run shares one
baseline, search space, objective, and budget; only the method and seed vary.
No significance test is applied: with a handful of seeds the spread, not a
p-value, is the honest summary.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..dataset import YoloDataset
from ..evaluation import DEFAULT_IOU
from ..evaluator import evaluate_baseline
from ..objective import ObjectiveConfig
from ..reporting import json_safe
from .random_search import run_random_search
from .runner import check_budget
from .space import SearchSpace
from .tpe_search import run_tpe_search
from .trials import SearchResult

SCHEMA_VERSION = "failureforge.search-comparison.v1"
RUNNERS = {"random": run_random_search, "tpe": run_tpe_search}


def best_so_far(result: SearchResult) -> list[float | None]:
    """Running best valid score after each trial (``None`` until one is valid)."""
    out: list[float | None] = []
    best: float | None = None
    for trial in sorted(result.trials, key=lambda t: t.index):
        if trial.valid and (best is None or trial.score > best):
            best = trial.score
        out.append(best)
    return out


def _stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


@dataclass
class MethodSummary:
    method: str
    seeds: list[int]
    best_scores: list[float | None]          # None = no valid trial for that seed
    valid_counts: list[int]
    curves: list[list[float | None]]
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method, "seeds": self.seeds,
            "best_scores": self.best_scores, "valid_counts": self.valid_counts,
            "best_so_far": self.curves, "best_score_stats": self.stats,
        }


@dataclass
class ComparisonResult:
    objective: ObjectiveConfig
    budget: int
    iou_threshold: float
    methods: list[MethodSummary]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return json_safe({
            "schema_version": SCHEMA_VERSION,
            "objective": self.objective.metric,
            "budget": self.budget,
            "iou_threshold": self.iou_threshold,
            "methods": [m.to_dict() for m in self.methods],
            "metadata": self.metadata,
        })


def compare_methods(
    detector: Any,
    dataset: YoloDataset,
    *,
    methods: Sequence[str] = ("random", "tpe"),
    seeds: Sequence[int] = (0, 1, 2),
    budget: int = 20,
    space: SearchSpace | None = None,
    objective: ObjectiveConfig | None = None,
    iou_threshold: float = DEFAULT_IOU,
) -> ComparisonResult:
    check_budget(budget)
    methods = list(dict.fromkeys(methods))
    unknown = set(methods) - set(RUNNERS)
    if not methods or unknown:
        raise ValueError(f"methods must be a non-empty subset of {sorted(RUNNERS)}, got {methods}")
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    if not seeds:
        raise ValueError("at least one seed is required")
    space = space or SearchSpace.default()
    objective = objective or ObjectiveConfig()
    baseline = evaluate_baseline(detector, dataset, iou_threshold=iou_threshold)

    summaries = []
    for method in methods:
        runs = [
            RUNNERS[method](detector, dataset, budget=budget, seed=s, space=space,
                            objective=objective, iou_threshold=iou_threshold,
                            baseline=baseline)
            for s in seeds
        ]
        bests = [r.best.score if r.best else None for r in runs]
        summaries.append(MethodSummary(
            method=method, seeds=seeds, best_scores=bests,
            valid_counts=[r.counts()["valid"] for r in runs],
            curves=[best_so_far(r) for r in runs],
            stats=_stats([b for b in bests if b is not None]),
        ))
    return ComparisonResult(
        objective=objective, budget=budget, iou_threshold=float(iou_threshold),
        methods=summaries,
        metadata={
            "split": baseline.split, "num_images": len(baseline.per_image),
            "space": space.describe(),
            "note": "same baseline, space, objective, and budget for every run",
        },
    )


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def render_comparison_markdown(result: ComparisonResult,
                               title: str = "FailureForge Search Comparison") -> str:
    metric = result.objective.metric
    lines = [
        f"# {title}", "",
        f"- Objective: `{metric}` degradation (baseline - perturbed; larger is worse)",
        f"- Budget per run: {result.budget}; seeds: {result.methods[0].seeds}",
        f"- Split: `{result.metadata['split']}` ({result.metadata['num_images']} images)", "",
        "## Best score per method", "",
        "| method | mean | std | min | max | runs with a valid trial |",
        "|---|---|---|---|---|---|",
    ]
    for m in result.methods:
        s = m.stats
        lines.append(f"| {m.method} | {_fmt(s['mean'])} | {_fmt(s['std'])} | {_fmt(s['min'])} "
                     f"| {_fmt(s['max'])} | {s['n']}/{len(m.seeds)} |")
    lines += ["", "## Per-seed best scores", "", "| method | " +
              " | ".join(f"seed {s}" for s in result.methods[0].seeds) + " |",
              "|---|" + "---|" * len(result.methods[0].seeds)]
    for m in result.methods:
        lines.append(f"| {m.method} | " + " | ".join(_fmt(b) for b in m.best_scores) + " |")
    lines += [
        "", "## Reading this", "",
        "- Descriptive only: no significance test is applied; with few seeds, compare "
        "the spread (std, min, max) before preferring a method.",
        "- Scores are on the compared split and are optimistic for selected configurations; "
        "use `search` confirmation for held-out evidence.",
        "- A seed with no valid trial is excluded from the statistics and shown as n/a.",
        "",
    ]
    return "\n".join(lines)


def write_comparison(result: ComparisonResult, output_dir: str | Path,
                     basename: str = "compare") -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{basename}.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    md_path = out / f"{basename}.md"
    md_path.write_text(render_comparison_markdown(result), encoding="utf-8")
    return {"json": json_path, "md": md_path}
