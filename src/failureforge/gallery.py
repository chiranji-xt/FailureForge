"""Portable, dependency-free HTML failure gallery.

The gallery is a single HTML file plus optional hashed image copies. It consumes
the schema-versioned run records from :mod:`failureforge.failures`; it does not
re-evaluate models, change TP/FP/FN, or turn Quick Mode instability into
accuracy claims.

Image artifacts are optional. Pass ``clean_images``/``perturbed_images`` maps
whose keys match record ``image`` values and whose values point to the **exact**
image those box coordinates refer to (clean for baseline/Quick-clean, transformed
for benchmark trials/Quick-perturbed). Only referenced files are copied, under
content-hashed relative names; source paths are never written to the HTML.
Without an artifact the gallery renders a clearly labelled box diagram instead
of claiming an image overlay.

The gallery has no timestamps and sorts entries deterministically. Existing HTML
is not overwritten unless ``overwrite=True``.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from .dataset import IMAGE_EXTS
from .failures import (
    FAILURE_CATEGORIES,
    QUICK_INSTABILITY_KINDS,
    BenchmarkRun,
    QuickRun,
    VerifiedFailure,
    QuickInstability,
    TrialRecord,
)
from .reporting import ranked_trials

GALLERY_SCHEMA_VERSION = "failureforge.failure-gallery.v1"

_LABELS = {
    "false_negative": "False negative",
    "false_positive": "False positive",
    "class_confusion": "Class confusion",
    "localization_error": "Localization error",
    "duplicate_detection": "Duplicate detection",
    "unmatched": "Unmatched / ambiguous",
    "disappeared": "Disappeared",
    "appeared": "Appeared",
    "class_changed": "Class changed",
    "confidence_changed": "Confidence changed",
    "box_drift": "Box drift",
}


@dataclass(frozen=True)
class GalleryImage:
    """A relative gallery image URL plus the dimensions used for box overlays."""

    url: str
    width: int
    height: int


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}g}" if abs(number) >= 1e5 else f"{number:.{digits}f}"


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _safe_transform(transformation: dict | None) -> dict | None:
    if not transformation:
        return None
    return {
        "name": transformation.get("name"),
        "family": transformation.get("family"),
        "params": dict(transformation.get("params") or {}),
        "seed": transformation.get("seed", 0),
    }


def _box(box: Iterable[float] | None, *, role: str, class_id: int | None,
         class_name: str | None, score: float | None = None,
         index: int | None = None) -> dict | None:
    if box is None:
        return None
    values = [float(value) for value in box]
    if len(values) != 4:
        return None
    return {
        "box": values, "role": role, "class_id": class_id,
        "class_name": class_name, "score": score, "index": index,
    }


def _class_text(class_id: int | None, class_name: str | None) -> str:
    if class_id is None and class_name is None:
        return "n/a"
    if class_name:
        return f"{class_name} (class {class_id})" if class_id is not None else class_name
    return f"class {class_id}"


def _benchmark_entry(
    failure: VerifiedFailure,
    *,
    baseline: bool,
    trial: TrialRecord | None,
    trial_index: int | None,
    rank: int | None,
) -> dict:
    transformation = None if baseline else _safe_transform(
        trial.transformation if trial is not None else failure.transformation
    )
    if baseline:
        context = "baseline"
        context_label = "Baseline (unperturbed)"
    else:
        context = "perturbed"
        rank_text = f"rank {rank}" if rank is not None else "unranked"
        context_label = f"Trial {int(trial_index) + 1} · {rank_text}"
    gt = None
    if failure.gt_box is not None or failure.gt_class_id is not None:
        gt = {
            "class_id": failure.gt_class_id,
            "class_name": failure.gt_class_name,
            "box": list(failure.gt_box) if failure.gt_box is not None else None,
            "index": failure.gt_index,
        }
    pred = None
    if (failure.pred_box is not None or failure.pred_class_id is not None):
        pred = {
            "class_id": failure.pred_class_id,
            "class_name": failure.pred_class_name,
            "box": list(failure.pred_box) if failure.pred_box is not None else None,
            "index": failure.pred_index,
            "score": failure.score,
        }
    boxes = [
        item for item in (
            _box(failure.gt_box, role="gt", class_id=failure.gt_class_id,
                 class_name=failure.gt_class_name, index=failure.gt_index),
            _box(failure.pred_box, role="pred", class_id=failure.pred_class_id,
                 class_name=failure.pred_class_name, score=failure.score,
                 index=failure.pred_index),
        ) if item is not None
    ]
    details = [
        ("Reason", failure.reason),
        ("Metric role", failure.counts_as),
        ("IoU", f"{_fmt(failure.iou)} (threshold {_fmt(failure.iou_threshold)})"),
        ("Score", _fmt(failure.score)),
        ("Localization threshold", _fmt(failure.localization_iou_threshold)),
    ]
    return {
        "id": (
            f"baseline-{trial_index if trial_index is not None else 0}-"
            if baseline else f"trial-{trial_index}-"
        ) + f"{failure.category}-{failure.image}-{failure.gt_index}-{failure.pred_index}",
        "mode": "benchmark",
        "verified": True,
        "category": failure.category,
        "label": _LABELS.get(failure.category, failure.category),
        "image": failure.image,
        "context": context,
        "context_label": context_label,
        "transformation": transformation,
        "rank": rank,
        "reason": failure.reason,
        "counts_as": failure.counts_as,
        "details": details,
        "gt": gt,
        "pred": pred,
        "boxes": boxes,
        "panels": [{
            "label": context_label,
            "artifact": "clean" if baseline else "perturbed",
            "boxes": boxes,
        }],
        "search": " ".join(str(part) for part in (
            failure.image, failure.category, failure.reason, failure.counts_as,
            failure.gt_class_name, failure.pred_class_name,
            (transformation or {}).get("name"),
        ) if part),
    }


def _quick_entry(record: QuickInstability) -> dict:
    transformation = _safe_transform(record.transformation)
    clean_boxes = [
        item for item in (
            _box(record.clean_box, role="clean", class_id=record.clean_class_id,
                 class_name=record.clean_class_name, score=record.clean_score,
                 index=record.clean_pred_index),
        ) if item is not None
    ]
    perturbed_boxes = [
        item for item in (
            _box(record.perturbed_box, role="perturbed",
                 class_id=record.perturbed_class_id,
                 class_name=record.perturbed_class_name,
                 score=record.perturbed_score,
                 index=record.perturbed_pred_index),
        ) if item is not None
    ]
    details = [
        ("Kind", _LABELS.get(record.kind, record.kind)),
        ("IoU", _fmt(record.iou)),
        ("Confidence delta", _fmt(record.confidence_delta)),
        ("Box drift", _fmt(record.drift)),
        ("Confidence epsilon", _fmt(record.confidence_epsilon)),
    ]
    return {
        "id": f"quick-{record.kind}-{record.image}-{record.clean_pred_index}-{record.perturbed_pred_index}",
        "mode": "quick",
        "verified": False,
        "category": record.kind,
        "label": _LABELS.get(record.kind, record.kind),
        "image": record.image,
        "context": "quick",
        "context_label": "Quick Mode · prediction instability",
        "transformation": transformation,
        "rank": None,
        "reason": "Clean/perturbed prediction change; no ground truth was used.",
        "counts_as": None,
        "details": details,
        "gt": None,
        "pred": None,
        "boxes": clean_boxes + perturbed_boxes,
        "panels": [
            {"label": "Clean predictions", "artifact": "clean",
             "boxes": clean_boxes},
            {"label": "Perturbed predictions", "artifact": "perturbed",
             "boxes": perturbed_boxes},
        ],
        "search": " ".join(str(part) for part in (
            record.image, record.kind, record.clean_class_name,
            record.perturbed_class_name, (transformation or {}).get("name"),
        ) if part),
    }


def _build_entries(
    run: BenchmarkRun | QuickRun,
    *,
    rank_metric: str,
    max_entries: int | None,
) -> tuple[list[dict], int]:
    entries: list[dict] = []
    if isinstance(run, BenchmarkRun):
        for failure in run.baseline.failures:
            entries.append(_benchmark_entry(
                failure, baseline=True, trial=None, trial_index=None, rank=None,
            ))
        ranked = ranked_trials(run, rank_metric)
        for entry in ranked:
            index = int(entry["index"])
            trial = run.trials[index]
            for failure in trial.failures:
                entries.append(_benchmark_entry(
                    failure, baseline=False, trial=trial, trial_index=index,
                    rank=entry["rank"],
                ))
    elif isinstance(run, QuickRun):
        entries = [_quick_entry(record) for record in run.analysis.records]
    else:
        raise TypeError(f"unsupported run type: {type(run).__name__}")
    total = len(entries)
    if max_entries is not None:
        if int(max_entries) <= 0:
            raise ValueError("max_entries must be positive or None")
        entries = entries[:int(max_entries)]
    return entries, total


def _entry_boxes(entry: dict) -> list[dict]:
    return [box for box in entry.get("boxes") or []]


def _artifact_names(entries: Iterable[dict]) -> set[str]:
    return {entry["image"] for entry in entries}


def _box_bounds(boxes: list[dict]) -> tuple[float, float]:
    max_x = max((box["box"][2] for box in boxes), default=0.0)
    max_y = max((box["box"][3] for box in boxes), default=0.0)
    return max(1.0, max_x), max(1.0, max_y)


def _panel_html(
    *, label: str, boxes: list[dict], image: GalleryImage | None,
    context: str,
) -> str:
    parts = [f'<div class="panel"><h4>{_h(label)}</h4>']
    if not boxes and image is None:
        parts.append(
            '<p class="muted">No image artifact or box coordinates available '
            'for this entry.</p></div>'
        )
        return "".join(parts)

    if image is not None:
        width, height = image.width, image.height
    else:
        width_f, height_f = _box_bounds(boxes)
        width, height = int(math.ceil(width_f)), int(math.ceil(height_f))
    parts.append('<div class="canvas ' + ("has-image" if image else "diagram") + '">')
    if image is not None:
        parts.append(
            f'<img src="{_h(image.url)}" alt="{_h(context)} image artifact" '
            'loading="lazy">'
        )
    if boxes:
        parts.append(
            f'<svg class="overlay" viewBox="0 0 {width} {height}" '
            'preserveAspectRatio="none" role="img" '
            f'aria-label="{_h(context)} bounding boxes">'
        )
        for box in boxes:
            x1, y1, x2, y2 = box["box"]
            left, top = min(x1, x2), min(y1, y2)
            box_width, box_height = abs(x2 - x1), abs(y2 - y1)
            title = (
                f"{box['role']} {_class_text(box['class_id'], box['class_name'])}"
                f" box {box['index'] if box['index'] is not None else 'n/a'}"
            )
            parts.append(
                f'<rect class="bbox bbox-{_h(box["role"])}" '
                f'data-role="{_h(box["role"])}" '
                f'x="{_fmt(left)}" y="{_fmt(top)}" '
                f'width="{_fmt(box_width)}" height="{_fmt(box_height)}">'
                f'<title>{_h(title)}</title></rect>'
            )
        parts.append("</svg>")
        if image is not None:
            out_of_bounds = any(
                box["box"][0] < 0 or box["box"][1] < 0
                or box["box"][2] > image.width or box["box"][3] > image.height
                for box in boxes
            )
            if out_of_bounds:
                parts.append(
                    '<p class="note warn">Box coordinates exceed the supplied '
                    'artifact dimensions; the image may not match this record.</p>'
                )
    parts.append("</div>")
    note = (
        "Boxes are drawn over the supplied artifact."
        if image is not None else
        "Box diagram only: no image artifact was supplied, so boxes are not "
        "presented as a photo overlay."
    )
    parts.append(f'<p class="note">{_h(note)}</p>')
    if boxes:
        legend = ", ".join(
            dict.fromkeys(
                f"{box['role']}: {_class_text(box['class_id'], box['class_name'])}"
                + (f" ({_fmt(box['score'])} score)" if box["score"] is not None else "")
                for box in boxes
            )
        )
        parts.append(f'<p class="legend">{_h(legend)}</p>')
    parts.append("</div>")
    return "".join(parts)


def _entry_html(entry: dict, clean_images: Mapping[str, GalleryImage],
                perturbed_images: Mapping[str, GalleryImage]) -> str:
    badges = [
        f'<span class="badge category">{_h(entry["label"])}</span>',
        f'<span class="badge context">{_h(entry["context_label"])}</span>',
    ]
    if entry["mode"] == "quick":
        badges.append('<span class="badge quick">Not a verified failure</span>')
    if not entry["verified"]:
        badges.append('<span class="badge unverified">Prediction instability</span>')

    panels = "".join(
        _panel_html(
            label=panel["label"], boxes=panel["boxes"],
            image=(
                clean_images.get(entry["image"])
                if panel["artifact"] == "clean"
                else perturbed_images.get(entry["image"])
            ),
            context=panel["artifact"],
        )
        for panel in entry["panels"]
    )

    transformation = entry.get("transformation")
    transform_line = ""
    if transformation:
        transform_line = (
            '<p class="transform">'
            f'<strong>{_h(transformation.get("name", "n/a"))}</strong>'
            f' · family {_h(transformation.get("family") or "unknown")}'
            f' · seed {_h(transformation.get("seed", 0))}'
            f' · params <code>{_h(json.dumps(transformation.get("params") or {}, sort_keys=True))}</code>'
            '</p>'
        )
    detail_items = "".join(
        f"<dt>{_h(key)}</dt><dd>{_h(value)}</dd>"
        for key, value in entry.get("details") or []
    )
    class_line = ""
    if entry["mode"] == "benchmark":
        gt = entry.get("gt")
        pred = entry.get("pred")
        bits = []
        if gt is not None:
            bits.append(
                "GT " + _class_text(gt.get("class_id"), gt.get("class_name"))
                + f" #{gt.get('index') if gt.get('index') is not None else 'n/a'}"
            )
        if pred is not None:
            bits.append(
                "Prediction " + _class_text(pred.get("class_id"), pred.get("class_name"))
                + f" #{pred.get('index') if pred.get('index') is not None else 'n/a'}"
            )
        if bits:
            class_line = f'<p class="classes">{_h(" · ".join(bits))}</p>'

    return (
        f'<article class="entry" id="{_h(entry["id"])}" '
        f'data-category="{_h(entry["category"])}" '
        f'data-mode="{_h(entry["mode"])}" '
        f'data-context="{_h(entry["context"])}" '
        f'data-verified="{str(entry["verified"]).lower()}" '
        f'data-search="{_h(entry["search"].lower())}">'
        f'<header><div class="badges">{"".join(badges)}</div>'
        f'<span class="image-name">{_h(entry["image"])}</span></header>'
        f'<p class="reason">{_h(entry["reason"])}</p>'
        f'{class_line}{transform_line}'
        f'<dl class="details">{detail_items}</dl>'
        f'<div class="panels">{panels}</div>'
        '</article>'
    )


def _summary_rows(run: BenchmarkRun | QuickRun) -> list[tuple[str, int]]:
    if isinstance(run, BenchmarkRun):
        rows = []
        for category in FAILURE_CATEGORIES:
            total = int(run.baseline.failure_counts.get(category, 0))
            total += sum(int(t.failure_counts.get(category, 0))
                         for t in run.trials)
            rows.append((_LABELS[category], total))
        return rows
    return [
        (_LABELS[kind], int(run.analysis.counts.get(kind, 0)))
        for kind in QUICK_INSTABILITY_KINDS
    ]


def _render(
    run: BenchmarkRun | QuickRun,
    entries: list[dict],
    total_entries: int,
    *,
    title: str,
    clean_images: Mapping[str, GalleryImage],
    perturbed_images: Mapping[str, GalleryImage],
    extra_warnings: Iterable[str],
) -> str:
    mode = "benchmark" if isinstance(run, BenchmarkRun) else "quick"
    meta = run.metadata or {}
    dataset = meta.get("dataset") or {}
    warnings = list(run.warnings) + list(extra_warnings)
    summary = "".join(
        f"<tr><td>{_h(label)}</td><td>{count}</td></tr>"
        for label, count in _summary_rows(run)
    )
    warning_html = ""
    if warnings:
        warning_html = (
            '<section class="warnings"><h2>Warnings</h2><ul>'
            + "".join(f"<li>{_h(message)}</li>" for message in warnings)
            + "</ul></section>"
        )
    categories = sorted({entry["category"] for entry in entries})
    category_options = "".join(
        f'<option value="{_h(category)}">{_h(_LABELS.get(category, category))}</option>'
        for category in categories
    )
    contexts = "".join(
        f'<option value="{_h(context)}">{_h(label)}</option>'
        for context, label in (
            ("baseline", "Baseline (unperturbed)"),
            ("perturbed", "Perturbed trials"),
            ("quick", "Quick Mode instability"),
        )
        if any(entry["context"] == context for entry in entries)
    )
    scope = (
        "Benchmark Mode entries are verified against ground truth. Quick Mode "
        "entries are prediction instability only and are not verified false "
        "positives/negatives."
    )
    manifest = {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "mode": mode,
        "total_entries": total_entries,
        "shown_entries": len(entries),
        "verified_failures": mode == "benchmark",
        "warnings": [str(w) for w in warnings],
    }
    body = "".join(
        _entry_html(entry, clean_images, perturbed_images) for entry in entries
    )
    if not body:
        body = '<p class="empty">No entries to display.</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="failureforge-gallery-schema" content="{GALLERY_SCHEMA_VERSION}">
<title>{_h(title)}</title>
<style>
:root {{ color-scheme: light dark; --fg:#1f2933; --muted:#5f6b7a; --bg:#f7f9fc;
  --card:#ffffff; --line:#d9e2ec; --accent:#2563eb; --gt:#16a34a;
  --pred:#dc2626; --clean:#0284c7; --perturbed:#d97706; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e5e7eb; --muted:#9ca3af;
  --bg:#0f172a; --card:#111827; --line:#334155; --accent:#60a5fa; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:2rem; font:15px/1.5 system-ui,Segoe UI,sans-serif;
  color:var(--fg); background:var(--bg); }}
h1 {{ margin:0 0 .25rem; }} h2 {{ margin:1.5rem 0 .5rem; font-size:1.1rem; }}
.subtitle,.muted,.note {{ color:var(--muted); }}
.scope,.warnings {{ background:var(--card); border:1px solid var(--line);
  border-radius:8px; padding:.75rem 1rem; margin:1rem 0; }}
.warnings {{ border-left:4px solid var(--perturbed); }}
table {{ border-collapse:collapse; background:var(--card); }}
td,th {{ border:1px solid var(--line); padding:.25rem .6rem; text-align:left; }}
.controls {{ display:flex; flex-wrap:wrap; gap:.6rem; margin:1.25rem 0; }}
.controls input,.controls select {{ padding:.45rem; border:1px solid var(--line);
  border-radius:6px; background:var(--card); color:var(--fg); }}
.entries {{ display:grid; gap:1rem; }}
.entry {{ background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:1rem; }}
.entry[hidden] {{ display:none; }}
.entry header {{ display:flex; justify-content:space-between; gap:.75rem;
  align-items:flex-start; flex-wrap:wrap; }}
.badges {{ display:flex; gap:.35rem; flex-wrap:wrap; }}
.badge {{ border-radius:999px; padding:.12rem .55rem; font-size:.78rem;
  border:1px solid var(--line); }}
.badge.category {{ background:var(--accent); color:#fff; border-color:transparent; }}
.badge.quick,.badge.unverified {{ border-color:var(--perturbed); color:var(--perturbed); }}
.image-name {{ font-family:ui-monospace,Consolas,monospace; color:var(--muted); }}
.reason {{ margin:.6rem 0 .35rem; }}
.classes,.transform {{ margin:.25rem 0; color:var(--muted); }}
details,dl.details {{ margin:.5rem 0; }}
dl.details {{ display:grid; grid-template-columns:max-content 1fr; gap:.1rem .8rem; }}
dl.details dt {{ color:var(--muted); }} dl.details dd {{ margin:0; }}
.panels {{ display:flex; gap:1rem; flex-wrap:wrap; margin-top:.75rem; }}
.panel {{ flex:1 1 260px; min-width:220px; }}
.panel h4 {{ margin:.25rem 0; }}
.canvas {{ position:relative; border:1px solid var(--line); border-radius:6px;
  overflow:hidden; background:repeating-conic-gradient(
    color-mix(in srgb, var(--line) 35%, transparent) 0 25%,
    transparent 0 50%) 0 0/24px 24px; }}
.canvas img {{ display:block; width:100%; height:auto; }}
.canvas .overlay {{ position:absolute; inset:0; width:100%; height:100%; }}
.canvas.diagram {{ min-height:120px; }}
.bbox {{ fill:none; stroke-width:3; vector-effect:non-scaling-stroke; }}
.bbox-gt {{ stroke:var(--gt); }} .bbox-pred {{ stroke:var(--pred); }}
.bbox-clean {{ stroke:var(--clean); }} .bbox-perturbed {{ stroke:var(--perturbed); }}
.note {{ font-size:.85rem; margin:.35rem 0; }}
.note.warn {{ color:var(--perturbed); }}
.legend {{ font-size:.82rem; color:var(--muted); margin:.2rem 0 0; }}
.empty {{ color:var(--muted); }}
</style>
</head>
<body>
<header>
<h1>{_h(title)}</h1>
<p class="subtitle">FailureForge gallery · schema
<code>{GALLERY_SCHEMA_VERSION}</code> · mode <strong>{_h(mode)}</strong></p>
</header>
<section class="scope">
<p>{_h(scope)}</p>
<p>Dataset split: <strong>{_h(dataset.get("split", "n/a"))}</strong> ·
images: <strong>{_h(dataset.get("num_images", "n/a"))}</strong> ·
classes: {_h(", ".join(str(c) for c in (dataset.get("class_names") or [])) or "n/a")}</p>
<p>Showing <strong>{len(entries)}</strong> of <strong>{total_entries}</strong>
entries. Image artifacts are optional, copied only when supplied, and referenced
by relative content-hashed paths.</p>
<table><thead><tr><th>Category / instability</th><th>Count</th></tr></thead>
<tbody>{summary}</tbody></table>
</section>
{warning_html}
<div class="controls" role="search">
<label>Category <select id="category-filter"><option value="">All</option>
{category_options}</select></label>
<label>Context <select id="context-filter"><option value="">All</option>
{contexts}</select></label>
<label>Search <input id="search-filter" type="search"
placeholder="image, class, transformation…"></label>
</div>
<main class="entries">{body}</main>
<footer><p class="muted">Entries are observations for the evaluated model,
dataset, seed, and thresholds; they are not a universal claim about
perturbation damage. This file requires no network access.</p></footer>
<script id="gallery-data" type="application/json">{_json_for_script(manifest)}</script>
<script>
(function () {{
  const category = document.getElementById('category-filter');
  const context = document.getElementById('context-filter');
  const search = document.getElementById('search-filter');
  const entries = Array.from(document.querySelectorAll('.entry'));
  function apply() {{
    const cat = category ? category.value : '';
    const ctx = context ? context.value : '';
    const needle = search ? search.value.trim().toLowerCase() : '';
    entries.forEach(function (entry) {{
      const visible = (!cat || entry.dataset.category === cat)
        && (!ctx || entry.dataset.context === ctx)
        && (!needle || entry.dataset.search.indexOf(needle) !== -1);
      entry.hidden = !visible;
    }});
  }}
  [category, context, search].forEach(function (el) {{
    if (el) {{ el.addEventListener('input', apply); el.addEventListener('change', apply); }}
  }});
}})();
</script>
</body>
</html>
"""


def render_gallery(
    run: BenchmarkRun | QuickRun,
    *,
    title: str = "FailureForge Failure Gallery",
    rank_metric: str = "f1",
    max_entries: int | None = 500,
    clean_images: Mapping[str, GalleryImage] | None = None,
    perturbed_images: Mapping[str, GalleryImage] | None = None,
) -> str:
    """Render a deterministic HTML gallery from a run record.

    ``clean_images``/``perturbed_images`` must already use relative URLs (see
    :func:`write_gallery`, which copies supplied artifacts).
    """
    entries, total = _build_entries(
        run, rank_metric=rank_metric, max_entries=max_entries
    )
    return _render(
        run, entries, total, title=title,
        clean_images=dict(clean_images or {}),
        perturbed_images=dict(perturbed_images or {}),
        extra_warnings=(),
    )


def _copy_artifacts(
    sources: Mapping[str, str | Path] | None,
    needed: set[str],
    images_dir: Path,
) -> tuple[dict[str, GalleryImage], list[str]]:
    images: dict[str, GalleryImage] = {}
    warnings: list[str] = []
    if sources is None:
        return images, warnings
    copied: dict[str, GalleryImage] = {}
    for name, source in sources.items():
        if name not in needed:
            continue
        source_path = Path(source)
        if not source_path.is_file():
            warnings.append(f"Image artifact for {name!r} does not exist; omitted.")
            continue
        suffix = source_path.suffix.lower()
        if suffix not in IMAGE_EXTS:
            warnings.append(
                f"Image artifact for {name!r} has unsupported extension {suffix!r}; omitted."
            )
            continue
        try:
            with Image.open(source_path) as image:
                width, height = image.size
        except Exception as exc:
            warnings.append(
                f"Image artifact for {name!r} could not be read ({exc}); omitted."
            )
            continue
        digest = hashlib.sha256()
        try:
            with source_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            warnings.append(
                f"Image artifact for {name!r} could not be hashed ({exc}); omitted."
            )
            continue
        filename = digest.hexdigest()[:16] + suffix
        destination = images_dir / filename
        if not destination.exists():
            images_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(source_path, destination)
            except OSError as exc:
                warnings.append(
                    f"Image artifact for {name!r} could not be copied ({exc}); omitted."
                )
                continue
        copied[name] = GalleryImage(
            url=f"images/{filename}", width=int(width), height=int(height)
        )
    images.update(copied)
    return images, warnings


def write_gallery(
    run: BenchmarkRun | QuickRun,
    output_dir: str | Path,
    *,
    basename: str = "failure_gallery",
    title: str = "FailureForge Failure Gallery",
    rank_metric: str = "f1",
    max_entries: int | None = 500,
    clean_images: Mapping[str, str | Path] | None = None,
    perturbed_images: Mapping[str, str | Path] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write an HTML gallery (and only referenced image copies); returns its path."""
    if Path(basename).name != basename:
        raise ValueError("basename must not contain path separators")
    name = basename if basename.lower().endswith(".html") else f"{basename}.html"
    directory = Path(output_dir)
    path = directory / name
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"gallery already exists: {path}; pass overwrite=True to replace it"
        )
    directory.mkdir(parents=True, exist_ok=True)

    entries, total = _build_entries(
        run, rank_metric=rank_metric, max_entries=max_entries
    )
    needed = _artifact_names(entries)
    clean_map, clean_warnings = _copy_artifacts(
        clean_images, needed, directory / "images"
    )
    pert_map, pert_warnings = _copy_artifacts(
        perturbed_images, needed, directory / "images"
    )
    document = _render(
        run, entries, total, title=title,
        clean_images=clean_map, perturbed_images=pert_map,
        extra_warnings=clean_warnings + pert_warnings,
    )
    path.write_text(document, encoding="utf-8")
    return path
