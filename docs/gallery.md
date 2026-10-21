# Failure gallery (Day 7)

`failureforge.gallery` turns the Day-6 `BenchmarkRun` / `QuickRun` records into
one self-contained HTML file with inline CSS and a small filtering script. It
needs no network, web server, plotting library, or new dependency.

```python
from failureforge.gallery import write_gallery

gallery = write_gallery(
    run,                                  # BenchmarkRun or QuickRun
    "results/gallery",                    # caller-chosen output directory
    clean_images={"img0.png": "data/images/val/img0.png"},          # optional
    perturbed_images={"img0.png": "results/perturbed/img0.png"},   # optional
    basename="failure_gallery",           # writes failure_gallery.html
    max_entries=500,
)
# Open gallery directly in a browser (file://); no server required.
```

`write_gallery` returns the HTML `Path`. It refuses to replace an existing HTML
file unless `overwrite=True`. JSON and Markdown report writers are unchanged.

## What each entry shows

- Image/sample name.
- Baseline vs perturbed trial context; Quick Mode entries are labelled
  **prediction instability** and **not a verified failure**.
- Verified category (`false_negative`, `false_positive`, `class_confusion`,
  `localization_error`, `duplicate_detection`, `unmatched`) or Quick
  instability kind (`disappeared`, `appeared`, `class_changed`,
  `confidence_changed`, `box_drift`).
- Transformation name, family, parameters, seed, and trial rank when available.
- Reason, metric role (`tp`/`fp`/`fn` or none for Quick), score, IoU/thresholds,
  confidence delta, and box drift as applicable.
- GT/prediction classes and indices.
- GT/prediction boxes, or clean/perturbed boxes for Quick records.
- Optional source image with boxes drawn as an SVG overlay.

Filters for category, context (baseline/perturbed/quick), and free-text search
work locally in the browser. A JSON manifest with the gallery schema version and
counts is embedded in the page.

## Image artifacts and overlays

Image artifacts are **optional and caller-supplied**:

- `clean_images` maps record image names to the exact clean image.
- `perturbed_images` maps record image names to the exact transformed image.

Keys must match the `image` field in the records. Benchmark trials use
`perturbed_images`; baseline entries and Quick clean panels use `clean_images`.
Quick perturbed panels use `perturbed_images`. Supplying the wrong image would
misplace boxes, so the caller is responsible for matching the coordinate space.

Only images referenced by included entries are copied. They go into
`<output>/images/` under content-hashed names, and the HTML uses relative
`images/...` URLs. Original absolute paths are not written to the gallery.
Artifacts with missing files, unsupported extensions, or unreadable content are
omitted with a warning.

When an image is present, the boxes are rendered as an SVG overlay using the
image's real pixel dimensions. When it is absent, the gallery renders a
**box diagram only** and says so explicitly; it never pretends to overlay boxes
on a photo that was not supplied. Missing box details and missing images are
handled without broken links or crashes. If supplied box coordinates exceed the
artifact dimensions, the entry warns that the image may not match the record.

## Safety and determinism

- All user-controlled text and attributes (filenames, class names, reasons,
  warnings, titles, parameters) are HTML-escaped.
- Embedded JSON escapes `<`, `>`, and `&`, so record text cannot terminate the
  script element.
- No timestamps are generated; entry order is deterministic (baseline, then
  ranked trials; Quick records use their analysis order).
- Output uses relative artifact paths and works from a local `file://` page.
- The gallery does not read arbitrary paths from records: image files are read
  only from explicit caller mappings.
- JSON/Markdown schemas remain unchanged; the gallery has its own
  `failureforge.failure-gallery.v1` schema marker.

## Verified failures vs Quick instability

Benchmark entries come from class-aware GT matching and explain TP/FP/FN without
changing them. A matched localization error is an aspect of a TP. Quick entries
come from location-matching clean vs perturbed predictions without labels; they
are not false positives/negatives and carry no precision/recall/F1/mAP claims.

## Limitations

- Only records already present in a run are shown; this is not a full image
  browser or a gallery of every prediction.
- `max_entries` defaults to 500; the page reports how many entries were shown
  versus available.
- Image artifacts are copies, not live links. Large failure sets can duplicate
  many source images; only referenced images are copied.
- Supplied images are trusted to match the record's coordinate space; dimension
  mismatches are only heuristically detectable.
- No CSV export or gallery CLI command yet (CLI remains Day 13).
