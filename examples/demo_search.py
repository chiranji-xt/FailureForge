"""Weight-free end-to-end demo: search -> confirmation -> reports -> gallery.

Uses ``ColorBlobDetector`` (a rule-based stand-in, NOT a YOLO model), so the
numbers only demonstrate that the pipeline runs. With real weights use the CLI.
"""
import argparse
from pathlib import Path

from failureforge.cli import main

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="demo_out")
args = ap.parse_args()

from failureforge.demo import ColorBlobDetector, make_demo_dataset  # noqa: E402

out = Path(args.out)
data = make_demo_dataset(out / "data", n=10)
raise SystemExit(main(
    ["search", "--data", str(data), "--budget", "12", "--seed", "0",
     "--transforms", "gaussian_blur,darken,gaussian_noise,rotation",
     "--gallery", "--out-dir", str(out / "results")],
    detector_factory=lambda _settings: ColorBlobDetector(),
))
