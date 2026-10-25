# Robustness search (Days 9–12)

Search looks for perturbation configurations that degrade a detector most, using
the Day-8 objective (`docs/objective.md`): **score = baseline − perturbed** for one
higher-is-better metric (`f1` default; `precision`, `recall`, `map50`). Larger =
worse; negative = improvement. Failure counts and Quick Mode instability are not
part of the score.

## Pipeline

1. `YoloDataset.partition(holdout_fraction, seed)` → disjoint **discovery** and
   **holdout** datasets (deterministic; both non-empty; ≥ 2 images).
2. Search on the discovery split (`run_random_search` or `run_tpe_search`). One
   baseline is computed once and shared by all trials.
3. `confirm_trials` re-scores the top-k *distinct* valid configurations on the
   holdout over `n_seeds` fresh seeds.
4. `write_search_report` writes `search.json`, `search.md`, `search_trials.csv`.

The holdout is never touched during search, so confirmation scores are not
inflated by selection. With `holdout_fraction=0` confirmation is skipped and the
CLI says so.

## Search space

`SearchSpace.default(include=None, exclude=("combined",), overrides=None)` is built
from the perturbation registry's `ParamSpec`s (37 transforms by default: `combined` and the legacy `blur` alias are excluded). Numeric
parameters are sampled uniformly in range (ints uniformly; `motion_blur.kernel_size`
on odd values because the engine rejects even kernels). Non-searchable parameters
(choices, `search=False`) keep their registered defaults. `overrides` may narrow
a range but never widen it beyond registry bounds. `combined` is excluded because
it needs an ordered `steps` list.

## Determinism

- Random Search: `random.Random(seed)` drives configuration sampling.
- TPE: `TPESampler(seed=seed)`, sequential ask/tell (no parallelism).
- Per-trial perturbation seed = `derive_trial_seed(seed, index, name, params)`
  (SHA-256 based), so any trial can be replayed on its own.
- `SearchResult.warnings` flags uninformative searches (baseline metric 0 or very low, no
  degrading trial, no valid trial); they appear in JSON, Markdown and on stderr.
- Wall-clock `duration_s` is recorded but is the only non-deterministic field.

## Trial states

| status | meaning | ranked? |
|---|---|---|
| `ok` | finite, compatible objective | yes |
| `invalid` | objective missing/non-finite/incompatible (`score=None`, warnings kept) | no |
| `error` | evaluation raised (`error` holds the message) | no |

Invalid/errored trials are never scored as 0 or −∞, never ranked, and Optuna is
told the trial FAILED (no value). Ranking is by descending score, ties by lower
trial index; `ranked(unique=True)` keeps one trial per near-identical config
(floats rounded to 4 places).

## Optuna (optional)

`pip install "failureforge[optuna]"`. Without it, `run_tpe_search` raises
`OptionalDependencyError` with the install command; everything else works. TPE
tests are skipped when Optuna is absent.

## Confirmation record

Per candidate: per-seed scores (`None` for invalid seeds), `mean`, `std` (sample,
needs ≥ 2 valid seeds), `min`, `max`, `n_valid`/`n_invalid`, and
`degrades_in_all_seeds` (every seed valid and > 0). No pass/fail threshold is
invented. `ConfirmationResult.results` holds in-memory `PerturbedResult`s used by
the CLI to build a `BenchmarkRun` (`confirmation.json/md`) and gallery.

## Comparing methods

`failureforge compare --data data.yaml --weights best.pt --methods random,tpe --runs 5 --budget 30`
(or `compare_methods(...)`) runs each method for `runs` consecutive seeds with one
shared baseline, space, objective and budget, then writes `compare.json|md`: best
score mean/std/min/max per method, per-seed bests, and best-so-far curves. It
compares optimizers on the given split (no holdout); it does not report
held-out robustness. Seeds with no valid trial are excluded and shown as n/a.

## Python example

```python
from failureforge.dataset import YoloDataset
from failureforge.demo import ColorBlobDetector, make_demo_dataset
from failureforge.objective import ObjectiveConfig
from failureforge.search import (
    SearchSpace, run_random_search, confirm_trials, write_search_report,
)

ds = YoloDataset.from_config(make_demo_dataset("demo_data", n=8))
discovery, holdout = ds.partition(0.3, seed=123)
det = ColorBlobDetector()          # replace with YOLODetector("best.pt")
space = SearchSpace.default(include=["gaussian_blur", "darken", "rotation"])
result = run_random_search(det, discovery, budget=20, seed=0, space=space,
                           objective=ObjectiveConfig("f1"))
confirmation = confirm_trials(det, holdout, result.trials, top_k=3, n_seeds=3)
write_search_report(result, "results", confirmation)
```

## Limitations

- One metric at a time; no multi-objective or weighted score.
- Uniform sampling only; no log-scale or conditional ranges, no combined
  (multi-step) perturbations in the default space.
- Small holdouts give noisy confirmation; `std` needs ≥ 2 seeds and is not a
  confidence interval.
- `compare` reports descriptive statistics only; no significance test.
- TPE is sequential; trials are not parallelised.
- Detector inference is repeated for every trial and image; cost scales with
  `budget × images`.
