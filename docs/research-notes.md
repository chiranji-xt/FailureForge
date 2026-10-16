# FailureForge — research notes

## Research question

> Under an equal evaluation budget, does Optuna TPE identify more damaging
> image perturbation configurations than random search?

## Proposed experimental design

1. Evaluate the detector on unperturbed data → baseline metrics.
2. Split labeled data into a **discovery** split (search optimization) and a
   held-out **confirmation** split (final comparison), with a recorded seed.
3. Run random search and TPE with **equal trial budgets** on the discovery
   split, optimizing deterioration
   `objective = metric(baseline) − metric(perturbed)` (e.g. F1 drop).
4. Re-evaluate each method's winning configuration on the confirmation split.
5. The method whose configuration causes the larger *confirmed* deterioration
   wins that round. Repeat over multiple seeds before drawing conclusions.

## Fair-comparison requirements

- Same model weights, search space, parameter ranges, inference settings
  (confidence/IoU thresholds), metric, and discovery data for both methods.
- Same trial budget; recorded wall-clock time per method.
- Confirmation data excluded from all optimization decisions.
- All trials recorded (parameters, seed, score, errors, timing).

## Reproducibility

- Deterministic seeds for dataset splits, perturbation sampling, occlusion
  placement, and both search algorithms.
- Config file, package version, and timings stored alongside every result.

## Potential biases & limitations (to revisit with data)

- Small datasets make single-seed comparisons noisy; multiple seeds required.
- Axis-aligned boxes only approximate rotated objects (conservative bias).
- The objective rewards metric deterioration, which must be validated as
  genuine detection degradation, not an annotation/evaluation artifact —
  guard with degenerate-box recording and empty-case handling.
- Results describe the tested detector/dataset; they do not enumerate all
  real-world failures and do not transfer automatically to other models.

## Sources / verification status

- No paper references, citations, benchmark results, or conclusions are
  claimed at this stage. Any external source cited in later days must be
  verified (retrieved and readable) before being relied upon. **[to verify later]**
