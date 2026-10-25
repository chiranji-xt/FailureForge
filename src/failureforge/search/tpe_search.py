"""Optional Optuna TPE search; the core package works without Optuna."""
from __future__ import annotations

from typing import Any

from ..dataset import YoloDataset
from ..evaluation import DEFAULT_IOU
from ..evaluator import EvaluationReport
from ..objective import ObjectiveConfig
from .runner import TrialRunner, check_budget
from .space import SearchSpace
from .trials import SearchResult


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required but not installed."""


def optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
    except ImportError:
        return False
    return True


def run_tpe_search(
    detector: Any,
    dataset: YoloDataset,
    *,
    budget: int = 20,
    seed: int = 0,
    space: SearchSpace | None = None,
    objective: ObjectiveConfig | None = None,
    iou_threshold: float = DEFAULT_IOU,
    baseline: EvaluationReport | None = None,
) -> SearchResult:
    """TPE search maximizing objective degradation.

    Invalid or errored trials are reported to Optuna as FAILED (no value), so
    they never act as a zero- or ``-inf``-valued observation.
    """
    check_budget(budget)
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise OptionalDependencyError(
            'Optuna is not installed. Install with: pip install "failureforge[optuna]"'
        ) from exc

    space = space or SearchSpace.default()
    runner = TrialRunner(detector, dataset, objective=objective,
                         iou_threshold=iou_threshold, baseline=baseline)
    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        trials = []
        for index in range(budget):
            ot = study.ask()
            name = ot.suggest_categorical("transform", space.names)
            params: dict[str, Any] = {}
            for r in space.ranges(name):
                key = f"{name}.{r.name}"
                if r.integer:
                    params[r.name] = ot.suggest_int(key, int(r.low), int(r.high), step=r.step)
                else:
                    params[r.name] = ot.suggest_float(key, r.low, r.high)
            trial = runner.run(index, name, space.normalize(name, params), seed)
            trials.append(trial)
            if trial.valid:
                study.tell(ot, trial.score)
            else:
                study.tell(ot, state=optuna.trial.TrialState.FAIL)
    finally:
        optuna.logging.set_verbosity(previous_verbosity)
    return runner.package(
        "tpe", seed, budget, space, trials,
        extra={"sampler": "optuna.TPESampler", "optuna_version": optuna.__version__},
    )
