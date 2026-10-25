"""Seeded Random Search over a :class:`SearchSpace`."""
from __future__ import annotations

import random
from typing import Any

from ..dataset import YoloDataset
from ..evaluation import DEFAULT_IOU
from ..evaluator import EvaluationReport
from ..objective import ObjectiveConfig
from .runner import TrialRunner, check_budget
from .space import SearchSpace
from .trials import SearchResult


def run_random_search(
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
    """Sample ``budget`` configurations and score each against one baseline.

    The sample stream depends only on ``seed`` and ``space``; per-trial
    perturbation seeds are derived from ``seed``, the trial index, and the
    configuration, so the whole search is reproducible.
    """
    check_budget(budget)
    space = space or SearchSpace.default()
    runner = TrialRunner(detector, dataset, objective=objective,
                         iou_threshold=iou_threshold, baseline=baseline)
    rng = random.Random(seed)
    trials = []
    for index in range(budget):
        name, params = space.sample(rng)
        trials.append(runner.run(index, name, params, seed))
    return runner.package("random", seed, budget, space, trials)
