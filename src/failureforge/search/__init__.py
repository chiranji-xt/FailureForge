"""Robustness search: Random Search, optional Optuna TPE, and confirmation."""
from .confirm import ConfirmationRecord, ConfirmationResult, confirm_trials
from .random_search import run_random_search
from .report import (
    render_search_markdown,
    search_payload,
    write_search_report,
    write_trials_csv,
)
from .runner import TrialRunner
from .space import ParamRange, SearchSpace, SearchSpaceError
from .tpe_search import OptionalDependencyError, optuna_available, run_tpe_search
from .trials import SearchResult, SearchTrial, config_key, rank_trials

__all__ = [
    "ConfirmationRecord", "ConfirmationResult", "confirm_trials",
    "run_random_search", "run_tpe_search", "optuna_available",
    "OptionalDependencyError", "TrialRunner", "ParamRange", "SearchSpace",
    "SearchSpaceError", "SearchResult", "SearchTrial", "config_key", "rank_trials",
    "render_search_markdown", "search_payload", "write_search_report",
    "write_trials_csv",
]
