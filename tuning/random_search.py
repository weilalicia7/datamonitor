"""
Random search over tree-ensemble hyperparameters (§29.4 method 2).

Uses ``sklearn.model_selection.RandomizedSearchCV`` with
``TimeSeriesSplit(n_splits=5)`` so we evaluate "train on past, validate
on future" — appropriate for a scheduling system whose underlying
patterns evolve over time.

Two model families are tunable here:

- ``noshow_model``  — `RandomForestClassifier` + `GradientBoostingClassifier`,
  optimising ROC-AUC.
- ``duration_model`` — `RandomForestRegressor` + `GradientBoostingRegressor`,
  optimising negative MAE.

The function returns a structured payload suitable for
``record_tuning_run(...)``; it never mutates the live ML modules.  The
boot path applies the tuned hyperparameters only when the orchestrator
recorded the run with ``data_channel="real"``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Search space — small enough to finish in <30 s on a 2 k-row dataset
#: with n_iter=20, but rich enough to materially shift performance on
#: real distributions.  Operator may pass a custom ``param_distributions``.
DEFAULT_NOSHOW_GRID: Dict[str, List[Any]] = {
    "n_estimators":       [50, 100, 200, 300],
    "max_depth":          [3, 5, 8, 12, None],
    "min_samples_leaf":   [1, 2, 5, 10],
    "min_samples_split":  [2, 5, 10],
}

DEFAULT_DURATION_GRID: Dict[str, List[Any]] = {
    "n_estimators":       [50, 100, 200, 300],
    "max_depth":          [3, 5, 8, 12, None],
    "min_samples_leaf":   [1, 2, 5, 10],
    "learning_rate":      [0.01, 0.03, 0.05, 0.1],
}


@dataclass
class RandomSearchResult:
    method: str = "RandomizedSearchCV"
    target: str = ""
    estimator: str = ""
    cv: str = ""
    n_iter: int = 0
    n_samples: int = 0
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    baseline_score: float = 0.0
    improvement: float = 0.0
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _import_sklearn():
    try:
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.model_selection import (
            RandomizedSearchCV,
            TimeSeriesSplit,
            cross_val_score,
        )
    except Exception as exc:                          # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for tuning.random_search; "
            "install it via the requirements.txt pin"
        ) from exc
    return {
        "RFClassifier": RandomForestClassifier,
        "RFRegressor":  RandomForestRegressor,
        "GBClassifier": GradientBoostingClassifier,
        "GBRegressor":  GradientBoostingRegressor,
        "RandomizedSearchCV": RandomizedSearchCV,
        "TimeSeriesSplit":    TimeSeriesSplit,
        "cross_val_score":    cross_val_score,
    }


def _split_xy(df: pd.DataFrame, target_col: str) -> tuple:
    """Drop the target + non-numeric columns, return (X, y)."""
    if target_col not in df.columns:
        raise KeyError(f"target column '{target_col}' missing from DataFrame")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    # Keep numeric columns only — sklearn estimators choke on strings.
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    if X.shape[1] == 0:
        raise ValueError("no numeric feature columns after dropping target")
    return X, y


def tune_noshow_model(
    df: pd.DataFrame,
    *,
    target_col: str = "is_noshow",
    estimator: str = "rf",
    n_iter: int = 20,
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
    param_distributions: Optional[Dict[str, List[Any]]] = None,
) -> RandomSearchResult:
    """RandomizedSearchCV for the no-show classifier."""
    sk = _import_sklearn()
    X, y = _split_xy(df, target_col)
    param_dist = param_distributions or DEFAULT_NOSHOW_GRID
    cls_map = {"rf": sk["RFClassifier"], "gb": sk["GBClassifier"]}
    if estimator not in cls_map:
        raise ValueError(f"estimator must be one of {list(cls_map)}; got {estimator!r}")
    base = cls_map[estimator](random_state=random_state)
    cv = sk["TimeSeriesSplit"](n_splits=cv_splits)

    t0 = time.perf_counter()
    baseline_score = float(np.mean(sk["cross_val_score"](
        base, X, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs,
    )))

    search = sk["RandomizedSearchCV"](
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        random_state=random_state,
        refit=False,
    )
    search.fit(X, y)
    elapsed = time.perf_counter() - t0

    return RandomSearchResult(
        target="noshow",
        estimator={"rf": "RandomForestClassifier",
                   "gb": "GradientBoostingClassifier"}[estimator],
        cv=f"TimeSeriesSplit({cv_splits})",
        n_iter=n_iter,
        n_samples=int(len(y)),
        best_params=dict(search.best_params_),
        best_score=round(float(search.best_score_), 4),
        baseline_score=round(baseline_score, 4),
        improvement=round(float(search.best_score_) - baseline_score, 4),
        elapsed_s=round(elapsed, 3),
    )


def tune_duration_model(
    df: pd.DataFrame,
    *,
    target_col: str = "Actual_Duration",
    estimator: str = "gb",
    n_iter: int = 20,
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
    param_distributions: Optional[Dict[str, List[Any]]] = None,
) -> RandomSearchResult:
    """RandomizedSearchCV for the duration regressor."""
    sk = _import_sklearn()
    X, y = _split_xy(df, target_col)
    # GB regressor's built-in `learning_rate` only applies to GB; drop
    # it from the RF grid if the caller selected RF.
    grid = dict(param_distributions or DEFAULT_DURATION_GRID)
    if estimator == "rf" and "learning_rate" in grid:
        grid = {k: v for k, v in grid.items() if k != "learning_rate"}
    cls_map = {"rf": sk["RFRegressor"], "gb": sk["GBRegressor"]}
    if estimator not in cls_map:
        raise ValueError(f"estimator must be one of {list(cls_map)}; got {estimator!r}")
    base = cls_map[estimator](random_state=random_state)
    cv = sk["TimeSeriesSplit"](n_splits=cv_splits)

    t0 = time.perf_counter()
    baseline_score = float(np.mean(sk["cross_val_score"](
        base, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=n_jobs,
    )))

    search = sk["RandomizedSearchCV"](
        estimator=base,
        param_distributions=grid,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
        random_state=random_state,
        refit=False,
    )
    search.fit(X, y)
    elapsed = time.perf_counter() - t0

    return RandomSearchResult(
        target="duration",
        estimator={"rf": "RandomForestRegressor",
                   "gb": "GradientBoostingRegressor"}[estimator],
        cv=f"TimeSeriesSplit({cv_splits})",
        n_iter=n_iter,
        n_samples=int(len(y)),
        best_params=dict(search.best_params_),
        best_score=round(float(search.best_score_), 4),
        baseline_score=round(baseline_score, 4),
        improvement=round(float(search.best_score_) - baseline_score, 4),
        elapsed_s=round(elapsed, 3),
    )
