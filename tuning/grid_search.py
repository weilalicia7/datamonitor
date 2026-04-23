"""
Grid search over CP-SAT objective-weight profiles (§29.4 method 1).

Sweeps every weight profile in ``PARETO_WEIGHT_SETS`` (or a custom
shortlist) on the historical appointment dataset, measures a composite
operator-facing score, and returns the Pareto-non-dominated subset.

The heavy lifting reuses the existing ``ScheduleOptimizer`` so this
module never re-implements CP-SAT — it only orchestrates calls.

Composite score per profile (higher = better):

    score = 0.50 * scheduled_fraction
          + 0.25 * utilisation
          + 0.15 * robustness
          - 0.10 * mean_waiting_days_normalised

All four sub-scores are normalised to ``[0, 1]`` before weighting so the
composite is itself in ``[-1, 1]``.

The tuner is **safe to run on synthetic data** (the resulting manifest
will be tagged ``data_channel="synthetic"`` by the orchestrator and the
boot path will refuse to apply it).  See :mod:`tuning.manifest` for the
gating rules.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Composite-score weights — chosen by domain, NOT tuned themselves.
DEFAULT_COMPOSITE_WEIGHTS: Dict[str, float] = {
    "scheduled_fraction": 0.50,
    "utilisation":        0.25,
    "robustness":         0.15,
    "neg_waiting_days":   0.10,    # subtracted, not added
}


@dataclass
class ProfileScore:
    name: str
    weights: Dict[str, float]
    scheduled_fraction: float
    utilisation: float
    robustness: float
    mean_waiting_days: float
    composite: float
    solve_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GridSearchResult:
    method: str = "grid_search"
    candidates_evaluated: int = 0
    pareto_frontier: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[Dict[str, Any]] = None
    composite_weights: Dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Score normalisation helpers
# --------------------------------------------------------------------------- #


def _normalised_waiting(days: float, cap: float = 14.0) -> float:
    """Squash mean waiting days into [0, 1] (1 == capped at the upper end)."""
    if not math.isfinite(days) or days <= 0:
        return 0.0
    return min(1.0, days / cap)


def _safe_metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    val = metrics.get(key, default)
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _composite(
    scheduled_fraction: float,
    utilisation: float,
    robustness: float,
    mean_waiting_days: float,
    composite_weights: Dict[str, float],
) -> float:
    return (
        composite_weights.get("scheduled_fraction", 0.0) * scheduled_fraction
        + composite_weights.get("utilisation",      0.0) * utilisation
        + composite_weights.get("robustness",       0.0) * robustness
        - composite_weights.get("neg_waiting_days", 0.0)
            * _normalised_waiting(mean_waiting_days)
    )


# --------------------------------------------------------------------------- #
# Pareto filter (maximise composite + scheduled_fraction; minimise wait)
# --------------------------------------------------------------------------- #


def _dominates(a: ProfileScore, b: ProfileScore) -> bool:
    """``a`` Pareto-dominates ``b`` iff a is no worse on all axes + better on at least one."""
    no_worse = (
        a.composite >= b.composite
        and a.scheduled_fraction >= b.scheduled_fraction
        and a.utilisation >= b.utilisation
        and a.robustness >= b.robustness
        and a.mean_waiting_days <= b.mean_waiting_days
    )
    strictly_better = (
        a.composite > b.composite
        or a.scheduled_fraction > b.scheduled_fraction
        or a.utilisation > b.utilisation
        or a.robustness > b.robustness
        or a.mean_waiting_days < b.mean_waiting_days
    )
    return no_worse and strictly_better


def pareto_frontier(scores: Sequence[ProfileScore]) -> List[ProfileScore]:
    """Return the non-dominated subset of ``scores`` (order-preserving)."""
    out: List[ProfileScore] = []
    for s in scores:
        if not any(_dominates(other, s) for other in scores if other is not s):
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def evaluate_weight_profiles(
    *,
    patients: List[Any],
    weight_sets: Sequence[Dict[str, float]],
    solve_fn: Callable[[List[Any], Dict[str, float], int], Any],
    time_limit_s: int = 30,
    composite_weights: Optional[Dict[str, float]] = None,
) -> GridSearchResult:
    """Run one solve per weight profile, score, return Pareto + winner.

    Parameters
    ----------
    patients
        Patient objects already loaded by the caller.  Use the same list
        for every profile so scores are directly comparable.
    weight_sets
        Iterable of weight dicts.  Each must include a ``name`` key
        (echoed in the result) plus the six objective weights.
    solve_fn
        Callable that takes ``(patients, weights, time_limit_s)`` and
        returns an object with ``.metrics`` (dict) + ``.appointments``
        (list).  Pass ``optimizer.solve_with_weights`` from flask_app.
    time_limit_s
        Per-profile CP-SAT budget.
    composite_weights
        Override the composite-score weights.  Defaults to
        :data:`DEFAULT_COMPOSITE_WEIGHTS`.
    """
    cw = composite_weights or DEFAULT_COMPOSITE_WEIGHTS
    n_patients = max(1, len(patients))
    t0_total = time.perf_counter()

    scores: List[ProfileScore] = []
    for cfg in weight_sets:
        cfg_name = str(cfg.get("name", "custom"))
        weights_only = {k: v for k, v in cfg.items() if k != "name"}
        t0 = time.perf_counter()
        try:
            result = solve_fn(patients, weights_only, time_limit_s)
        except Exception as exc:                      # pragma: no cover
            logger.warning("grid_search: profile %s failed: %s", cfg_name, exc)
            continue
        solve_time = time.perf_counter() - t0

        metrics = getattr(result, "metrics", {}) or {}
        n_scheduled = len(getattr(result, "appointments", []) or [])
        sched_frac = n_scheduled / n_patients
        util = _safe_metric(metrics, "utilization", 0.0)
        robust = _safe_metric(metrics, "robustness_score",
                              _safe_metric(metrics, "robustness", 0.0))
        wait_days = _safe_metric(metrics, "avg_waiting_days",
                                 _safe_metric(metrics, "average_waiting_days", 0.0))

        composite = _composite(sched_frac, util, robust, wait_days, cw)
        scores.append(ProfileScore(
            name=cfg_name,
            weights=weights_only,
            scheduled_fraction=round(sched_frac, 4),
            utilisation=round(util, 4),
            robustness=round(robust, 4),
            mean_waiting_days=round(wait_days, 2),
            composite=round(composite, 4),
            solve_time_s=round(solve_time, 3),
        ))

    if not scores:
        return GridSearchResult(
            candidates_evaluated=0,
            composite_weights=cw,
            elapsed_s=round(time.perf_counter() - t0_total, 3),
        )

    frontier = pareto_frontier(scores)
    # Winner: highest composite among the Pareto frontier.
    winner = max(frontier, key=lambda s: s.composite)
    return GridSearchResult(
        candidates_evaluated=len(scores),
        pareto_frontier=[s.to_dict() for s in frontier],
        winner=winner.to_dict(),
        composite_weights=cw,
        elapsed_s=round(time.perf_counter() - t0_total, 3),
    )
