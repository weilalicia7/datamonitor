"""
Bayesian optimisation over expensive scalar hyperparameters (§29.4 method 3).

Uses ``skopt.gp_minimize`` with 30 random initial points + 20 Expected
Improvement (EI) acquisitions over a single scalar in each tuner:

- DRO Wasserstein radius ``epsilon`` ∈ [0.005, 0.20]
- CVaR quantile ``alpha``           ∈ [0.05, 0.30]
- Lipschitz constant ``L``          ∈ [0.5, 5.0]

The objective each tuner minimises is a **negated** composite operator
score ``- (0.5·util - 0.3·wait_norm + 0.2·fairness_ratio)`` so larger
operator score → smaller skopt loss.

The tuners are intentionally **scalar-only** because:

- The objective is opaque (it requires a real schedule solve), so the
  surrogate has to learn quickly with few samples — multivariate
  Bayesian opt would need ~10× more solves.
- Production rarely tunes more than one scalar at a time anyway —
  the operator brings a domain prior on the others.

The orchestrator in :mod:`tuning.run` decides which scalar(s) to tune
based on the tuning request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

#: Default search bounds.  Operator-tunable via the ``run`` orchestrator.
DEFAULT_BOUNDS: Dict[str, tuple] = {
    "dro_epsilon": (0.005, 0.20),
    "cvar_alpha":  (0.05,  0.30),
    "lipschitz_l": (0.5,   5.0),
}

#: Composite-objective coefficients (matches the §29.4 brief).
DEFAULT_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "utilisation":    0.50,
    "neg_waiting":    0.30,    # subtracted (smaller wait → larger reward)
    "fairness_ratio": 0.20,
}


@dataclass
class BayesOptResult:
    method: str = "skopt.gp_minimize"
    target: str = ""
    n_initial_points: int = 30
    n_calls: int = 50
    bounds: tuple = (0.0, 0.0)
    best_value: float = 0.0
    best_objective: float = 0.0     # the SCORE we wanted (positive)
    elapsed_s: float = 0.0
    n_samples: int = 0
    objective_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _composite_score(
    metrics: Dict[str, Any],
    weights: Dict[str, float],
) -> float:
    """Operator-facing composite (higher = better). Range roughly [0, 1]."""
    util = float(metrics.get("utilisation", metrics.get("utilization", 0.0)) or 0.0)
    wait = float(metrics.get("avg_waiting_days",
                             metrics.get("average_waiting_days", 0.0)) or 0.0)
    wait_norm = max(0.0, min(1.0, wait / 14.0))
    fairness = float(metrics.get("fairness_ratio",
                                 metrics.get("fairness_pass_ratio", 1.0)) or 1.0)
    return (
        weights.get("utilisation", 0.0)    * util
        - weights.get("neg_waiting", 0.0)  * wait_norm
        + weights.get("fairness_ratio", 0.0) * fairness
    )


def _import_skopt():
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except Exception as exc:                          # pragma: no cover
        raise RuntimeError(
            "scikit-optimize (skopt) is required for tuning.bayes_opt; "
            "install via 'pip install scikit-optimize'"
        ) from exc
    return gp_minimize, Real


def tune_scalar(
    *,
    target: str,
    evaluate_fn: Callable[[float], Dict[str, Any]],
    bounds: Optional[tuple] = None,
    n_initial_points: int = 30,
    n_calls: int = 50,
    random_state: int = 42,
    objective_weights: Optional[Dict[str, float]] = None,
    n_samples: int = 0,
) -> BayesOptResult:
    """Generic scalar Bayesian-opt loop.

    Parameters
    ----------
    target
        Logical name of the parameter (e.g. ``"dro_epsilon"``).  Used
        only for logging + result attribution.
    evaluate_fn
        ``value -> metrics dict``.  Caller wires this to a real solve
        whose metrics include ``utilisation`` / ``avg_waiting_days`` /
        ``fairness_ratio``.
    bounds
        ``(low, high)``.  Defaults to ``DEFAULT_BOUNDS[target]``.
    n_initial_points, n_calls
        skopt budget.  ``n_calls`` is total (initial + acquisition).
    random_state
        Reproducible seed.
    objective_weights
        Override composite-score weights.
    n_samples
        Optional: dataset row count, recorded in the result for
        provenance only.
    """
    gp_minimize, Real = _import_skopt()
    weights = objective_weights or DEFAULT_OBJECTIVE_WEIGHTS
    lo, hi = bounds or DEFAULT_BOUNDS.get(target, (0.0, 1.0))
    space = [Real(lo, hi, name=target)]
    n_initial_points = max(1, min(n_initial_points, n_calls))

    def _loss(point):
        value = float(point[0])
        try:
            metrics = evaluate_fn(value) or {}
        except Exception as exc:                      # pragma: no cover
            logger.warning("bayes_opt(%s): evaluate(%g) failed: %s", target, value, exc)
            return 1.0     # large positive loss
        score = _composite_score(metrics, weights)
        return -score      # skopt minimises; we want max score

    t0 = time.perf_counter()
    result = gp_minimize(
        _loss,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        acq_func="EI",
    )
    elapsed = time.perf_counter() - t0

    best_value = float(result.x[0])
    best_loss = float(result.fun)
    best_objective = -best_loss

    return BayesOptResult(
        target=target,
        n_initial_points=n_initial_points,
        n_calls=n_calls,
        bounds=(round(lo, 6), round(hi, 6)),
        best_value=round(best_value, 6),
        best_objective=round(best_objective, 6),
        elapsed_s=round(elapsed, 3),
        n_samples=int(n_samples),
        objective_weights=dict(weights),
    )
