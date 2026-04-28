"""
Multi-Criteria Decision Analysis (MCDA) on the Pareto-weight frontiers.

Each weight-sensitivity sweep emits a frontier of points in 2-D objective
space (utilisation vs mean wait; no-show risk vs robustness).  A
day-unit manager's tuning question is "which weight setting gives the
best compromise?"  This module answers that with two formal tools:

knee_point(frontier)
    The "knee" of a Pareto front is the point with the maximum
    perpendicular distance from the chord joining the two extremes
    (utopia / nadir).  This is the Kneedle-style detection on the
    normalised front.  Geometrically, it identifies the position where
    a small move along the front buys the largest joint improvement
    on the other axis - the point of diminishing returns.

trade_off_ratios(frontier)
    For each interior point, dY/dX in normalised axis units.  Adjacent
    high ratios mean improving X further is expensive in Y; the knee
    sits where the ratio sharply changes.

The output of ``analyse_frontier()`` is suitable for both:
- the dissertation tables / annotated figures (point estimate + weights)
- the Flask diagnostic endpoint
  (``GET /api/ml/optimisation/pareto-knee``).

The module reads the JSONL written by ``ml/benchmark_weight_sensitivity.py``
so it can run silently as part of the training pipeline; it is callable
on a list of point dicts directly for unit-tests.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple

import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Frontier configuration
# --------------------------------------------------------------------------- #

# Each frontier reports the metric to plot and whether higher is better.
FRONTIER_AXES: Dict[str, Dict[str, Any]] = {
    "frontier_util_vs_wait": {
        "x_metric":      "mean_wait_min",
        "x_higher_better": False,
        "x_label":       "Mean wait (min)",
        "y_metric":      "utilisation",
        "y_higher_better": True,
        "y_label":       "Chair utilisation",
        "axis_label":    "utilisation_vs_waiting_time",
    },
    "frontier_noshow_vs_robust": {
        "x_metric":      "mean_scheduled_noshow_rate",
        "x_higher_better": False,
        "x_label":       "Scheduled no-show risk",
        "y_metric":      "robustness_score",
        "y_higher_better": True,
        "y_label":       "Schedule robustness",
        "axis_label":    "noshow_risk_vs_robustness",
    },
}


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class FrontierMCDA:
    name:                   str
    n_points:               int
    knee_index:             Optional[int]
    knee_x:                 Optional[float]
    knee_y:                 Optional[float]
    knee_weights:           Optional[Dict[str, float]]
    knee_distance_to_chord: Optional[float]    # in normalised units
    trade_off_at_knee:      Optional[float]    # dY/dX in normalised units
    utopia_x:               Optional[float]
    utopia_y:               Optional[float]
    nadir_x:                Optional[float]
    nadir_y:                Optional[float]
    points:                 List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParetoMCDAResult:
    source:    str                                 # path or "<inline>"
    frontiers: Dict[str, FrontierMCDA] = field(default_factory=dict)
    n_patients: Optional[int] = None
    n_chairs:   Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source":     self.source,
            "n_patients": self.n_patients,
            "n_chairs":   self.n_chairs,
            "frontiers":  {k: v.to_dict() for k, v in self.frontiers.items()},
        }


# --------------------------------------------------------------------------- #
# Core MCDA primitives
# --------------------------------------------------------------------------- #


def _normalise(values: List[float], higher_better: bool) -> Tuple[np.ndarray, float, float]:
    """Map values to [0, 1] where 1 = best.  Returns (normalised, min, max)."""
    arr = np.array(values, dtype=float)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if vmax == vmin:
        return np.zeros_like(arr), vmin, vmax
    if higher_better:
        norm = (arr - vmin) / (vmax - vmin)
    else:
        norm = (vmax - arr) / (vmax - vmin)
    return norm, vmin, vmax


def _knee_point_index(x_norm: np.ndarray, y_norm: np.ndarray) -> Optional[int]:
    """Index of the front point with maximum perpendicular distance from the
    chord joining the two extremes.  Both axes already in [0, 1] with 1=best.
    """
    n = len(x_norm)
    if n < 3:
        return None
    # Sort by x_norm so chord runs from worst-X to best-X
    order = np.argsort(x_norm)
    xs, ys = x_norm[order], y_norm[order]
    p1 = np.array([xs[0],  ys[0]])
    p2 = np.array([xs[-1], ys[-1]])
    chord = p2 - p1
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm < 1e-12:
        return None
    distances = []
    for i in range(n):
        p = np.array([xs[i], ys[i]])
        # Signed distance from chord; sign positive when above chord
        cross = chord[0] * (p[1] - p1[1]) - chord[1] * (p[0] - p1[0])
        distances.append(abs(cross) / chord_norm)
    knee_local = int(np.argmax(distances))
    return int(order[knee_local])


def _trade_off_ratios(x_norm: np.ndarray, y_norm: np.ndarray
                      ) -> List[Optional[float]]:
    """dY/dX in normalised axis units, central differences for interior
    points; endpoints get one-sided differences.  None when undefined."""
    n = len(x_norm)
    order = np.argsort(x_norm)
    xs, ys = x_norm[order], y_norm[order]
    ratios = [None] * n
    for k in range(n):
        if k == 0:
            dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        elif k == n - 1:
            dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
        else:
            dx, dy = xs[k+1] - xs[k-1], ys[k+1] - ys[k-1]
        ratios[order[k]] = (float(dy / dx) if abs(dx) > 1e-12 else None)
    return ratios


def analyse_frontier(points: List[Dict[str, Any]],
                     name: str = "frontier_util_vs_wait") -> FrontierMCDA:
    """Run knee-point + trade-off MCDA on a single frontier."""
    cfg = FRONTIER_AXES.get(name, FRONTIER_AXES["frontier_util_vs_wait"])
    if not points:
        return FrontierMCDA(name=name, n_points=0,
                            knee_index=None, knee_x=None, knee_y=None,
                            knee_weights=None, knee_distance_to_chord=None,
                            trade_off_at_knee=None,
                            utopia_x=None, utopia_y=None,
                            nadir_x=None, nadir_y=None, points=[])

    xs = [p[cfg["x_metric"]] for p in points]
    ys = [p[cfg["y_metric"]] for p in points]
    x_norm, x_min, x_max = _normalise(xs, cfg["x_higher_better"])
    y_norm, y_min, y_max = _normalise(ys, cfg["y_higher_better"])

    knee_idx = _knee_point_index(x_norm, y_norm)
    ratios   = _trade_off_ratios(x_norm, y_norm)

    knee_dist = None
    if knee_idx is not None:
        order = np.argsort(x_norm)
        xs_sorted, ys_sorted = x_norm[order], y_norm[order]
        chord = np.array([xs_sorted[-1] - xs_sorted[0],
                          ys_sorted[-1] - ys_sorted[0]])
        cnorm = float(np.linalg.norm(chord))
        p   = np.array([x_norm[knee_idx], y_norm[knee_idx]])
        p1  = np.array([xs_sorted[0], ys_sorted[0]])
        cross = chord[0] * (p[1] - p1[1]) - chord[1] * (p[0] - p1[0])
        knee_dist = abs(float(cross / cnorm))

    knee_pt = points[knee_idx] if knee_idx is not None else None
    knee_weights = (knee_pt or {}).get("weights")
    if knee_weights is None and knee_pt is not None:
        knee_weights = {
            "axis_a_weight": knee_pt.get("axis_a_weight"),
            "axis_b_weight": knee_pt.get("axis_b_weight"),
        }

    enriched: List[Dict[str, Any]] = []
    for i, p in enumerate(points):
        ep = dict(p)
        ep["x_norm"] = float(x_norm[i])
        ep["y_norm"] = float(y_norm[i])
        ep["trade_off_dY_dX"] = ratios[i]
        ep["is_knee"] = (knee_idx is not None and i == knee_idx)
        enriched.append(ep)

    return FrontierMCDA(
        name                   = name,
        n_points               = len(points),
        knee_index             = knee_idx,
        knee_x                 = (None if knee_idx is None else float(xs[knee_idx])),
        knee_y                 = (None if knee_idx is None else float(ys[knee_idx])),
        knee_weights           = knee_weights,
        knee_distance_to_chord = knee_dist,
        trade_off_at_knee      = (None if knee_idx is None else ratios[knee_idx]),
        utopia_x               = x_min if cfg["x_higher_better"] else x_min,  # x best
        utopia_y               = y_max if cfg["y_higher_better"] else y_min,  # y best
        nadir_x                = x_max if cfg["x_higher_better"] else x_max,
        nadir_y                = y_min if cfg["y_higher_better"] else y_max,
        points                 = enriched,
    )


def analyse_latest(jsonl_path: Path) -> Optional[ParetoMCDAResult]:
    """Read the latest row from the weight-sensitivity JSONL and run MCDA
    on every frontier.  Returns ``None`` if the file is absent or empty."""
    if not jsonl_path.exists():
        return None
    last_line = None
    try:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last_line = line
    except OSError:
        return None
    if last_line is None:
        return None
    try:
        row = json.loads(last_line)
    except Exception:
        return None

    out = ParetoMCDAResult(
        source     = str(jsonl_path),
        n_patients = row.get("n_patients"),
        n_chairs   = row.get("n_chairs"),
    )
    for frontier_key in ("frontier_util_vs_wait", "frontier_noshow_vs_robust"):
        pts = row.get(frontier_key) or []
        out.frontiers[frontier_key] = analyse_frontier(pts, name=frontier_key)
    return out
