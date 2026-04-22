"""
Automated Root-Cause Analysis on Drift (Dissertation §3.4)
==========================================================

The legacy pipeline (``ml/drift_detection.py``) tells an operator
*that* a distribution has shifted via PSI / KS / CUSUM.  §3.4 adds
*why*: which features contributed, and within those features, which
histogram bins drove the movement.  The output is a human-readable
summary such as::

    72% of PSI increase due to 'Travel_Time_Min' shift
    (more remote patients: bin [42, 58) min grew from 12% → 28%)

Math
----
For each numeric feature *j*, we take reference values $\\mathbf{r}_j$
and current values $\\mathbf{c}_j$, histogram both on the same
percentile-based breakpoints, and compute the per-bin PSI
contribution

.. math::
    \\delta_{j,i} = (P_{j,i}^{\\text{cur}} - P_{j,i}^{\\text{ref}})
                \\cdot \\ln\\left(
                      \\frac{P_{j,i}^{\\text{cur}}}
                           {P_{j,i}^{\\text{ref}}}\\right).

The total per-feature PSI is $\\mathrm{PSI}_j = \\sum_i \\delta_{j,i}$
(matching the existing ``DriftDetector.compute_psi``).  The overall
drift score used by the rest of the system is the *max* feature-wise
PSI, but for attribution we treat the **sum** of feature PSIs as the
pie we slice:

.. math::
    \\mathrm{share}_j = \\frac{\\mathrm{PSI}_j}{\\sum_{k} \\mathrm{PSI}_k}.

Within the top-ranked feature we pick the bins whose $|\\delta|$ is
largest, translate the percentile bin into a human-readable range
(``[42 min, 58 min)``), and emit a one-sentence narrative.  All of
this is independent of the rest of the pipeline — it *reuses* the
same percentile bin scheme and Laplace smoothing so the summed
per-feature PSIs equal the detector's reported per-feature score.

Wiring
------
* ``DriftAttributor.attribute(reference, current)`` is a pure
  function returning a ``DriftAttribution`` dataclass.
* ``flask_app.py`` injects the attributor into the drift pipeline
  so ``/api/ml/drift/check`` returns ``attribution`` automatically
  (invisible to clients that ignore it).
* Every run writes one JSONL row to
  ``data_cache/drift_attribution/attributions.jsonl``; the §25
  dissertation R script reads this log.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_BINS: int = 10
DEFAULT_TOP_BINS: int = 3                  # bins to call out in the narrative
DEFAULT_TOP_FEATURES: int = 5              # features to include in breakdown
MIN_SAMPLES: int = 10                      # below this we refuse to attribute
ATTRIBUTION_DIR: Path = DATA_CACHE_DIR / "drift_attribution"
ATTRIBUTION_LOG: Path = ATTRIBUTION_DIR / "attributions.jsonl"


# ---------------------------------------------------------------------------
# Dataclasses — public API shapes
# ---------------------------------------------------------------------------


@dataclass
class BinContribution:
    """Per-bin contribution to a single feature's PSI."""
    bin_index: int
    lower: float
    upper: float                 # exclusive upper for all but the last bin
    p_ref: float
    p_cur: float
    delta: float                 # p_cur - p_ref
    psi_contribution: float      # (p_cur - p_ref) * ln(p_cur / p_ref)
    direction: str               # "grew" | "shrank" | "unchanged"


@dataclass
class FeatureAttribution:
    """PSI decomposition for a single feature."""
    feature: str
    psi: float                   # Σ bin contributions
    share_of_total: float        # psi / Σ feature psi  (∈ [0, 1])
    n_ref: int
    n_cur: int
    bins: List[BinContribution] = field(default_factory=list)
    top_bin_summary: str = ""    # one-sentence description of top bin


@dataclass
class DriftAttribution:
    """Top-level result returned by ``DriftAttributor.attribute``."""
    computed_ts: str
    total_psi: float                         # Σ feature PSI
    overall_severity: str                    # "none" | "moderate" | "significant"
    top_feature: Optional[str]
    top_feature_share: float                 # 0..1
    narrative: str                           # human-readable summary
    feature_breakdown: List[FeatureAttribution]
    n_features_analysed: int
    n_ref_rows: int
    n_cur_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Attributor
# ---------------------------------------------------------------------------


class DriftAttributor:
    """Decomposes the drift signal per-feature and per-bin.

    The attributor is stateless — every call computes from scratch —
    but it owns the JSONL event log and a light in-memory cache of
    the most recent attribution so the /api/drift/attribution/last
    endpoint can serve it without a re-fit.
    """

    def __init__(
        self,
        *,
        n_bins: int = DEFAULT_N_BINS,
        top_bins: int = DEFAULT_TOP_BINS,
        top_features: int = DEFAULT_TOP_FEATURES,
        storage_dir: Path = ATTRIBUTION_DIR,
    ):
        self.n_bins = int(n_bins)
        self.top_bins = int(top_bins)
        self.top_features = int(top_features)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = self.storage_dir / "attributions.jsonl"
        self._lock = threading.Lock()
        self._last: Optional[DriftAttribution] = None
        self._n_runs: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def attribute(
        self,
        reference: Dict[str, Any],
        current: Dict[str, Any],
        feature_hints: Optional[Dict[str, str]] = None,
    ) -> DriftAttribution:
        """Compute per-feature and per-bin attribution.

        Args
        ----
        reference:
            ``{feature_name: array-like}`` — the reference window.
        current:
            ``{feature_name: array-like}`` — the recent window.
        feature_hints:
            Optional mapping ``{feature_name: human-hint}`` used in
            the narrative (e.g.\\ ``"Travel_Time_Min": "travel time
            (more remote patients)"``).  Unknown features just use
            the column name.
        """
        fh = feature_hints or {}
        per_feature: List[FeatureAttribution] = []
        n_ref_all = 0
        n_cur_all = 0

        # Intersect feature sets and compute per-feature PSI
        common = sorted(set(reference.keys()) & set(current.keys()))
        for feat in common:
            fa = self._attribute_one_feature(
                feat, reference[feat], current[feat],
                hint=fh.get(feat, ""),
            )
            if fa is None:
                continue
            per_feature.append(fa)
            n_ref_all = max(n_ref_all, fa.n_ref)
            n_cur_all = max(n_cur_all, fa.n_cur)

        total_psi = float(sum(fa.psi for fa in per_feature))
        # Rank
        per_feature.sort(key=lambda fa: fa.psi, reverse=True)
        for fa in per_feature:
            fa.share_of_total = (fa.psi / total_psi) if total_psi > 1e-12 else 0.0

        # Keep only top K in the response
        per_feature = per_feature[: self.top_features]

        # Overall severity uses the same thresholds as DriftDetector
        max_psi = per_feature[0].psi if per_feature else 0.0
        if max_psi > 0.25:
            severity = "significant"
        elif max_psi > 0.1:
            severity = "moderate"
        else:
            severity = "none"

        top_feat = per_feature[0].feature if per_feature else None
        top_share = per_feature[0].share_of_total if per_feature else 0.0

        narrative = self._build_narrative(per_feature, total_psi, severity)

        attribution = DriftAttribution(
            computed_ts=datetime.utcnow().isoformat(timespec="seconds"),
            total_psi=total_psi,
            overall_severity=severity,
            top_feature=top_feat,
            top_feature_share=float(top_share),
            narrative=narrative,
            feature_breakdown=per_feature,
            n_features_analysed=len(per_feature),
            n_ref_rows=int(n_ref_all),
            n_cur_rows=int(n_cur_all),
        )

        with self._lock:
            self._last = attribution
            self._n_runs += 1
        self._append_event(attribution)
        return attribution

    def last(self) -> Optional[DriftAttribution]:
        with self._lock:
            return self._last

    def status(self) -> Dict[str, Any]:
        last = self.last()
        return {
            "n_bins": self.n_bins,
            "top_bins": self.top_bins,
            "top_features": self.top_features,
            "total_runs": self._n_runs,
            "log_path": str(self.event_log),
            "last_computed_ts": last.computed_ts if last else None,
            "last_total_psi": last.total_psi if last else None,
            "last_top_feature": last.top_feature if last else None,
            "last_top_share": last.top_feature_share if last else None,
            "last_narrative": last.narrative if last else None,
            "last_severity": last.overall_severity if last else None,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _attribute_one_feature(
        self, feat: str, ref_raw: Any, cur_raw: Any, hint: str = ""
    ) -> Optional[FeatureAttribution]:
        """Compute per-bin PSI decomposition for one feature."""
        ref = np.asarray(ref_raw, dtype=float)
        cur = np.asarray(cur_raw, dtype=float)
        ref = ref[~np.isnan(ref)]
        cur = cur[~np.isnan(cur)]
        if len(ref) < MIN_SAMPLES or len(cur) < MIN_SAMPLES:
            return None

        # Reference-percentile breakpoints, matching DriftDetector
        bps = np.percentile(ref, np.linspace(0, 100, self.n_bins + 1))
        bps = np.unique(bps)
        if len(bps) < 3:
            return None

        ref_counts = np.histogram(ref, bins=bps)[0]
        cur_counts = np.histogram(cur, bins=bps)[0]

        n_bins_real = len(ref_counts)
        # Laplace smoothing (same scheme as DriftDetector)
        p_ref = (ref_counts + 1) / (len(ref) + n_bins_real)
        p_cur = (cur_counts + 1) / (len(cur) + n_bins_real)

        delta = p_cur - p_ref
        # Guard against log(0) — Laplace smoothing already prevents it,
        # but be defensive in case of weird inputs.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(np.clip(p_cur / p_ref, 1e-12, 1e12))
        bin_psi = delta * log_ratio
        total = float(np.sum(bin_psi))

        # Build BinContribution list
        bins: List[BinContribution] = []
        for i in range(n_bins_real):
            direction = (
                "grew" if delta[i] > 1e-4
                else "shrank" if delta[i] < -1e-4
                else "unchanged"
            )
            bins.append(
                BinContribution(
                    bin_index=int(i),
                    lower=float(bps[i]),
                    upper=float(bps[i + 1]),
                    p_ref=float(p_ref[i]),
                    p_cur=float(p_cur[i]),
                    delta=float(delta[i]),
                    psi_contribution=float(bin_psi[i]),
                    direction=direction,
                )
            )
        # Top bins by absolute contribution
        top = sorted(bins, key=lambda b: abs(b.psi_contribution), reverse=True)[: self.top_bins]

        top_bin = top[0] if top else None
        top_summary = _describe_top_bin(feat, top_bin, hint) if top_bin else ""

        return FeatureAttribution(
            feature=feat,
            psi=total,
            share_of_total=0.0,  # filled after the outer sum
            n_ref=int(len(ref)),
            n_cur=int(len(cur)),
            bins=sorted(bins, key=lambda b: abs(b.psi_contribution), reverse=True),
            top_bin_summary=top_summary,
        )

    def _build_narrative(
        self,
        per_feature: List[FeatureAttribution],
        total_psi: float,
        severity: str,
    ) -> str:
        """Produce a human-readable summary in the §3.4 style."""
        if not per_feature or total_psi < 1e-6:
            return "No material drift detected across analysed features."
        top = per_feature[0]
        pct = 100.0 * top.share_of_total
        lead = (
            f"{pct:.0f}% of PSI increase due to '{top.feature}' shift"
        )
        if top.top_bin_summary:
            lead += f" ({top.top_bin_summary})"
        if severity == "significant":
            lead += "; overall severity = SIGNIFICANT -- retrain recommended."
        elif severity == "moderate":
            lead += "; overall severity = moderate -- monitor."
        else:
            lead += "; overall severity = within limits."
        # Append the next driver if it contributes ≥10%
        if len(per_feature) > 1:
            second = per_feature[1]
            if second.share_of_total >= 0.10:
                lead += (
                    f"  Second driver: '{second.feature}' ("
                    f"{100.0 * second.share_of_total:.0f}% of PSI)."
                )
        return lead

    def _append_event(self, attribution: DriftAttribution) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": attribution.computed_ts,
                "total_psi": attribution.total_psi,
                "severity": attribution.overall_severity,
                "top_feature": attribution.top_feature,
                "top_feature_share": attribution.top_feature_share,
                "narrative": attribution.narrative,
                "features": [
                    {
                        "feature": fa.feature,
                        "psi": fa.psi,
                        "share": fa.share_of_total,
                    }
                    for fa in attribution.feature_breakdown
                ],
            }
            with open(self.event_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Attribution event-log write failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _describe_top_bin(
    feat: str, top: BinContribution, hint: str
) -> str:
    """Render a bin range into a short phrase operators can read."""
    lo = _fmt_bin(top.lower)
    hi = _fmt_bin(top.upper)
    pct_ref = 100.0 * top.p_ref
    pct_cur = 100.0 * top.p_cur
    direction = top.direction
    narrative = f"bin [{lo}, {hi}) {direction} from {pct_ref:.1f}% -> {pct_cur:.1f}%"
    if hint:
        narrative = f"{hint}: " + narrative
    return narrative


def _fmt_bin(v: float) -> str:
    if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01):
        return f"{v:.2g}"
    if abs(v - round(v)) < 0.05:
        return f"{int(round(v))}"
    return f"{v:.2f}"


# ---------------------------------------------------------------------------
# Module-level convenience singleton (mirrors §3.1-§3.3 style)
# ---------------------------------------------------------------------------

_GLOBAL: Optional[DriftAttributor] = None


def get_attributor() -> DriftAttributor:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = DriftAttributor()
    return _GLOBAL


def set_attributor(a: DriftAttributor) -> None:
    global _GLOBAL
    _GLOBAL = a
