"""
Distributionally Robust Fairness Certificates (Dissertation §4.1)
================================================================

The legacy pipeline (``ml/fairness_audit.py``) treats demographic
parity as a \\emph{soft} objective-function penalty: the optimiser
can trade it off against throughput if the scheduling weights
permit.  The §4.1 brief asks for the opposite — a hard constraint
with a \\emph{certificate} that holds even when the evaluation
distribution is perturbed.  Formally, for protected groups
$g_1, g_2$ and outcome indicator $Y = \\mathbf{1}[\\text{scheduled}]$,
we want

.. math::
    \\sup_{Q \\in B_\\varepsilon(\\widehat{P})}
       \\big|\\,P_Q(Y = 1 \\mid G = g_1)
             - P_Q(Y = 1 \\mid G = g_2)\\,\\big|
       \\;\\le\\; \\delta,

where $B_\\varepsilon(\\widehat{P})$ is a 1-Wasserstein ball of
radius $\\varepsilon$ centred on the empirical joint $\\widehat{P}$.

Upper bound used
----------------
Because $Y \\in \\{0, 1\\}$ and we use the Wasserstein-1 metric
with the discrete $0$–$1$ cost on the label, the supremum admits
the closed-form dual bound (see Taskesen et al., FAccT 2021; and
Shafieezadeh-Abadeh, Kuhn & Esfahani, 2019):

.. math::
    \\sup_Q \\big| \\Delta_Q \\big|
    \\;\\le\\; \\underbrace{|\\widehat{\\Delta}|}_{\\text{empirical gap}}
    \\;+\\; \\varepsilon \\cdot
          \\Big(\\tfrac{1}{\\pi_1} + \\tfrac{1}{\\pi_2}\\Big),

where $\\widehat{\\Delta}$ is the plug-in parity gap and $\\pi_g$
the empirical mass on group $g$.  The $\\varepsilon/\\pi_g$ term
captures the adversary's leverage: the smaller the group, the more
a unit of mass can shift its conditional rate.  The implementation
is deliberately distribution-free — no classifier score, no
density estimate — so it runs in $O(|\\text{groups}|)$ per pair
and composes with the existing ``FairnessAuditor`` without any
refactor.

Finite-sample adjustment
------------------------
For small samples we tighten the bound by subtracting a
Clopper–Pearson-style standard error $z_{\\alpha/2} \\sqrt{\\hat{p}
(1 - \\hat{p})/n_g}$ \\emph{from below} (i.e.\\ we inflate by the
SE in both directions).  This gives operators a two-sided
confidence-adjusted worst-case.  Details: see MATH_LOGIC §11b.1.

Wiring
------
* ``DROFairnessCertifier.certify(patients, scheduled_ids,
  group_column, epsilon, delta)`` returns a
  :class:`DROFairnessCertificate`.
* ``flask_app.py`` exposes three diagnostics routes under
  ``/api/fairness/dro/*`` and auto-attaches a certificate to every
  ``/api/fairness/audit`` response (invisible integration).
* Every run logs one JSONL row to
  ``data_cache/dro_fairness/certificates.jsonl``; the §26
  dissertation R script reads this log.

This module intentionally does NOT mutate the existing optimiser
or fairness objective.  It is a strictly additive hard-constraint
certificate layer — the operator chooses whether to treat a failed
certificate as advisory or blocking, via
``DROFairnessCertifier.enforce_as_hard_constraint``.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults — kept aligned with the existing FairnessAuditor thresholds
# ---------------------------------------------------------------------------
DEFAULT_EPSILON: float = 0.02          # 1-Wasserstein ball radius, fraction of total mass
DEFAULT_DELTA: float = 0.15            # parity-gap budget (same as DEMOGRAPHIC_PARITY_THRESHOLD)
DEFAULT_CONFIDENCE: float = 0.95       # two-sided confidence for SE adjustment
MIN_GROUP_SIZE: int = 5
CERTIFICATE_DIR: Path = DATA_CACHE_DIR / "dro_fairness"
CERTIFICATE_LOG: Path = CERTIFICATE_DIR / "certificates.jsonl"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PairCertificate:
    """Wasserstein-DRO certificate for a single group pair."""
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    pi_a: float
    pi_b: float
    rate_a: float
    rate_b: float
    empirical_gap: float
    wasserstein_epsilon: float
    worst_case_gap: float
    se_adjusted_upper: float       # finite-sample inflated bound
    delta_budget: float
    certified: bool                # worst_case_gap <= delta_budget
    certified_conservative: bool   # se_adjusted_upper <= delta_budget
    slack_to_budget: float         # delta_budget - worst_case_gap (+ = room, - = breach)
    method: str = "wasserstein_1_plugin"


@dataclass
class DROFairnessCertificate:
    """Top-level certificate returned by ``DROFairnessCertifier.certify``."""
    computed_ts: str
    group_column: str
    epsilon: float
    delta: float
    overall_certified: bool
    overall_certified_conservative: bool
    worst_pair_gap: float                  # max worst_case_gap across pairs
    worst_pair: Optional[Tuple[str, str]]  # (group_a, group_b) achieving the max
    narrative: str
    pair_certificates: List[PairCertificate] = field(default_factory=list)
    n_groups: int = 0
    n_patients: int = 0
    n_scheduled: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Tuples → lists for JSON
        if d["worst_pair"] is not None:
            d["worst_pair"] = list(d["worst_pair"])
        return d


# ---------------------------------------------------------------------------
# Certifier
# ---------------------------------------------------------------------------


class DROFairnessCertifier:
    """Hard-constraint fairness certifier with Wasserstein-1 DRO guarantees."""

    def __init__(
        self,
        *,
        epsilon: float = DEFAULT_EPSILON,
        delta: float = DEFAULT_DELTA,
        confidence: float = DEFAULT_CONFIDENCE,
        enforce_as_hard_constraint: bool = False,
        storage_dir: Path = CERTIFICATE_DIR,
    ):
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.confidence = float(confidence)
        self.enforce_as_hard_constraint = bool(enforce_as_hard_constraint)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.certificate_log = self.storage_dir / "certificates.jsonl"

        self._lock = threading.Lock()
        self._last: Optional[DROFairnessCertificate] = None
        self._n_runs: int = 0

    # ---------- public API ----------------------------------------------- #

    def certify(
        self,
        patients: Iterable[Dict[str, Any]],
        scheduled_ids: Iterable[Any],
        group_column: str = "Age_Band",
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> DROFairnessCertificate:
        """Compute a per-pair DRO certificate and an overall verdict."""
        patients = list(patients)
        scheduled_set = set(
            s if not hasattr(s, "patient_id") else s.patient_id
            for s in scheduled_ids
        )
        eps = float(self.epsilon if epsilon is None else epsilon)
        d_bud = float(self.delta if delta is None else delta)

        groups = _group_by(patients, scheduled_set, group_column)
        n_total = sum(g["total"] for g in groups.values())
        n_scheduled_total = sum(g["scheduled"] for g in groups.values())

        pair_certs: List[PairCertificate] = []
        names = [g for g in groups if groups[g]["total"] >= MIN_GROUP_SIZE]
        for i, g_a in enumerate(names):
            for g_b in names[i + 1 :]:
                pc = self._certify_pair(
                    g_a, g_b, groups[g_a], groups[g_b], eps, d_bud, n_total
                )
                pair_certs.append(pc)

        # Aggregate: overall_certified iff every pair is certified
        overall = all(pc.certified for pc in pair_certs) if pair_certs else True
        overall_conservative = (
            all(pc.certified_conservative for pc in pair_certs)
            if pair_certs else True
        )

        worst_gap = (
            max((pc.worst_case_gap for pc in pair_certs), default=0.0)
        )
        worst_pair: Optional[Tuple[str, str]] = None
        for pc in pair_certs:
            if pc.worst_case_gap == worst_gap and worst_gap > 0.0:
                worst_pair = (pc.group_a, pc.group_b)
                break

        narrative = _build_narrative(
            pair_certs, overall, overall_conservative,
            worst_gap, worst_pair, eps, d_bud, group_column,
        )

        cert = DROFairnessCertificate(
            computed_ts=datetime.utcnow().isoformat(timespec="seconds"),
            group_column=group_column,
            epsilon=eps,
            delta=d_bud,
            overall_certified=overall,
            overall_certified_conservative=overall_conservative,
            worst_pair_gap=worst_gap,
            worst_pair=worst_pair,
            narrative=narrative,
            pair_certificates=pair_certs,
            n_groups=len(names),
            n_patients=int(n_total),
            n_scheduled=int(n_scheduled_total),
        )
        with self._lock:
            self._last = cert
            self._n_runs += 1
        self._append_event(cert)
        return cert

    def last(self) -> Optional[DROFairnessCertificate]:
        with self._lock:
            return self._last

    def status(self) -> Dict[str, Any]:
        last = self.last()
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "confidence": self.confidence,
            "enforce_as_hard_constraint": self.enforce_as_hard_constraint,
            "total_runs": self._n_runs,
            "log_path": str(self.certificate_log),
            "last_computed_ts": last.computed_ts if last else None,
            "last_overall_certified": last.overall_certified if last else None,
            "last_worst_pair_gap": last.worst_pair_gap if last else None,
            "last_worst_pair": (
                list(last.worst_pair) if last and last.worst_pair else None
            ),
            "last_narrative": last.narrative if last else None,
        }

    def update_config(
        self,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        confidence: Optional[float] = None,
        enforce_as_hard_constraint: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if epsilon is not None:
            self.epsilon = float(epsilon)
        if delta is not None:
            self.delta = float(delta)
        if confidence is not None:
            self.confidence = float(confidence)
        if enforce_as_hard_constraint is not None:
            self.enforce_as_hard_constraint = bool(enforce_as_hard_constraint)
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "confidence": self.confidence,
            "enforce_as_hard_constraint": self.enforce_as_hard_constraint,
        }

    # ---------- internals ------------------------------------------------ #

    def _certify_pair(
        self,
        g_a: str,
        g_b: str,
        rec_a: Dict[str, Any],
        rec_b: Dict[str, Any],
        epsilon: float,
        delta: float,
        n_total: int,
    ) -> PairCertificate:
        n_a = int(rec_a["total"])
        n_b = int(rec_b["total"])
        s_a = int(rec_a["scheduled"])
        s_b = int(rec_b["scheduled"])
        pi_a = n_a / max(n_total, 1)
        pi_b = n_b / max(n_total, 1)
        rate_a = s_a / max(n_a, 1)
        rate_b = s_b / max(n_b, 1)
        emp_gap = abs(rate_a - rate_b)

        # Wasserstein-1 DRO upper bound
        pi_min_a = max(pi_a, 1e-6)
        pi_min_b = max(pi_b, 1e-6)
        adv_inflation = epsilon * (1.0 / pi_min_a + 1.0 / pi_min_b)
        worst_case = emp_gap + adv_inflation

        # Finite-sample SE adjustment (two-sided, z_{α/2} at `confidence`)
        z = _z_score(self.confidence)
        se_a = math.sqrt(max(rate_a * (1.0 - rate_a) / max(n_a, 1), 0.0))
        se_b = math.sqrt(max(rate_b * (1.0 - rate_b) / max(n_b, 1), 0.0))
        se_gap = math.sqrt(se_a ** 2 + se_b ** 2)
        se_upper = worst_case + z * se_gap

        certified = bool(worst_case <= delta)
        certified_cons = bool(se_upper <= delta)
        slack = delta - worst_case

        return PairCertificate(
            group_a=str(g_a),
            group_b=str(g_b),
            n_a=n_a, n_b=n_b,
            pi_a=float(pi_a), pi_b=float(pi_b),
            rate_a=float(rate_a), rate_b=float(rate_b),
            empirical_gap=float(emp_gap),
            wasserstein_epsilon=float(epsilon),
            worst_case_gap=float(worst_case),
            se_adjusted_upper=float(se_upper),
            delta_budget=float(delta),
            certified=certified,
            certified_conservative=certified_cons,
            slack_to_budget=float(slack),
        )

    def _append_event(self, cert: DROFairnessCertificate) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": cert.computed_ts,
                "group_column": cert.group_column,
                "epsilon": cert.epsilon,
                "delta": cert.delta,
                "overall_certified": cert.overall_certified,
                "worst_pair_gap": cert.worst_pair_gap,
                "worst_pair": list(cert.worst_pair) if cert.worst_pair else None,
                "narrative": cert.narrative,
                "n_groups": cert.n_groups,
                "n_patients": cert.n_patients,
                "pairs": [
                    {
                        "group_a": pc.group_a,
                        "group_b": pc.group_b,
                        "empirical_gap": pc.empirical_gap,
                        "worst_case_gap": pc.worst_case_gap,
                        "certified": pc.certified,
                    }
                    for pc in cert.pair_certificates
                ],
            }
            with open(self.certificate_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"DRO fairness log write failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_by(
    patients: List[Dict[str, Any]],
    scheduled_set: set,
    group_column: str,
) -> Dict[str, Dict[str, int]]:
    """Bucket patients by group + count scheduled."""
    groups: Dict[str, Dict[str, int]] = {}
    for p in patients:
        pid = p.get("Patient_ID", p.get("patient_id", None))
        raw = p.get(group_column, None)
        if raw is None:
            # Try lower-case variant — schedule payloads sometimes use lower
            raw = p.get(group_column.lower(), "unknown")
        g = str(raw) if raw is not None else "unknown"
        rec = groups.setdefault(g, {"total": 0, "scheduled": 0})
        rec["total"] += 1
        if pid in scheduled_set:
            rec["scheduled"] += 1
    return groups


def _build_narrative(
    pair_certs: List[PairCertificate],
    overall: bool,
    overall_conservative: bool,
    worst_gap: float,
    worst_pair: Optional[Tuple[str, str]],
    epsilon: float,
    delta: float,
    group_column: str,
) -> str:
    if not pair_certs:
        return (
            f"DRO fairness check skipped: too few samples per group "
            f"for '{group_column}' (need >= {MIN_GROUP_SIZE})."
        )
    if overall:
        msg = (
            f"DRO certificate: PASS on '{group_column}' with Wasserstein "
            f"radius e={epsilon:.3f} and parity budget d={delta:.3f}; "
            f"worst-case pair gap = {worst_gap:.3f}"
        )
        if worst_pair:
            msg += f" (between '{worst_pair[0]}' and '{worst_pair[1]}')"
        msg += "."
        if not overall_conservative:
            msg += (
                "  Conservative (SE-inflated) bound would breach the "
                "budget -- consider a smaller epsilon or a larger "
                "sample before relying on this as a hard constraint."
            )
        return msg
    # Failure narrative — cite the pair with the LARGEST worst-case gap,
    # not just the first failing one, so the message matches `worst_pair_gap`.
    worst_pc = max(pair_certs, key=lambda pc: pc.worst_case_gap)
    msg = (
        f"DRO certificate: FAIL on '{group_column}' at e={epsilon:.3f}, "
        f"d={delta:.3f}."
    )
    msg += (
        f"  Worst pair '{worst_pc.group_a}' vs '{worst_pc.group_b}': "
        f"empirical gap = {worst_pc.empirical_gap:.3f}, "
        f"worst-case gap = {worst_pc.worst_case_gap:.3f} "
        f"(> {worst_pc.delta_budget:.3f})."
    )
    return msg


def _z_score(confidence: float) -> float:
    """Two-sided normal z-score for a given confidence level."""
    # Small lookup table; avoids scipy dependency in tests.
    table = {
        0.80: 1.2816,
        0.90: 1.6449,
        0.95: 1.9600,
        0.975: 2.2414,
        0.99: 2.5758,
    }
    # Nearest key
    if not table:
        return 1.96
    nearest = min(table.keys(), key=lambda k: abs(k - confidence))
    return float(table[nearest])


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors §3.1-§3.4 style)
# ---------------------------------------------------------------------------

_GLOBAL: Optional[DROFairnessCertifier] = None


def get_certifier() -> DROFairnessCertifier:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = DROFairnessCertifier()
    return _GLOBAL


def set_certifier(c: DROFairnessCertifier) -> None:
    global _GLOBAL
    _GLOBAL = c
