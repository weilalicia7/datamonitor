"""
Counterfactual Fairness Audit (Dissertation §4.4)
=================================================

For every \\emph{rejected} scheduling request, generate a
counterfactual by flipping a proxy-sensitive attribute and ask the
system: \\emph{would this patient have been scheduled if the only
thing that changed was their postcode?}  The brief gives the
motivating example:

    "Would this patient have been scheduled if their postcode were
     CF10 (affluent) instead of CF44 (deprived)?"

A high flip rate — rejects that would have been accepted under a
counterfactual affluent postcode — is evidence of \\emph{proxy
discrimination}: the optimiser is using postcode (or travel
distance / no-show rate, which correlate with postcode) as a proxy
for class or race.

Design
------
1. **Scheduleability predictor.**  A thin logistic regression on
   the historical ``(features, was\\_scheduled)`` pair.  Features
   are numeric-only and chosen so a counterfactual flip of
   postcode propagates through the pipeline:

   * ``age``, ``priority``, ``expected\\_duration`` — intrinsic.
   * ``deprivation\\_score`` (0--10, from a built-in Welsh
     postcode lookup) — the primary proxy.
   * ``distance\\_km``, ``travel\\_time\\_min`` — downstream effect
     of postcode on access.
   * ``no\\_show\\_rate`` — indirectly picks up deprivation via
     historical attendance.

   The model is fitted on-the-fly from whatever historical data
   the caller passes.  If scikit-learn is unavailable we fall
   back to a closed-form Platt-scaled mean-ratio heuristic.

2. **Postcode deprivation lookup.**  Small in-module dict,
   calibrated to the Welsh Index of Multiple Deprivation (WIMD
   2019) quintile scale; ``CF10``/``CF23``/``CF11`` are affluent
   Cardiff centre / bay areas, ``CF44``/``CF40``/``CF42``/``CF43``
   are the deprived Rhondda / Cynon Valley postcodes that sit in
   WIMD decile 1 or 2.  The lookup is overridable via
   ``POST /api/fairness/counterfactual/config`` --- operators
   should bring their own WIMD data for production use.

3. **Flip rule.**  The predictor produces $\\Pr(\\text{scheduled})$
   under the patient's own postcode $p_0$ and under a
   counterfactual $p_c$.  We declare a flip when

   .. math::
       \\Pr_{\\text{cf}} \\ge \\tau_{\\text{decision}} \\;
       \\text{AND}\\;
       \\Pr_{\\text{cf}} - \\Pr_{\\text{orig}} \\ge \\Delta_{\\min},

   where $\\tau_{\\text{decision}}$ is the median predicted
   probability of the actually-scheduled cohort and
   $\\Delta_{\\min} = 0.05$ is a minimum effect size.

4. **Integration.**  A certificate is auto-attached to
   ``/api/fairness/audit`` and every ``run\\_optimization()``; a
   narrative in the §4.4 brief's style is rendered.  JSONL log
   at ``data_cache/counterfactual_fairness/audits.jsonl``.  No UI
   panel.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_EFFECT_SIZE: float = 0.05
DEFAULT_FLIP_BUDGET: float = 0.10         # certified iff flip rate ≤ budget
DEFAULT_TOP_FLIPS: int = 10
DEFAULT_MIN_REJECTED: int = 5             # fewer than this ⇒ vacuous pass

# Decision-threshold safety bounds.  The threshold MUST sit strictly between
# 0 and 1 so the predictor's "flip" criterion (cf_prob ≥ threshold) is
# meaningful: 0 means every patient trivially flips; 1 means none ever can.
# Tightening to [0.05, 0.95] also keeps the heuristic predict_proba away
# from sigmoid saturation when it falls back to the mean-ratio path.
# Regression for §4.5.15: a degenerate y (all-rejected or all-scheduled
# input cohort) used to push the threshold to y.mean() ∈ {0.0, 1.0},
# which made every audit a vacuous PASS with delta_prob = 0 everywhere.
DECISION_THRESHOLD_FLOOR: float = 0.05
DECISION_THRESHOLD_CEILING: float = 0.95
DECISION_THRESHOLD_NEUTRAL: float = 0.5   # used when the cohort gives no signal
COUNTERFACTUAL_DIR: Path = DATA_CACHE_DIR / "counterfactual_fairness"
COUNTERFACTUAL_LOG: Path = COUNTERFACTUAL_DIR / "audits.jsonl"

# Welsh Index of Multiple Deprivation (WIMD) 2019 buckets
# Source: stats.wales.gov.uk/Catalogue/Community-Safety-and-Social-Inclusion/
#         Welsh-Index-of-Multiple-Deprivation
# This is a compact hand-curated map — operators in production should
# replace it with the authoritative LSOA-level lookup.
# Deprivation score: 0 (most affluent) … 10 (most deprived).
DEFAULT_POSTCODE_DEPRIVATION: Dict[str, float] = {
    # Cardiff — affluent centre / bay
    "CF10": 2.0,   # Cardiff Bay, city centre
    "CF11": 3.0,   # Canton / Grangetown mix
    "CF14": 3.5,   # Cardiff north, mixed
    "CF23": 2.5,   # Cyncoed / Pontprennau (affluent)
    "CF5":  3.5,   # Llandaff / Fairwater
    # Vale of Glamorgan — affluent suburbs
    "CF61": 2.0,   # Llantwit Major
    "CF62": 3.0,   # Barry
    "CF63": 3.5,
    "CF64": 2.5,   # Penarth (affluent)
    # Rhondda Cynon Taff — mid-to-deprived valleys
    "CF37": 7.0,   # Pontypridd
    "CF38": 6.5,
    "CF39": 8.0,   # Porth
    "CF40": 8.5,   # Tonypandy
    "CF41": 8.0,
    "CF42": 8.5,   # Treherbert
    "CF43": 8.0,   # Ferndale
    "CF44": 9.0,   # Aberdare — cited in the brief
    "CF45": 8.0,   # Mountain Ash
    "CF46": 7.5,
    "CF47": 7.5,   # Merthyr Tydfil
    "CF48": 7.0,
    # Bridgend / Porthcawl
    "CF31": 5.0,
    "CF32": 5.5,
    "CF33": 4.5,
    "CF34": 6.0,
    "CF35": 5.0,
    "CF36": 4.0,   # Porthcawl (affluent coast)
    # Newport — mixed
    "NP10": 5.5, "NP11": 6.5, "NP12": 6.5, "NP13": 5.5,
    "NP15": 3.5, "NP16": 3.0, "NP18": 3.5, "NP19": 6.0, "NP20": 5.5,
}
AFFLUENT_CUTOFF: float = 4.0
DEPRIVED_CUTOFF: float = 6.5

# Representative counterfactual postcodes
COUNTERFACTUAL_AFFLUENT: str = "CF10"
COUNTERFACTUAL_DEPRIVED: str = "CF44"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FlipCase:
    patient_id: str
    original_postcode: str
    original_deprivation: float
    counterfactual_postcode: str
    counterfactual_deprivation: float
    original_prob: float
    counterfactual_prob: float
    delta_prob: float                 # cf - orig
    would_flip: bool                  # crosses decision threshold AND delta >= min effect
    flipped_features: List[str]       # feature names the flip changed


@dataclass
class CounterfactualFairnessCertificate:
    computed_ts: str
    n_rejected: int
    n_flipped: int
    flip_rate: float                  # n_flipped / n_rejected
    flip_budget: float
    certified: bool                   # flip_rate <= budget
    decision_threshold: float         # median predicted prob of scheduled cohort
    counterfactual_postcode: str
    narrative: str
    top_flips: List[FlipCase] = field(default_factory=list)
    mean_delta_prob: float = 0.0
    max_delta_prob: float = 0.0
    affluent_deprivation: float = 0.0
    predictor_method: str = "logistic_regression"
    predictor_coefficients: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scheduleability predictor
# ---------------------------------------------------------------------------


class ScheduleabilityPredictor:
    """
    A thin logistic regression on numeric features.  Falls back to
    a mean-ratio heuristic if scikit-learn is unavailable.
    """

    FEATURES: Tuple[str, ...] = (
        "age",
        "priority",
        "expected_duration",
        "deprivation_score",
        "distance_km",
        "travel_time_min",
        "no_show_rate",
    )

    def __init__(self):
        self._coef: Optional[Dict[str, float]] = None
        self._intercept: float = 0.0
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}
        self._decision_threshold: float = 0.5
        self._method: str = "uninitialised"

    def fit(
        self,
        patients: Sequence[Dict[str, Any]],
        scheduled_ids: set,
        postcode_deprivation: Dict[str, float],
    ) -> "ScheduleabilityPredictor":
        X, y = _build_xy(patients, scheduled_ids, postcode_deprivation, self.FEATURES)
        if X.shape[0] < 10 or y.sum() == 0 or (y == 0).sum() == 0:
            # Degenerate: at least one class missing.  Fall back to a
            # neutral threshold (NOT y.mean(), which would be 0 or 1 here
            # and make every audit a vacuous "PASS" with delta_prob = 0
            # everywhere — see §4.5.15 regression test).  Log loudly so
            # operators see why the audit will be vacuous.
            logger.warning(
                "ScheduleabilityPredictor.fit on degenerate input: "
                f"n={X.shape[0]}, n_scheduled={int(y.sum())}, "
                f"n_rejected={int((y == 0).sum())}; "
                f"falling back to neutral threshold {DECISION_THRESHOLD_NEUTRAL} "
                "(audit will report vacuously)."
            )
            self._method = "degenerate_fallback"
            self._means = {
                f: float(X[:, k].mean()) for k, f in enumerate(self.FEATURES)
            } if X.size else {}
            self._decision_threshold = DECISION_THRESHOLD_NEUTRAL
            return self

        # Standardise for numerical stability
        self._means = {f: float(X[:, k].mean()) for k, f in enumerate(self.FEATURES)}
        self._stds = {f: float(max(X[:, k].std(), 1e-6)) for k, f in enumerate(self.FEATURES)}
        Xs = np.zeros_like(X, dtype=float)
        for k, f in enumerate(self.FEATURES):
            Xs[:, k] = (X[:, k] - self._means[f]) / self._stds[f]

        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
            clf.fit(Xs, y)
            coef = clf.coef_[0]
            self._coef = {f: float(coef[k]) for k, f in enumerate(self.FEATURES)}
            self._intercept = float(clf.intercept_[0])
            probs_all = clf.predict_proba(Xs)[:, 1]
            raw_threshold = float(
                np.median(probs_all[y == 1]) if (y == 1).any() else DECISION_THRESHOLD_NEUTRAL
            )
            self._decision_threshold = _clamp_threshold(raw_threshold)
            self._method = "logistic_regression"
        except Exception as exc:
            logger.warning(f"sklearn LogReg unavailable ({exc}); using mean-ratio fallback")
            self._method = "mean_ratio"
            # Use the base rate as a starting point, then clamp to keep the
            # threshold strictly inside (0, 1) — degenerate y.mean() values
            # of 0.0 or 1.0 would otherwise saturate the predictor and zero
            # out every delta_prob in the downstream audit.
            self._decision_threshold = _clamp_threshold(float(y.mean()))
            # Tiny approximation: negative β on deprivation, negative β on distance
            # scaled to their inverse std — produces sensible flip signals.
            self._coef = {
                "age": 0.0,
                "priority": -0.5,
                "expected_duration": 0.0,
                "deprivation_score": -0.5,
                "distance_km": -0.3,
                "travel_time_min": -0.2,
                "no_show_rate": -0.5,
            }
            self._intercept = float(math.log(
                self._decision_threshold / (1 - self._decision_threshold)
            ))
        return self

    def predict_proba(self, feature_dict: Dict[str, float]) -> float:
        if self._coef is None:
            return float(self._decision_threshold)
        z = self._intercept
        for f, b in self._coef.items():
            v = float(feature_dict.get(f, self._means.get(f, 0.0)) or 0.0)
            if self._stds:
                std = self._stds.get(f, 1.0)
                z += b * ((v - self._means.get(f, 0.0)) / max(std, 1e-6))
            else:
                z += b * v
        # Clip z to avoid overflow
        z = max(min(z, 40.0), -40.0)
        return 1.0 / (1.0 + math.exp(-z))

    @property
    def decision_threshold(self) -> float:
        return self._decision_threshold

    @property
    def method(self) -> str:
        return self._method

    @property
    def coefficients(self) -> Dict[str, float]:
        return dict(self._coef or {})


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


class CounterfactualFairnessAuditor:
    """Audit rejected scheduling requests for proxy discrimination."""

    def __init__(
        self,
        *,
        postcode_deprivation: Optional[Dict[str, float]] = None,
        counterfactual_postcode: str = COUNTERFACTUAL_AFFLUENT,
        min_effect_size: float = DEFAULT_MIN_EFFECT_SIZE,
        flip_budget: float = DEFAULT_FLIP_BUDGET,
        top_flips: int = DEFAULT_TOP_FLIPS,
        flip_downstream: bool = False,
        storage_dir: Path = COUNTERFACTUAL_DIR,
    ):
        self.postcode_deprivation: Dict[str, float] = dict(
            postcode_deprivation or DEFAULT_POSTCODE_DEPRIVATION
        )
        self.counterfactual_postcode = str(counterfactual_postcode)
        self.min_effect_size = float(min_effect_size)
        self.flip_budget = float(flip_budget)
        self.top_flips = int(top_flips)
        self.flip_downstream = bool(flip_downstream)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.audits_log = self.storage_dir / "audits.jsonl"

        self._lock = threading.Lock()
        self._last: Optional[CounterfactualFairnessCertificate] = None
        self._n_runs: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def audit(
        self,
        patients: Sequence[Dict[str, Any]],
        scheduled_ids: Sequence[Any],
        rejected_ids: Optional[Sequence[Any]] = None,
        counterfactual_postcode: Optional[str] = None,
        min_effect_size: Optional[float] = None,
    ) -> CounterfactualFairnessCertificate:
        cf_postcode = str(counterfactual_postcode or self.counterfactual_postcode)
        min_eff = float(
            self.min_effect_size if min_effect_size is None else min_effect_size
        )
        sched_set = set(
            _pid(x) for x in scheduled_ids
        )
        if rejected_ids is None:
            rejected_ids = [
                _pid(p) for p in patients if _pid(p) not in sched_set
            ]
        rej_set = set(_pid(x) for x in rejected_ids)

        # Fit the predictor on the full (patients, scheduled) data
        predictor = ScheduleabilityPredictor().fit(
            patients, sched_set, self.postcode_deprivation,
        )

        # Build a fast index of patients by id
        pat_by_id: Dict[str, Dict[str, Any]] = {}
        for p in patients:
            pid = _pid(p)
            if pid is None:
                continue
            pat_by_id[pid] = p

        # Pre-compute the counterfactual deprivation
        cf_depr = float(self.postcode_deprivation.get(cf_postcode, 2.0))

        flips: List[FlipCase] = []
        n_rejected = 0
        deltas: List[float] = []

        for pid in rej_set:
            p = pat_by_id.get(pid)
            if p is None:
                continue
            orig_post = _postcode(p)
            orig_depr = float(self.postcode_deprivation.get(orig_post, 5.0))
            # Skip patients whose postcode is already in the affluent bucket —
            # the brief asks about deprived->affluent flips.
            if orig_depr <= AFFLUENT_CUTOFF:
                continue

            n_rejected += 1

            orig_feats = _features(p, self.postcode_deprivation)
            # Counterfactual: flip ONLY the deprivation label.  This is the
            # "proxy discrimination" reading of the brief's question — we want
            # to isolate the effect of the postcode *label* on the decision,
            # holding the patient's actual distance / travel / no-show fixed.
            # (If an operator wants the richer "would their circumstances have
            # been different?" reading, they can pass flip_downstream=True.)
            cf_feats = dict(orig_feats)
            cf_feats["deprivation_score"] = cf_depr
            if self.flip_downstream and cf_depr < orig_depr:
                cf_feats["distance_km"] = orig_feats.get("distance_km", 15.0) * 0.6
                cf_feats["travel_time_min"] = orig_feats.get("travel_time_min", 30.0) * 0.6
                cf_feats["no_show_rate"] = max(
                    0.0, orig_feats.get("no_show_rate", 0.15) - 0.05
                )

            orig_prob = predictor.predict_proba(orig_feats)
            cf_prob = predictor.predict_proba(cf_feats)
            delta = cf_prob - orig_prob
            deltas.append(delta)

            would_flip = (
                cf_prob >= predictor.decision_threshold
                and delta >= min_eff
            )

            flipped_features = [
                f for f in ("deprivation_score", "distance_km",
                            "travel_time_min", "no_show_rate")
                if abs(float(orig_feats.get(f, 0) or 0)
                       - float(cf_feats.get(f, 0) or 0)) > 1e-9
            ]

            flips.append(FlipCase(
                patient_id=pid,
                original_postcode=orig_post,
                original_deprivation=orig_depr,
                counterfactual_postcode=cf_postcode,
                counterfactual_deprivation=cf_depr,
                original_prob=float(orig_prob),
                counterfactual_prob=float(cf_prob),
                delta_prob=float(delta),
                would_flip=bool(would_flip),
                flipped_features=flipped_features,
            ))

        n_flipped = sum(1 for f in flips if f.would_flip)
        rate = n_flipped / max(n_rejected, 1)
        certified = rate <= self.flip_budget

        # Keep top flips by delta_prob (most suspicious)
        flips.sort(key=lambda f: f.delta_prob, reverse=True)
        top = [f for f in flips if f.would_flip][: self.top_flips]
        if not top:
            top = flips[: self.top_flips]

        narrative = _build_narrative(
            n_rejected=n_rejected, n_flipped=n_flipped,
            rate=rate, certified=certified,
            cf_postcode=cf_postcode, affluent_depr=cf_depr,
            min_effect=min_eff, top=top[0] if top else None,
            predictor_method=predictor.method,
        )

        max_delta = max(deltas) if deltas else 0.0
        mean_delta = float(np.mean(deltas)) if deltas else 0.0

        cert = CounterfactualFairnessCertificate(
            computed_ts=datetime.utcnow().isoformat(timespec="seconds"),
            n_rejected=int(n_rejected),
            n_flipped=int(n_flipped),
            flip_rate=float(rate),
            flip_budget=float(self.flip_budget),
            certified=bool(certified),
            decision_threshold=float(predictor.decision_threshold),
            counterfactual_postcode=cf_postcode,
            narrative=narrative,
            top_flips=top,
            mean_delta_prob=float(mean_delta),
            max_delta_prob=float(max_delta),
            affluent_deprivation=float(cf_depr),
            predictor_method=predictor.method,
            predictor_coefficients=predictor.coefficients,
        )
        with self._lock:
            self._last = cert
            self._n_runs += 1
        self._append_event(cert)
        return cert

    def last(self) -> Optional[CounterfactualFairnessCertificate]:
        with self._lock:
            return self._last

    def status(self) -> Dict[str, Any]:
        last = self.last()
        return {
            "counterfactual_postcode": self.counterfactual_postcode,
            "min_effect_size": self.min_effect_size,
            "flip_budget": self.flip_budget,
            "top_flips": self.top_flips,
            "n_known_postcodes": len(self.postcode_deprivation),
            "affluent_cutoff": AFFLUENT_CUTOFF,
            "deprived_cutoff": DEPRIVED_CUTOFF,
            "total_runs": self._n_runs,
            "log_path": str(self.audits_log),
            "last_computed_ts": last.computed_ts if last else None,
            "last_certified": last.certified if last else None,
            "last_n_rejected": last.n_rejected if last else None,
            "last_n_flipped": last.n_flipped if last else None,
            "last_flip_rate": last.flip_rate if last else None,
            "last_narrative": last.narrative if last else None,
            "last_predictor_method": last.predictor_method if last else None,
        }

    def update_config(
        self,
        counterfactual_postcode: Optional[str] = None,
        min_effect_size: Optional[float] = None,
        flip_budget: Optional[float] = None,
        postcode_deprivation: Optional[Dict[str, float]] = None,
        flip_downstream: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if counterfactual_postcode is not None:
            self.counterfactual_postcode = str(counterfactual_postcode)
        if min_effect_size is not None:
            self.min_effect_size = float(min_effect_size)
        if flip_budget is not None:
            self.flip_budget = float(flip_budget)
        if postcode_deprivation is not None:
            self.postcode_deprivation.update(postcode_deprivation)
        if flip_downstream is not None:
            self.flip_downstream = bool(flip_downstream)
        return {
            "counterfactual_postcode": self.counterfactual_postcode,
            "min_effect_size": self.min_effect_size,
            "flip_budget": self.flip_budget,
            "flip_downstream": self.flip_downstream,
            "n_known_postcodes": len(self.postcode_deprivation),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _append_event(self, cert: CounterfactualFairnessCertificate) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": cert.computed_ts,
                "n_rejected": cert.n_rejected,
                "n_flipped": cert.n_flipped,
                "flip_rate": cert.flip_rate,
                "certified": cert.certified,
                "decision_threshold": cert.decision_threshold,
                "counterfactual_postcode": cert.counterfactual_postcode,
                "narrative": cert.narrative,
                "predictor_method": cert.predictor_method,
                "mean_delta_prob": cert.mean_delta_prob,
                "max_delta_prob": cert.max_delta_prob,
                "top_flips": [
                    {
                        "patient_id": f.patient_id,
                        "original_postcode": f.original_postcode,
                        "delta_prob": f.delta_prob,
                        "would_flip": f.would_flip,
                    }
                    for f in cert.top_flips
                ],
            }
            with open(self.audits_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Counterfactual audit log write failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_threshold(value: float) -> float:
    """
    Clamp a candidate decision threshold into the safe range
    ``[DECISION_THRESHOLD_FLOOR, DECISION_THRESHOLD_CEILING]``.  If the
    candidate is NaN or missing, return ``DECISION_THRESHOLD_NEUTRAL``.

    The audit's "would_flip" check is ``cf_prob >= threshold``; a
    threshold of 0 means every patient trivially flips and a threshold
    of 1 means none ever can — both make the certificate vacuous.
    Clamping at the source guarantees ``decision_threshold ∈ (0, 1)``
    on every code path, which is exactly the §4.5.15 invariant the
    dissertation table claims.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return DECISION_THRESHOLD_NEUTRAL
    if not math.isfinite(v):
        return DECISION_THRESHOLD_NEUTRAL
    if v < DECISION_THRESHOLD_FLOOR:
        return DECISION_THRESHOLD_FLOOR
    if v > DECISION_THRESHOLD_CEILING:
        return DECISION_THRESHOLD_CEILING
    return v


def _pid(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("Patient_ID") or v.get("patient_id")
    if isinstance(v, str):
        return v
    return getattr(v, "patient_id", None)


def _postcode(p: Dict[str, Any]) -> str:
    # Prefer Patient_Postcode, fall back to postcode
    pc = (
        p.get("Patient_Postcode")
        or p.get("postcode")
        or p.get("Postcode")
        or ""
    )
    if isinstance(pc, str):
        pc = pc.strip().upper().split()[0] if pc.strip() else ""
    return str(pc)


def _features(p: Dict[str, Any], postcode_depr: Dict[str, float]) -> Dict[str, float]:
    postcode = _postcode(p)
    depr = float(postcode_depr.get(postcode, 5.0))

    age_val = p.get("Age") or p.get("age")
    if age_val is None:
        age_band = str(p.get("Age_Band", "") or "")
        age_val = {"<40": 35, "40-60": 50, "60-75": 67, ">75": 82}.get(age_band, 60)
    try:
        age_val = float(age_val)
    except Exception:
        age_val = 60.0

    prio = p.get("Priority") or p.get("priority") or 3
    if isinstance(prio, str):
        if len(prio) >= 2 and prio[0] in "Pp" and prio[1:].isdigit():
            prio = int(prio[1:])
        else:
            try:
                prio = int(prio)
            except Exception:
                prio = 3
    try:
        prio = int(prio)
    except Exception:
        prio = 3

    return {
        "age": float(age_val),
        "priority": float(prio),
        "expected_duration": float(p.get("Planned_Duration")
                                   or p.get("expected_duration") or 60.0),
        "deprivation_score": depr,
        "distance_km": float(p.get("Distance_km")
                             or p.get("distance_km") or 15.0),
        "travel_time_min": float(p.get("Travel_Time_Min")
                                 or p.get("travel_time_minutes") or 30.0),
        "no_show_rate": float(p.get("Patient_NoShow_Rate")
                              or p.get("no_show_rate") or 0.15),
    }


def _build_xy(
    patients: Sequence[Dict[str, Any]],
    scheduled_set: set,
    postcode_depr: Dict[str, float],
    features: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for p in patients:
        pid = _pid(p)
        if pid is None:
            continue
        f = _features(p, postcode_depr)
        rows.append([float(f.get(col, 0.0)) for col in features])
        labels.append(1 if pid in scheduled_set else 0)
    if not rows:
        return np.zeros((0, len(features))), np.zeros(0, dtype=int)
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def _build_narrative(
    *, n_rejected: int, n_flipped: int, rate: float,
    certified: bool, cf_postcode: str, affluent_depr: float,
    min_effect: float, top: Optional[FlipCase],
    predictor_method: str = "logistic_regression",
) -> str:
    if predictor_method == "degenerate_fallback":
        return (
            f"Counterfactual audit: vacuously PASS -- predictor fitted on "
            f"degenerate input (only one class present in the scheduling "
            f"signal, so no Pr(scheduled) signal to compare against). "
            f"Decision threshold defaulted to neutral; re-run once a real "
            f"schedule has been computed."
        )
    if n_rejected < DEFAULT_MIN_REJECTED:
        return (
            f"Counterfactual audit: vacuously PASS -- only {n_rejected} "
            f"rejected patients (need >= {DEFAULT_MIN_REJECTED})."
        )
    if certified:
        msg = (
            f"Counterfactual audit: PASS -- {n_flipped}/{n_rejected} "
            f"({rate * 100:.1f}%) rejects would have been scheduled at "
            f"postcode '{cf_postcode}'; within flip-budget."
        )
    else:
        msg = (
            f"Counterfactual audit: FAIL -- {n_flipped}/{n_rejected} "
            f"({rate * 100:.1f}%) rejects would have been scheduled at "
            f"affluent postcode '{cf_postcode}' (WIMD={affluent_depr:.1f}); "
            f"proxy-discrimination signal exceeds flip-budget."
        )
    if top is not None and top.would_flip:
        msg += (
            f"  Most-suspicious case: patient '{top.patient_id}' at "
            f"'{top.original_postcode}' (WIMD={top.original_deprivation:.1f}) "
            f"would have Pr(scheduled) jump from {top.original_prob:.2f} to "
            f"{top.counterfactual_prob:.2f} (delta={top.delta_prob:+.2f})."
        )
    return msg


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_GLOBAL: Optional[CounterfactualFairnessAuditor] = None


def get_auditor() -> CounterfactualFairnessAuditor:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = CounterfactualFairnessAuditor()
    return _GLOBAL


def set_auditor(a: CounterfactualFairnessAuditor) -> None:
    global _GLOBAL
    _GLOBAL = a
