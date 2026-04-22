"""
Human-in-the-Loop Override Learning (Dissertation §5.2)
======================================================

Every time a clinician manually moves or cancels an appointment the
scheduler has just produced, the override is a fragment of
preference information: something about the proposed slot was
wrong.  S5.2 asks us to harvest this signal --- log every
override, fit a model that predicts override probability, and
preemptively suggest the clinician's preferred alternative when
$Pr(text{override} \\mid text{slot}) > 0.80$.  The system
becomes an assistant that defers to its human overseers, rather
than a black box they have to fight.

Design contract
---------------
1. **Event log**.  ``OverrideLearner.log_event`` appends one
   JSONL row per override to
   ``data_cache/override_learning/events.jsonl``.  Events are
   structured as :class:`OverrideEvent` with the original
   appointment, the corrected appointment, and a clinician-supplied
   reason.
2. **Model**.  When the log reaches ``min_events_for_fit`` rows we
   fit a logistic regression on
   :math:`Pr(text{override} = 1 \\mid text{features})` where
   features are the appointment's hour-of-day, day-of-week, site
   code, priority, planned duration, and forecasted no-show rate.
   sklearn is the primary path; a count-based mean-rate heuristic
   provides a safe fallback when sklearn is absent or degenerate
   (all labels the same).
3. **Suggestion**.  For each incoming appointment we compute
   ``p = predict_override_probability(appt)``.  If
   :math:`p ge tau_{text{suggest}}` (default 0.80) we search
   a small neighbourhood of candidate alternatives (adjacent hours,
   sibling chairs) and return the one with the lowest
   override probability --- the clinician's emph{preferred}
   slot, inferred from their own past corrections.
4. **Invisibility**.  Flask wires ``run_optimization`` so every
   optimisation result carries an ``override_suggestions`` block;
   no UI panel.  Diagnostics on
   ``/api/override/status``, ``/api/override/last``,
   ``/api/override/log``, ``/api/override/suggest``, and
   ``/api/override/config``.

Positive (emph{negative}) labels
---------------------------------
* Label **1** = override happened (the scheduler got it wrong).
* Label **0** = implicit acceptance (the scheduler's proposal was
  kept).  Because the system only sees overrides directly, the
  negative class is synthesised from the final
  ``app_state['appointments']`` list: an appointment that ends
  up in the committed schedule and has emph{no} corresponding
  override event is implicitly labelled 0.
"""

from __future__ import annotations

import json
import math
import pickle
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SUGGEST_THRESHOLD: float = 0.80      # brief calls for > 80%
DEFAULT_MIN_EVENTS_FOR_FIT: int = 20
DEFAULT_NEIGHBOURHOOD_HOURS: Tuple[int, ...] = (-2, -1, 1, 2)
DEFAULT_COLD_START_PRIOR: float = 0.10       # base rate when model is untrained
OVERRIDE_DIR: Path = DATA_CACHE_DIR / "override_learning"
EVENT_LOG_FILE: Path = OVERRIDE_DIR / "events.jsonl"
MODEL_FILE: Path = OVERRIDE_DIR / "model.pkl"
SUGGESTIONS_LOG_FILE: Path = OVERRIDE_DIR / "suggestions.jsonl"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OverrideEvent:
    """A clinician moved / cancelled / rescheduled an appointment."""
    ts: str
    patient_id: str
    original_chair_id: Optional[str]
    original_start_time: Optional[str]
    original_duration: Optional[int]
    new_chair_id: Optional[str]
    new_start_time: Optional[str]
    new_duration: Optional[int]
    priority: int = 3
    site_code: Optional[str] = None
    noshow_prob: Optional[float] = None
    clinician_id: Optional[str] = None
    reason: str = ""


@dataclass
class OverrideSuggestion:
    """Preemptive alternative when Pr(override) exceeds threshold."""
    patient_id: str
    original_chair_id: Optional[str]
    original_start_time: Optional[str]
    original_override_prob: float
    suggested_chair_id: Optional[str]
    suggested_start_time: Optional[str]
    suggested_override_prob: float
    delta_prob: float                  # original - suggested (positive = improvement)
    reason: str                        # human-readable rationale


@dataclass
class OverrideStatus:
    """Shape of /api/override/status."""
    n_events: int
    n_suggestions_generated: int
    model_method: str                  # 'logistic_regression' | 'count_rate' | 'cold_start'
    is_trained: bool
    suggest_threshold: float
    min_events_for_fit: int
    cold_start_prior: float
    last_fit_ts: Optional[str]
    last_suggestion_ts: Optional[str]
    last_narrative: Optional[str]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


FEATURE_COLS: Tuple[str, ...] = (
    "hour_of_day",
    "day_of_week",
    "priority",
    "duration_min",
    "noshow_prob",
    "site_idx",
)


def _extract_features(appt: Dict[str, Any]) -> Dict[str, float]:
    """Return numeric feature dict for a proposed appointment."""
    start = _coerce_dt(appt.get("start_time"))
    hour = start.hour if start else 12
    dow = start.weekday() if start else 2
    dur = appt.get("duration") or appt.get("expected_duration") or 60
    try:
        dur = int(dur)
    except Exception:
        dur = 60
    prio = appt.get("priority")
    if prio is None:
        prio = 3
    try:
        prio = int(prio)
    except Exception:
        if isinstance(prio, str) and len(prio) > 1 and prio[0] in "Pp":
            try:
                prio = int(prio[1:])
            except Exception:
                prio = 3
        else:
            prio = 3
    noshow = appt.get("noshow_prob")
    if noshow is None:
        noshow = 0.15
    try:
        noshow = float(noshow)
    except Exception:
        noshow = 0.15
    site_code = appt.get("site_code") or appt.get("site") or "UNKNOWN"
    site_idx = _hash_bucket(str(site_code))
    return {
        "hour_of_day": float(hour),
        "day_of_week": float(dow),
        "priority": float(prio),
        "duration_min": float(dur),
        "noshow_prob": float(max(0.0, min(1.0, noshow))),
        "site_idx": float(site_idx),
    }


def _hash_bucket(s: str, n_buckets: int = 20) -> int:
    return int(abs(hash(s)) % n_buckets)


def _coerce_dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except Exception:
            try:
                base = datetime.now().replace(hour=0, minute=0,
                                              second=0, microsecond=0)
                parts = v.strip().split(":")
                hh = int(parts[0])
                mm = int(parts[1]) if len(parts) > 1 else 0
                return base.replace(hour=hh, minute=mm)
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# OverrideLearner
# ---------------------------------------------------------------------------


class OverrideLearner:
    """Harvest clinician overrides, predict override probability, and
    suggest preferred alternatives."""

    def __init__(
        self,
        *,
        suggest_threshold: float = DEFAULT_SUGGEST_THRESHOLD,
        min_events_for_fit: int = DEFAULT_MIN_EVENTS_FOR_FIT,
        neighbourhood_hours: Sequence[int] = DEFAULT_NEIGHBOURHOOD_HOURS,
        cold_start_prior: float = DEFAULT_COLD_START_PRIOR,
        storage_dir: Path = OVERRIDE_DIR,
    ):
        self.suggest_threshold = float(suggest_threshold)
        self.min_events_for_fit = int(min_events_for_fit)
        self.neighbourhood_hours = tuple(int(h) for h in neighbourhood_hours)
        self.cold_start_prior = float(cold_start_prior)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.events_log = self.storage_dir / "events.jsonl"
        self.suggestions_log = self.storage_dir / "suggestions.jsonl"
        self.model_path = self.storage_dir / "model.pkl"

        self._lock = threading.Lock()
        self._events: List[OverrideEvent] = []
        self._accepted_appts: List[Dict[str, Any]] = []
        self._model = None
        self._feature_means: Dict[str, float] = {}
        self._feature_stds: Dict[str, float] = {}
        self._model_method = "cold_start"
        self._is_trained = False
        self._last_fit_ts: Optional[str] = None
        self._last_suggestion_ts: Optional[str] = None
        self._last_narrative: Optional[str] = None
        self._n_suggestions_generated: int = 0

        self._load_events()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def log_event(self, event: OverrideEvent) -> None:
        """Record one override event and (optionally) refit the model."""
        with self._lock:
            self._events.append(event)
        self._append_event_log(event)
        # Refit eagerly once we hit the threshold
        if len(self._events) >= self.min_events_for_fit and not self._is_trained:
            self.fit()

    def register_accepted_appointments(
        self, appts: Sequence[Dict[str, Any]]
    ) -> None:
        """
        Tell the learner which appointments were committed without override.

        This provides the negative class for fitting — appointments that
        made it into app_state['appointments'] without a subsequent
        OverrideEvent are implicit 0s.
        """
        with self._lock:
            self._accepted_appts = [dict(a) for a in appts]

    def fit(self) -> bool:
        """Fit the override-probability model.  Returns True on success."""
        with self._lock:
            events = list(self._events)
            accepted = list(self._accepted_appts)

        if len(events) < self.min_events_for_fit:
            self._model_method = "cold_start"
            self._is_trained = False
            return False

        # Build (X, y) from events + negatives
        X, y = self._build_xy(events, accepted)
        if X.shape[0] < self.min_events_for_fit or y.sum() == 0 or (y == 0).sum() == 0:
            # Degenerate (all positive or all negative) -> count-rate fallback
            self._fit_count_rate(events)
            return True

        # Standardise
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds = np.where(stds < 1e-6, 1.0, stds)
        Xs = (X - means) / stds
        self._feature_means = {c: float(means[k]) for k, c in enumerate(FEATURE_COLS)}
        self._feature_stds = {c: float(stds[k]) for k, c in enumerate(FEATURE_COLS)}

        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=300, C=1.0)
            clf.fit(Xs, y)
            self._model = {
                "type": "logistic_regression",
                "coef": [float(c) for c in clf.coef_[0]],
                "intercept": float(clf.intercept_[0]),
                "features": list(FEATURE_COLS),
            }
            self._model_method = "logistic_regression"
        except Exception as exc:
            logger.warning(f"sklearn LogReg unavailable ({exc}); count-rate fallback")
            self._fit_count_rate(events)
            return True

        self._is_trained = True
        self._last_fit_ts = datetime.utcnow().isoformat(timespec="seconds")
        self._persist_model()
        return True

    def predict_override_probability(
        self, appt: Dict[str, Any]
    ) -> float:
        """Return :math:`Pr(text{override} \\mid text{slot})`."""
        if not self._is_trained:
            return self.cold_start_prior
        feats = _extract_features(appt)
        if self._model_method == "logistic_regression":
            z = float(self._model["intercept"])
            for k, col in enumerate(self._model["features"]):
                mean = self._feature_means.get(col, 0.0)
                std = self._feature_stds.get(col, 1.0)
                v = feats.get(col, 0.0)
                z += self._model["coef"][k] * ((v - mean) / std)
            z = max(min(z, 40.0), -40.0)
            return 1.0 / (1.0 + math.exp(-z))
        if self._model_method == "count_rate":
            key = self._count_key(feats)
            rates = self._model.get("rates", {})
            return float(rates.get(key, self.cold_start_prior))
        return self.cold_start_prior

    def suggest(
        self, appt: Dict[str, Any], alternatives: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[OverrideSuggestion]:
        """Return an ``OverrideSuggestion`` if Pr(override) > threshold.

        If ``alternatives`` is provided, pick the best from that set.
        Otherwise generate adjacent-hour candidates automatically.
        """
        p = self.predict_override_probability(appt)
        if p < self.suggest_threshold:
            return None

        candidates = list(alternatives) if alternatives is not None else \
            self._generate_neighbourhood(appt)
        if not candidates:
            return None

        # Score every candidate; pick the one with the lowest override prob
        scored = sorted(
            candidates,
            key=lambda c: self.predict_override_probability(c),
        )
        best = scored[0]
        p_best = self.predict_override_probability(best)
        if p_best >= p - 1e-9:
            # No improvement available
            return None

        suggestion = OverrideSuggestion(
            patient_id=str(appt.get("patient_id") or ""),
            original_chair_id=appt.get("chair_id"),
            original_start_time=_fmt(appt.get("start_time")),
            original_override_prob=float(p),
            suggested_chair_id=best.get("chair_id"),
            suggested_start_time=_fmt(best.get("start_time")),
            suggested_override_prob=float(p_best),
            delta_prob=float(p - p_best),
            reason=(
                f"Pr(override) on current slot = {p:.2f} exceeds "
                f"threshold {self.suggest_threshold:.2f}; alternative "
                f"reduces it to {p_best:.2f} ({p - p_best:+.2f})."
            ),
        )
        self._append_suggestion_log(suggestion)
        with self._lock:
            self._n_suggestions_generated += 1
            self._last_suggestion_ts = suggestion.original_start_time
            self._last_narrative = suggestion.reason
        return suggestion

    def status(self) -> OverrideStatus:
        with self._lock:
            n_events = len(self._events)
        return OverrideStatus(
            n_events=n_events,
            n_suggestions_generated=self._n_suggestions_generated,
            model_method=self._model_method,
            is_trained=self._is_trained,
            suggest_threshold=self.suggest_threshold,
            min_events_for_fit=self.min_events_for_fit,
            cold_start_prior=self.cold_start_prior,
            last_fit_ts=self._last_fit_ts,
            last_suggestion_ts=self._last_suggestion_ts,
            last_narrative=self._last_narrative,
        )

    def update_config(
        self,
        suggest_threshold: Optional[float] = None,
        min_events_for_fit: Optional[int] = None,
        cold_start_prior: Optional[float] = None,
        neighbourhood_hours: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        if suggest_threshold is not None:
            self.suggest_threshold = float(suggest_threshold)
        if min_events_for_fit is not None:
            self.min_events_for_fit = int(min_events_for_fit)
        if cold_start_prior is not None:
            self.cold_start_prior = float(cold_start_prior)
        if neighbourhood_hours is not None:
            self.neighbourhood_hours = tuple(int(h) for h in neighbourhood_hours)
        return {
            "suggest_threshold": self.suggest_threshold,
            "min_events_for_fit": self.min_events_for_fit,
            "cold_start_prior": self.cold_start_prior,
            "neighbourhood_hours": list(self.neighbourhood_hours),
        }

    def get_events(self) -> List[OverrideEvent]:
        with self._lock:
            return list(self._events)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_xy(
        self, events: List[OverrideEvent], accepted: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble (features, label) where label=1 for overrides, 0 for accepted.

        To stay balanced when one class dominates, we undersample the
        majority class to at most 4x the minority class size.
        """
        pos_rows: List[List[float]] = []
        for e in events:
            appt = {
                "start_time": e.original_start_time,
                "duration": e.original_duration,
                "priority": e.priority,
                "noshow_prob": e.noshow_prob,
                "site_code": e.site_code,
            }
            f = _extract_features(appt)
            pos_rows.append([f[c] for c in FEATURE_COLS])

        neg_rows: List[List[float]] = []
        override_ids = {e.patient_id for e in events}
        for a in accepted:
            if str(a.get("patient_id")) in override_ids:
                continue
            f = _extract_features(a)
            neg_rows.append([f[c] for c in FEATURE_COLS])

        if not pos_rows:
            return np.zeros((0, len(FEATURE_COLS))), np.zeros(0, dtype=int)

        # Balance: cap negatives at 4x positives
        if neg_rows and len(neg_rows) > 4 * len(pos_rows):
            idx = np.random.RandomState(42).choice(
                len(neg_rows), size=4 * len(pos_rows), replace=False
            )
            neg_rows = [neg_rows[i] for i in idx]

        X = np.asarray(pos_rows + neg_rows, dtype=float)
        y = np.asarray([1] * len(pos_rows) + [0] * len(neg_rows), dtype=int)
        return X, y

    def _fit_count_rate(self, events: List[OverrideEvent]) -> None:
        """Simple count-based override rate per (hour-bucket, priority, site)."""
        with self._lock:
            accepted = list(self._accepted_appts)
        # Count each (hour, priority, site) override occurrence
        buckets: Dict[str, Dict[str, int]] = {}
        for e in events:
            f = _extract_features({
                "start_time": e.original_start_time,
                "priority": e.priority,
                "site_code": e.site_code,
            })
            key = self._count_key(f)
            buckets.setdefault(key, {"over": 0, "seen": 0})
            buckets[key]["over"] += 1
            buckets[key]["seen"] += 1
        override_ids = {e.patient_id for e in events}
        for a in accepted:
            if str(a.get("patient_id")) in override_ids:
                continue
            f = _extract_features(a)
            key = self._count_key(f)
            buckets.setdefault(key, {"over": 0, "seen": 0})
            buckets[key]["seen"] += 1

        # Laplace-smoothed rates
        rates = {
            k: (b["over"] + 1) / (b["seen"] + 2)
            for k, b in buckets.items()
        }
        self._model = {"type": "count_rate", "rates": rates}
        self._model_method = "count_rate"
        self._is_trained = True
        self._last_fit_ts = datetime.utcnow().isoformat(timespec="seconds")
        self._persist_model()

    def _count_key(self, feats: Dict[str, float]) -> str:
        hour_bucket = int(feats.get("hour_of_day", 12)) // 3        # 3-hour bins
        prio = int(feats.get("priority", 3))
        site = int(feats.get("site_idx", 0))
        return f"{hour_bucket}|{prio}|{site}"

    def _generate_neighbourhood(
        self, appt: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build candidate adjacent-hour slots on the same chair."""
        out: List[Dict[str, Any]] = []
        start = _coerce_dt(appt.get("start_time"))
        if start is None:
            return out
        dur = int(appt.get("duration") or appt.get("expected_duration") or 60)
        for dh in self.neighbourhood_hours:
            new_start = start + timedelta(hours=dh)
            if not (8 <= new_start.hour < 18):
                continue
            c = dict(appt)
            c["start_time"] = new_start
            c["end_time"] = new_start + timedelta(minutes=dur)
            out.append(c)
        return out

    def _append_event_log(self, event: OverrideEvent) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            with open(self.events_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event)) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"override event log write failed: {exc}")

    def _append_suggestion_log(self, suggestion: OverrideSuggestion) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.utcnow().isoformat(timespec="seconds"),
                "patient_id": suggestion.patient_id,
                "original_override_prob": suggestion.original_override_prob,
                "suggested_override_prob": suggestion.suggested_override_prob,
                "delta_prob": suggestion.delta_prob,
                "reason": suggestion.reason,
            }
            with open(self.suggestions_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"override suggestion log write failed: {exc}")

    def _persist_model(self) -> None:
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "feature_means": self._feature_means,
                    "feature_stds": self._feature_stds,
                    "method": self._model_method,
                    "fit_ts": self._last_fit_ts,
                }, f)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"override model persist failed: {exc}")

    def _load_events(self) -> None:
        if not self.events_log.exists():
            return
        try:
            with open(self.events_log, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        self._events.append(OverrideEvent(**d))
                    except Exception:
                        continue
        except Exception as exc:  # pragma: no cover
            logger.warning(f"override event log read failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(t: Any) -> Optional[str]:
    t = _coerce_dt(t)
    return t.isoformat(timespec="seconds") if t else (
        str(t) if t is not None else None
    )


def compute_suggestions_for_schedule(
    learner: OverrideLearner, appointments: Sequence[Any],
) -> List[OverrideSuggestion]:
    """Bulk-apply ``learner.suggest`` over every appointment in a schedule.

    Returns only the appointments that crossed the threshold.  Invoked
    by the Flask layer from ``run_optimization`` to tack suggestions
    onto the result dict.
    """
    out: List[OverrideSuggestion] = []
    for a in appointments:
        appt: Dict[str, Any]
        if isinstance(a, dict):
            appt = dict(a)
        else:
            appt = {
                "patient_id": getattr(a, "patient_id", None),
                "chair_id": getattr(a, "chair_id", None),
                "site_code": getattr(a, "site_code", None),
                "start_time": getattr(a, "start_time", None),
                "end_time": getattr(a, "end_time", None),
                "duration": getattr(a, "duration", None),
                "priority": getattr(a, "priority", None),
                "noshow_prob": getattr(a, "noshow_probability", None),
            }
        s = learner.suggest(appt)
        if s is not None:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors §3.1-§5.1 style)
# ---------------------------------------------------------------------------

_GLOBAL: Optional[OverrideLearner] = None


def get_learner() -> OverrideLearner:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = OverrideLearner()
    return _GLOBAL


def set_learner(l: OverrideLearner) -> None:
    global _GLOBAL
    _GLOBAL = l
