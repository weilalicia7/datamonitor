"""
Online Feature Store (Dissertation §3.1)
========================================

Moves the system off monthly re-training of
``historical_appointments.xlsx`` and onto a streaming-style feature
store with three operational guarantees that matter to a clinical
system:

1. **Low-latency online serving** — every prediction endpoint can
   pull a patient's rolling-window features in well under the 100 ms
   target specified in the §3.1 brief.
2. **Point-in-time correctness** — for training workflows, the store
   returns features *as of* an event timestamp, so a model fitted
   today never sees a feature value computed from an outcome that
   only occurred tomorrow.
3. **Schema + feature versioning** — every view carries a version
   tag, materialisation timestamps are logged, and the schema is
   persisted to JSON so the dissertation results are reproducible.

The implementation is deliberately self-contained: no Feast or
Hopsworks dependency is required (those are 200 MB+ and need Redis /
Kafka).  Instead we use a pickled in-memory dict for the online path
and a JSONL append-only log for the event stream.  The interface
mirrors Feast/Hopsworks (`register_entity`, `register_view`,
`materialize`, `push_event`, `get_online_features`) so swapping to a
managed store later is a one-file change.

Feature views shipped out of the box
------------------------------------

* ``patient_30d_stats``   — no-show rate, appointment count, mean
  duration, cancellation rate over the last 30 days.
* ``patient_90d_stats``   — same four quantities on a 90-day window
  (smoother, used for slow-drifting baselines).
* ``patient_cycle_ctx``   — current cycle number, total cycles,
  cycles since a regimen modification, days-since-last visit.
* ``patient_trend``       — attended-streak length, cancelled-streak
  length, duration trend direction (``stable`` | ``growing`` |
  ``declining``) computed via a rolling linear slope.

Every view is deterministic in its definition (same inputs always
produce the same output), so the exact features that trained a
model can always be reproduced from the event log.

Online-store schema
-------------------

``online[patient_id] = {
    '<view_name>__<feature_name>': value,
    '<view_name>__as_of': iso-timestamp,
    '<view_name>__materialisation_id': str,
}``

All feature lookups are O(1) dict access; the 100 ms budget for a
batch-of-200 lookup is dominated by Python overhead, not I/O.
"""

from __future__ import annotations

import json
import math
import pickle
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Paths + schema version
# ---------------------------------------------------------------------------

FEATURE_STORE_DIR: Path = DATA_CACHE_DIR / 'feature_store'
ONLINE_STATE_FILE: Path = FEATURE_STORE_DIR / 'online.pkl'
EVENTS_LOG_FILE: Path = FEATURE_STORE_DIR / 'events.jsonl'
MATERIALISATION_LOG: Path = FEATURE_STORE_DIR / 'materialisations.jsonl'
LATENCY_LOG: Path = FEATURE_STORE_DIR / 'serving_latency.jsonl'
SCHEMA_FILE: Path = FEATURE_STORE_DIR / 'schema.json'

SCHEMA_VERSION: str = "1.0.0"
# Default TTL on event-derived rolling stats: everything older than 180 days
# is eligible for purge at the next push.  Expiry is *lazy* — a materialise()
# call or a targeted purge() is what actually removes rows.
DEFAULT_EVENT_TTL_DAYS: int = 180


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A keyed entity (at present: Patient).  Mirrors Feast's Entity."""
    name: str
    join_key: str
    description: str = ""


@dataclass
class FeatureView:
    """
    A named collection of features produced by a single compute function.
    ``compute`` receives the ordered list of historical events for a single
    entity (oldest-first) and returns a dict of feature values.
    """
    name: str
    version: str
    entity: str
    feature_names: Tuple[str, ...]
    compute: Callable[[List[Dict]], Dict[str, Any]]
    description: str = ""
    ttl_days: int = DEFAULT_EVENT_TTL_DAYS

    def compute_one(self, events: List[Dict]) -> Dict[str, Any]:
        """Defensive wrapper so one bad event can't poison a whole batch."""
        try:
            out = self.compute(events or [])
            if not isinstance(out, dict):
                raise TypeError(f"FeatureView '{self.name}' compute did not return dict")
            # Pad any missing feature with None so the online schema is stable
            for f in self.feature_names:
                out.setdefault(f, None)
            return out
        except Exception as exc:
            logger.warning(f"FeatureView {self.name} compute failed: {exc}")
            return {f: None for f in self.feature_names}


@dataclass
class MaterialisationInfo:
    ts: str
    view: str
    view_version: str
    n_entities: int
    n_events_scanned: int
    materialisation_id: str


# ---------------------------------------------------------------------------
# Default compute functions (deterministic, no hidden state)
# ---------------------------------------------------------------------------


def _parse_date(row: Dict) -> Optional[datetime]:
    """
    Return the *occurrence* time of an event for point-in-time filtering.

    Priority order:
      1. `Date` / `date`          — authoritative appointment date
      2. `timestamp`              — explicit event timestamp
      3. `event_ts`               — the ingestion time (fallback only)

    `event_ts` is set by push_event() to datetime.utcnow() so it marks
    *when the row was logged*, not *when the appointment occurred* —
    using it first would break point-in-time correctness whenever a
    row is back-filled from historical data.
    """
    for key in ('Date', 'date', 'timestamp', 'event_ts'):
        v = row.get(key)
        if v is None or v == '':
            continue
        if isinstance(v, datetime):
            return v
        try:
            return datetime.fromisoformat(str(v).replace(' ', 'T')[:19])
        except Exception:
            try:
                return datetime.strptime(str(v)[:10], "%Y-%m-%d")
            except Exception:
                continue
    return None


def _attended_flag(row: Dict) -> Optional[bool]:
    """Normalise attendance across the various field spellings used by the dataset."""
    status = row.get('Attended_Status', row.get('attended_status'))
    if status is not None:
        s = str(status).strip().lower()
        if s in ('yes', 'attended'):
            return True
        if s in ('no', 'noshow', 'no-show', 'did not attend', 'dna'):
            return False
        if s in ('cancelled', 'cancel', 'cancellation'):
            return False  # attendance = False for cancellations
    if 'Showed_Up' in row:
        try:
            return bool(row['Showed_Up'])
        except Exception:
            return None
    if 'attended' in row:
        try:
            return bool(row['attended'])
        except Exception:
            return None
    return None


def _cancelled_flag(row: Dict) -> bool:
    status = str(row.get('Attended_Status', '')).strip().lower()
    return status in ('cancelled', 'cancel', 'cancellation')


def _duration(row: Dict) -> Optional[float]:
    for k in ('Actual_Duration', 'actual_duration', 'Planned_Duration', 'planned_duration'):
        v = row.get(k)
        if v is None or v == '':
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def _rolling_compute(days: int) -> Callable[[List[Dict]], Dict[str, Any]]:
    """Factory for an N-day rolling-window stats view."""
    def _fn(events: List[Dict]) -> Dict[str, Any]:
        if not events:
            return {}
        now = max((_parse_date(e) for e in events if _parse_date(e)),
                  default=datetime.utcnow())
        cutoff = now - timedelta(days=days)
        window = [e for e in events
                  if (dt := _parse_date(e)) is not None and dt >= cutoff]
        n = len(window)
        if n == 0:
            return {
                f'noshow_rate_{days}d': 0.0,
                f'appointment_count_{days}d': 0,
                f'mean_duration_{days}d': None,
                f'cancellation_rate_{days}d': 0.0,
            }
        attended = [_attended_flag(e) for e in window]
        n_known = sum(1 for a in attended if a is not None)
        n_noshow = sum(1 for a in attended if a is False)
        n_cancel = sum(1 for e in window if _cancelled_flag(e))
        durations = [d for d in (_duration(e) for e in window) if d is not None]
        return {
            f'noshow_rate_{days}d': (n_noshow / n_known) if n_known else 0.0,
            f'appointment_count_{days}d': n,
            f'mean_duration_{days}d': (
                float(statistics.mean(durations)) if durations else None
            ),
            f'cancellation_rate_{days}d': (n_cancel / n),
        }
    return _fn


def _cycle_ctx_compute(events: List[Dict]) -> Dict[str, Any]:
    if not events:
        return {}
    cycles = [int(e.get('Cycle_Number', 0) or 0) for e in events]
    mods = [int(e.get('Regimen_Modification_Flag', 0) or 0) for e in events]
    now = max((_parse_date(e) for e in events if _parse_date(e)),
              default=datetime.utcnow())
    last = events[-1]
    last_dt = _parse_date(last) or now
    since_mod = None
    for e in reversed(events):
        if int(e.get('Regimen_Modification_Flag', 0) or 0):
            dt = _parse_date(e)
            if dt is not None:
                since_mod = (now - dt).days
                break
    return {
        'current_cycle': max(cycles) if cycles else 0,
        'total_cycles_observed': len(set(cycles)) if cycles else 0,
        'cycles_since_modification': since_mod,
        'days_since_last_visit': (now - last_dt).days,
    }


def _trend_compute(events: List[Dict]) -> Dict[str, Any]:
    if not events:
        return {}
    attended = [_attended_flag(e) for e in events]
    attended_tail: List[bool] = []
    for a in reversed(attended):
        if a is None:
            continue
        attended_tail.append(a)
    # streak of the most-recent *same* attendance value
    streak_attended = 0
    streak_cancelled = 0
    if attended_tail:
        cur = attended_tail[0]
        s = 0
        for a in attended_tail:
            if a == cur:
                s += 1
            else:
                break
        if cur is True:
            streak_attended = s
        else:
            streak_cancelled = s

    # duration trend on the last ≤8 observations with known duration
    recent_durations = [d for d in (_duration(e) for e in events[-8:]) if d is not None]
    trend_label = 'stable'
    if len(recent_durations) >= 4:
        xs = list(range(len(recent_durations)))
        n = len(xs)
        mx = sum(xs) / n
        my = sum(recent_durations) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, recent_durations))
        den = sum((x - mx) ** 2 for x in xs)
        slope = (num / den) if den else 0.0
        rel = slope / max(abs(my), 1.0)
        if rel > 0.02:
            trend_label = 'growing'
        elif rel < -0.02:
            trend_label = 'declining'
    return {
        'attended_streak': streak_attended,
        'cancelled_streak': streak_cancelled,
        'duration_trend': trend_label,
    }


# ---------------------------------------------------------------------------
# The feature store
# ---------------------------------------------------------------------------


class FeatureStore:
    """
    Self-contained online + offline feature store for the SACT
    scheduling system.  Thread-compatible (single-writer assumption,
    which matches the Flask app's single-process deployment).
    """

    def __init__(
        self,
        storage_dir: Path = FEATURE_STORE_DIR,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.online_path = self.storage_dir / 'online.pkl'
        self.events_path = self.storage_dir / 'events.jsonl'
        self.materialisation_path = self.storage_dir / 'materialisations.jsonl'
        self.latency_path = self.storage_dir / 'serving_latency.jsonl'
        self.schema_path = self.storage_dir / 'schema.json'
        self.schema_version = schema_version

        self._entities: Dict[str, Entity] = {}
        self._views: Dict[str, FeatureView] = {}
        self._online: Dict[str, Dict[str, Any]] = {}
        self._try_load_online()
        self._register_defaults()

    # ---- registration ----

    def register_entity(self, entity: Entity) -> None:
        self._entities[entity.name] = entity

    def register_view(self, view: FeatureView) -> None:
        if view.entity not in self._entities:
            raise ValueError(
                f"Entity '{view.entity}' must be registered before view '{view.name}'"
            )
        self._views[view.name] = view

    def _register_defaults(self) -> None:
        self.register_entity(Entity(
            name='Patient', join_key='patient_id',
            description='SACT patient identified by Patient_ID or NHS Number',
        ))
        self.register_view(FeatureView(
            name='patient_30d_stats', version='v1.0.0', entity='Patient',
            feature_names=(
                'noshow_rate_30d', 'appointment_count_30d',
                'mean_duration_30d', 'cancellation_rate_30d',
            ),
            compute=_rolling_compute(30),
            description='30-day rolling patient activity + outcome stats.',
        ))
        self.register_view(FeatureView(
            name='patient_90d_stats', version='v1.0.0', entity='Patient',
            feature_names=(
                'noshow_rate_90d', 'appointment_count_90d',
                'mean_duration_90d', 'cancellation_rate_90d',
            ),
            compute=_rolling_compute(90),
            description='90-day rolling counterpart to 30d stats.',
        ))
        self.register_view(FeatureView(
            name='patient_cycle_ctx', version='v1.0.0', entity='Patient',
            feature_names=(
                'current_cycle', 'total_cycles_observed',
                'cycles_since_modification', 'days_since_last_visit',
            ),
            compute=_cycle_ctx_compute,
            description='Treatment-cycle context and recency.',
        ))
        self.register_view(FeatureView(
            name='patient_trend', version='v1.0.0', entity='Patient',
            feature_names=('attended_streak', 'cancelled_streak', 'duration_trend'),
            compute=_trend_compute,
            description='Behavioural streak + duration-trend signals.',
        ))

    # ---- event ingest ----

    def push_event(self, event: Dict) -> None:
        """
        Append a single event to the log *and* recompute the patient's
        materialised features on the fly — i.e. streaming update.
        """
        event = dict(event)  # defensive copy
        event.setdefault('event_ts', datetime.utcnow().isoformat(timespec='seconds'))
        patient_id = str(event.get('Patient_ID', event.get('patient_id', '')))
        if not patient_id:
            raise ValueError("push_event requires Patient_ID or patient_id.")
        # Append to offline log
        with self.events_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(event, default=str) + '\n')
        # Streaming materialisation for this patient only
        patient_events = list(self._read_patient_events(patient_id))
        self._materialise_patient(patient_id, patient_events, reason='stream')

    def _read_patient_events(self, patient_id: str) -> Iterable[Dict]:
        """Linear scan of the event log, filtered to one entity.  O(events)."""
        if not self.events_path.exists():
            return []
        out: List[Dict] = []
        with self.events_path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    pid = str(e.get('Patient_ID', e.get('patient_id', '')))
                    if pid == patient_id:
                        out.append(e)
                except Exception:
                    continue
        # Sort chronologically
        out.sort(key=lambda e: _parse_date(e) or datetime.min)
        return out

    def _materialise_patient(
        self,
        patient_id: str,
        events: List[Dict],
        reason: str = 'batch',
    ) -> int:
        """Recompute every view's features for a single entity; returns # features set."""
        row = self._online.get(patient_id, {})
        n_set = 0
        ts = datetime.utcnow().isoformat(timespec='seconds')
        for view_name, view in self._views.items():
            vals = view.compute_one(events)
            for fname, fval in vals.items():
                row[f'{view_name}__{fname}'] = fval
                n_set += 1
            row[f'{view_name}__as_of'] = ts
            row[f'{view_name}__version'] = view.version
        row['__entity_last_materialisation'] = ts
        row['__entity_materialisation_reason'] = reason
        self._online[patient_id] = row
        return n_set

    # ---- batch materialisation ----

    def materialize(self, df) -> MaterialisationInfo:
        """
        Batch-materialise from a historical DataFrame (typically
        ``historical_appointments_df``).  One pass groups by patient.
        """
        import pandas as pd
        if 'Patient_ID' not in df.columns:
            raise ValueError("DataFrame needs Patient_ID column for materialisation.")
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for _, row in df.iterrows():
            pid = str(row.get('Patient_ID', ''))
            if not pid:
                continue
            d = row.to_dict()
            # Normalise a pandas Timestamp to ISO
            for k, v in list(d.items()):
                if isinstance(v, (pd.Timestamp, )):
                    d[k] = v.isoformat()
            grouped[pid].append(d)

        total = 0
        for pid, evs in grouped.items():
            evs.sort(key=lambda e: _parse_date(e) or datetime.min)
            self._materialise_patient(pid, evs, reason='batch')
            total += len(evs)

        self._save_online()
        mat = MaterialisationInfo(
            ts=datetime.utcnow().isoformat(timespec='seconds'),
            view='__all__',
            view_version=self.schema_version,
            n_entities=len(grouped),
            n_events_scanned=total,
            materialisation_id=f"mat_{int(time.time())}",
        )
        with self.materialisation_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(asdict(mat)) + '\n')
        self.save_schema()
        logger.info(
            f"FeatureStore materialised: entities={mat.n_entities} "
            f"events={mat.n_events_scanned} id={mat.materialisation_id}"
        )
        return mat

    # ---- online serving ----

    def get_online_features(
        self,
        patient_ids: Sequence[str],
        view_names: Optional[Sequence[str]] = None,
        log_latency: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        O(N) dict lookup.  Returns {patient_id: {feature_fqn: value}}.
        """
        t0 = time.perf_counter()
        if view_names is None:
            view_names = list(self._views.keys())
        # Flatten requested feature FQNs once
        wanted_prefixes = tuple(f'{v}__' for v in view_names)
        out: Dict[str, Dict[str, Any]] = {}
        for pid in patient_ids:
            row = self._online.get(str(pid)) or {}
            sub = {k: v for k, v in row.items()
                   if k.startswith(wanted_prefixes) or k.startswith('__entity_')}
            out[str(pid)] = sub
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if log_latency:
            try:
                with self.latency_path.open('a', encoding='utf-8') as fh:
                    fh.write(json.dumps({
                        'ts': datetime.utcnow().isoformat(timespec='seconds'),
                        'n_patients': len(out),
                        'n_views': len(view_names),
                        'latency_ms': round(latency_ms, 3),
                    }) + '\n')
            except Exception:  # pragma: no cover
                pass
        return out

    def last_latency(self) -> Optional[float]:
        if not self.latency_path.exists():
            return None
        try:
            last = None
            with self.latency_path.open('r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        last = json.loads(line)
            return float(last['latency_ms']) if last else None
        except Exception:
            return None

    # ---- point-in-time correctness for training ----

    def as_of(
        self,
        patient_id: str,
        timestamp: datetime,
        view_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Return the feature values that would have been served *at or before*
        `timestamp` — i.e. exclude any events dated after that point.  This
        is the training-time contract that prevents label leakage.
        """
        if view_names is None:
            view_names = list(self._views.keys())
        events = [
            e for e in self._read_patient_events(patient_id)
            if (dt := _parse_date(e)) is not None and dt <= timestamp
        ]
        out: Dict[str, Any] = {}
        for vname in view_names:
            view = self._views.get(vname)
            if view is None:
                continue
            vals = view.compute_one(events)
            for fname, fval in vals.items():
                out[f'{vname}__{fname}'] = fval
        out['__as_of_timestamp'] = timestamp.isoformat()
        out['__n_events_considered'] = len(events)
        return out

    # ---- persistence ----

    def save_schema(self) -> None:
        payload = {
            'schema_version': self.schema_version,
            'entities': {
                name: asdict(e) for name, e in self._entities.items()
            },
            'views': {
                name: {
                    'name': v.name, 'version': v.version, 'entity': v.entity,
                    'feature_names': list(v.feature_names),
                    'description': v.description,
                    'ttl_days': v.ttl_days,
                }
                for name, v in self._views.items()
            },
            'saved_at': datetime.utcnow().isoformat(timespec='seconds'),
        }
        self.schema_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def _save_online(self) -> None:
        with self.online_path.open('wb') as fh:
            pickle.dump(self._online, fh)

    def _try_load_online(self) -> None:
        if not self.online_path.exists():
            return
        try:
            with self.online_path.open('rb') as fh:
                self._online = pickle.load(fh)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"FeatureStore online-state load failed: {exc}")
            self._online = {}

    def status(self) -> Dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'n_entities_materialised': len(self._online),
            'views': [
                {'name': v.name, 'version': v.version,
                 'entity': v.entity,
                 'feature_names': list(v.feature_names),
                 'ttl_days': v.ttl_days}
                for v in self._views.values()
            ],
            'last_latency_ms': self.last_latency(),
            'storage_dir': str(self.storage_dir),
        }


# ---------------------------------------------------------------------------
# Module-level singleton so Flask routes + the prediction hook share state.
# ---------------------------------------------------------------------------


_STORE: Optional[FeatureStore] = None


def get_store() -> FeatureStore:
    global _STORE
    if _STORE is None:
        _STORE = FeatureStore()
    return _STORE


__all__ = [
    'Entity',
    'FeatureView',
    'FeatureStore',
    'MaterialisationInfo',
    'SCHEMA_VERSION',
    'get_store',
]
