"""
Explainable Scheduling Reports for Rejected Patients (Dissertation §5.3)
=======================================================================

When the CP-SAT optimiser cannot place a patient, the only signal
the caller receives today is a patient-id in
``OptimizationResult.unscheduled``.  §5.3 requires an honest,
operator-facing explanation of \\emph{why} each rejection
happened and a concrete counterfactual: the closest available slot
the patient could be offered instead.  The brief's motivating
example is literally:

    "Patient P12345 (P2, previous no-shows: 3) not scheduled because:
    - Chair 4 has 45 min slack (requires 60 min for this patient)
    - Chair 7 would require bumping P1 patient (not permitted)
    - Alternative: offer Monday 2pm (15 min wait increase)
    [Accept] [Decline]"

Implementation contract
-----------------------
1. **Per-chair blocker classification**.  ``RejectionExplainer.explain``
   walks every chair in ``DEFAULT_SITES`` (or the chair list passed
   in), measures the gap the patient needed, and classifies the
   blocker into one of 7 stable categories: ``insufficient_slack``,
   ``would_bump_higher_priority``, ``would_bump_same_or_lower_priority``,
   ``travel_exceeded``, ``outside_operating_hours``,
   ``earliest_time_exceeded``, ``duration_too_long_for_day``.
2. **Alternative slot**.  A bounded forward scan
   (``look_ahead_days`` days × ``DEFAULT_SITES`` chairs) finds the
   earliest slot that *would* fit, returning a
   :class:`AlternativeSlot` with wait-time delta vs the patient's
   original ``earliest_time``.
3. **Narrative**.  The output :class:`RejectionExplanation`
   contains a human-readable string matching the brief's template
   so UIs can display it verbatim.
4. **Invisible integration**.  Flask attaches every
   ``run_optimization`` unscheduled-list to the explainer; no UI
   panel per the brief.

Blocker types
-------------
=============================== ==================================================
``insufficient_slack``          Best gap on this chair is smaller than the
                                patient's expected duration.
``would_bump_higher_priority``  The only candidate window overlaps with a
                                strictly higher-priority (lower number)
                                appointment.  Swap not permitted.
``would_bump_same_or_lower``    The window overlaps a same/lower-priority
                                appointment but ``allow_reschedule`` is
                                ``False``.
``travel_exceeded``             Chair's site is farther than
                                ``max_travel_minutes`` from the patient.
``outside_operating_hours``     All windows fall outside the 08:00--18:00
                                day (or the chair's configured hours).
``earliest_time_exceeded``      Every window starts before the patient's
                                ``earliest_time`` gate.
``duration_too_long_for_day``   Patient's duration does not fit in the
                                remaining day on any chair.
=============================== ==================================================
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, DEFAULT_SITES, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DAY_START_HOUR: int = 8
DEFAULT_DAY_END_HOUR: int = 18
DEFAULT_MAX_TRAVEL_MIN: int = 120
DEFAULT_LOOK_AHEAD_DAYS: int = 5
DEFAULT_SLACK_BUFFER_MIN: int = 5
REJECTION_DIR: Path = DATA_CACHE_DIR / "rejection_explainer"
REJECTION_LOG: Path = REJECTION_DIR / "explanations.jsonl"

# Blocker categories (stable strings — used in JSONL + UI)
BLOCKER_INSUFFICIENT_SLACK = "insufficient_slack"
BLOCKER_BUMP_HIGHER = "would_bump_higher_priority"
BLOCKER_BUMP_SAME_OR_LOWER = "would_bump_same_or_lower_priority"
BLOCKER_TRAVEL_EXCEEDED = "travel_exceeded"
BLOCKER_OUTSIDE_HOURS = "outside_operating_hours"
BLOCKER_EARLIEST_EXCEEDED = "earliest_time_exceeded"
BLOCKER_DURATION_TOO_LONG = "duration_too_long_for_day"
BLOCKER_CHAIR_CLOSED = "chair_closed"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChairBlocker:
    chair_id: str
    site_code: str
    blocker_type: str
    detail: str
    max_slack_min: Optional[int] = None           # only for slack/bump reports
    colliding_patient_id: Optional[str] = None    # only for bump reports
    colliding_priority: Optional[int] = None
    travel_minutes: Optional[int] = None


@dataclass
class AlternativeSlot:
    chair_id: str
    site_code: str
    date: str               # ISO date
    start_time: str         # ISO datetime
    duration_minutes: int
    wait_increase_minutes: int     # vs patient's earliest_time
    narrative: str          # e.g. "Monday 2pm (15 min wait increase)"


@dataclass
class RejectionExplanation:
    computed_ts: str
    patient_id: str
    priority: int
    previous_no_shows: int
    expected_duration_min: int
    requested_earliest: Optional[str]
    chairs_checked: int
    blockers: List[ChairBlocker] = field(default_factory=list)
    alternative: Optional[AlternativeSlot] = None
    narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------


class RejectionExplainer:
    """Generate per-patient rejection explanations."""

    def __init__(
        self,
        *,
        day_start_hour: int = DEFAULT_DAY_START_HOUR,
        day_end_hour: int = DEFAULT_DAY_END_HOUR,
        max_travel_minutes: int = DEFAULT_MAX_TRAVEL_MIN,
        look_ahead_days: int = DEFAULT_LOOK_AHEAD_DAYS,
        slack_buffer_min: int = DEFAULT_SLACK_BUFFER_MIN,
        allow_reschedule: bool = False,
        storage_dir: Path = REJECTION_DIR,
    ):
        self.day_start_hour = int(day_start_hour)
        self.day_end_hour = int(day_end_hour)
        self.max_travel_minutes = int(max_travel_minutes)
        self.look_ahead_days = int(look_ahead_days)
        self.slack_buffer_min = int(slack_buffer_min)
        self.allow_reschedule = bool(allow_reschedule)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.storage_dir / "explanations.jsonl"

        self._lock = threading.Lock()
        self._last: Optional[List[RejectionExplanation]] = None
        self._last_narrative: Optional[str] = None
        self._n_runs: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def explain(
        self,
        patient: Dict[str, Any],
        schedule: Sequence[Any],
        chairs: Optional[Sequence[Dict[str, Any]]] = None,
        patient_history: Optional[Dict[str, Any]] = None,
        current_date: Optional[datetime] = None,
    ) -> RejectionExplanation:
        """Produce one :class:`RejectionExplanation` for a single patient.

        ``schedule`` is the committed list of ``ScheduledAppointment``
        objects (or the dict-shaped equivalent) on the day in question.
        ``chairs`` defaults to ``DEFAULT_SITES``-derived list.
        """
        current_date = current_date or datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        chairs_resolved = list(chairs) if chairs is not None else self._default_chairs()
        pid = str(
            patient.get("Patient_ID")
            or patient.get("patient_id")
            or ""
        )
        priority = int(
            patient.get("priority")
            or patient.get("Priority")
            or 3
        )
        duration = int(
            patient.get("expected_duration")
            or patient.get("Planned_Duration")
            or 60
        )
        earliest_raw = patient.get("earliest_time") or patient.get("Earliest_Time")
        earliest = _coerce_dt(earliest_raw) if earliest_raw else current_date.replace(
            hour=self.day_start_hour
        )

        history = patient_history or {}
        prev_ns = int(
            history.get("no_shows")
            or patient.get("no_shows")
            or 0
        )

        travel_for_site = _default_travel_lookup(
            patient, chairs_resolved, default=30
        )

        # Group schedule by chair
        by_chair: Dict[str, List[Dict[str, Any]]] = {}
        for a in schedule:
            d = _serialise_appt(a)
            if d.get("start_time") is not None:
                by_chair.setdefault(d.get("chair_id", "_UNK"), []).append(d)
        for k in by_chair:
            by_chair[k].sort(key=lambda d: d["start_time"])

        blockers: List[ChairBlocker] = []
        for chair in chairs_resolved:
            cid = chair.get("chair_id") or chair.get("id") or ""
            site = chair.get("site_code") or chair.get("site") or "UNKNOWN"

            # Travel check
            travel_min = int(travel_for_site.get(site, 30))
            if travel_min > self.max_travel_minutes:
                blockers.append(ChairBlocker(
                    chair_id=cid, site_code=site,
                    blocker_type=BLOCKER_TRAVEL_EXCEEDED,
                    detail=(
                        f"site {site} is {travel_min} min away "
                        f"(>{self.max_travel_minutes} min ceiling)"
                    ),
                    travel_minutes=travel_min,
                ))
                continue

            # Gather existing appointments on this chair within the day
            items = by_chair.get(cid, [])
            best_slack, max_slack_any, bump_candidate = self._analyse_chair(
                items, earliest, duration, current_date,
            )

            if best_slack is None:
                # No gap fits.  Classify by which constraint bit:
                #   * duration > whole-day window  -> DURATION_TOO_LONG
                #   * earliest_time is after day-end -> EARLIEST_EXCEEDED
                #   * some gap exists (just too small) -> INSUFFICIENT_SLACK
                #   * no gaps at all + colliding appt with higher priority -> BUMP_HIGHER
                #   * no gaps at all + colliding appt same/lower -> BUMP_SAME_OR_LOWER
                #   * no items at all (chair closed / outside hours) -> OUTSIDE_HOURS
                if duration > (self.day_end_hour - self.day_start_hour) * 60:
                    reason = BLOCKER_DURATION_TOO_LONG
                elif earliest.hour >= self.day_end_hour:
                    reason = BLOCKER_EARLIEST_EXCEEDED
                elif (max_slack_any or 0) > 0:
                    reason = BLOCKER_INSUFFICIENT_SLACK
                elif bump_candidate is not None:
                    coll_prio = int(bump_candidate.get("priority", 3))
                    if coll_prio < priority:
                        reason = BLOCKER_BUMP_HIGHER
                    elif not self.allow_reschedule:
                        reason = BLOCKER_BUMP_SAME_OR_LOWER
                    else:
                        reason = BLOCKER_INSUFFICIENT_SLACK
                else:
                    reason = BLOCKER_OUTSIDE_HOURS

                if reason == BLOCKER_INSUFFICIENT_SLACK:
                    detail = (
                        f"Chair {cid} has {int(max_slack_any or 0)} min slack "
                        f"(requires {duration} min for this patient)"
                    )
                elif reason == BLOCKER_BUMP_HIGHER and bump_candidate is not None:
                    detail = (
                        f"Chair {cid} would require bumping priority-"
                        f"{int(bump_candidate.get('priority', 3))} patient "
                        f"{bump_candidate.get('patient_id', '?')} "
                        "(not permitted)"
                    )
                elif reason == BLOCKER_BUMP_SAME_OR_LOWER and bump_candidate is not None:
                    detail = (
                        f"Chair {cid} collision with priority-"
                        f"{int(bump_candidate.get('priority', 3))} patient "
                        f"{bump_candidate.get('patient_id', '?')}; "
                        "allow_reschedule is off"
                    )
                elif reason == BLOCKER_DURATION_TOO_LONG:
                    detail = (
                        f"Chair {cid}: patient needs {duration} min which "
                        f"exceeds the full "
                        f"{self.day_start_hour:02d}:00-{self.day_end_hour:02d}:00 day"
                    )
                elif reason == BLOCKER_EARLIEST_EXCEEDED:
                    detail = (
                        f"Chair {cid}: patient's earliest_time "
                        f"{earliest.strftime('%H:%M')} is past the "
                        f"{self.day_end_hour:02d}:00 day end"
                    )
                else:  # OUTSIDE_HOURS
                    detail = (
                        f"Chair {cid} has no open window today for a "
                        f"{duration}-min appointment"
                    )

                blockers.append(ChairBlocker(
                    chair_id=cid, site_code=site,
                    blocker_type=reason, detail=detail,
                    max_slack_min=int(max_slack_any or 0),
                    colliding_patient_id=(str(bump_candidate.get("patient_id"))
                                          if bump_candidate else None),
                    colliding_priority=(int(bump_candidate.get("priority", 3))
                                        if bump_candidate else None),
                ))
                continue

            if best_slack >= duration + self.slack_buffer_min:
                # This chair COULD fit — the optimiser's packing decided not to.
                # We still emit a "would-have-fit" blocker so the operator sees it.
                # This normally only shows up during secondary audits after CP-SAT
                # has picked a slightly different global trade-off.
                blockers.append(ChairBlocker(
                    chair_id=cid, site_code=site,
                    blocker_type="would_have_fit_global_trade_off",
                    detail=(
                        f"Chair {cid} has {best_slack} min slack "
                        f"(requires {duration} min); CP-SAT chose a global "
                        f"trade-off"
                    ),
                    max_slack_min=int(best_slack),
                ))
                continue

            # Bump classification
            if bump_candidate is not None:
                colliding_prio = int(bump_candidate.get("priority", 3))
                if colliding_prio < priority:
                    # Colliding appt is strictly higher priority
                    blockers.append(ChairBlocker(
                        chair_id=cid, site_code=site,
                        blocker_type=BLOCKER_BUMP_HIGHER,
                        detail=(
                            f"Chair {cid} would require bumping "
                            f"priority-{colliding_prio} patient "
                            f"{bump_candidate.get('patient_id', '?')} "
                            "(not permitted)"
                        ),
                        max_slack_min=int(best_slack),
                        colliding_patient_id=str(bump_candidate.get("patient_id")),
                        colliding_priority=colliding_prio,
                    ))
                    continue
                elif not self.allow_reschedule:
                    blockers.append(ChairBlocker(
                        chair_id=cid, site_code=site,
                        blocker_type=BLOCKER_BUMP_SAME_OR_LOWER,
                        detail=(
                            f"Chair {cid} collision with priority-"
                            f"{colliding_prio} patient "
                            f"{bump_candidate.get('patient_id', '?')}; "
                            "allow_reschedule is off"
                        ),
                        max_slack_min=int(best_slack),
                        colliding_patient_id=str(bump_candidate.get("patient_id")),
                        colliding_priority=colliding_prio,
                    ))
                    continue

            # Generic slack
            blockers.append(ChairBlocker(
                chair_id=cid, site_code=site,
                blocker_type=BLOCKER_INSUFFICIENT_SLACK,
                detail=(
                    f"Chair {cid} has {best_slack} min slack "
                    f"(requires {duration} min for this patient)"
                ),
                max_slack_min=int(best_slack),
            ))

        # Find the earliest alternative
        alternative = self._find_alternative(
            patient, schedule, chairs_resolved, earliest,
            duration, current_date,
        )

        narrative = self._build_narrative(
            pid=pid, priority=priority, prev_ns=prev_ns,
            duration=duration, blockers=blockers, alternative=alternative,
        )

        explanation = RejectionExplanation(
            computed_ts=datetime.utcnow().isoformat(timespec="seconds"),
            patient_id=pid,
            priority=priority,
            previous_no_shows=prev_ns,
            expected_duration_min=duration,
            requested_earliest=_fmt(earliest),
            chairs_checked=len(chairs_resolved),
            blockers=blockers,
            alternative=alternative,
            narrative=narrative,
        )
        self._append_event(explanation)
        return explanation

    def explain_all(
        self,
        unscheduled_patients: Sequence[Dict[str, Any]],
        schedule: Sequence[Any],
        chairs: Optional[Sequence[Dict[str, Any]]] = None,
        current_date: Optional[datetime] = None,
    ) -> List[RejectionExplanation]:
        out: List[RejectionExplanation] = []
        for p in unscheduled_patients:
            out.append(self.explain(
                p, schedule=schedule, chairs=chairs, current_date=current_date,
            ))
        with self._lock:
            self._last = out
            self._n_runs += 1
            if out:
                self._last_narrative = out[0].narrative
        return out

    def last(self) -> Optional[List[RejectionExplanation]]:
        with self._lock:
            return self._last

    def status(self) -> Dict[str, Any]:
        return {
            "day_start_hour": self.day_start_hour,
            "day_end_hour": self.day_end_hour,
            "max_travel_minutes": self.max_travel_minutes,
            "look_ahead_days": self.look_ahead_days,
            "slack_buffer_min": self.slack_buffer_min,
            "allow_reschedule": self.allow_reschedule,
            "total_runs": self._n_runs,
            "log_path": str(self.log_path),
            "last_n_explanations": len(self._last) if self._last else 0,
            "last_narrative": self._last_narrative,
        }

    def update_config(
        self,
        day_start_hour: Optional[int] = None,
        day_end_hour: Optional[int] = None,
        max_travel_minutes: Optional[int] = None,
        look_ahead_days: Optional[int] = None,
        slack_buffer_min: Optional[int] = None,
        allow_reschedule: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if day_start_hour is not None:
            self.day_start_hour = int(day_start_hour)
        if day_end_hour is not None:
            self.day_end_hour = int(day_end_hour)
        if max_travel_minutes is not None:
            self.max_travel_minutes = int(max_travel_minutes)
        if look_ahead_days is not None:
            self.look_ahead_days = int(look_ahead_days)
        if slack_buffer_min is not None:
            self.slack_buffer_min = int(slack_buffer_min)
        if allow_reschedule is not None:
            self.allow_reschedule = bool(allow_reschedule)
        return {
            "day_start_hour": self.day_start_hour,
            "day_end_hour": self.day_end_hour,
            "max_travel_minutes": self.max_travel_minutes,
            "look_ahead_days": self.look_ahead_days,
            "slack_buffer_min": self.slack_buffer_min,
            "allow_reschedule": self.allow_reschedule,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _default_chairs(self) -> List[Dict[str, Any]]:
        """Build a chair list from DEFAULT_SITES."""
        out: List[Dict[str, Any]] = []
        for s in DEFAULT_SITES:
            total = int(s.get("chairs", 0)) + int(s.get("recliners", 0))
            for i in range(total):
                out.append({
                    "chair_id": f"{s['code']}-C{i+1:02d}",
                    "site_code": s["code"],
                })
        if not out:
            out = [{"chair_id": "DEFAULT-C01", "site_code": "DEFAULT"}]
        return out

    def _analyse_chair(
        self,
        existing_items: List[Dict[str, Any]],
        earliest: datetime,
        duration: int,
        current_date: datetime,
    ) -> Tuple[Optional[int], int, Optional[Dict[str, Any]]]:
        """Return ``(best_slack, max_slack_any, colliding_appt)``.

        * ``best_slack`` = minutes of the largest gap >= duration found
          that also respects ``earliest``.  ``None`` if no such gap.
        * ``max_slack_any`` = biggest gap we saw at all (used to give
          the operator a number even when no gap fits).
        * ``colliding_appt`` = the nearest appointment whose occupancy
          prevents a fit, used for bump classification.
        """
        day_start = current_date.replace(
            hour=self.day_start_hour, minute=0, second=0, microsecond=0
        )
        day_end = current_date.replace(
            hour=self.day_end_hour, minute=0, second=0, microsecond=0
        )
        t_min = max(earliest, day_start)

        # Build gap list
        cursor = t_min
        gaps: List[Tuple[datetime, int, Optional[Dict[str, Any]]]] = []
        for it in existing_items:
            if it["end_time"] <= cursor:
                continue
            if it["start_time"] >= day_end:
                break
            gap_min = int((it["start_time"] - cursor).total_seconds() // 60)
            gaps.append((cursor, max(gap_min, 0), it))
            cursor = max(cursor, it["end_time"])
        # Trailing gap up to day end
        if cursor < day_end:
            trailing = int((day_end - cursor).total_seconds() // 60)
            gaps.append((cursor, trailing, None))

        max_any = max((g[1] for g in gaps), default=0)
        best_fit = next(
            (g for g in gaps if g[1] >= duration),
            None,
        )
        if best_fit is None:
            # No gap fits — pick the HIGHEST-priority colliding appt we'd
            # have to displace.  This matches the brief's phrasing "would
            # require bumping P1 patient" (the toughest swap to justify).
            colliders = [g[2] for g in gaps if g[2] is not None]
            colliders += [it for it in existing_items if it not in colliders]
            # Priority in this codebase is "lower number = higher priority".
            if colliders:
                coll = min(colliders, key=lambda a: int(a.get("priority", 3)))
            else:
                coll = None
            return None, int(max_any), coll
        return int(best_fit[1]), int(max_any), None

    def _find_alternative(
        self,
        patient: Dict[str, Any],
        schedule: Sequence[Any],
        chairs: List[Dict[str, Any]],
        earliest: datetime,
        duration: int,
        current_date: datetime,
    ) -> Optional[AlternativeSlot]:
        """Scan up to ``look_ahead_days`` days for the first fitting slot."""
        by_chair_day: Dict[str, List[Dict[str, Any]]] = {}
        for a in schedule:
            d = _serialise_appt(a)
            if d.get("start_time") is None:
                continue
            key = d.get("chair_id", "_UNK")
            by_chair_day.setdefault(key, []).append(d)
        for k in by_chair_day:
            by_chair_day[k].sort(key=lambda d: d["start_time"])

        # Respect the patient's earliest.hour / earliest.minute on every
        # look-ahead day — they don't become more available tomorrow.
        step_start_today = max(earliest, current_date.replace(
            hour=self.day_start_hour, minute=0, second=0, microsecond=0
        ))
        for day_offset in range(self.look_ahead_days):
            day = current_date + timedelta(days=day_offset)
            if day_offset == 0:
                day_start = step_start_today
            else:
                # On future days, minimum = max(patient.earliest HH:MM, day_start)
                e_h = max(earliest.hour, self.day_start_hour)
                e_m = earliest.minute if earliest.hour >= self.day_start_hour else 0
                day_start = day.replace(
                    hour=e_h, minute=e_m, second=0, microsecond=0,
                )
            for chair in chairs:
                cid = chair.get("chair_id")
                items = [
                    d for d in by_chair_day.get(cid, [])
                    if d["start_time"].date() == day.date()
                ]
                items.sort(key=lambda d: d["start_time"])
                best_gap, _, _ = self._analyse_chair(
                    items, day_start, duration, day,
                )
                if best_gap is not None:
                    # Find the actual start datetime of that gap
                    gap_start = self._first_fitting_start(
                        items, day_start, duration, day,
                    )
                    if gap_start is None:
                        continue
                    wait_increase = max(
                        0,
                        int((gap_start - earliest).total_seconds() // 60),
                    )
                    narrative = _friendly_day_phrase(gap_start, earliest)
                    return AlternativeSlot(
                        chair_id=str(cid),
                        site_code=str(chair.get("site_code") or ""),
                        date=gap_start.date().isoformat(),
                        start_time=gap_start.isoformat(timespec="seconds"),
                        duration_minutes=int(duration),
                        wait_increase_minutes=int(wait_increase),
                        narrative=narrative,
                    )
        return None

    def _first_fitting_start(
        self,
        existing_items: List[Dict[str, Any]],
        earliest: datetime,
        duration: int,
        current_date: datetime,
    ) -> Optional[datetime]:
        day_start = current_date.replace(
            hour=self.day_start_hour, minute=0, second=0, microsecond=0
        )
        day_end = current_date.replace(
            hour=self.day_end_hour, minute=0, second=0, microsecond=0
        )
        cursor = max(earliest, day_start)
        for it in existing_items:
            if it["end_time"] <= cursor:
                continue
            if it["start_time"] >= day_end:
                break
            gap_start = cursor
            gap_min = int((it["start_time"] - cursor).total_seconds() // 60)
            if gap_min >= duration + self.slack_buffer_min:
                return gap_start
            cursor = max(cursor, it["end_time"])
        # Trailing window
        if cursor + timedelta(minutes=duration) <= day_end:
            return cursor
        return None

    def _build_narrative(
        self,
        *, pid: str, priority: int, prev_ns: int, duration: int,
        blockers: List[ChairBlocker], alternative: Optional[AlternativeSlot],
    ) -> str:
        lines: List[str] = [
            f"Patient {pid} (P{priority}, previous no-shows: {prev_ns}) "
            f"not scheduled because:"
        ]
        # De-duplicate by blocker_type+chair_id+detail (keep chronological)
        seen = set()
        top_blockers: List[ChairBlocker] = []
        for b in blockers:
            key = (b.blocker_type, b.chair_id)
            if key in seen:
                continue
            seen.add(key)
            top_blockers.append(b)
            if len(top_blockers) >= 3:
                break
        for b in top_blockers:
            lines.append(f"- {b.detail}")
        if alternative is not None:
            lines.append(
                f"- Alternative: offer {alternative.narrative} "
                f"[Accept] [Decline]"
            )
        else:
            lines.append(
                f"- No alternative within "
                f"{self.look_ahead_days} days lookahead."
            )
        return "\n".join(lines)

    def _append_event(self, explanation: RejectionExplanation) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": explanation.computed_ts,
                "patient_id": explanation.patient_id,
                "priority": explanation.priority,
                "previous_no_shows": explanation.previous_no_shows,
                "expected_duration_min": explanation.expected_duration_min,
                "n_blockers": len(explanation.blockers),
                "top_blocker_types": list({b.blocker_type for b in explanation.blockers})[:5],
                "has_alternative": explanation.alternative is not None,
                "wait_increase_minutes": (
                    explanation.alternative.wait_increase_minutes
                    if explanation.alternative else None
                ),
                "narrative": explanation.narrative,
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"rejection explanation log write failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialise_appt(a: Any) -> Dict[str, Any]:
    if isinstance(a, dict):
        d = dict(a)
    else:
        d = {
            "patient_id": getattr(a, "patient_id", None),
            "chair_id": getattr(a, "chair_id", None),
            "site_code": getattr(a, "site_code", None),
            "start_time": getattr(a, "start_time", None),
            "end_time": getattr(a, "end_time", None),
            "duration": getattr(a, "duration", None),
            "priority": getattr(a, "priority", None),
        }
    d["start_time"] = _coerce_dt(d.get("start_time"))
    d["end_time"] = _coerce_dt(d.get("end_time"))
    return d


def _coerce_dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except Exception:
            pass
    return None


def _fmt(t: Any) -> Optional[str]:
    t = _coerce_dt(t)
    return t.isoformat(timespec="seconds") if t else None


def _default_travel_lookup(
    patient: Dict[str, Any],
    chairs: List[Dict[str, Any]],
    default: int = 30,
) -> Dict[str, int]:
    """Best-effort per-site travel estimate."""
    base_tt = int(patient.get("travel_time_minutes")
                  or patient.get("travel_time") or default)
    home_site = patient.get("preferred_site")
    tt = {}
    for chair in chairs:
        site = chair.get("site_code") or chair.get("site") or "UNKNOWN"
        if home_site and str(home_site).lower() == str(site).lower():
            tt[site] = min(tt.get(site, base_tt), base_tt)
        else:
            # Penalise unfamiliar sites
            tt[site] = max(tt.get(site, base_tt), base_tt + 10)
    return tt


def _friendly_day_phrase(slot: datetime, earliest: datetime) -> str:
    """Turn a datetime into 'Monday 2pm (15 min wait increase)' style."""
    weekday = slot.strftime("%A")
    hour = slot.hour
    meridiem = "am" if hour < 12 else "pm"
    disp_hour = hour if hour <= 12 else hour - 12
    if disp_hour == 0:
        disp_hour = 12
    if slot.minute:
        time_str = f"{disp_hour}:{slot.minute:02d}{meridiem}"
    else:
        time_str = f"{disp_hour}{meridiem}"
    wait_delta_min = max(0, int((slot - earliest).total_seconds() // 60))
    if wait_delta_min == 0:
        return f"{weekday} {time_str}"
    hours = wait_delta_min // 60
    mins = wait_delta_min % 60
    if hours == 0:
        return f"{weekday} {time_str} ({mins} min wait increase)"
    if mins == 0:
        return f"{weekday} {time_str} ({hours} h wait increase)"
    return f"{weekday} {time_str} ({hours} h {mins} min wait increase)"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_GLOBAL: Optional[RejectionExplainer] = None


def get_explainer() -> RejectionExplainer:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = RejectionExplainer()
    return _GLOBAL


def set_explainer(e: RejectionExplainer) -> None:
    global _GLOBAL
    _GLOBAL = e
