"""
Safety Guardrails with Runtime Monitoring (Dissertation §4.3)
=============================================================

Wraps the CP-SAT optimiser with a post-hoc monitor that rejects or
flags schedules violating hard clinical-safety rules.  The brief
calls for a ``reject_schedule`` gate on outputs that are
"technically optimal but clinically dangerous" — e.g.

    if any(remaining_slack < 5 for critical_patient):
        reject_schedule("Would create cascade risk for high-priority patient")

This module is the general form of that gate.  It owns a registry
of safety rules, runs them against every candidate schedule, and
produces a single :class:`SafetyReport` with verdict
``accept``/``warn``/``reject`` together with an itemised list of
violations.

Design contract
---------------
* **Invisible integration.**  ``flask_app.py`` calls
  ``monitor.evaluate(...)`` on every ``run_optimization()`` result
  and attaches the report to
  ``app_state['optimization_results']['safety_report']``.
  When ``enforce_as_hard_gate`` is set, any CRITICAL violation
  raises ``safety_blocked = True`` and the caller can refuse to
  commit.  No UI panel per brief.
* **Pluggable rules.**  Each rule is a pure function
  ``rule(schedule, patients_by_id, context) -> List[SafetyViolation]``.
  A small built-in set ships (see ``_register_default_rules``);
  new rules can be registered at runtime via
  ``monitor.register_rule``.
* **JSONL persistence.**  Every evaluation writes one row to
  ``data_cache/safety_guardrails/reports.jsonl``; the §28
  dissertation R script consumes that log.
* **Distribution-free.**  No classifier, no distance metric, no
  dependency beyond ``datetime`` arithmetic — rules operate on
  already-computed schedule fields (``start_time``, ``end_time``,
  ``chair_id``, ``priority``, …) and on the patient catalogue.

Built-in rule set
-----------------
The default set focuses on the failure modes the SACT brief
cites:

* ``critical_slack_floor`` (CRITICAL) — high-priority patients
  must have ≥ 5 min slack on both sides, else a cascade from any
  small delay.
* ``chair_capacity`` (CRITICAL) — no site's chairs + recliners are
  over-booked.
* ``concurrent_chair_overlap`` (CRITICAL) — no chair has two
  appointments overlapping unless both are marked double-booked.
* ``long_infusion_cutoff`` (HIGH) — infusions > 180 min must start
  by 13:00 so the chair is free by closing.
* ``travel_time_ceiling`` (HIGH) — no patient assigned a site
  farther than a configurable max distance.
* ``wait_time_ceiling`` (HIGH) — no patient waits > cut-off past
  their ``earliest_time``.
* ``consecutive_high_noshow`` (MODERATE) — ≥ 3 high-NS-risk
  patients in a row on the same chair.
* ``double_book_density`` (MODERATE) — no chair-hour has > 1
  double-booking.

Any rule can be disabled at runtime via
``POST /api/safety/config {rule_name: enabled=false}``.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SAFETY_DIR: Path = DATA_CACHE_DIR / "safety_guardrails"
SAFETY_LOG: Path = SAFETY_DIR / "reports.jsonl"
VERDICT_ACCEPT = "accept"
VERDICT_WARN = "warn"
VERDICT_REJECT = "reject"

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_HIGH = "HIGH"
SEVERITY_MODERATE = "MODERATE"
SEVERITY_LOW = "LOW"

# Built-in thresholds — tuneable via /api/safety/config
DEFAULT_CRITICAL_PRIORITIES: Tuple[int, ...] = (1, 2)
DEFAULT_MIN_SLACK_MIN: int = 5
DEFAULT_LONG_INFUSION_MIN: int = 180
DEFAULT_LONG_INFUSION_CUTOFF_HOUR: int = 13
DEFAULT_MAX_TRAVEL_MIN: int = 120
DEFAULT_MAX_WAIT_MIN: int = 180
DEFAULT_HIGH_NOSHOW_THRESHOLD: float = 0.40
DEFAULT_HIGH_NOSHOW_RUN: int = 3
DEFAULT_MAX_DOUBLE_BOOK_PER_CHAIR_HOUR: int = 1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SafetyViolation:
    rule_name: str
    severity: str
    detail: str
    patient_id: Optional[str] = None
    chair_id: Optional[str] = None
    site_code: Optional[str] = None
    start_time: Optional[str] = None
    suggested_fix: str = ""


@dataclass
class SafetyRuleConfig:
    name: str
    enabled: bool = True
    severity: str = SEVERITY_HIGH
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyReport:
    computed_ts: str
    verdict: str                          # accept / warn / reject
    n_appointments: int
    n_violations: int
    n_critical: int
    n_high: int
    n_moderate: int
    n_low: int
    rules_evaluated: int
    rules_tripped: List[str]
    narrative: str
    enforce_as_hard_gate: bool
    violations: List[SafetyViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


# A rule is a callable that takes the schedule, patient catalogue, and
# a context dict, and returns a list of violations.  Kept as a plain
# callable protocol rather than a Protocol class to stay import-free.
RuleFn = Callable[[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]], List[SafetyViolation]]


class SafetyGuardrailsMonitor:
    """Runtime monitor: evaluate safety rules, emit a verdict."""

    def __init__(
        self,
        *,
        enforce_as_hard_gate: bool = True,
        storage_dir: Path = SAFETY_DIR,
    ):
        self.enforce_as_hard_gate = bool(enforce_as_hard_gate)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.reports_log = self.storage_dir / "reports.jsonl"

        self._lock = threading.Lock()
        self._rules: Dict[str, Tuple[RuleFn, SafetyRuleConfig]] = {}
        self._last: Optional[SafetyReport] = None
        self._n_runs: int = 0

        # Register default rule set
        _register_default_rules(self)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def register_rule(
        self,
        name: str,
        fn: RuleFn,
        severity: str = SEVERITY_HIGH,
        description: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = SafetyRuleConfig(
            name=name,
            severity=severity,
            description=description,
            params=dict(params or {}),
        )
        with self._lock:
            self._rules[name] = (fn, cfg)

    def evaluate(
        self,
        appointments: Sequence[Any],
        patients: Optional[Sequence[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyReport:
        """Run every enabled rule and return a report."""
        appts = [_normalise_appointment(a) for a in appointments]
        patients = list(patients or [])
        patient_by_id: Dict[str, Dict[str, Any]] = {}
        for p in patients:
            pid = p.get("Patient_ID") or p.get("patient_id")
            if pid is None:
                continue
            patient_by_id[str(pid)] = p
        ctx = dict(context or {})

        violations: List[SafetyViolation] = []
        rules_tripped: List[str] = []
        rules_evaluated = 0
        with self._lock:
            rule_items = list(self._rules.items())
        for name, (fn, cfg) in rule_items:
            if not cfg.enabled:
                continue
            rules_evaluated += 1
            try:
                result = fn(appts, patient_by_id, {**ctx, "_params": cfg.params})
            except Exception as exc:  # pragma: no cover — keep monitor robust
                logger.warning(f"Safety rule '{name}' crashed: {exc}")
                continue
            if not result:
                continue
            rules_tripped.append(name)
            for v in result:
                # Ensure every violation carries the rule-level severity
                if not getattr(v, "severity", None):
                    v.severity = cfg.severity
                # Promote the rule name if the rule forgot to set it
                if not getattr(v, "rule_name", None):
                    v.rule_name = name
                violations.append(v)

        n_c = sum(1 for v in violations if v.severity == SEVERITY_CRITICAL)
        n_h = sum(1 for v in violations if v.severity == SEVERITY_HIGH)
        n_m = sum(1 for v in violations if v.severity == SEVERITY_MODERATE)
        n_l = sum(1 for v in violations if v.severity == SEVERITY_LOW)

        # Verdict rule: any CRITICAL → reject; any HIGH or MODERATE → warn; else accept
        if n_c > 0:
            verdict = VERDICT_REJECT
        elif n_h > 0 or n_m > 0:
            verdict = VERDICT_WARN
        else:
            verdict = VERDICT_ACCEPT

        narrative = _build_narrative(
            verdict=verdict,
            n_violations=len(violations),
            n_c=n_c, n_h=n_h, n_m=n_m, n_l=n_l,
            rules_tripped=rules_tripped,
            top=violations[0] if violations else None,
        )

        report = SafetyReport(
            computed_ts=datetime.utcnow().isoformat(timespec="seconds"),
            verdict=verdict,
            n_appointments=len(appts),
            n_violations=len(violations),
            n_critical=n_c,
            n_high=n_h,
            n_moderate=n_m,
            n_low=n_l,
            rules_evaluated=rules_evaluated,
            rules_tripped=rules_tripped,
            narrative=narrative,
            enforce_as_hard_gate=self.enforce_as_hard_gate,
            violations=violations,
        )
        with self._lock:
            self._last = report
            self._n_runs += 1
        self._append_event(report)
        return report

    def last(self) -> Optional[SafetyReport]:
        with self._lock:
            return self._last

    def status(self) -> Dict[str, Any]:
        last = self.last()
        with self._lock:
            rules_meta = [
                {
                    "name": c.name,
                    "enabled": c.enabled,
                    "severity": c.severity,
                    "description": c.description,
                    "params": dict(c.params),
                }
                for (_, c) in self._rules.values()
            ]
        return {
            "enforce_as_hard_gate": self.enforce_as_hard_gate,
            "total_runs": self._n_runs,
            "log_path": str(self.reports_log),
            "rules": rules_meta,
            "last_verdict": last.verdict if last else None,
            "last_n_violations": last.n_violations if last else None,
            "last_n_critical": last.n_critical if last else None,
            "last_rules_tripped": list(last.rules_tripped) if last else None,
            "last_narrative": last.narrative if last else None,
            "last_computed_ts": last.computed_ts if last else None,
        }

    def update_config(
        self,
        enforce_as_hard_gate: Optional[bool] = None,
        rules: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Retune either the global gate flag or per-rule overrides.

        ``rules`` is ``{rule_name: {enabled?: bool, severity?: str,
        params?: {...}}}``.
        """
        if enforce_as_hard_gate is not None:
            self.enforce_as_hard_gate = bool(enforce_as_hard_gate)
        if rules:
            with self._lock:
                for rname, overrides in rules.items():
                    if rname not in self._rules:
                        continue
                    fn, cfg = self._rules[rname]
                    if "enabled" in overrides:
                        cfg.enabled = bool(overrides["enabled"])
                    if "severity" in overrides:
                        cfg.severity = str(overrides["severity"])
                    if "params" in overrides:
                        cfg.params.update(dict(overrides["params"]))
                    self._rules[rname] = (fn, cfg)
        return self.status()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _append_event(self, report: SafetyReport) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": report.computed_ts,
                "verdict": report.verdict,
                "n_appointments": report.n_appointments,
                "n_violations": report.n_violations,
                "n_critical": report.n_critical,
                "n_high": report.n_high,
                "n_moderate": report.n_moderate,
                "n_low": report.n_low,
                "rules_tripped": list(report.rules_tripped),
                "narrative": report.narrative,
                "enforce_as_hard_gate": report.enforce_as_hard_gate,
                "top_violations": [
                    {
                        "rule_name": v.rule_name,
                        "severity": v.severity,
                        "detail": v.detail,
                        "patient_id": v.patient_id,
                    }
                    for v in report.violations[:10]
                ],
            }
            with open(self.reports_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Safety-guardrails log write failed: {exc}")


# ---------------------------------------------------------------------------
# Built-in rules — each returns List[SafetyViolation]
# ---------------------------------------------------------------------------


def _register_default_rules(monitor: SafetyGuardrailsMonitor) -> None:
    monitor.register_rule(
        name="critical_slack_floor",
        fn=_rule_critical_slack_floor,
        severity=SEVERITY_CRITICAL,
        description=(
            "High-priority patients (priority in critical_priorities) must "
            "have >= min_slack_minutes of remaining slack on both sides."
        ),
        params={
            "critical_priorities": list(DEFAULT_CRITICAL_PRIORITIES),
            "min_slack_minutes": DEFAULT_MIN_SLACK_MIN,
        },
    )
    monitor.register_rule(
        name="chair_capacity",
        fn=_rule_chair_capacity,
        severity=SEVERITY_CRITICAL,
        description="No site can dispatch more concurrent appointments than its chair+recliner count.",
        params={},
    )
    monitor.register_rule(
        name="concurrent_chair_overlap",
        fn=_rule_concurrent_chair_overlap,
        severity=SEVERITY_CRITICAL,
        description="No chair can hold two overlapping appointments unless explicitly double-booked.",
        params={},
    )
    monitor.register_rule(
        name="long_infusion_cutoff",
        fn=_rule_long_infusion_cutoff,
        severity=SEVERITY_HIGH,
        description="Infusions >= long_infusion_minutes must start on or before cutoff_hour:00.",
        params={
            "long_infusion_minutes": DEFAULT_LONG_INFUSION_MIN,
            "cutoff_hour": DEFAULT_LONG_INFUSION_CUTOFF_HOUR,
        },
    )
    monitor.register_rule(
        name="travel_time_ceiling",
        fn=_rule_travel_time_ceiling,
        severity=SEVERITY_HIGH,
        description="No patient may be assigned a site whose travel time exceeds max_travel_minutes.",
        params={"max_travel_minutes": DEFAULT_MAX_TRAVEL_MIN},
    )
    monitor.register_rule(
        name="wait_time_ceiling",
        fn=_rule_wait_time_ceiling,
        severity=SEVERITY_HIGH,
        description="No patient may wait > max_wait_minutes past their earliest_time.",
        params={"max_wait_minutes": DEFAULT_MAX_WAIT_MIN},
    )
    monitor.register_rule(
        name="consecutive_high_noshow",
        fn=_rule_consecutive_high_noshow,
        severity=SEVERITY_MODERATE,
        description=(
            ">= run_length consecutive high-noshow-risk patients on the same "
            "chair creates a cascade risk if all three actually no-show."
        ),
        params={
            "high_noshow_threshold": DEFAULT_HIGH_NOSHOW_THRESHOLD,
            "run_length": DEFAULT_HIGH_NOSHOW_RUN,
        },
    )
    monitor.register_rule(
        name="double_book_density",
        fn=_rule_double_book_density,
        severity=SEVERITY_MODERATE,
        description="No chair-hour may carry more than max_double_book_per_chair_hour double-bookings.",
        params={
            "max_double_book_per_chair_hour": DEFAULT_MAX_DOUBLE_BOOK_PER_CHAIR_HOUR,
        },
    )


def _rule_critical_slack_floor(
    appts: List[Dict[str, Any]],
    patients_by_id: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
) -> List[SafetyViolation]:
    params = context.get("_params", {}) or {}
    crit = set(int(p) for p in params.get("critical_priorities",
                                          DEFAULT_CRITICAL_PRIORITIES))
    min_slack = int(params.get("min_slack_minutes", DEFAULT_MIN_SLACK_MIN))
    # Group by chair → sorted
    by_chair: Dict[str, List[Dict[str, Any]]] = {}
    for a in appts:
        by_chair.setdefault(a.get("chair_id", "_UNK"), []).append(a)
    for k in by_chair:
        by_chair[k].sort(key=lambda a: a["start_time"] or datetime.min)
    out: List[SafetyViolation] = []
    for chair_id, items in by_chair.items():
        for idx, a in enumerate(items):
            pid = str(a.get("patient_id") or "")
            priority = int(a.get("priority") or 3)
            if priority not in crit:
                continue
            slack_before = _gap_minutes(
                items[idx - 1]["end_time"] if idx > 0 else None,
                a["start_time"],
            )
            slack_after = _gap_minutes(
                a["end_time"],
                items[idx + 1]["start_time"] if idx + 1 < len(items) else None,
            )
            if slack_before is None:
                slack_before = 999
            if slack_after is None:
                slack_after = 999
            slack = min(slack_before, slack_after)
            if slack < min_slack:
                out.append(
                    SafetyViolation(
                        rule_name="critical_slack_floor",
                        severity=SEVERITY_CRITICAL,
                        patient_id=pid,
                        chair_id=chair_id,
                        start_time=_fmt(a.get("start_time")),
                        detail=(
                            f"priority-{priority} patient has only {slack} min "
                            f"slack (min required={min_slack})"
                        ),
                        suggested_fix=(
                            "move the neighbouring appointment to open >= "
                            f"{min_slack} min slack either side"
                        ),
                    )
                )
    return out


def _rule_chair_capacity(
    appts: List[Dict[str, Any]],
    patients_by_id: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
) -> List[SafetyViolation]:
    # Count concurrent appointments per site at start-time of each appt
    by_site: Dict[str, List[Dict[str, Any]]] = {}
    for a in appts:
        by_site.setdefault(a.get("site_code", "_UNK"), []).append(a)
    try:
        from config import DEFAULT_SITES
        cap = {s["code"]: int(s.get("chairs", 0)) + int(s.get("recliners", 0))
               for s in DEFAULT_SITES}
    except Exception:
        cap = {}
    out: List[SafetyViolation] = []
    for sc, items in by_site.items():
        # For each appt, count how many others overlap
        for i, a in enumerate(items):
            t_start = a["start_time"]
            if t_start is None:
                continue
            concurrent = 1 + sum(
                1 for j, b in enumerate(items)
                if j != i and _overlap(a, b)
            )
            site_cap = cap.get(sc, 999)
            if site_cap and concurrent > site_cap:
                out.append(
                    SafetyViolation(
                        rule_name="chair_capacity",
                        severity=SEVERITY_CRITICAL,
                        patient_id=str(a.get("patient_id") or ""),
                        chair_id=a.get("chair_id"),
                        site_code=sc,
                        start_time=_fmt(t_start),
                        detail=(
                            f"{concurrent} concurrent appointments at {sc} "
                            f"exceed capacity ({site_cap})"
                        ),
                        suggested_fix=(
                            f"reassign overflow to a neighbouring site"
                        ),
                    )
                )
    return out


def _rule_concurrent_chair_overlap(
    appts: List[Dict[str, Any]],
    patients_by_id: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
) -> List[SafetyViolation]:
    by_chair: Dict[str, List[Dict[str, Any]]] = {}
    for a in appts:
        by_chair.setdefault(a.get("chair_id", "_UNK"), []).append(a)
    out: List[SafetyViolation] = []
    seen_pairs: set = set()
    for chair_id, items in by_chair.items():
        for i, a in enumerate(items):
            for j, b in enumerate(items):
                if j <= i:
                    continue
                if not _overlap(a, b):
                    continue
                # Allow explicit double-bookings
                if bool(a.get("double_booked")) or bool(b.get("double_booked")):
                    continue
                key = tuple(sorted([
                    str(a.get("patient_id") or ""),
                    str(b.get("patient_id") or ""),
                ]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                out.append(
                    SafetyViolation(
                        rule_name="concurrent_chair_overlap",
                        severity=SEVERITY_CRITICAL,
                        patient_id=key[0],
                        chair_id=chair_id,
                        start_time=_fmt(a["start_time"]),
                        detail=(
                            f"overlap: {key[0]} and {key[1]} on chair {chair_id}"
                        ),
                        suggested_fix="move one of the overlapping appointments",
                    )
                )
    return out


def _rule_long_infusion_cutoff(
    appts: List[Dict[str, Any]],
    patients_by_id: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
) -> List[SafetyViolation]:
    params = context.get("_params", {}) or {}
    long_min = int(params.get("long_infusion_minutes", DEFAULT_LONG_INFUSION_MIN))
    cutoff_hour = int(params.get("cutoff_hour", DEFAULT_LONG_INFUSION_CUTOFF_HOUR))
    out: List[SafetyViolation] = []
    for a in appts:
        dur = int(a.get("duration") or 0)
        if dur < long_min:
            continue
        t = a.get("start_time")
        if t is None:
            continue
        if t.hour > cutoff_hour or (t.hour == cutoff_hour and t.minute > 0):
            out.append(
                SafetyViolation(
                    rule_name="long_infusion_cutoff",
                    severity=SEVERITY_HIGH,
                    patient_id=str(a.get("patient_id") or ""),
                    chair_id=a.get("chair_id"),
                    start_time=_fmt(t),
                    detail=(
                        f"{dur}-min infusion starts at {t.strftime('%H:%M')} "
                        f"(cutoff {cutoff_hour:02d}:00)"
                    ),
                    suggested_fix="move to a morning chair slot",
                )
            )
    return out


def _rule_travel_time_ceiling(
    appts: List[Dict[str, Any]],
    patients_by_id: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
) -> List[SafetyViolation]:
    params = context.get("_params", {}) or {}
    max_tt = int(params.get("max_travel_minutes", DEFAULT_MAX_TRAVEL_MIN))
    out: List[SafetyViolation] = []
    for a in appts:
        tt = int(a.get("travel_time") or 0)
        if tt > max_tt:
            out.append(
                SafetyViolation(
                    rule_name="travel_time_ceiling",
                    severity=SEVERITY_HIGH,
                    patient_id=str(a.get("patient_id") or ""),
                    chair_id=a.get("chair_id"),
                    start_time=_fmt(a.get("start_time")),
                    detail=f"travel {tt} min exceeds {max_tt} min ceiling",
                    suggested_fix="reassign to a closer site",
                )
            )
    return out


def _rule_wait_time_ceiling(
    appts: List[Dict[str, Any]],
    patients_by_id: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
) -> List[SafetyViolation]:
    params = context.get("_params", {}) or {}
    max_wait = int(params.get("max_wait_minutes", DEFAULT_MAX_WAIT_MIN))
    out: List[SafetyViolation] = []
    for a in appts:
        pid = str(a.get("patient_id") or "")
        p = patients_by_id.get(pid)
        if not p:
            continue
        earliest = _coerce_dt(p.get("earliest_time") or p.get("Earliest_Time"))
        start = a.get("start_time")
        if earliest is None or start is None:
            continue
        wait = int((start - earliest).total_seconds() // 60)
        if wait > max_wait:
            out.append(
                SafetyViolation(
                    rule_name="wait_time_ceiling",
                    severity=SEVERITY_HIGH,
                    patient_id=pid,
                    chair_id=a.get("chair_id"),
                    start_time=_fmt(start),
                    detail=f"wait {wait} min exceeds {max_wait} min ceiling",
                    suggested_fix="bring earlier in the day",
                )
            )
    return out


def _rule_consecutive_high_noshow(
    appts: List[Dict[str, Any]],
    patients_by_id: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
) -> List[SafetyViolation]:
    params = context.get("_params", {}) or {}
    thresh = float(params.get("high_noshow_threshold", DEFAULT_HIGH_NOSHOW_THRESHOLD))
    run_len = int(params.get("run_length", DEFAULT_HIGH_NOSHOW_RUN))

    def noshow_rate(a: Dict[str, Any]) -> float:
        pid = str(a.get("patient_id") or "")
        p = patients_by_id.get(pid, {})
        r = (
            p.get("no_show_rate")
            or p.get("Patient_NoShow_Rate")
            or p.get("noshow_rate")
        )
        try:
            return float(r) if r is not None else 0.0
        except Exception:
            return 0.0

    by_chair: Dict[str, List[Dict[str, Any]]] = {}
    for a in appts:
        by_chair.setdefault(a.get("chair_id", "_UNK"), []).append(a)
    out: List[SafetyViolation] = []
    for chair_id, items in by_chair.items():
        items.sort(key=lambda a: a["start_time"] or datetime.min)
        run = 0
        run_ids: List[str] = []
        for a in items:
            if noshow_rate(a) >= thresh:
                run += 1
                run_ids.append(str(a.get("patient_id") or ""))
                if run >= run_len:
                    out.append(
                        SafetyViolation(
                            rule_name="consecutive_high_noshow",
                            severity=SEVERITY_MODERATE,
                            chair_id=chair_id,
                            start_time=_fmt(a["start_time"]),
                            patient_id=run_ids[-1],
                            detail=(
                                f"{run} consecutive high-no-show patients on "
                                f"{chair_id} ({', '.join(run_ids[-run_len:])})"
                            ),
                            suggested_fix=(
                                "interleave a low-risk patient to break the run"
                            ),
                        )
                    )
                    run = 0
                    run_ids = []
            else:
                run = 0
                run_ids = []
    return out


def _rule_double_book_density(
    appts: List[Dict[str, Any]],
    patients_by_id: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
) -> List[SafetyViolation]:
    params = context.get("_params", {}) or {}
    max_db = int(
        params.get("max_double_book_per_chair_hour", DEFAULT_MAX_DOUBLE_BOOK_PER_CHAIR_HOUR)
    )
    by_chair_hour: Dict[Tuple[str, int], int] = {}
    example: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for a in appts:
        if not bool(a.get("double_booked")):
            continue
        t = a.get("start_time")
        if t is None:
            continue
        key = (str(a.get("chair_id") or "_UNK"), int(t.hour))
        by_chair_hour[key] = by_chair_hour.get(key, 0) + 1
        example.setdefault(key, a)
    out: List[SafetyViolation] = []
    for key, count in by_chair_hour.items():
        if count > max_db:
            a = example[key]
            out.append(
                SafetyViolation(
                    rule_name="double_book_density",
                    severity=SEVERITY_MODERATE,
                    chair_id=key[0],
                    patient_id=str(a.get("patient_id") or ""),
                    start_time=_fmt(a.get("start_time")),
                    detail=(
                        f"{count} double-bookings on chair {key[0]} in hour "
                        f"{key[1]:02d}:00-{key[1]+1:02d}:00 (max {max_db})"
                    ),
                    suggested_fix="move excess double-books to adjacent hours",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_appointment(a: Any) -> Dict[str, Any]:
    """Convert a ScheduledAppointment / dict to a uniform dict with parsed times."""
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
            "travel_time": getattr(a, "travel_time", None),
            "double_booked": getattr(a, "double_booked", False),
        }
    d["start_time"] = _coerce_dt(d.get("start_time"))
    d["end_time"] = _coerce_dt(d.get("end_time"))
    # Derive end_time from start + duration if missing
    if d["end_time"] is None and d["start_time"] is not None:
        dur = d.get("duration") or 0
        try:
            d["end_time"] = d["start_time"] + timedelta(minutes=int(dur))
        except Exception:
            pass
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
            # "HH:MM" or "HH:MM:SS" — attach today's date
            try:
                base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                parts = v.strip().split(":")
                hh = int(parts[0]); mm = int(parts[1]) if len(parts) > 1 else 0
                return base.replace(hour=hh, minute=mm)
            except Exception:
                return None
    return None


def _gap_minutes(a: Any, b: Any) -> Optional[int]:
    a_dt = _coerce_dt(a)
    b_dt = _coerce_dt(b)
    if a_dt is None or b_dt is None:
        return None
    return int((b_dt - a_dt).total_seconds() // 60)


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    s1, e1 = a.get("start_time"), a.get("end_time")
    s2, e2 = b.get("start_time"), b.get("end_time")
    if None in (s1, e1, s2, e2):
        return False
    return s1 < e2 and s2 < e1


def _fmt(t: Any) -> Optional[str]:
    t = _coerce_dt(t)
    return t.isoformat(timespec="seconds") if t else None


def _build_narrative(
    *, verdict: str, n_violations: int, n_c: int, n_h: int, n_m: int, n_l: int,
    rules_tripped: List[str], top: Optional[SafetyViolation],
) -> str:
    if verdict == VERDICT_ACCEPT:
        return "Safety guardrails: ACCEPT -- 0 violations across enabled rules."
    base = (
        f"Safety guardrails: {verdict.upper()} -- {n_violations} total "
        f"violations (C={n_c}, H={n_h}, M={n_m}, L={n_l}), rules tripped: "
        f"[{', '.join(sorted(set(rules_tripped)))}]"
    )
    if verdict == VERDICT_REJECT and top is not None:
        base += (
            f".  Top trip: '{top.rule_name}' -- {top.detail}"
        )
        if top.suggested_fix:
            base += f"  -> Fix: {top.suggested_fix}."
    elif verdict == VERDICT_WARN and top is not None:
        base += f".  Representative: {top.detail}."
    return base + ("" if base.endswith(".") else ".")


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors §3.1-§4.2 style)
# ---------------------------------------------------------------------------

_GLOBAL: Optional[SafetyGuardrailsMonitor] = None


def get_monitor() -> SafetyGuardrailsMonitor:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = SafetyGuardrailsMonitor()
    return _GLOBAL


def set_monitor(m: SafetyGuardrailsMonitor) -> None:
    global _GLOBAL
    _GLOBAL = m
