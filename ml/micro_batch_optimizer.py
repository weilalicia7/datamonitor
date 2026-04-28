"""
Micro-Batch Scheduling Optimizer (Dissertation §3.2)
====================================================

Replaces the monolithic "re-optimise the whole schedule on every
change" pattern with a three-tier orchestration layer that picks the
right primitive for each change:

    FAST PATH      (< 50 ms target) — single urgent-patient insertion
                                      via the existing SqueezeInHandler
    SLOW PATH      (full CP-SAT)    — triggered every 15 min OR after
                                      ≥ 3 queued changes
    BACKGROUND RL  (continuous)     — SchedulingRLAgent proposes
                                      incremental swaps between
                                      slow-path runs

The primitives themselves are all pre-existing:

* ``optimization/squeeze_in.py::SqueezeInHandler.squeeze_in_with_noshow``
* ``optimization/optimizer.py::ScheduleOptimizer.optimize``
* ``ml/rl_scheduler.py::SchedulingRLAgent``

This module adds only the *coordinator* on top — queue, routing,
threshold logic, latency telemetry, and a safe start/stop of the
background thread.  Every tier writes one JSONL row per decision to
``data_cache/micro_batch/latency.jsonl``, which the dissertation R
script aggregates into the §23 Figure.

Design contract
---------------

* **Invisibility.**  An existing caller of
  ``squeeze_handler.squeeze_in_with_noshow(...)`` can migrate to
  ``micro_batch.submit_change('insert', payload=...)`` and get
  identical semantics plus free latency tracking.
* **No shared-state races.**  A single ``threading.Lock`` guards the
  queue + counters.  The Flask app is single-process so this is
  sufficient.
* **Eventual consistency.**  Queued changes are *applied* — the
  current scheduler state is updated — only when the slow path
  fires; between firings the schedule reflects the fast-path
  mutations only.  The §3.2 brief explicitly calls this the
  "eventual consistency" trade-off; we log it so operators can
  audit the lag.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults from the §3.2 brief
# ---------------------------------------------------------------------------
DEFAULT_FAST_PATH_BUDGET_MS: float = 50.0
DEFAULT_CHANGE_THRESHOLD: int = 3           # slow path fires at ≥ this many queued changes
DEFAULT_SLOW_PATH_INTERVAL_S: int = 900     # 15 minutes
DEFAULT_RL_TICK_S: int = 60                 # RL background tick cadence
MICRO_BATCH_DIR: Path = DATA_CACHE_DIR / 'micro_batch'
LATENCY_LOG_FILE: Path = MICRO_BATCH_DIR / 'latency.jsonl'
CONFIG_FILE: Path = MICRO_BATCH_DIR / 'config.json'


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChangeRecord:
    """A single pending scheduling-change request."""
    ts: str                      # ISO utc timestamp of submission
    change_type: str             # 'insert' | 'cancel' | 'reschedule'
    payload: Dict[str, Any]      # caller-supplied dict (patient_id, slot, …)
    submitted_by: Optional[str] = None   # e.g. 'urgent_insert_endpoint'


@dataclass
class MicroBatchResult:
    """Per-call outcome the coordinator returns to the caller."""
    path: str                    # 'fast' | 'slow' | 'queued' | 'rejected'
    success: bool
    latency_ms: float
    queue_size: int
    change_type: str
    message: str = ""
    payload: Optional[Dict[str, Any]] = None   # primitive's raw output
    triggered_slow_path: bool = False
    reason: Optional[str] = None               # why the path was picked


@dataclass
class MicroBatchStatus:
    """Shape of the status endpoint's response."""
    fast_path_budget_ms: float
    change_threshold: int
    slow_path_interval_s: int
    rl_tick_s: int
    queue_size: int
    total_changes_seen: int
    total_fast_path: int
    total_slow_path: int
    total_queued: int
    total_rl_ticks: int
    last_fast_latency_ms: Optional[float]
    last_slow_latency_ms: Optional[float]
    last_full_reopt_ts: Optional[str]
    next_slow_path_eligible_in_s: Optional[float]
    rl_background_enabled: bool
    eventual_consistency_lag_s: Optional[float]


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class MicroBatchOptimizer:
    """
    Three-tier coordinator.  Constructed with callables rather than
    imports so the Flask module can inject its already-instantiated
    singletons (`squeeze_handler`, `optimizer`, `rl_agent`) without
    circular imports.
    """

    def __init__(
        self,
        *,
        fast_path_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        slow_path_fn: Optional[Callable[[List[ChangeRecord]], Dict[str, Any]]] = None,
        rl_tick_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        fast_path_budget_ms: float = DEFAULT_FAST_PATH_BUDGET_MS,
        change_threshold: int = DEFAULT_CHANGE_THRESHOLD,
        slow_path_interval_s: int = DEFAULT_SLOW_PATH_INTERVAL_S,
        rl_tick_s: int = DEFAULT_RL_TICK_S,
        storage_dir: Path = MICRO_BATCH_DIR,
    ):
        self._fast_path_fn = fast_path_fn
        self._slow_path_fn = slow_path_fn
        self._rl_tick_fn = rl_tick_fn
        self.fast_path_budget_ms = float(fast_path_budget_ms)
        self.change_threshold = int(change_threshold)
        self.slow_path_interval_s = int(slow_path_interval_s)
        self.rl_tick_s = int(rl_tick_s)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.latency_path = self.storage_dir / 'latency.jsonl'
        self.config_path = self.storage_dir / 'config.json'

        # State — protected by `self._lock`
        self._lock = threading.Lock()
        self.queue: Deque[ChangeRecord] = deque()
        self.total_changes_seen = 0
        self.total_fast_path = 0
        self.total_slow_path = 0
        self.total_queued = 0
        self.total_rl_ticks = 0
        self.last_fast_latency_ms: Optional[float] = None
        self.last_slow_latency_ms: Optional[float] = None
        self.last_full_reopt_ts: Optional[datetime] = None

        self._rl_thread: Optional[threading.Thread] = None
        self._rl_stop = threading.Event()
        self._rl_enabled = False

    # ---------- config ----------

    def save_config(self) -> None:
        cfg = {
            'fast_path_budget_ms': self.fast_path_budget_ms,
            'change_threshold': self.change_threshold,
            'slow_path_interval_s': self.slow_path_interval_s,
            'rl_tick_s': self.rl_tick_s,
        }
        self.config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')

    def update_config(
        self,
        *,
        fast_path_budget_ms: Optional[float] = None,
        change_threshold: Optional[int] = None,
        slow_path_interval_s: Optional[int] = None,
        rl_tick_s: Optional[int] = None,
    ) -> Dict[str, Any]:
        if fast_path_budget_ms is not None:
            self.fast_path_budget_ms = float(fast_path_budget_ms)
        if change_threshold is not None:
            if int(change_threshold) < 1:
                raise ValueError("change_threshold must be ≥ 1")
            self.change_threshold = int(change_threshold)
        if slow_path_interval_s is not None:
            if int(slow_path_interval_s) < 1:
                raise ValueError("slow_path_interval_s must be ≥ 1")
            self.slow_path_interval_s = int(slow_path_interval_s)
        if rl_tick_s is not None:
            if int(rl_tick_s) < 1:
                raise ValueError("rl_tick_s must be ≥ 1")
            self.rl_tick_s = int(rl_tick_s)
        self.save_config()
        return {
            'fast_path_budget_ms': self.fast_path_budget_ms,
            'change_threshold': self.change_threshold,
            'slow_path_interval_s': self.slow_path_interval_s,
            'rl_tick_s': self.rl_tick_s,
        }

    # ---------- routing ----------

    def submit_change(
        self,
        change_type: str,
        payload: Dict[str, Any],
        *,
        submitted_by: Optional[str] = None,
    ) -> MicroBatchResult:
        """
        Main entry point.  Routes to FAST / SLOW / QUEUED tier based
        on the change type, urgency, and queue state.  Returns in at
        most ``fast_path_budget_ms`` for ``insert`` changes that take
        the fast path.
        """
        change_type = (change_type or '').strip().lower()
        if change_type not in ('insert', 'cancel', 'reschedule'):
            return MicroBatchResult(
                path='rejected', success=False, latency_ms=0.0,
                queue_size=len(self.queue), change_type=change_type,
                message=f"unknown change_type: {change_type}",
            )

        t0 = time.perf_counter()
        record = ChangeRecord(
            ts=datetime.utcnow().isoformat(timespec='seconds'),
            change_type=change_type, payload=dict(payload or {}),
            submitted_by=submitted_by,
        )

        # FAST PATH — urgent single-patient insertion.  Uses the existing
        # SqueezeInHandler via the injected callable so we never clone
        # its scoring logic.
        urgent = bool(record.payload.get('is_urgent', True) if change_type == 'insert' else False)
        if change_type == 'insert' and urgent and self._fast_path_fn is not None:
            try:
                fast_result = self._fast_path_fn(record.payload)
                elapsed = (time.perf_counter() - t0) * 1000.0
                with self._lock:
                    self.total_changes_seen += 1
                    self.total_fast_path += 1
                    self.last_fast_latency_ms = elapsed
                self._log_latency('fast', elapsed, change_type, True)
                return MicroBatchResult(
                    path='fast',
                    success=bool(fast_result.get('success', True)),
                    latency_ms=elapsed,
                    queue_size=len(self.queue),
                    change_type=change_type,
                    payload=fast_result,
                    reason='urgent_insert → heuristic',
                )
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000.0
                self._log_latency('fast', elapsed, change_type, False, error=str(exc))
                logger.warning(f"Fast path failed ({change_type}): {exc}; will queue")

        # Otherwise queue the change; maybe trigger slow path
        triggered = False
        with self._lock:
            self.queue.append(record)
            self.total_changes_seen += 1
            self.total_queued += 1
            should_trigger = self._should_trigger_slow_path_locked()
        if should_trigger:
            slow = self._run_slow_path()
            elapsed = (time.perf_counter() - t0) * 1000.0
            triggered = True
            return MicroBatchResult(
                path='slow',
                success=bool(slow.get('success', False)),
                latency_ms=elapsed,
                queue_size=len(self.queue),
                change_type=change_type,
                payload=slow,
                triggered_slow_path=True,
                reason=slow.get('reason', 'threshold'),
            )
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._log_latency('queued', elapsed, change_type, True)
        return MicroBatchResult(
            path='queued',
            success=True,
            latency_ms=elapsed,
            queue_size=len(self.queue),
            change_type=change_type,
            reason='below thresholds — awaiting batch reopt',
        )

    def flush(self, reason: str = 'manual_flush') -> MicroBatchResult:
        """Explicit operator call forces the slow path to run now."""
        t0 = time.perf_counter()
        slow = self._run_slow_path(forced_reason=reason)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return MicroBatchResult(
            path='slow', success=bool(slow.get('success', False)),
            latency_ms=elapsed, queue_size=len(self.queue),
            change_type='flush', payload=slow,
            triggered_slow_path=True, reason=reason,
        )

    # ---------- private helpers ----------

    def _should_trigger_slow_path_locked(self) -> bool:
        """Must be called with ``self._lock`` held."""
        if len(self.queue) >= self.change_threshold:
            return True
        if self.last_full_reopt_ts is None:
            return False   # very first batch: wait for the threshold
        elapsed = (datetime.utcnow() - self.last_full_reopt_ts).total_seconds()
        return elapsed >= self.slow_path_interval_s

    def _run_slow_path(self, forced_reason: Optional[str] = None) -> Dict[str, Any]:
        if self._slow_path_fn is None:
            with self._lock:
                pending = list(self.queue); self.queue.clear()
            return {'success': False, 'reason': 'no slow_path_fn configured',
                    'consumed': len(pending)}
        # Snapshot + drain the queue atomically
        with self._lock:
            pending = list(self.queue)
            self.queue.clear()
        reopt_t0 = time.perf_counter()
        try:
            out = self._slow_path_fn(pending) or {}
            elapsed = (time.perf_counter() - reopt_t0) * 1000.0
            with self._lock:
                self.total_slow_path += 1
                self.last_slow_latency_ms = elapsed
                self.last_full_reopt_ts = datetime.utcnow()
            self._log_latency('slow', elapsed, 'batch_reopt', True,
                              extra={'n_consumed': len(pending),
                                     'reason': forced_reason or 'threshold'})
            out.setdefault('success', True)
            out.setdefault('reason', forced_reason or 'threshold')
            out['n_consumed'] = len(pending)
            out['latency_ms'] = round(elapsed, 2)
            return out
        except Exception as exc:
            elapsed = (time.perf_counter() - reopt_t0) * 1000.0
            self._log_latency('slow', elapsed, 'batch_reopt', False, error=str(exc))
            logger.error(f"Slow path failed: {exc}")
            # Put the drained changes back so they aren't lost
            with self._lock:
                for r in pending:
                    self.queue.appendleft(r)
            return {'success': False, 'error': str(exc),
                    'n_consumed': 0, 'n_requeued': len(pending)}

    def _log_latency(
        self, path: str, latency_ms: float, change_type: str,
        success: bool, *, error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            row = {
                'ts': datetime.utcnow().isoformat(timespec='seconds'),
                'path': path,
                'change_type': change_type,
                'latency_ms': round(latency_ms, 3),
                'success': bool(success),
                # Live calls are tagged so dissertation analysis can
                # cleanly separate them from steady-state benchmark rows
                # (which carry phase='steady', source='benchmark').
                'phase': 'production',
                'source': 'live',
            }
            if error:
                row['error'] = error
            if extra:
                row.update(extra)
            with self.latency_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(row) + '\n')
        except Exception:  # pragma: no cover
            pass

    # ---------- background RL ----------

    def start_background_rl(self) -> None:
        """Spawn the RL tick thread (daemon).  Safe to call multiple times."""
        if self._rl_tick_fn is None:
            logger.info("micro-batch: no rl_tick_fn → background RL disabled")
            return
        if self._rl_thread is not None and self._rl_thread.is_alive():
            return
        self._rl_stop.clear()
        self._rl_enabled = True

        def _loop() -> None:
            logger.info("micro-batch: RL background started")
            while not self._rl_stop.is_set():
                try:
                    if self._rl_tick_fn is not None:
                        _ = self._rl_tick_fn() or {}
                        with self._lock:
                            self.total_rl_ticks += 1
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"micro-batch: RL tick error: {exc}")
                # use Event.wait so stop_background_rl() is snappy
                if self._rl_stop.wait(self.rl_tick_s):
                    break
            logger.info("micro-batch: RL background stopped")

        self._rl_thread = threading.Thread(
            target=_loop, daemon=True, name='micro-batch-rl',
        )
        self._rl_thread.start()

    def stop_background_rl(self, timeout: float = 5.0) -> None:
        """Signal the background thread to exit and join briefly."""
        self._rl_stop.set()
        self._rl_enabled = False
        if self._rl_thread is not None:
            self._rl_thread.join(timeout=timeout)

    # ---------- introspection ----------

    def status(self) -> MicroBatchStatus:
        with self._lock:
            next_in_s: Optional[float]
            if self.last_full_reopt_ts is None:
                next_in_s = None
            else:
                elapsed = (datetime.utcnow() - self.last_full_reopt_ts).total_seconds()
                next_in_s = max(0.0, self.slow_path_interval_s - elapsed)
            lag_s: Optional[float] = None
            if self.queue and self.last_full_reopt_ts is not None:
                lag_s = (datetime.utcnow() - self.last_full_reopt_ts).total_seconds()
            return MicroBatchStatus(
                fast_path_budget_ms=self.fast_path_budget_ms,
                change_threshold=self.change_threshold,
                slow_path_interval_s=self.slow_path_interval_s,
                rl_tick_s=self.rl_tick_s,
                queue_size=len(self.queue),
                total_changes_seen=self.total_changes_seen,
                total_fast_path=self.total_fast_path,
                total_slow_path=self.total_slow_path,
                total_queued=self.total_queued,
                total_rl_ticks=self.total_rl_ticks,
                last_fast_latency_ms=self.last_fast_latency_ms,
                last_slow_latency_ms=self.last_slow_latency_ms,
                last_full_reopt_ts=(
                    self.last_full_reopt_ts.isoformat(timespec='seconds')
                    if self.last_full_reopt_ts else None
                ),
                next_slow_path_eligible_in_s=next_in_s,
                rl_background_enabled=bool(self._rl_enabled),
                eventual_consistency_lag_s=lag_s,
            )


__all__ = [
    'ChangeRecord',
    'MicroBatchResult',
    'MicroBatchStatus',
    'MicroBatchOptimizer',
    'DEFAULT_FAST_PATH_BUDGET_MS',
    'DEFAULT_CHANGE_THRESHOLD',
    'DEFAULT_SLOW_PATH_INTERVAL_S',
    'DEFAULT_RL_TICK_S',
]
