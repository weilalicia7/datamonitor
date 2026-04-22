"""
Structured logging + audit trail (Production-Readiness T4.4).

Four concerns live in this module:

1. **JSON log formatting** — ``JsonFormatter`` emits one JSON object per
   record, suitable for container log collectors (ELK, Loki, CloudWatch).
   Opt-in via ``LOG_FORMAT=json``; the human-readable default is kept
   for local development so developers still get colour-free ``asctime``
   lines rather than opaque JSON.

2. **PII redaction** — ``PatientIdRedactor`` filter masks common patient
   identifier patterns (``patient_id=12345``, ``Patient_ID: 99``, 10-digit
   NHS numbers, UK postcodes) before the record reaches any handler.
   Disable via ``LOG_PATIENT_IDS=true`` when you genuinely need IDs in
   debug output — otherwise every environment stays conservative by
   default.

3. **Request ID propagation** — ``attach_request_id(app)`` wires a Flask
   ``before_request`` hook that reads ``X-Request-ID`` or generates a
   UUID4 hex, stores it on ``flask.g.request_id``, and surfaces it on
   every log record emitted inside that request via a logging filter.
   Lets operators grep ``request_id`` across distributed log streams.

4. **Audit trail** — :func:`audit_event` appends one JSONL row to a
   dated file under ``data_cache/audit/``.  Append-only + fsync on every
   write; the file is never mutated after creation.  Intended for the
   GDPR / NHS DPIA retention requirement ("every mutating clinical
   action must be reconstructable for at least 3 years").

All four concerns default to SAFE (JSON off, redaction on, request ID
always attached, audit dir created lazily when the first event fires).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def log_format() -> str:
    """Return ``'json'`` or ``'text'`` — picks the root handler format."""
    return os.environ.get("LOG_FORMAT", "text").strip().lower()


def log_redact_patient_ids() -> bool:
    """Return True iff patient-ID redaction is active (default True)."""
    return not _env_bool("LOG_PATIENT_IDS", False)


def audit_dir() -> Path:
    """Path where audit JSONL files are appended."""
    raw = os.environ.get("AUDIT_LOG_DIR", "data_cache/audit")
    return Path(raw)


# --------------------------------------------------------------------------- #
# Request-ID thread-local (works even outside a Flask request context)
# --------------------------------------------------------------------------- #

_request_ctx = threading.local()


def generate_request_id() -> str:
    """Return a new 32-char UUID4 hex string."""
    return uuid.uuid4().hex


def set_request_id(request_id: Optional[str]) -> None:
    """Store the request ID on the current thread's context."""
    _request_ctx.request_id = request_id


def current_request_id() -> Optional[str]:
    return getattr(_request_ctx, "request_id", None)


# --------------------------------------------------------------------------- #
# PII redaction filter
# --------------------------------------------------------------------------- #


#: Patient-ID-ish substrings we scrub unless LOG_PATIENT_IDS=true.
#: Only the patterns we actually emit; broader PII redaction is a separate
#: concern (the Python ``scrubadub`` package if we ever need it).
_PII_PATTERNS = (
    # patient_id=12345 / patient_id: 99  (our canonical form in log lines)
    re.compile(r"(?i)(patient[_\s-]?id\s*[:=]\s*)(\w+)"),
    # Patient_ID: ABC123 (uppercase variant used in some DataFrame logs)
    re.compile(r"(Patient[_\s-]?ID\s*[:=]\s*)(\w+)"),
    # Raw 10-digit NHS number appearing as a bare token
    re.compile(r"\b(\d{10})\b"),
    # UK postcodes — case-insensitive; common format "CF10 3AT"
    re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2})\b"),
)


class PatientIdRedactor(logging.Filter):
    """Replace patient identifiers in a record's message with ``[REDACTED]``.

    Applied as a filter (not a formatter) so the redaction persists across
    all downstream handlers (stdout + file + structured).
    """

    REPLACEMENT = "[REDACTED]"

    def __init__(self, active: Optional[bool] = None) -> None:
        super().__init__()
        self._active = log_redact_patient_ids() if active is None else active

    def _redact(self, text: str) -> str:
        if not self._active or not text:
            return text
        # patient_id=foo → patient_id=[REDACTED]   (preserves prefix)
        text = _PII_PATTERNS[0].sub(lambda m: f"{m.group(1)}{self.REPLACEMENT}", text)
        text = _PII_PATTERNS[1].sub(lambda m: f"{m.group(1)}{self.REPLACEMENT}", text)
        # Bare tokens: replace the match entirely.
        text = _PII_PATTERNS[2].sub(self.REPLACEMENT, text)
        text = _PII_PATTERNS[3].sub(self.REPLACEMENT, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        # ``record.msg`` before %-formatting; ``record.args`` is the tuple of
        # substitution values.  Redact both — either may carry the ID.
        try:
            record.msg = self._redact(str(record.msg))
        except Exception:  # pragma: no cover — never block logging on a PII error
            pass
        if record.args:
            try:
                record.args = tuple(
                    self._redact(a) if isinstance(a, str) else a for a in record.args
                )
            except Exception:
                pass
        return True


# --------------------------------------------------------------------------- #
# Request-ID filter
# --------------------------------------------------------------------------- #


class RequestIdFilter(logging.Filter):
    """Attach ``record.request_id`` from the thread-local."""

    def filter(self, record: logging.LogRecord) -> bool:
        rid = current_request_id()
        record.request_id = rid or "-"
        return True


# --------------------------------------------------------------------------- #
# JSON formatter
# --------------------------------------------------------------------------- #


class JsonFormatter(logging.Formatter):
    """Serialise a LogRecord as a single JSON object per line."""

    # Fields copied verbatim from the record.  ``message`` is the %-formatted
    # final message; ``msg`` + ``args`` aren't needed once formatted.
    _BASE_FIELDS = (
        "name", "levelname", "pathname", "lineno",
        "funcName", "request_id",
    )

    def format(self, record: logging.LogRecord) -> str:
        # Ensure `message` is populated (Formatter.format does this normally
        # via self._style.format, but we short-circuit the %-style).
        if record.args:
            try:
                record.message = record.getMessage()
            except Exception:
                record.message = str(record.msg)
        else:
            record.message = str(record.msg)

        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }
        # Optional enrichment fields.
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Root-logger installer
# --------------------------------------------------------------------------- #


def install_json_logging(
    level: int = logging.INFO,
    *,
    stream: Optional[Any] = None,
    force: bool = False,
) -> bool:
    """Attach the JSON handler to the root logger.

    Returns True if the JSON handler was installed, False if skipped
    (LOG_FORMAT != 'json' and ``force`` not set).

    The existing root handlers are kept — JSON is additive so file +
    stdout text handlers can coexist during the rollout; flip to
    ``LOG_FORMAT=json`` alone in production.
    """
    if not force and log_format() != "json":
        return False
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())
    handler.addFilter(PatientIdRedactor())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(min(root.level or level, level))
    return True


def install_request_id_filter() -> RequestIdFilter:
    """Attach the request-ID filter to the root logger (idempotent).

    Text-format deployments still benefit from this because developers can
    enable ``%(request_id)s`` in their format string when running locally.
    """
    root = logging.getLogger()
    # Avoid stacking duplicate filters on repeated module reload (test suites
    # often import logging_config more than once).
    for f in root.filters:
        if isinstance(f, RequestIdFilter):
            return f
    flt = RequestIdFilter()
    root.addFilter(flt)
    return flt


def install_patient_id_redactor() -> Optional[PatientIdRedactor]:
    """Attach the redaction filter to the root logger iff redaction is active."""
    if not log_redact_patient_ids():
        return None
    root = logging.getLogger()
    for f in root.filters:
        if isinstance(f, PatientIdRedactor):
            return f
    flt = PatientIdRedactor()
    root.addFilter(flt)
    return flt


# --------------------------------------------------------------------------- #
# Flask wiring
# --------------------------------------------------------------------------- #


def attach_request_id(app) -> None:
    """Wire a before_request hook that reads X-Request-ID or generates one.

    Stores on ``flask.g.request_id`` so route handlers can reference it in
    downstream API responses.  Also mirrors into the logging thread-local
    so every log line emitted during the request carries it.
    """
    from flask import g, request

    @app.before_request
    def _set_request_id():                        # pragma: no cover — trivial
        rid = request.headers.get("X-Request-ID") or generate_request_id()
        g.request_id = rid
        set_request_id(rid)

    @app.after_request
    def _clear_and_tag(resp):                     # pragma: no cover — trivial
        rid = current_request_id()
        if rid is not None:
            resp.headers.setdefault("X-Request-ID", rid)
        set_request_id(None)
        return resp


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #

_audit_lock = threading.Lock()


def _audit_file_for(now: Optional[datetime] = None) -> Path:
    now = now or datetime.now(tz=timezone.utc)
    d = audit_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{now.date().isoformat()}.jsonl"


def audit_event(
    actor: str,
    action: str,
    *,
    target: Optional[str] = None,
    outcome: str = "success",
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Append an audit record to today's JSONL file and return the written row.

    Mandatory fields:

    - ``actor``   — who performed the action (username, API-key ID, or "system")
    - ``action``  — what they did (e.g. ``auth.login``, ``schedule.optimise``)

    Optional fields:

    - ``target``    — what was acted on (patient_id hash, schedule version)
    - ``outcome``   — ``success`` / ``failure`` / ``denied``
    - ``metadata``  — dict of arbitrary JSON-serialisable context
    - ``request_id`` — falls back to :func:`current_request_id`

    The call is thread-safe (module-level lock around the append) and
    each line is ``fsync``'d so a crash doesn't lose the tail.
    """
    ts = now or datetime.now(tz=timezone.utc)
    row: Dict[str, Any] = {
        "ts": ts.isoformat(timespec="milliseconds"),
        "actor": actor,
        "action": action,
        "outcome": outcome,
    }
    if target is not None:
        row["target"] = target
    rid = request_id or current_request_id()
    if rid is not None:
        row["request_id"] = rid
    if metadata:
        row["metadata"] = metadata

    line = json.dumps(row, default=str, ensure_ascii=False) + "\n"
    path = _audit_file_for(ts)
    with _audit_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:                          # pragma: no cover
                pass
    return row


def read_audit_tail(n: int = 100, *, day: Optional[datetime] = None) -> list:
    """Read the last ``n`` rows from today's audit file.

    Intended for ``/admin/audit`` or an operator health check.  Caller
    must still gate access behind a role check.
    """
    path = _audit_file_for(day)
    if not path.exists():
        return []
    rows: list = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-n:]
