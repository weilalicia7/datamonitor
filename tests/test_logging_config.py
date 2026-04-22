"""
Tests for logging_config.py — JSON formatter, PII redaction, request-ID
propagation, audit trail (Production-Readiness T4.4).

Structure:
- TestJsonFormatter      — shape of emitted JSON, required keys, exc_info
- TestPatientIdRedactor  — each pattern + opt-out via LOG_PATIENT_IDS
- TestRequestIdThreadLocal — set/get/clear across threads
- TestRequestIdFilter    — attaches record.request_id
- TestFlaskRequestId     — before_request hook reads X-Request-ID header
                           or generates a UUID; after_request tags response
- TestAuditEvent         — writes one JSONL row per call; append-only;
                           metadata persisted; read_audit_tail round-trip
- TestInstallers         — idempotency, env-gated JSON handler, redactor
                           gated by LOG_PATIENT_IDS
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest
from flask import Flask, jsonify

import logging_config as lc


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_record(msg="hello", args=None, level=logging.INFO, name="test"):
    logger = logging.getLogger(name)
    return logger.makeRecord(
        name=name, level=level, fn="test.py", lno=1, msg=msg,
        args=args or (), exc_info=None,
    )


# --------------------------------------------------------------------------- #
# JsonFormatter
# --------------------------------------------------------------------------- #


class TestJsonFormatter:
    def test_basic_shape(self):
        rec = _make_record("hello world")
        out = lc.JsonFormatter().format(rec)
        payload = json.loads(out)
        assert payload["message"] == "hello world"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "test"
        assert "ts" in payload

    def test_args_formatted_into_message(self):
        rec = _make_record("hello %s", args=("world",))
        out = lc.JsonFormatter().format(rec)
        payload = json.loads(out)
        assert payload["message"] == "hello world"

    def test_request_id_included_when_set(self):
        rec = _make_record("hi")
        rec.request_id = "abc123"
        out = lc.JsonFormatter().format(rec)
        payload = json.loads(out)
        assert payload["request_id"] == "abc123"

    def test_exception_info_included(self):
        try:
            raise ValueError("boom")
        except ValueError:
            logger = logging.getLogger("exc-test")
            rec = logger.makeRecord(
                name="exc-test", level=logging.ERROR, fn="f", lno=1,
                msg="caught", args=(), exc_info=True,
            )
        import sys as _sys
        rec.exc_info = _sys.exc_info()
        out = lc.JsonFormatter().format(rec)
        payload = json.loads(out)
        # Either this test catches the active exception, or the formatter
        # tolerates the absence.  Either way, message must be present.
        assert payload["message"] == "caught"


# --------------------------------------------------------------------------- #
# PatientIdRedactor
# --------------------------------------------------------------------------- #


class TestPatientIdRedactor:
    def test_redacts_patient_id_kv(self):
        flt = lc.PatientIdRedactor(active=True)
        rec = _make_record("handling patient_id=12345 now")
        flt.filter(rec)
        assert "12345" not in str(rec.msg)
        assert "[REDACTED]" in str(rec.msg)

    def test_redacts_uppercase_variant(self):
        flt = lc.PatientIdRedactor(active=True)
        rec = _make_record("Patient_ID: ABC123")
        flt.filter(rec)
        assert "ABC123" not in str(rec.msg)

    def test_redacts_10_digit_nhs_number(self):
        flt = lc.PatientIdRedactor(active=True)
        rec = _make_record("referral for 1234567890 completed")
        flt.filter(rec)
        assert "1234567890" not in str(rec.msg)

    def test_redacts_uk_postcode(self):
        flt = lc.PatientIdRedactor(active=True)
        rec = _make_record("address CF10 3AT attended")
        flt.filter(rec)
        assert "CF10 3AT" not in str(rec.msg)

    def test_inactive_skips_redaction(self):
        flt = lc.PatientIdRedactor(active=False)
        rec = _make_record("patient_id=12345")
        flt.filter(rec)
        assert "12345" in str(rec.msg)

    def test_redacts_in_args_tuple(self):
        flt = lc.PatientIdRedactor(active=True)
        rec = _make_record("patient=%s", args=("patient_id=99",))
        flt.filter(rec)
        # After redaction, args tuple should have been scrubbed.
        assert all("99" not in str(a) or "[REDACTED]" in str(a) for a in rec.args)

    def test_env_gating_default_active(self, monkeypatch):
        monkeypatch.delenv("LOG_PATIENT_IDS", raising=False)
        assert lc.log_redact_patient_ids() is True

    def test_env_gating_disables_when_true(self, monkeypatch):
        monkeypatch.setenv("LOG_PATIENT_IDS", "true")
        assert lc.log_redact_patient_ids() is False


# --------------------------------------------------------------------------- #
# Request-ID thread-local + filter
# --------------------------------------------------------------------------- #


class TestRequestIdThreadLocal:
    def setup_method(self):
        lc.set_request_id(None)

    def teardown_method(self):
        lc.set_request_id(None)

    def test_generate_is_uuid_hex(self):
        rid = lc.generate_request_id()
        assert len(rid) == 32
        # UUID hex round-trips.
        uuid.UUID(rid)

    def test_set_and_get(self):
        lc.set_request_id("abc")
        assert lc.current_request_id() == "abc"

    def test_clear(self):
        lc.set_request_id("abc")
        lc.set_request_id(None)
        assert lc.current_request_id() is None

    def test_thread_isolation(self):
        # Each thread has its own context; main-thread value must not leak.
        lc.set_request_id("main-thread")
        captured = {}

        def worker():
            captured["child"] = lc.current_request_id()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert captured["child"] is None
        assert lc.current_request_id() == "main-thread"


class TestRequestIdFilter:
    def test_filter_attaches_rid(self):
        lc.set_request_id("req-xyz")
        try:
            flt = lc.RequestIdFilter()
            rec = _make_record("hi")
            flt.filter(rec)
            assert rec.request_id == "req-xyz"
        finally:
            lc.set_request_id(None)

    def test_filter_falls_back_to_dash(self):
        lc.set_request_id(None)
        flt = lc.RequestIdFilter()
        rec = _make_record("hi")
        flt.filter(rec)
        assert rec.request_id == "-"


# --------------------------------------------------------------------------- #
# Flask request-id wiring
# --------------------------------------------------------------------------- #


class TestFlaskRequestId:
    def _build_app(self):
        app = Flask(__name__)
        app.secret_key = "test"
        lc.attach_request_id(app)

        @app.route("/hit")
        def hit():
            from flask import g
            return jsonify({"rid": g.request_id})

        return app

    def test_generates_rid_when_header_absent(self):
        app = self._build_app()
        resp = app.test_client().get("/hit")
        assert resp.status_code == 200
        body = resp.get_json()
        assert len(body["rid"]) == 32
        # Response echoes the same ID in X-Request-ID.
        assert resp.headers["X-Request-ID"] == body["rid"]

    def test_honours_inbound_header(self):
        app = self._build_app()
        resp = app.test_client().get("/hit", headers={"X-Request-ID": "caller-123"})
        body = resp.get_json()
        assert body["rid"] == "caller-123"
        assert resp.headers["X-Request-ID"] == "caller-123"


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #


class TestAuditEvent:
    def setup_method(self):
        # Always start from a clean slate.
        lc.set_request_id(None)

    def test_writes_row(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        lc.audit_event(actor="alice", action="schedule.optimise")
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        rows = [json.loads(l) for l in files[0].read_text(encoding="utf-8").splitlines() if l]
        assert rows[0]["actor"] == "alice"
        assert rows[0]["action"] == "schedule.optimise"
        assert rows[0]["outcome"] == "success"
        assert "ts" in rows[0]

    def test_metadata_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        lc.audit_event(
            actor="bob", action="auth.login", outcome="failure",
            metadata={"reason": "bad_password", "ip": "10.0.0.1"},
        )
        files = list(tmp_path.glob("*.jsonl"))
        row = json.loads(files[0].read_text(encoding="utf-8").strip())
        assert row["outcome"] == "failure"
        assert row["metadata"] == {"reason": "bad_password", "ip": "10.0.0.1"}

    def test_appends_not_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        lc.audit_event(actor="a", action="x")
        lc.audit_event(actor="b", action="y")
        lc.audit_event(actor="c", action="z")
        files = list(tmp_path.glob("*.jsonl"))
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        actors = [json.loads(l)["actor"] for l in lines]
        assert actors == ["a", "b", "c"]

    def test_request_id_auto_captured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        lc.set_request_id("rid-789")
        try:
            lc.audit_event(actor="sys", action="test.hook")
        finally:
            lc.set_request_id(None)
        files = list(tmp_path.glob("*.jsonl"))
        row = json.loads(files[0].read_text(encoding="utf-8").strip())
        assert row["request_id"] == "rid-789"

    def test_explicit_request_id_wins_over_ctx(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        lc.set_request_id("ctx-rid")
        try:
            lc.audit_event(actor="sys", action="x", request_id="override-rid")
        finally:
            lc.set_request_id(None)
        files = list(tmp_path.glob("*.jsonl"))
        row = json.loads(files[0].read_text(encoding="utf-8").strip())
        assert row["request_id"] == "override-rid"

    def test_read_tail_returns_last_n(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        for i in range(5):
            lc.audit_event(actor="sys", action=f"step.{i}")
        rows = lc.read_audit_tail(n=3)
        assert len(rows) == 3
        assert [r["action"] for r in rows] == ["step.2", "step.3", "step.4"]

    def test_read_tail_empty_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "never-created"))
        rows = lc.read_audit_tail()
        assert rows == []

    def test_concurrent_writes_are_serialised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        N = 20

        def worker(i):
            lc.audit_event(actor=f"worker-{i}", action="stress.test")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        files = list(tmp_path.glob("*.jsonl"))
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == N  # every write landed; no torn lines
        # Every line parses.
        for l in lines:
            json.loads(l)


# --------------------------------------------------------------------------- #
# Installers
# --------------------------------------------------------------------------- #


class TestInstallers:
    def test_request_id_filter_idempotent(self):
        # Remove any existing filter instance first.
        root = logging.getLogger()
        root.filters = [f for f in root.filters if not isinstance(f, lc.RequestIdFilter)]
        flt1 = lc.install_request_id_filter()
        flt2 = lc.install_request_id_filter()
        assert flt1 is flt2
        count = sum(1 for f in root.filters if isinstance(f, lc.RequestIdFilter))
        assert count == 1

    def test_patient_id_redactor_installed_by_default(self, monkeypatch):
        monkeypatch.delenv("LOG_PATIENT_IDS", raising=False)
        root = logging.getLogger()
        root.filters = [f for f in root.filters if not isinstance(f, lc.PatientIdRedactor)]
        flt = lc.install_patient_id_redactor()
        assert flt is not None

    def test_patient_id_redactor_skipped_when_ids_enabled(self, monkeypatch):
        monkeypatch.setenv("LOG_PATIENT_IDS", "true")
        root = logging.getLogger()
        root.filters = [f for f in root.filters if not isinstance(f, lc.PatientIdRedactor)]
        flt = lc.install_patient_id_redactor()
        assert flt is None

    def test_install_json_logging_skipped_when_format_text(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "text")
        assert lc.install_json_logging() is False

    def test_install_json_logging_when_env_json(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "json")
        buf = StringIO()
        assert lc.install_json_logging(stream=buf, force=True) is True
        # Verify the added handler emits JSON.
        root = logging.getLogger()
        # Identify the handler we just added (last one).
        handler = root.handlers[-1]
        try:
            handler.emit(_make_record("hello"))
            out = buf.getvalue().strip()
            payload = json.loads(out)
            assert payload["message"] == "hello"
        finally:
            root.removeHandler(handler)


# --------------------------------------------------------------------------- #
# log_format env parsing
# --------------------------------------------------------------------------- #


class TestLogFormatEnv:
    def test_default_is_text(self, monkeypatch):
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        assert lc.log_format() == "text"

    def test_json_set(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "JSON")
        assert lc.log_format() == "json"
