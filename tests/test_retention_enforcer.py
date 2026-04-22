"""
Tests for scripts/retention_enforcer.py — TTL pruning + right-to-erasure
(Production-Readiness T4.8).

Structure:
- TestFilesOlderThan    — mtime cutoff, empty dir, mixed ages
- TestPrune             — dry-run leaves files, real run deletes
- TestEraseHashInFile   — removes matching rows, preserves non-matching,
                          keeps unparseable lines, atomic rename
- TestEraseAcross       — walks multiple dirs, summarises by file
- TestCLI               — argparse + main() integration
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# Dynamic import of the script module (scripts/ isn't a package).
# --------------------------------------------------------------------------- #


SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "retention_enforcer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "retention_enforcer", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def re_mod():
    return _load_module()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _age_file(path: Path, days: int):
    """Set mtime to `days` in the past."""
    past = time.time() - days * 86400
    os.utime(path, (past, past))


# --------------------------------------------------------------------------- #
# files_older_than
# --------------------------------------------------------------------------- #


class TestFilesOlderThan:
    def test_empty_dir_returns_empty(self, tmp_path, re_mod):
        assert re_mod.files_older_than(tmp_path, ttl_days=7) == []

    def test_missing_dir_returns_empty(self, tmp_path, re_mod):
        assert re_mod.files_older_than(tmp_path / "never", ttl_days=7) == []

    def test_returns_only_old_files(self, tmp_path, re_mod):
        fresh = tmp_path / "fresh.jsonl"
        stale = tmp_path / "stale.jsonl"
        fresh.write_text("{}\n")
        stale.write_text("{}\n")
        _age_file(stale, days=40)

        out = re_mod.files_older_than(tmp_path, ttl_days=30)
        assert stale in out
        assert fresh not in out

    def test_pattern_respected(self, tmp_path, re_mod):
        (tmp_path / "audit.jsonl").write_text("{}\n")
        (tmp_path / "leaveme.txt").write_text("x")
        _age_file(tmp_path / "audit.jsonl", days=100)
        _age_file(tmp_path / "leaveme.txt", days=100)

        out = re_mod.files_older_than(tmp_path, ttl_days=1, pattern="*.jsonl")
        assert tmp_path / "audit.jsonl" in out
        assert tmp_path / "leaveme.txt" not in out


# --------------------------------------------------------------------------- #
# prune
# --------------------------------------------------------------------------- #


class TestPrune:
    def test_dry_run_does_not_delete(self, tmp_path, re_mod):
        victim = tmp_path / "old.jsonl"
        victim.write_text("{}\n")
        _age_file(victim, days=100)

        victims = re_mod.prune(tmp_path, ttl_days=30, dry_run=True)
        assert victim in victims
        assert victim.exists()

    def test_real_run_deletes(self, tmp_path, re_mod):
        victim = tmp_path / "old.jsonl"
        victim.write_text("{}\n")
        _age_file(victim, days=100)

        victims = re_mod.prune(tmp_path, ttl_days=30, dry_run=False)
        assert victim in victims
        assert not victim.exists()

    def test_keeps_fresh(self, tmp_path, re_mod):
        fresh = tmp_path / "fresh.jsonl"
        fresh.write_text("{}\n")
        victims = re_mod.prune(tmp_path, ttl_days=30)
        assert victims == []
        assert fresh.exists()


# --------------------------------------------------------------------------- #
# erase_hash_in_file
# --------------------------------------------------------------------------- #


class TestEraseHashInFile:
    def test_removes_matching_patient_id(self, tmp_path, re_mod):
        path = tmp_path / "events.jsonl"
        _write_jsonl(path, [
            {"patient_id": "keep-me", "x": 1},
            {"patient_id": "DELETE", "x": 2},
            {"patient_id": "keep-me-too", "x": 3},
        ])
        kept, removed = re_mod.erase_hash_in_file(path, "DELETE")
        assert kept == 2
        assert removed == 1
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        ids = [r["patient_id"] for r in rows]
        assert "DELETE" not in ids
        assert "keep-me" in ids

    def test_matches_actor_field(self, tmp_path, re_mod):
        path = tmp_path / "audit.jsonl"
        _write_jsonl(path, [
            {"actor": "DELETE", "action": "login"},
            {"actor": "alice", "action": "login"},
        ])
        kept, removed = re_mod.erase_hash_in_file(path, "DELETE")
        assert removed == 1
        assert kept == 1

    def test_matches_target_field(self, tmp_path, re_mod):
        path = tmp_path / "audit.jsonl"
        _write_jsonl(path, [
            {"actor": "sys", "target": "DELETE"},
            {"actor": "sys", "target": "keep"},
        ])
        _, removed = re_mod.erase_hash_in_file(path, "DELETE")
        assert removed == 1

    def test_preserves_unparseable_lines(self, tmp_path, re_mod):
        path = tmp_path / "mixed.jsonl"
        path.write_text('{"patient_id": "DELETE"}\ngarbage line\n{"patient_id": "keep"}\n')
        kept, removed = re_mod.erase_hash_in_file(path, "DELETE")
        # Unparseable line counts toward kept — operator sees it next scan.
        assert kept == 2
        assert removed == 1

    def test_missing_file_no_ops(self, tmp_path, re_mod):
        kept, removed = re_mod.erase_hash_in_file(tmp_path / "nope.jsonl", "x")
        assert (kept, removed) == (0, 0)

    def test_preserves_mtime(self, tmp_path, re_mod):
        path = tmp_path / "events.jsonl"
        _write_jsonl(path, [
            {"patient_id": "keep"},
            {"patient_id": "DELETE"},
        ])
        _age_file(path, days=10)
        mtime_before = path.stat().st_mtime
        re_mod.erase_hash_in_file(path, "DELETE")
        mtime_after = path.stat().st_mtime
        # Mtime preserved so TTL enforcement still works after erasure.
        assert abs(mtime_before - mtime_after) < 2


# --------------------------------------------------------------------------- #
# erase_hash_across
# --------------------------------------------------------------------------- #


class TestEraseAcross:
    def test_walks_multiple_dirs(self, tmp_path, re_mod):
        d1 = tmp_path / "events"
        d2 = tmp_path / "audit"
        d1.mkdir()
        d2.mkdir()
        _write_jsonl(d1 / "a.jsonl", [{"patient_id": "DELETE"}])
        _write_jsonl(d2 / "b.jsonl", [{"actor": "DELETE"}, {"actor": "sys"}])
        results = re_mod.erase_hash_across([d1, d2], "DELETE")
        assert str(d1 / "a.jsonl") in results
        assert str(d2 / "b.jsonl") in results
        assert sum(r[1] for r in results.values()) == 2

    def test_missing_dir_skipped(self, tmp_path, re_mod):
        results = re_mod.erase_hash_across(
            [tmp_path / "never1", tmp_path / "never2"], "x"
        )
        assert results == {}


# --------------------------------------------------------------------------- #
# CLI integration
# --------------------------------------------------------------------------- #


class TestCLI:
    def test_main_prune_exit_zero(self, tmp_path, monkeypatch, re_mod, capsys):
        events = tmp_path / "events"
        audit = tmp_path / "audit"
        events.mkdir()
        audit.mkdir()
        stale = events / "old.jsonl"
        stale.write_text("{}\n")
        _age_file(stale, days=100)

        monkeypatch.setenv("EVENT_CACHE_DIR", str(events))
        monkeypatch.setenv("AUDIT_LOG_DIR", str(audit))
        monkeypatch.setenv("EVENT_RETENTION_DAYS", "30")
        monkeypatch.setenv("AUDIT_RETENTION_DAYS", "30")

        code = re_mod.main([])
        assert code == 0
        out = capsys.readouterr().out
        assert "events:deleted" in out
        assert not stale.exists()

    def test_main_dry_run_keeps_file(self, tmp_path, monkeypatch, re_mod, capsys):
        events = tmp_path / "events"
        audit = tmp_path / "audit"
        events.mkdir()
        audit.mkdir()
        stale = events / "old.jsonl"
        stale.write_text("{}\n")
        _age_file(stale, days=100)

        monkeypatch.setenv("EVENT_CACHE_DIR", str(events))
        monkeypatch.setenv("AUDIT_LOG_DIR", str(audit))
        code = re_mod.main(["--dry-run", "--event-days", "30"])
        assert code == 0
        assert stale.exists()
        out = capsys.readouterr().out
        assert "events:would-delete" in out

    def test_main_erase_flow(self, tmp_path, monkeypatch, re_mod, capsys):
        events = tmp_path / "events"
        events.mkdir()
        f = events / "a.jsonl"
        _write_jsonl(f, [
            {"patient_id": "KEEPER"},
            {"patient_id": "HASH-TO-DELETE"},
            {"patient_id": "HASH-TO-DELETE"},
        ])
        monkeypatch.setenv("EVENT_CACHE_DIR", str(events))
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "nope"))
        code = re_mod.main(["--erase", "HASH-TO-DELETE"])
        assert code == 0
        out = capsys.readouterr().out
        assert "total rows removed: 2" in out
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        assert all(r["patient_id"] == "KEEPER" for r in rows)
