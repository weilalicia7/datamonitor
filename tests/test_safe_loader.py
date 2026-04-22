"""
Tests for safe_loader.py — SHA-256-verified pickle loader (T2.3).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from safe_loader import (
    SIDECAR_SUFFIX,
    UnsafeLoadError,
    file_sha256,
    safe_load,
    safe_save,
    verify_only,
)


# --------------------------------------------------------------------------- #
# file_sha256
# --------------------------------------------------------------------------- #


class TestFileSha256:
    def test_known_digest(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        # Pre-computed SHA-256 of b"hello"
        assert file_sha256(p) == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )

    def test_chunks_large_file(self, tmp_path):
        p = tmp_path / "big.bin"
        p.write_bytes(b"\0" * (3 * 1024 * 1024))   # 3 MiB
        # Just check it doesn't raise + returns a 64-char hex digest.
        d = file_sha256(p)
        assert len(d) == 64 and all(c in "0123456789abcdef" for c in d)


# --------------------------------------------------------------------------- #
# safe_save → sidecar + digest round-trip
# --------------------------------------------------------------------------- #


class TestSafeSave:
    def test_writes_pickle_and_sidecar(self, tmp_path):
        path = tmp_path / "model.pkl"
        digest = safe_save({"x": 1, "y": [1, 2, 3]}, path)
        assert path.exists()
        side = path.with_suffix(path.suffix + SIDECAR_SUFFIX)
        assert side.exists()
        assert side.read_text(encoding="utf-8").strip() == digest
        assert digest == file_sha256(path)

    def test_no_sidecar_when_disabled(self, tmp_path):
        path = tmp_path / "model.pkl"
        safe_save({"x": 1}, path, write_sidecar=False)
        assert not path.with_suffix(path.suffix + SIDECAR_SUFFIX).exists()


# --------------------------------------------------------------------------- #
# safe_load — verification paths
# --------------------------------------------------------------------------- #


class TestSafeLoad:
    def _save(self, tmp_path, obj):
        path = tmp_path / "model.pkl"
        digest = safe_save(obj, path)
        return path, digest

    def test_round_trip_with_sidecar(self, tmp_path):
        obj = {"k": "v"}
        path, _ = self._save(tmp_path, obj)
        loaded = safe_load(path)
        assert loaded == obj

    def test_pinned_digest_matches(self, tmp_path):
        obj = [1, 2, 3]
        path, digest = self._save(tmp_path, obj)
        assert safe_load(path, expected_sha256=digest) == obj

    def test_pinned_digest_mismatch_raises(self, tmp_path):
        path, _ = self._save(tmp_path, {"x": 1})
        with pytest.raises(UnsafeLoadError) as exc:
            safe_load(path, expected_sha256="0" * 64)
        assert "mismatch" in str(exc.value)

    def test_sidecar_mismatch_raises(self, tmp_path):
        path, _ = self._save(tmp_path, {"x": 1})
        # Tamper with the file AFTER the sidecar was written.
        path.write_bytes(b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00\x00.")
        with pytest.raises(UnsafeLoadError):
            safe_load(path)

    def test_missing_sidecar_warns_then_loads(self, tmp_path, caplog):
        path = tmp_path / "model.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"plain": True}, fh)
        # No sidecar — default allow_unverified=True path.
        with caplog.at_level("WARNING"):
            obj = safe_load(path)
        assert obj == {"plain": True}
        assert any("WITHOUT integrity check" in r.message for r in caplog.records)

    def test_strict_mode_refuses_unverified(self, tmp_path):
        path = tmp_path / "model.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"plain": True}, fh)
        with pytest.raises(UnsafeLoadError):
            safe_load(path, allow_unverified=False)

    def test_pinned_digest_takes_precedence_over_sidecar(self, tmp_path):
        # If a caller passes expected_sha256, the sidecar should NOT be
        # consulted at all — pin wins.
        path, digest = self._save(tmp_path, {"x": 1})
        # Tamper the sidecar to a wrong value.
        side = path.with_suffix(path.suffix + SIDECAR_SUFFIX)
        side.write_text("0" * 64, encoding="utf-8")
        # Pinned digest still matches → load succeeds.
        assert safe_load(path, expected_sha256=digest) == {"x": 1}

    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            safe_load(tmp_path / "nope.pkl")


# --------------------------------------------------------------------------- #
# verify_only
# --------------------------------------------------------------------------- #


class TestVerifyOnly:
    def test_true_when_sidecar_matches(self, tmp_path):
        path = tmp_path / "m.pkl"
        safe_save({"x": 1}, path)
        assert verify_only(path) is True

    def test_false_when_tampered(self, tmp_path):
        path = tmp_path / "m.pkl"
        safe_save({"x": 1}, path)
        path.write_bytes(b"corrupted")
        assert verify_only(path) is False

    def test_pinned_digest_overrides(self, tmp_path):
        path = tmp_path / "m.pkl"
        digest = safe_save({"x": 1}, path)
        assert verify_only(path, expected_sha256=digest) is True
        assert verify_only(path, expected_sha256="0" * 64) is False

    def test_missing_file_returns_false(self, tmp_path):
        assert verify_only(tmp_path / "nope.pkl") is False

    def test_no_sidecar_no_pin_returns_false(self, tmp_path):
        path = tmp_path / "m.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"x": 1}, fh)
        assert verify_only(path) is False
