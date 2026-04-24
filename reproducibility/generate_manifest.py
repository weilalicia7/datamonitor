"""
Reproducibility manifest generator (§3.7 / Improvement H).
==========================================================

Run:

    python -m reproducibility.generate_manifest

to regenerate ``reproducibility/manifest.json`` with the current:

  * git SHA + branch + dirty-working-tree flag
  * Python version + platform
  * Pinned pip dependency list (pip freeze)
  * SHA-256 checksums of every JSONL under ``data_cache/`` so an
    auditor can verify that the benchmark outputs they reproduce
    match the ones the dissertation quotes

The manifest is read by

  * ``dissertation_analysis.R §22`` → emits \\repro* macros cited
    in the Reproducibility Statement
  * ``flask_app.py::/api/reproducibility/status`` → read-only
    diagnostic endpoint for compliance reviewers
  * ``tests/test_reproducibility.py::TestReproducibilityArtefacts``
    → locks the schema so a future refactor cannot silently drop
    a field

This script has no non-stdlib dependencies so it runs inside any
Python 3.9+ environment — including a minimal Docker image that
might not even have pytest installed yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(
            cmd, cwd=_REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_state() -> Dict[str, Optional[str]]:
    sha = _run(["git", "rev-parse", "HEAD"])
    short_sha = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "commit_sha": sha,
        "short_sha": short_sha,
        "branch": branch,
        "working_tree_dirty": bool(dirty) if dirty is not None else None,
        "last_commit_subject": _run(["git", "log", "-1", "--format=%s"]),
        "last_commit_ts": _run(["git", "log", "-1", "--format=%cI"]),
    }


def _pip_freeze() -> List[str]:
    out = _run([sys.executable, "-m", "pip", "freeze"])
    if out is None:
        return []
    return [line for line in out.splitlines() if line and not line.startswith("#")]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl_checksums(cache_root: Path) -> Dict[str, Dict[str, object]]:
    """For every JSONL in data_cache/, record size + sha256 +
    line_count so an auditor can verify the benchmark outputs their
    Docker build produces match the dissertation's cited numbers."""
    if not cache_root.exists():
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for p in sorted(cache_root.rglob("*.jsonl")):
        rel = p.relative_to(_REPO_ROOT).as_posix()
        try:
            with p.open("r", encoding="utf-8") as fh:
                lines = sum(1 for _ in fh)
            out[rel] = {
                "sha256": _sha256(p),
                "size_bytes": p.stat().st_size,
                "lines": lines,
            }
        except Exception as exc:                             # pragma: no cover
            out[rel] = {"error": str(exc)}
    return out


def _key_file_checksums() -> Dict[str, str]:
    """Hashes of the exact files the reviewer cares about for
    reproducibility: the requirements pin, the Dockerfile, and the
    env.yml.  A divergence between the manifest and the source tree
    is a loud signal that a reproduction isn't faithful."""
    out = {}
    for rel in ("requirements.txt", "Dockerfile", "environment.yml",
                ".zenodo.json", ".dockerignore"):
        p = _REPO_ROOT / rel
        if p.exists():
            out[rel] = _sha256(p)
    return out


def _dissertation_checksum() -> Optional[str]:
    """Hash of the last-built main.pdf if present — lets the auditor
    confirm their container-driven rebuild matches what the reviewer
    saw.  Absent is fine; the dissertation PDF isn't required for
    reproducibility-of-the-code."""
    pdf = _REPO_ROOT.parent / "dissertation" / "main.pdf"
    if pdf.exists():
        return _sha256(pdf)
    return None


def generate(output_path: Path, *, include_pip_freeze: bool = True) -> Dict:
    cache_root = _REPO_ROOT / "data_cache"
    manifest = {
        "format_version": "1.0",
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "reproducibility/generate_manifest.py",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": _git_state(),
        "key_file_checksums": _key_file_checksums(),
        "data_cache_jsonl_checksums": _jsonl_checksums(cache_root),
        "dissertation_pdf_sha256": _dissertation_checksum(),
        "pip_freeze": _pip_freeze() if include_pip_freeze else None,
        "docker_repro_command": "docker run --rm sact-scheduler:repro",
        "docker_build_command": (
            "docker build --target reproducibility -t sact-scheduler:repro ."
        ),
        "notes": (
            "Regenerate via `python -m reproducibility.generate_manifest`.  "
            "Manifest consumed by dissertation §3.7 Reproducibility Statement, "
            "/api/reproducibility/status, and TestReproducibilityArtefacts."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    print(f"[reproducibility] wrote {output_path}  "
          f"(sha={manifest['git']['short_sha'] or 'n/a'}, "
          f"n_jsonl={len(manifest['data_cache_jsonl_checksums'])}, "
          f"n_deps={len(manifest.get('pip_freeze') or [])})")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output", default="reproducibility/manifest.json",
        help="Output path (relative to repo root)",
    )
    parser.add_argument(
        "--skip-pip-freeze", action="store_true",
        help="Skip pip-freeze (faster; recommended only for test runs)",
    )
    args = parser.parse_args()
    generate(
        _REPO_ROOT / args.output,
        include_pip_freeze=not args.skip_pip_freeze,
    )


if __name__ == "__main__":
    main()
