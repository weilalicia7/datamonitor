"""
Retention enforcer (Production-Readiness T4.8).

Runs daily (cron / systemd-timer / k8s CronJob).  Two concerns:

1. **TTL pruning** — delete files under ``data_cache/events/`` older than
   ``EVENT_RETENTION_DAYS`` and files under ``data_cache/audit/`` older
   than ``AUDIT_RETENTION_DAYS``.  Both directories contain append-only
   JSONL; deletion is by file-mtime (simple, auditable).

2. **Right-to-erasure** — ``--erase <hash>`` scans every JSONL file
   and rewrites it to drop rows whose ``patient_id`` / ``actor`` /
   ``target`` matches the provided hash.  The rewrite uses a tempfile
   + atomic rename, preserving original mtime, so TTL enforcement
   still works after the erasure.

Usage:

    # TTL prune (what cron runs):
    python scripts/retention_enforcer.py

    # Dry run (list what would be removed, don't touch):
    python scripts/retention_enforcer.py --dry-run

    # Right-to-erasure for a specific salted hash:
    python scripts/retention_enforcer.py --erase 7d3abf...

    # Override retention windows at CLI:
    python scripts/retention_enforcer.py --event-days 7 --audit-days 90

Exit codes:
    0 — ran to completion (with possibly zero deletions)
    2 — argument error
    3 — IO error on a file it couldn't skip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_EVENT_TTL_DAYS = 30
DEFAULT_AUDIT_TTL_DAYS = 2557  # 7 years

EVENT_DIR_ENV = "EVENT_CACHE_DIR"
AUDIT_DIR_ENV = "AUDIT_LOG_DIR"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _event_dir() -> Path:
    return Path(os.environ.get(EVENT_DIR_ENV, "data_cache/events"))


def _audit_dir() -> Path:
    return Path(os.environ.get(AUDIT_DIR_ENV, "data_cache/audit"))


# --------------------------------------------------------------------------- #
# TTL pruning
# --------------------------------------------------------------------------- #


def files_older_than(
    directory: Path,
    *,
    ttl_days: int,
    now: Optional[datetime] = None,
    pattern: str = "*.jsonl",
) -> List[Path]:
    """Return files in ``directory`` whose mtime is older than ``ttl_days``."""
    if not directory.exists():
        return []
    now = now or datetime.now(tz=timezone.utc)
    threshold = now - timedelta(days=ttl_days)
    threshold_epoch = threshold.timestamp()
    matches: List[Path] = []
    for p in directory.glob(pattern):
        if not p.is_file():
            continue
        try:
            if p.stat().st_mtime < threshold_epoch:
                matches.append(p)
        except OSError:
            continue
    return sorted(matches)


def prune(
    directory: Path,
    *,
    ttl_days: int,
    dry_run: bool = False,
    now: Optional[datetime] = None,
) -> List[Path]:
    """Delete expired files.  Returns the list of files acted on."""
    victims = files_older_than(directory, ttl_days=ttl_days, now=now)
    if dry_run:
        return victims
    for v in victims:
        try:
            v.unlink()
        except OSError:
            pass  # best-effort — the next cron cycle will retry
    return victims


# --------------------------------------------------------------------------- #
# Right-to-erasure
# --------------------------------------------------------------------------- #


def _row_matches_hash(row: Dict, target_hash: str) -> bool:
    """Return True iff the JSONL row references ``target_hash``."""
    for key in ("patient_id", "actor", "target"):
        val = row.get(key)
        if isinstance(val, str) and val == target_hash:
            return True
    return False


def erase_hash_in_file(path: Path, target_hash: str) -> Tuple[int, int]:
    """Rewrite ``path`` without rows referencing ``target_hash``.

    Returns ``(rows_kept, rows_removed)``.  The rewrite is atomic — we
    write to a sibling tempfile + rename — so a crash mid-write leaves
    either the original or the cleaned file, never a truncated mix.
    """
    if not path.exists():
        return 0, 0
    tmp = path.with_suffix(path.suffix + ".tmp-erase")
    kept = removed = 0
    try:
        with open(path, "r", encoding="utf-8") as src, \
             open(tmp, "w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Preserve lines we can't parse — operator will see them
                    # on the next scan; they're not real patient rows.
                    dst.write(line + "\n")
                    kept += 1
                    continue
                if _row_matches_hash(row, target_hash):
                    removed += 1
                    continue
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise
    # Preserve mtime so TTL enforcement is consistent.
    mtime = path.stat().st_mtime
    shutil.move(str(tmp), str(path))
    os.utime(path, (mtime, mtime))
    return kept, removed


def erase_hash_across(
    directories: Iterable[Path],
    target_hash: str,
    *,
    pattern: str = "*.jsonl",
) -> Dict[str, Tuple[int, int]]:
    results: Dict[str, Tuple[int, int]] = {}
    for d in directories:
        if not d.exists():
            continue
        for p in d.glob(pattern):
            if not p.is_file():
                continue
            results[str(p)] = erase_hash_in_file(p, target_hash)
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SACT Scheduler retention enforcer")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be deleted, don't touch files.")
    p.add_argument("--event-days", type=int, default=None,
                   help="Override EVENT_RETENTION_DAYS.")
    p.add_argument("--audit-days", type=int, default=None,
                   help="Override AUDIT_RETENTION_DAYS.")
    p.add_argument("--erase", metavar="HASH",
                   help="Right-to-erasure: remove every row where patient_id/"
                        "actor/target equals this hash across data_cache/**.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    event_ttl = args.event_days if args.event_days is not None else \
        _env_int("EVENT_RETENTION_DAYS", DEFAULT_EVENT_TTL_DAYS)
    audit_ttl = args.audit_days if args.audit_days is not None else \
        _env_int("AUDIT_RETENTION_DAYS", DEFAULT_AUDIT_TTL_DAYS)

    if args.erase:
        erasure = erase_hash_across(
            [_event_dir(), _audit_dir()], target_hash=args.erase,
        )
        total_removed = sum(r[1] for r in erasure.values())
        for path, (kept, removed) in erasure.items():
            print(f"{path}  kept={kept}  removed={removed}")
        print(f"[erase] total rows removed: {total_removed}")
        return 0

    event_victims = prune(_event_dir(), ttl_days=event_ttl, dry_run=args.dry_run)
    audit_victims = prune(_audit_dir(), ttl_days=audit_ttl, dry_run=args.dry_run)

    mode = "would-delete" if args.dry_run else "deleted"
    for v in event_victims:
        print(f"[events:{mode}] {v}")
    for v in audit_victims:
        print(f"[audit:{mode}]  {v}")
    print(f"Summary: events {mode}={len(event_victims)} "
          f"audit {mode}={len(audit_victims)} "
          f"event_ttl={event_ttl}d audit_ttl={audit_ttl}d")
    return 0


if __name__ == "__main__":                           # pragma: no cover
    sys.exit(main())
