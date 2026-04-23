"""
Tuning manifest — single source of truth for all tuned hyperparameters.

Layout::

    {
      "version": 1,
      "data_channel": "synthetic" | "real",
      "tuned_at": "2026-04-23T03:00:00+00:00",
      "git_sha": "<commit>",
      "n_records": 1899,
      "results": {
        "noshow_model":   {...},
        "duration_model": {...},
        "cpsat_weights":  {...},
        "dro_epsilon":    {...},
        "lipschitz_l":    {...},
        "cvar_alpha":     {...}
      }
    }

The manifest carries a ``data_channel`` discriminator so the boot code
can decide whether the tuned values are safe to apply:

- ``data_channel == "synthetic"`` → smoke run only.  Boot **does NOT**
  override defaults, because tuning against the synthetic generator
  picks up its quirks rather than real distributional patterns.
- ``data_channel == "real"`` → tuned on Channel 2.  Boot applies
  every recorded override on top of the module-level ``DEFAULT_*``
  constants.

This single gate is what makes the tuners safe to run today against
synthetic data while ensuring no override leaks into the live
prediction pipeline until real data is wired.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Manifest format version.  Bump on any breaking schema change.
MANIFEST_VERSION: int = 1

#: Default location.  Operator may override via ``TUNING_MANIFEST_PATH``.
MANIFEST_PATH: Path = Path("data_cache/tuning/manifest.json")

#: Channel discriminator values.
CHANNEL_SYNTHETIC: str = "synthetic"
CHANNEL_REAL: str = "real"
VALID_CHANNELS = (CHANNEL_SYNTHETIC, CHANNEL_REAL)


def manifest_path() -> Path:
    raw = os.environ.get("TUNING_MANIFEST_PATH")
    return Path(raw) if raw else MANIFEST_PATH


def _git_sha() -> str:                                # pragma: no cover
    """Best-effort current git SHA — empty string if unavailable."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _empty_manifest(channel: str) -> Dict[str, Any]:
    if channel not in VALID_CHANNELS:
        raise ValueError(f"data_channel must be one of {VALID_CHANNELS}; got {channel!r}")
    return {
        "version": MANIFEST_VERSION,
        "data_channel": channel,
        "tuned_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "n_records": 0,
        "results": {},
    }


def load_manifest(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Read the manifest, returning ``None`` if absent or unparseable."""
    p = path or manifest_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:    # pragma: no cover
        logger.warning("tuning: manifest at %s is unreadable (%s); ignoring", p, exc)
        return None


def save_manifest(manifest: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Atomically write the manifest as pretty JSON."""
    p = path or manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
    return p


def record_tuning_run(
    *,
    channel: str,
    tuner_key: str,
    payload: Dict[str, Any],
    n_records: int = 0,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Merge a single tuner's result into the manifest, then write it.

    Creates the manifest if missing.  Returns the post-write manifest dict
    so callers (tests, the orchestrator) can assert on the result.
    """
    if channel not in VALID_CHANNELS:
        raise ValueError(f"data_channel must be one of {VALID_CHANNELS}; got {channel!r}")
    p = path or manifest_path()
    existing = load_manifest(p) or _empty_manifest(channel)
    # If the file existed but with a different channel, preserve the new
    # channel value — operator is moving from synthetic→real or vice versa.
    existing["data_channel"] = channel
    existing["tuned_at"] = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    existing["git_sha"] = _git_sha()
    existing["n_records"] = max(existing.get("n_records", 0), n_records)
    results = existing.setdefault("results", {})
    results[tuner_key] = payload
    save_manifest(existing, p)
    return existing


def load_overrides(path: Optional[Path] = None) -> Dict[str, Any]:
    """Return the set of overrides the boot path should apply.

    - When the manifest is missing: returns ``{}``.
    - When ``data_channel == "synthetic"``: returns ``{}`` and logs once
      that the manifest exists but is in smoke mode.  This is the gate
      that keeps synthetic-data tuning out of the live pipeline.
    - When ``data_channel == "real"``: returns the ``results`` dict so
      the boot code can apply every override.
    """
    manifest = load_manifest(path)
    if manifest is None:
        return {}
    channel = manifest.get("data_channel", CHANNEL_SYNTHETIC)
    if channel != CHANNEL_REAL:
        logger.info(
            "tuning: manifest at %s is in '%s' mode; "
            "overrides are NOT applied to the runtime "
            "(set data_channel='real' after Channel 2 cutover)",
            (path or manifest_path()), channel,
        )
        return {}
    return dict(manifest.get("results", {}))


def summary(path: Optional[Path] = None) -> Dict[str, Any]:
    """Compact summary suitable for the ``GET /api/tuning/status`` payload."""
    manifest = load_manifest(path)
    if manifest is None:
        return {
            "present": False,
            "data_channel": None,
            "tuned_at": None,
            "git_sha": None,
            "n_records": 0,
            "tuners": [],
            "overrides_active": False,
            "manifest_path": str(path or manifest_path()),
        }
    return {
        "present": True,
        "data_channel": manifest.get("data_channel"),
        "tuned_at": manifest.get("tuned_at"),
        "git_sha": manifest.get("git_sha"),
        "n_records": manifest.get("n_records", 0),
        "tuners": sorted((manifest.get("results") or {}).keys()),
        "overrides_active": manifest.get("data_channel") == CHANNEL_REAL,
        "manifest_path": str(path or manifest_path()),
    }


def detect_channel(env_var: str = "SACT_CHANNEL") -> str:
    """Pick the channel based on env or the presence of real-data files.

    - If ``SACT_CHANNEL`` is set explicitly, honour it.
    - Else, if ``datasets/real_data/historical_appointments.xlsx`` exists,
      assume ``real``.
    - Else default to ``synthetic``.
    """
    explicit = (os.environ.get(env_var) or "").strip().lower()
    if explicit in VALID_CHANNELS:
        return explicit
    real_marker = Path("datasets/real_data/historical_appointments.xlsx")
    if real_marker.exists():
        return CHANNEL_REAL
    return CHANNEL_SYNTHETIC
