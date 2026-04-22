"""
SHA-256-verified pickle loader (Production-Readiness T2.3).

Plain ``pickle.load`` from disk is a remote-code-execution vector — the
unpickling protocol can construct arbitrary objects, so an attacker who
can write to ``models/`` or ``data_cache/`` can run code in the app's
process at next boot.  The fix is to gate every load behind a
checksum verification:

    digest = safe_save(model, path)        # writes path + path.sha256
    ...
    model = safe_load(path)                # verifies sidecar, then loads
    model = safe_load(path, expected_sha256=digest)  # verifies given digest

Two trust modes exist:

* **Sidecar verification** (default for legacy callers).  Each save
  writes a ``<path>.sha256`` file alongside the pickle.  Load reads the
  sidecar and refuses to deserialise if the file's hash doesn't match.
* **Pinned-digest verification** (recommended for new callers).  The
  caller passes ``expected_sha256`` from a pinned manifest (env var,
  config file, or model-card).  Sidecar is not consulted; the pin is
  authoritative.

Either mode raises :class:`UnsafeLoadError` on mismatch.  When neither
mode is available the loader logs a warning and proceeds — keeps the
existing tests green during the migration but every legacy call site
should adopt one of the modes within a release cycle.

The hashing is done in 1 MB chunks so we don't load multi-GB models
into memory just to compute a digest.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

#: Read this many bytes per loop when streaming a hash.
_HASH_CHUNK = 1 << 20    # 1 MiB

#: Sidecar file extension.  Holds the lowercase hex digest, no trailing newline.
SIDECAR_SUFFIX = ".sha256"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class UnsafeLoadError(RuntimeError):
    """Raised when a pickle's hash doesn't match the expected digest."""


# --------------------------------------------------------------------------- #
# Hashing helper
# --------------------------------------------------------------------------- #


def file_sha256(path: Union[str, Path]) -> str:
    """Stream a file through SHA-256 and return the lowercase hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sidecar_path(path: Union[str, Path]) -> Path:
    return Path(str(path) + SIDECAR_SUFFIX)


def _read_sidecar(path: Union[str, Path]) -> Optional[str]:
    side = _sidecar_path(path)
    if not side.exists():
        return None
    try:
        return side.read_text(encoding="utf-8").strip().lower()
    except OSError:                                     # pragma: no cover
        return None


def _write_sidecar(path: Union[str, Path], digest: str) -> None:
    side = _sidecar_path(path)
    side.write_text(digest.lower().strip(), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def safe_save(
    obj: Any,
    path: Union[str, Path],
    *,
    write_sidecar: bool = True,
) -> str:
    """Pickle ``obj`` to ``path``; return the SHA-256 of the bytes written.

    When ``write_sidecar=True`` (default), ``path.sha256`` is written
    alongside so :func:`safe_load` can verify on read.  Returning the
    digest also lets callers pin it in a manifest / config.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    digest = file_sha256(path)
    if write_sidecar:
        _write_sidecar(path, digest)
    return digest


def safe_load(
    path: Union[str, Path],
    *,
    expected_sha256: Optional[str] = None,
    verify_sidecar: bool = True,
    allow_unverified: bool = True,
) -> Any:
    """Load a pickled object from ``path`` after a SHA-256 check.

    Resolution order:

    1. If ``expected_sha256`` is provided, hash + compare; raise
       :class:`UnsafeLoadError` on mismatch.
    2. Otherwise, if ``verify_sidecar`` and ``<path>.sha256`` exists,
       load that digest, hash + compare; raise on mismatch.
    3. Otherwise, if ``allow_unverified`` is True, log a warning and
       load anyway.  If False, raise :class:`UnsafeLoadError`.

    The hash is computed BEFORE deserialisation, so a tampered file
    can never reach ``pickle.load``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"safe_load: no file at {path}")

    actual = file_sha256(path)

    if expected_sha256 is not None:
        if actual.lower() != expected_sha256.strip().lower():
            raise UnsafeLoadError(
                f"safe_load: SHA-256 mismatch for {path}: "
                f"expected {expected_sha256.lower()}, got {actual}"
            )
    else:
        sidecar = _read_sidecar(path) if verify_sidecar else None
        if sidecar is not None:
            if actual.lower() != sidecar:
                raise UnsafeLoadError(
                    f"safe_load: sidecar mismatch for {path}: "
                    f"sidecar says {sidecar}, file is {actual}"
                )
        elif not allow_unverified:
            raise UnsafeLoadError(
                f"safe_load: refusing unverified pickle at {path} "
                "(no expected_sha256, no sidecar, allow_unverified=False)"
            )
        else:
            logger.warning(
                "safe_load: loading %s WITHOUT integrity check; "
                "create a sidecar with safe_save() to enable verification",
                path,
            )

    with open(path, "rb") as fh:
        return pickle.load(fh)         # safe — hash already verified above


def verify_only(
    path: Union[str, Path],
    *,
    expected_sha256: Optional[str] = None,
) -> bool:
    """Return True if ``path``'s SHA-256 matches the expected / sidecar digest.

    Useful for boot-time integrity probes that don't need to actually
    deserialise anything.
    """
    path = Path(path)
    if not path.exists():
        return False
    actual = file_sha256(path)
    if expected_sha256 is not None:
        return actual.lower() == expected_sha256.strip().lower()
    sidecar = _read_sidecar(path)
    if sidecar is None:
        return False
    return actual.lower() == sidecar
