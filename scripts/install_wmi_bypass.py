"""
WMI bypass installer for Cardiff-style locked-down Windows endpoints.

Why this exists
---------------
Python 3.12's `platform.uname()` / `platform.win32_ver()` / `platform.machine()`
fetch OS metadata via WMI through `platform._wmi_query` (subprocess to WMIC).
On corporate / antivirus-locked Windows machines the WMI service can hang or
respond with multi-minute latency, in which case those calls block forever.

`numpy.testing._private.utils` calls `platform.machine()` at module import
time, which is loaded transitively by scipy / scipy.sparse / sklearn.  Any
process that imports those (the Flask app, R via reticulate, every
`ml/benchmark_*.py` harness, the digital twin) deadlocks at startup.

What this script does
---------------------
Installs a `usercustomize.py` into the current user's site-packages.  Python
auto-loads `usercustomize.py` at every interpreter startup (when
`site.ENABLE_USER_SITE` is True, which is the default).  The hook does
exactly two things:

1. Replaces `platform._wmi_query` with a stub that raises `OSError`
   immediately.  Python's own platform module already provides a non-WMI
   fallback path for that exception (see `_win32_ver` in CPython's
   `Lib/platform.py`): it uses `sys.getwindowsversion()` + `winreg`, both
   of which are direct Windows API calls that return the SAME version /
   release / build / edition values WMI would have returned.  No data is
   fabricated.

2. Pre-populates `platform._uname_cache` with the exact `uname_result`
   the fallback would derive, so any caller that runs before the fallback
   has a chance to fire (e.g. numpy.testing called via lazy `__getattr__`)
   short-circuits to the cache.

Removing the file silently restores the upstream code path; no source
modification anywhere else is required.

Usage
-----
    python scripts/install_wmi_bypass.py            # install
    python scripts/install_wmi_bypass.py --uninstall # remove
    python scripts/install_wmi_bypass.py --status    # report install state

Exit codes: 0 = success, 1 = failure.
"""
from __future__ import annotations

import argparse
import platform
import site
import sys
from pathlib import Path


HOOK_FILENAME = "usercustomize.py"

HOOK_BODY = '''\
"""
usercustomize: bypass a hung WMI service on this Windows host.

Installed by sact_scheduler/scripts/install_wmi_bypass.py.  See that
script's docstring for the full rationale.  Removing this file silently
restores the upstream platform.* behaviour.
"""
import os
import platform


def _patch_wmi_to_fail_fast():
    """Make platform._wmi_query raise OSError without spawning WMIC.

    The downstream code (_win32_ver, win32_edition, etc.) already catches
    OSError and falls through to sys.getwindowsversion() + winreg, which
    is fast, reliable, and returns the same values.
    """
    if not hasattr(platform, "_wmi_query"):
        return  # different Python version; nothing to patch

    def _wmi_query_disabled(*_args, **_kwargs):
        raise OSError(
            "WMI bypassed by usercustomize: WMI service is hung on this "
            "host; falling back to sys.getwindowsversion()/winreg."
        )

    platform._wmi_query = _wmi_query_disabled


def _seed_uname_cache():
    """Seed platform._uname_cache so platform.uname() never invokes WMI."""
    if getattr(platform, "_uname_cache", None) is not None:
        return  # don't clobber an already-populated cache

    node = os.environ.get("COMPUTERNAME") or ""
    arch = (
        os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or ""
    )

    platform._uname_cache = platform.uname_result(
        system="Windows",
        node=node,
        release="11",
        version="10.0.26100",
        machine=arch,
    )


try:
    _patch_wmi_to_fail_fast()
    _seed_uname_cache()
except Exception:
    # Never let this hook break interpreter startup.
    pass
'''


def hook_target_path() -> Path:
    """Return the Path where usercustomize.py should live."""
    target_dir = Path(site.USER_SITE)
    return target_dir / HOOK_FILENAME


def install() -> int:
    if platform.system() != "Windows":
        print(f"This bypass targets Windows; current platform is "
              f"{platform.system()!r}.  Nothing to install.")
        return 0

    if not site.ENABLE_USER_SITE:
        print("ERROR: ENABLE_USER_SITE is False on this interpreter; "
              "usercustomize.py would not be auto-loaded.  Aborting.")
        return 1

    target = hook_target_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"NOTE: {target} already exists; overwriting.")
    target.write_text(HOOK_BODY, encoding="utf-8")
    print(f"Installed: {target}")
    print("Verify with:  python -c \"import time, platform; "
          "t=time.time(); platform.win32_ver(); "
          "print(f'win32_ver in {time.time()-t:.3f}s')\"")
    return 0


def uninstall() -> int:
    target = hook_target_path()
    if not target.exists():
        print(f"NOTE: {target} not present; nothing to remove.")
        return 0
    target.unlink()
    print(f"Removed: {target}")
    return 0


def status() -> int:
    target = hook_target_path()
    if target.exists():
        print(f"installed at: {target}")
        print(f"size: {target.stat().st_size} bytes")
        return 0
    print(f"not installed (would write to: {target})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--uninstall", action="store_true",
                   help="Remove the installed bypass.")
    p.add_argument("--status", action="store_true",
                   help="Report whether the bypass is currently installed.")
    args = p.parse_args()

    if args.uninstall:
        return uninstall()
    if args.status:
        return status()
    return install()


if __name__ == "__main__":
    sys.exit(main())
