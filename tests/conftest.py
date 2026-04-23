"""
Shared pytest fixtures + import-time patches for flask_app integration tests.

``flask_app.py`` runs a full ``initialize_data()`` at module import, which
includes a CP-SAT / column-generation solve over ~200 patients.  That solve
takes ~15 minutes and is irrelevant to route-level testing, so we patch
``ScheduleOptimizer.optimize`` to return an empty result BEFORE flask_app is
imported.

BUT: once flask_app is imported, the stub is no longer needed — and in fact
COLLIDES with tests like ``test_optimization.py::TestColumnGeneration`` that
rely on the real CP-SAT + CG code paths.  So the flow is:

  1. Save references to the originals.
  2. Install the stub + raise COLUMN_GEN_THRESHOLD.
  3. Import flask_app (runs initialize_data with the stub).
  4. Restore the originals.

After step 4 every test — including ``test_optimization.py`` — sees the real
optimiser, and the flask-routes tests share an app instance whose state was
seeded through the cheap path.
"""

from __future__ import annotations

import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Project root on sys.path
# --------------------------------------------------------------------------- #

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Pre-import patch: stub the expensive optimiser solve
# --------------------------------------------------------------------------- #

import optimization.optimizer as _opt_mod  # noqa: E402


_ORIGINAL_OPTIMIZE = _opt_mod.ScheduleOptimizer.optimize
_ORIGINAL_CG_THRESHOLD = _opt_mod.COLUMN_GEN_THRESHOLD


def _stub_optimize(self, patients, *args, **kwargs):
    """Fast no-op replacement for the real CP-SAT + CG solve."""
    return _opt_mod.OptimizationResult(
        success=True,
        appointments=[],
        unscheduled=[getattr(p, "patient_id", str(p)) for p in patients],
        metrics={"stubbed": True, "n_patients": len(patients)},
        solve_time=0.0,
        status="STUBBED",
    )


# Install the stub.
_opt_mod.ScheduleOptimizer.optimize = _stub_optimize
_opt_mod.COLUMN_GEN_THRESHOLD = 100_000

# Only import flask_app if one of the flask-routes tests will actually run
# (cheap optimisation — skip import for a pure-unit-test run).
_flask_routes_files = [
    "test_flask_routes_health", "test_flask_routes_status",
    "test_flask_routes_ml", "test_flask_routes_data",
    "test_flask_routes_optimization", "test_flask_routes_twin",
    "test_flask_routes_mpc", "test_flask_routes_validation_errors",
]
_need_flask_import = any(
    any(token in arg for token in _flask_routes_files) for arg in sys.argv
) or not any(arg.startswith("tests/test_") or arg.startswith("tests\\test_")
             for arg in sys.argv)

if _need_flask_import:
    try:
        import flask_app  # noqa: F401  — triggers initialize_data() under the stub
    except Exception:
        # If flask_app import fails, restore originals then re-raise so
        # the rest of the suite still sees the real optimiser.
        _opt_mod.ScheduleOptimizer.optimize = _ORIGINAL_OPTIMIZE
        _opt_mod.COLUMN_GEN_THRESHOLD = _ORIGINAL_CG_THRESHOLD
        raise

# Regardless of whether flask_app got imported, restore the originals so
# the rest of the suite (test_optimization.py etc.) sees the real code.
_opt_mod.ScheduleOptimizer.optimize = _ORIGINAL_OPTIMIZE
_opt_mod.COLUMN_GEN_THRESHOLD = _ORIGINAL_CG_THRESHOLD
