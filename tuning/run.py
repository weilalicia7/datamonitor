"""
Orchestrator + CLI for the three tuners.

Reads the historical-appointment dataset for the configured channel
(synthetic by default), runs the requested tuner(s), and writes the
results to ``data_cache/tuning/manifest.json`` tagged with the channel.

The boot path in ``flask_app.py`` calls
:func:`tuning.manifest.load_overrides` which returns ``{}`` whenever
the manifest is in synthetic mode — so this orchestrator is safe to
run today without affecting the live prediction pipeline.

CLI::

    # Smoke run on synthetic data (channel auto-detected)
    python -m tuning.run --tuner=all

    # Force the real-data path (only if datasets/real_data/* present)
    SACT_CHANNEL=real python -m tuning.run --tuner=random_search
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tuning.manifest import (
    CHANNEL_REAL,
    CHANNEL_SYNTHETIC,
    detect_channel,
    record_tuning_run,
    summary,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def _data_dir(channel: str) -> Path:
    if channel == CHANNEL_REAL:
        return Path("datasets/real_data")
    return Path("datasets/sample_data")


def load_historical(channel: Optional[str] = None) -> tuple:
    """Return ``(historical_df, channel)``.

    ``historical_df`` is the appointment-level DataFrame with at least
    ``Attended_Status`` + numeric features.  Adds an ``is_noshow``
    integer column (0/1) for classifier targets.
    """
    import pandas as pd
    ch = channel or detect_channel()
    path = _data_dir(ch) / "historical_appointments.xlsx"
    if not path.exists():
        raise FileNotFoundError(
            f"tuning: no historical dataset for channel={ch!r} at {path}"
        )
    df = pd.read_excel(path)
    if "is_noshow" not in df.columns and "Attended_Status" in df.columns:
        df["is_noshow"] = (df["Attended_Status"] == "No").astype(int)
    return df, ch


# --------------------------------------------------------------------------- #
# Per-tuner runners (each writes its own manifest entry)
# --------------------------------------------------------------------------- #


def run_random_search(
    df, channel: str, *, n_iter: int = 20, cv_splits: int = 5,
) -> Dict[str, Any]:
    """RandomizedSearchCV for both no-show + duration targets."""
    from tuning.random_search import tune_duration_model, tune_noshow_model

    ns = tune_noshow_model(df, n_iter=n_iter, cv_splits=cv_splits)
    record_tuning_run(
        channel=channel, tuner_key="noshow_model",
        payload=ns.to_dict(), n_records=int(len(df)),
    )

    if "Actual_Duration" in df.columns:
        dur = tune_duration_model(
            df.dropna(subset=["Actual_Duration"]),
            n_iter=n_iter, cv_splits=cv_splits,
        )
        record_tuning_run(
            channel=channel, tuner_key="duration_model",
            payload=dur.to_dict(), n_records=int(len(df)),
        )
    else:
        dur = None

    return {"noshow_model": ns.to_dict(),
            "duration_model": (dur.to_dict() if dur else None)}


def run_grid_search(
    patients: List[Any],
    channel: str,
    *,
    solve_fn: Callable[[List[Any], Dict[str, float], int], Any],
    weight_sets: Optional[List[Dict[str, float]]] = None,
    time_limit_s: int = 5,
) -> Dict[str, Any]:
    """Grid sweep over PARETO_WEIGHT_SETS."""
    from tuning.grid_search import evaluate_weight_profiles
    if weight_sets is None:
        from config import PARETO_WEIGHT_SETS
        weight_sets = list(PARETO_WEIGHT_SETS)
    res = evaluate_weight_profiles(
        patients=patients,
        weight_sets=weight_sets,
        solve_fn=solve_fn,
        time_limit_s=time_limit_s,
    )
    record_tuning_run(
        channel=channel, tuner_key="cpsat_weights",
        payload=res.to_dict(), n_records=len(patients),
    )
    return res.to_dict()


def run_bayes_opt(
    *,
    channel: str,
    target: str,
    evaluate_fn: Callable[[float], Dict[str, Any]],
    n_initial_points: int = 30,
    n_calls: int = 50,
    bounds: Optional[tuple] = None,
    n_samples: int = 0,
) -> Dict[str, Any]:
    """Bayesian opt for one of {dro_epsilon, cvar_alpha, lipschitz_l}."""
    from tuning.bayes_opt import tune_scalar
    res = tune_scalar(
        target=target,
        evaluate_fn=evaluate_fn,
        bounds=bounds,
        n_initial_points=n_initial_points,
        n_calls=n_calls,
        n_samples=n_samples,
    )
    record_tuning_run(
        channel=channel, tuner_key=target,
        payload=res.to_dict(), n_records=int(n_samples),
    )
    return res.to_dict()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SACT tuning orchestrator")
    p.add_argument("--tuner", choices=["random_search", "grid_search", "bayes_opt", "all"],
                   default="random_search")
    p.add_argument("--channel", choices=["auto", "synthetic", "real"], default="auto")
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--bayes-target",
                   choices=["dro_epsilon", "cvar_alpha", "lipschitz_l"],
                   default="dro_epsilon")
    p.add_argument("--bayes-init", type=int, default=30)
    p.add_argument("--bayes-calls", type=int, default=50)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    channel = detect_channel() if args.channel == "auto" else args.channel
    print(f"[tuning] channel={channel!r}")

    if args.tuner in ("random_search", "all"):
        df, ch = load_historical(channel)
        print(f"[tuning] random_search: {len(df)} rows from channel={ch!r}")
        run_random_search(df, ch, n_iter=args.n_iter, cv_splits=args.cv_splits)

    if args.tuner in ("grid_search", "all"):
        # Grid search needs the live optimiser; only run via the Flask
        # endpoint, not the CLI.  Skip here with a friendly message.
        print("[tuning] grid_search via CLI is unsupported "
              "(requires the running optimiser); use POST /api/tuning/run "
              "with body {\"tuner\": \"grid_search\"} instead")

    if args.tuner in ("bayes_opt", "all"):
        # Bayes opt needs an evaluate_fn that calls the live solver too.
        print("[tuning] bayes_opt via CLI is unsupported "
              "(requires the running optimiser); use POST /api/tuning/run "
              f"with body {{\"tuner\": \"bayes_opt\", \"target\": \"{args.bayes_target}\"}} instead")

    print("[tuning] manifest summary:")
    for k, v in summary().items():
        print(f"   {k}: {v}")
    return 0


if __name__ == "__main__":                           # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
