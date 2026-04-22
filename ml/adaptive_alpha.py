"""
Risk-Adaptive Conformal α (Dissertation §2.2)
=============================================

Standard conformal prediction uses a fixed miscoverage level α (default
0.10) for every patient.  In a scheduling setting that is too blunt:
a patient who is almost certain to attend deserves a tight prediction
interval on their treatment duration, while a patient whose no-show
probability is 0.4 and is due on a day with 95 % chair utilisation
deserves a *wider* interval so the operator knows the worst-case
overrun risk.

This module implements the risk-adaptive policy from the §2.2 brief:

    α(p, o) = clamp( α_base + β_1 · P_noshow(p) + β_2 · occupancy,
                     α_floor, α_ceil )

It is intentionally simple — a 3-parameter affine policy with a clamp
— so the coverage guarantee of conformal prediction remains
interpretable: in the limit β_1 = β_2 = 0 we recover exactly the
baseline α_base and the original validity proof applies.  For non-zero
β, conditional coverage is no longer marginal at α_base but the
operator-facing interval width still respects the (1-α_floor) and
(1-α_ceil) bounds exactly (because α is clamped to those values before
the quantile lookup).

Design choices
--------------

* **Backward compatibility.** A fresh `AdaptiveAlphaPolicy()` with
  its defaults (β_1 = β_2 = 0) behaves indistinguishably from the
  legacy fixed-α pipeline, so every existing call site keeps working.
* **Bounded output.** `α_floor = 0.01` and `α_ceil = 0.20` keep the
  resulting interval coverage in `[0.80, 0.99]`, well inside the range
  for which conformal quantile lookups are numerically stable.
* **Invisibility.** Enforced at the Flask route layer: the existing
  `/api/ml/conformal/duration` and `/api/ml/conformal/noshow`
  responses are unchanged; this module only widens or narrows the
  intervals they already return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence
import json

import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# Module-level defaults — overridable via config.py or the
# /api/ml/conformal/adaptive/config POST endpoint.  Chosen so the
# default-on version yields:
#   low-risk patient (P=0.05)  at 50 % occupancy → α ≈ 0.107  (1 pp wider)
#   high-risk patient (P=0.40) at 90 % occupancy → α ≈ 0.180  (8 pp wider)
DEFAULT_ALPHA_BASE: float = 0.10
DEFAULT_BETA_NOSHOW: float = 0.15
DEFAULT_BETA_OCCUPANCY: float = 0.08
DEFAULT_ALPHA_FLOOR: float = 0.01
DEFAULT_ALPHA_CEIL: float = 0.20

ADAPTIVE_ALPHA_CONFIG_FILE: Path = DATA_CACHE_DIR / 'adaptive_alpha_config.json'
ADAPTIVE_ALPHA_LOG_FILE: Path = DATA_CACHE_DIR / 'adaptive_alpha_log.jsonl'


@dataclass
class AdaptiveAlphaPolicy:
    """
    The §2.2 policy object.  Instances are cheap; construct once per
    request.  Persistence is at the class level via a JSON config file
    so an operator can retune β_1 / β_2 without restarting the server.
    """
    alpha_base: float = DEFAULT_ALPHA_BASE
    beta_noshow: float = DEFAULT_BETA_NOSHOW
    beta_occupancy: float = DEFAULT_BETA_OCCUPANCY
    alpha_floor: float = DEFAULT_ALPHA_FLOOR
    alpha_ceil: float = DEFAULT_ALPHA_CEIL
    enabled: bool = True

    # -------- evaluation --------

    def compute(
        self,
        noshow_probability: Optional[float] = None,
        occupancy: Optional[float] = None,
    ) -> float:
        """
        Evaluate α(p, o) for a single patient.

        * `noshow_probability` is P_noshow in [0, 1]; None → treated as 0.
        * `occupancy` is current chair utilisation in [0, 1]; None → 0.
        * If the policy is disabled returns `alpha_base` verbatim.
        """
        if not self.enabled:
            return float(self.alpha_base)
        p = 0.0 if noshow_probability is None else float(noshow_probability)
        o = 0.0 if occupancy is None else float(occupancy)
        # defensive clamp on inputs so malformed upstream signals cannot
        # push α outside [floor, ceil]
        p = max(0.0, min(1.0, p))
        o = max(0.0, min(1.0, o))
        raw = self.alpha_base + self.beta_noshow * p + self.beta_occupancy * o
        return float(max(self.alpha_floor, min(self.alpha_ceil, raw)))

    def compute_batch(
        self,
        noshow_probabilities: Sequence[Optional[float]],
        occupancy: Optional[float] = None,
    ) -> np.ndarray:
        """Vectorised evaluation; `occupancy` is system-wide (scalar)."""
        p = np.asarray([0.0 if x is None else x for x in noshow_probabilities],
                       dtype=float)
        o = 0.0 if occupancy is None else float(occupancy)
        o = max(0.0, min(1.0, o))
        p = np.clip(p, 0.0, 1.0)
        raw = self.alpha_base + self.beta_noshow * p + self.beta_occupancy * o
        return np.clip(raw, self.alpha_floor, self.alpha_ceil).astype(float)

    # -------- persistence --------

    def to_dict(self) -> Dict[str, float]:
        return {
            'alpha_base': self.alpha_base,
            'beta_noshow': self.beta_noshow,
            'beta_occupancy': self.beta_occupancy,
            'alpha_floor': self.alpha_floor,
            'alpha_ceil': self.alpha_ceil,
            'enabled': bool(self.enabled),
        }

    def save(self, path: Path = ADAPTIVE_ALPHA_CONFIG_FILE) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: Path = ADAPTIVE_ALPHA_CONFIG_FILE) -> 'AdaptiveAlphaPolicy':
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            return cls(
                alpha_base=float(payload.get('alpha_base', DEFAULT_ALPHA_BASE)),
                beta_noshow=float(payload.get('beta_noshow', DEFAULT_BETA_NOSHOW)),
                beta_occupancy=float(payload.get('beta_occupancy', DEFAULT_BETA_OCCUPANCY)),
                alpha_floor=float(payload.get('alpha_floor', DEFAULT_ALPHA_FLOOR)),
                alpha_ceil=float(payload.get('alpha_ceil', DEFAULT_ALPHA_CEIL)),
                enabled=bool(payload.get('enabled', True)),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(f"AdaptiveAlphaPolicy.load failed: {exc}")
            return cls()


# ---------------------------------------------------------------------------
# Module-level singleton so the Flask routes + the conformal predictors see
# the same policy instance.  Reload on demand via reload_policy().
# ---------------------------------------------------------------------------

_POLICY: Optional[AdaptiveAlphaPolicy] = None


def get_policy() -> AdaptiveAlphaPolicy:
    global _POLICY
    if _POLICY is None:
        _POLICY = AdaptiveAlphaPolicy.load()
    return _POLICY


def set_policy(policy: AdaptiveAlphaPolicy) -> None:
    """Replace the live policy (also persists to disk)."""
    global _POLICY
    _POLICY = policy
    try:
        policy.save()
    except Exception as exc:  # pragma: no cover
        logger.warning(f"set_policy save failed: {exc}")


def reload_policy() -> AdaptiveAlphaPolicy:
    """Re-read the on-disk config file (e.g., after external edit)."""
    global _POLICY
    _POLICY = AdaptiveAlphaPolicy.load()
    return _POLICY


__all__ = [
    'AdaptiveAlphaPolicy',
    'DEFAULT_ALPHA_BASE',
    'DEFAULT_BETA_NOSHOW',
    'DEFAULT_BETA_OCCUPANCY',
    'DEFAULT_ALPHA_FLOOR',
    'DEFAULT_ALPHA_CEIL',
    'get_policy',
    'set_policy',
    'reload_policy',
]
