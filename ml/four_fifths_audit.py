"""
Four-Fifths Rule fairness audit on no-show rates across protected
characteristics.  Mirrors ``dissertation/dissertation_analysis.R``
Section 10 so that the production pipeline and the dissertation
report identical Four-Fifths ratios from the same data.

Methodology (per the EEOC original and the UK Equality Act 2010 s.19
indirect-discrimination test): a protected characteristic passes when
``min(no_show_rate) / max(no_show_rate) >= 0.80`` across its groups.

The Person_Stated_Gender_Code = 9 (Not_Stated) value is a SACT v4.0
data-quality category, not a protected attribute.  The primary audit
therefore restricts gender to codes 1 (Male) and 2 (Female); a
secondary "Unknown-included" ratio is reported for transparency so
the reader can see how the exclusion changes the headline number.

The output is exposed via ``GET /api/ml/fairness/four-fifths`` for
diagnostics; the audit runs silently as part of the model-training
pipeline (no UI panel).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


THRESHOLD = 0.80         # Four-Fifths rule
MIN_GROUP_N = 10         # min observations per group for inclusion
DEFAULT_BOOTSTRAP_B = 1000   # bootstrap resamples for ratio CIs
DEFAULT_BOOTSTRAP_SEED = 2026


@dataclass
class GroupRatio:
    name:           str
    ratio:          float
    max_disparity:  float
    passes:         bool
    n_groups:       int
    group_sizes:    Dict[str, int]
    ratio_ci_lo:    Optional[float] = None  # 95% percentile bootstrap CI
    ratio_ci_hi:    Optional[float] = None
    n_bootstrap:    Optional[int]   = None


@dataclass
class FourFifthsResult:
    groups:                    List[GroupRatio]
    n_passing:                 int
    n_failing:                 int
    gender_ratio_incl_unknown: Optional[float] = None  # transparency disclosure
    threshold:                 float = THRESHOLD
    method:                    str = (
        "Four-Fifths Rule (EEOC; UK Equality Act 2010 s.19). "
        "Gender restricted to coded Male/Female; 'Not_Stated' is a "
        "data-quality category and excluded from the primary audit."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'threshold':                 self.threshold,
            'n_passing':                 self.n_passing,
            'n_failing':                 self.n_failing,
            'gender_ratio_incl_unknown': self.gender_ratio_incl_unknown,
            'method':                    self.method,
            'groups': [asdict(g) for g in self.groups],
        }


def _compute_ratio(sub: pd.DataFrame, group_col: str,
                   outcome: str) -> Optional[float]:
    """Return min/max group rate ratio, or None if insufficient groups."""
    rates = sub.groupby(group_col)[outcome].mean()
    if len(rates) < 2:
        return None
    rmin, rmax = float(rates.min()), float(rates.max())
    return rmin / rmax if rmax > 0 else 0.0


def _bootstrap_ratio_ci(
    sub: pd.DataFrame, group_col: str, outcome: str,
    *, n_bootstrap: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Optional[Dict[str, float]]:
    """
    Stratified percentile bootstrap CI for the Four-Fifths ratio.

    Each resample draws rows with replacement (size = original n)
    from the cohort and recomputes the ratio.  Resamples that lose
    a group entirely (resulting in <2 groups) are skipped.  The 95%
    CI is the (2.5, 97.5) percentile of the resample distribution.

    Returns ``None`` when ``n_bootstrap == 0`` (caller opts out) or
    when the cohort is too small (n < 50).
    """
    if n_bootstrap <= 0 or len(sub) < 50:
        return None
    rng = np.random.RandomState(seed)
    idx_max = len(sub)
    arr = sub.reset_index(drop=True)
    ratios: List[float] = []
    skipped = 0
    for _ in range(n_bootstrap):
        boot_idx = rng.randint(0, idx_max, size=idx_max)
        b_sub = arr.iloc[boot_idx]
        r = _compute_ratio(b_sub, group_col, outcome)
        if r is not None:
            ratios.append(r)
        else:
            skipped += 1
    if len(ratios) < n_bootstrap // 2:
        return None
    arr_r = np.array(ratios)
    return {
        "ci_lo":         float(np.percentile(arr_r, 2.5)),
        "ci_hi":         float(np.percentile(arr_r, 97.5)),
        "n_bootstrap":   int(len(ratios)),
        "n_skipped":     int(skipped),
    }


def _ratio_for(df: pd.DataFrame, group_col: str, outcome: str,
               *, n_bootstrap: int = DEFAULT_BOOTSTRAP_B,
               seed: int = DEFAULT_BOOTSTRAP_SEED) -> Optional[GroupRatio]:
    if group_col not in df.columns or outcome not in df.columns:
        return None
    sub = df[[group_col, outcome]].dropna()
    if sub.empty:
        return None
    rates = (
        sub.groupby(group_col)[outcome]
           .agg(['mean', 'count'])
           .rename(columns={'mean': 'rate', 'count': 'n'})
    )
    rates = rates[rates['n'] >= MIN_GROUP_N]
    if len(rates) < 2:
        return None
    keep_groups = rates.index
    sub_kept = sub[sub[group_col].isin(keep_groups)].reset_index(drop=True)
    rmin, rmax = float(rates['rate'].min()), float(rates['rate'].max())
    ratio = rmin / rmax if rmax > 0 else 0.0

    boot = _bootstrap_ratio_ci(sub_kept, group_col, outcome,
                                n_bootstrap=n_bootstrap, seed=seed)

    return GroupRatio(
        name           = group_col,
        ratio          = round(ratio, 3),
        max_disparity  = round(rmax - rmin, 3),
        passes         = ratio >= THRESHOLD,
        n_groups       = int(len(rates)),
        group_sizes    = {str(k): int(v) for k, v in rates['n'].items()},
        ratio_ci_lo    = round(boot["ci_lo"], 3) if boot else None,
        ratio_ci_hi    = round(boot["ci_hi"], 3) if boot else None,
        n_bootstrap    = boot["n_bootstrap"] if boot else None,
    )


def audit_four_fifths(
    historical_df: pd.DataFrame,
    noshow_col:        str = "is_noshow",
    age_col:           str = "Age_Band",
    site_col:          str = "Site_Code",
    priority_col:      str = "Priority",
    gender_code_col:   str = "Person_Stated_Gender_Code",
    *,
    n_bootstrap:       int = DEFAULT_BOOTSTRAP_B,
    bootstrap_seed:    int = DEFAULT_BOOTSTRAP_SEED,
) -> Optional[FourFifthsResult]:
    """Run the Four-Fifths Rule audit on the given cohort.

    Returns ``None`` when the outcome column is absent.

    Each per-group ratio carries a 95 % percentile bootstrap CI
    over ``n_bootstrap`` row resamples (default 1 000); set
    ``n_bootstrap=0`` to skip the bootstrap (fast-path for tests).
    """
    if noshow_col not in historical_df.columns:
        return None
    df = historical_df.copy()

    groups: List[GroupRatio] = []
    for col in (age_col, site_col, priority_col):
        gr = _ratio_for(df, col, noshow_col,
                        n_bootstrap=n_bootstrap, seed=bootstrap_seed)
        if gr is not None:
            groups.append(gr)

    # Primary gender audit: codes 1, 2 only
    gender_ratio_incl_unknown: Optional[float] = None
    if gender_code_col in df.columns:
        coded = df[df[gender_code_col].isin([1, 2])].copy()
        coded['Gender'] = np.where(coded[gender_code_col] == 1, 'Male', 'Female')
        coded[noshow_col] = df.loc[coded.index, noshow_col].values
        gr = _ratio_for(coded, 'Gender', noshow_col,
                        n_bootstrap=n_bootstrap, seed=bootstrap_seed)
        if gr is not None:
            groups.append(gr)

        # Transparency disclosure: gender ratio with Unknown INCLUDED
        coded_inc = df[df[gender_code_col].isin([1, 2, 9])].copy()
        coded_inc['Gender'] = coded_inc[gender_code_col].map(
            {1: 'Male', 2: 'Female', 9: 'Not_Stated'}
        )
        gr_inc = _ratio_for(coded_inc, 'Gender', noshow_col,
                             n_bootstrap=0, seed=bootstrap_seed)
        if gr_inc is not None:
            gender_ratio_incl_unknown = gr_inc.ratio

    if not groups:
        return None

    return FourFifthsResult(
        groups                    = groups,
        n_passing                 = sum(1 for g in groups if g.passes),
        n_failing                 = sum(1 for g in groups if not g.passes),
        gender_ratio_incl_unknown = gender_ratio_incl_unknown,
    )
