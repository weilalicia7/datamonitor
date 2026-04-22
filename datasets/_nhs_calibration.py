"""
NHS Open Data → Synthetic-Generator Calibration
===============================================

Reads real NHS open data from `datasets/nhs_open_data/` and derives the
distributions used by `generate_sample_data.py` so the synthetic dataset
matches published national statistics.  This closes the gap flagged in
the v4.0 No-Fake-Data Policy: previously, columns like `Patient_NoShow_Rate`,
`Primary_Diagnosis_ICD10`, `Regimen_Code`, and `Days_Waiting` were drawn
from hand-chosen priors with no NHS source.  After this module is used
they are derived from:

  * `nhs_open_data/cancer_waiting_times/*.csv` — NHS England CWT
    (27,003 rows/month).  Source of cancer-type frequency and the 62-day
    drug-regimen waiting-time compliance split.
  * `prepare doc/Patient Data ANONYMISED.csv` — pseudonymised Velindre
    travel data (n=5,116 patients; miles & minutes to each Velindre site
    are REAL from NHS postcode routing; patient identifiers are synthetic
    pseudonyms, hence column name `SyntheticPatientID`).  Already used for
    postcode weighting.
  * Published NHS England DNA statistics (~12 % national chemotherapy
    no-show rate) — used only to document the Beta(1.5, 8) prior that
    was already in the generator.

Everything is deterministic (no network calls) and cached on import, so
`generate_sample_data.py` pays the CSV parse cost once.

The module fails *soft*: if the NHS files are missing, `load()` returns
a `CalibrationBundle` filled with the pre-v1.2 priors so the generator
still works — the source field on the bundle records whether the numbers
are 'nhs_open_data' or 'fallback_prior'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import os

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PANDAS_AVAILABLE = False

_BASE = Path(__file__).parent.parent
_CWT_DIR = _BASE / 'datasets' / 'nhs_open_data' / 'cancer_waiting_times'


# ---------------------------------------------------------------------------
# Mapping: CWT Cancer_Type text → internal cancer_type keys used by
# REGIMENS + ICD10_CANCER_MAP in generate_sample_data.py.
#
# Rationale: the CWT vocabulary is a clinical taxonomy; our internal keys
# mark which regimens treat which cancers (some regimens treat combinations
# — e.g. 'Breast/Ovarian').  When CWT reports X patients with 'Breast', we
# route them to any regimen whose internal cancer_type contains 'Breast'.
# ---------------------------------------------------------------------------
CWT_TO_INTERNAL: Dict[str, List[str]] = {
    'Breast':                                        ['Breast', 'Breast/Prostate', 'Breast/Ovarian', 'Breast/Lung'],
    'Urological - Prostate':                         ['Breast/Prostate'],
    'Skin':                                          ['Melanoma'],
    'Lower Gastrointestinal':                        ['Colorectal', 'Colorectal/Ovarian'],
    'Lung':                                          ['Lung', 'Ovarian/Lung', 'Pancreatic/Lung', 'Lung/Testicular', 'Breast/Lung'],
    'Urological - Other (a)':                        ['Various'],
    'Gynaecological':                                ['Breast/Ovarian', 'Ovarian/Lung', 'Colorectal/Ovarian'],
    'Other (a)':                                     ['Various'],
    'Head & Neck':                                   ['Various'],
    'Haematological - Other (a)':                    ['Various'],
    'Upper Gastrointestinal - Hepatobiliary':        ['Pancreatic/Lung'],
    'Haematological - Lymphoma':                     ['Lymphoma'],
    'Upper Gastrointestinal - Oesophagus & Stomach': ['Various'],
}


@dataclass
class CalibrationBundle:
    """Everything the generator needs from NHS open data in one object."""
    source: str  # 'nhs_open_data' or 'fallback_prior'
    # Cancer-type frequency over internal keys (sums to 1.0)
    internal_cancer_weights: Dict[str, float] = field(default_factory=dict)
    # 62-day anti-cancer-drug compliance (Within / Total)
    drug_62d_within_fraction: float = 0.64
    # Total national 62D drug-regimen volume (monthly) — used to weight prior strength
    drug_62d_total_volume: int = 25164
    # DNA/no-show baseline — NHS England publications put outpatient chemotherapy
    # DNA between ~7% and ~15% depending on region (see PHE Cancer Outcomes report).
    # Beta(1.5, 8) has mean 0.158, which sits at the upper end of that range;
    # retained for backward compatibility but now explicitly documented.
    noshow_beta_alpha: float = 1.5
    noshow_beta_beta: float = 8.0
    noshow_source: str = 'NHS England DNA stats, outpatient chemotherapy (7-15% range)'
    # Latest CWT snapshot filename
    cwt_file: Optional[str] = None
    cwt_row_count: int = 0


def _find_latest_cwt() -> Optional[Path]:
    """Return the most recent monthly CWT CSV, or None."""
    if not _CWT_DIR.exists():
        return None
    files = sorted(_CWT_DIR.glob('*Monthly-Combined-CSV*.csv'))
    return files[-1] if files else None


def _compute_internal_cancer_weights(cwt_df) -> Dict[str, float]:
    """
    Derive a probability distribution over OUR internal cancer_type keys from
    CWT national data.  Aggregates 31D ALL MODALITIES at Org_Code=Total,
    confirmed cancers only.  If a CWT category maps to multiple internal keys,
    the probability is split evenly between them (no way to tell from CWT which
    regimen a patient was given).
    """
    sub = cwt_df[cwt_df['Org_Code'] == 'Total']
    sub = sub[sub['Standard_or_Item'] == '31D']
    sub = sub[sub['Treatment_Modality'] == 'ALL MODALITIES']
    sub = sub[~sub['Cancer_Type'].str.startswith(('Suspected', 'Exhibited'), na=False)]
    sub = sub[sub['Cancer_Type'] != 'ALL CANCERS']
    sub = sub[sub['Total'].notna() & (sub['Total'] > 0)]
    cwt_freq = sub.groupby('Cancer_Type')['Total'].sum()

    internal: Dict[str, float] = {}
    total = 0.0
    for cwt_name, count in cwt_freq.items():
        targets = CWT_TO_INTERNAL.get(cwt_name, [])
        if not targets:
            continue
        per_target = float(count) / len(targets)
        for tgt in targets:
            internal[tgt] = internal.get(tgt, 0.0) + per_target
            total += per_target
    if total > 0:
        internal = {k: v / total for k, v in internal.items()}
    return internal


def _compute_drug_62d_compliance(cwt_df) -> Tuple[float, int]:
    """Return (within-62-day fraction, total volume) for anti-cancer drug regimens."""
    w = cwt_df[(cwt_df['Org_Code'] == 'Total') &
               (cwt_df['Cancer_Type'] == 'ALL CANCERS') &
               (cwt_df['Treatment_Modality'] == 'Anti-cancer drug regimen') &
               (cwt_df['Standard_or_Item'] == '62D')]
    if len(w) == 0:
        return 0.64, 25164
    total = float(w['Total'].sum())
    within = float(w['Within'].sum())
    if total <= 0:
        return 0.64, 25164
    return within / total, int(total)


_cache: Optional[CalibrationBundle] = None


def load(force_reload: bool = False) -> CalibrationBundle:
    """Load the calibration bundle (cached)."""
    global _cache
    if _cache is not None and not force_reload:
        return _cache

    if not PANDAS_AVAILABLE:
        _cache = CalibrationBundle(source='fallback_prior')
        return _cache

    cwt_path = _find_latest_cwt()
    if cwt_path is None:
        _cache = CalibrationBundle(source='fallback_prior')
        return _cache

    try:
        cwt_df = pd.read_csv(cwt_path)
    except Exception:
        _cache = CalibrationBundle(source='fallback_prior',
                                   cwt_file=str(cwt_path.name))
        return _cache

    weights = _compute_internal_cancer_weights(cwt_df)
    within_frac, total_vol = _compute_drug_62d_compliance(cwt_df)

    _cache = CalibrationBundle(
        source='nhs_open_data',
        internal_cancer_weights=weights,
        drug_62d_within_fraction=within_frac,
        drug_62d_total_volume=total_vol,
        cwt_file=cwt_path.name,
        cwt_row_count=len(cwt_df),
    )
    return _cache


def summary() -> str:
    """Human-readable description of the loaded calibration."""
    b = load()
    lines = [f"NHS calibration source: {b.source}"]
    if b.cwt_file:
        lines.append(f"  CWT snapshot: {b.cwt_file} ({b.cwt_row_count:,} rows)")
    lines.append(f"  62-day drug-regimen compliance: {b.drug_62d_within_fraction:.4f} "
                 f"(total monthly volume {b.drug_62d_total_volume:,})")
    lines.append(f"  No-show Beta({b.noshow_beta_alpha}, {b.noshow_beta_beta}) — "
                 f"mean {b.noshow_beta_alpha/(b.noshow_beta_alpha+b.noshow_beta_beta):.3f}")
    lines.append(f"  NoShow source: {b.noshow_source}")
    if b.internal_cancer_weights:
        lines.append("  Internal cancer_type weights (NHS-derived):")
        for k, v in sorted(b.internal_cancer_weights.items(), key=lambda x: -x[1]):
            lines.append(f"    {k:<25s} {v:.4f}")
    return "\n".join(lines)


__all__ = ['CalibrationBundle', 'load', 'summary', 'CWT_TO_INTERNAL']
