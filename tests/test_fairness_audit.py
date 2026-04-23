"""
Tests for ml.fairness_audit — FairnessAuditor, FairnessMetric, FairnessReport.

Covers:
    * Construction defaults.
    * Happy-path audit on synthetic patient records.
    * full_audit over all protected attributes.
    * Equal-opportunity computation on attended patients.
    * get_summary keys.
    * Fallback when groups are smaller than the n>=2 guard (no violations).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.fairness_audit import FairnessAuditor, FairnessMetric, FairnessReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_patients(n_per_group: int = 10):
    """Synthetic cohort with two age-bands and mixed outcomes."""
    patients = []
    idx = 0
    for band in ('Under 40', 'Over 75'):
        for i in range(n_per_group):
            patients.append({
                'Patient_ID': f'P{idx:04d}',
                'Age_Band': band,
                'Person_Stated_Gender_Code': 1 if i % 2 == 0 else 2,
                'Travel_Time_Min': 10 if i % 3 == 0 else 60,
                'Attended_Status': 'Yes' if i % 5 != 0 else 'No',
                'Showed_Up': i % 5 != 0,
            })
            idx += 1
    return patients


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_auditor_default_construction():
    auditor = FairnessAuditor()
    assert auditor.audit_history == []
    assert auditor.DISPARATE_IMPACT_THRESHOLD == 0.80
    assert auditor.DEMOGRAPHIC_PARITY_THRESHOLD == 0.15
    assert auditor.EQUAL_OPPORTUNITY_THRESHOLD == 0.10


def test_audit_schedule_returns_report():
    auditor = FairnessAuditor()
    patients = _make_patients(10)
    # Schedule everyone in Under 40 and only half of Over 75 — guaranteed
    # demographic-parity violation.
    scheduled_ids = {
        p['Patient_ID'] for p in patients
        if p['Age_Band'] == 'Under 40' or int(p['Patient_ID'][1:]) % 2 == 0
    }

    report = auditor.audit_schedule(patients, scheduled_ids, 'Age_Band')

    assert isinstance(report, FairnessReport)
    assert report.total_patients == len(patients)
    assert 0.0 <= report.overall_rate <= 1.0
    assert 'Age_Band' in report.protected_groups_checked
    assert len(report.metrics) >= 1
    assert all(isinstance(m, FairnessMetric) for m in report.metrics)
    assert isinstance(report.passes_audit, bool)


def test_audit_violation_detection_under_skew():
    """A heavily skewed schedule must report at least one violation."""
    auditor = FairnessAuditor()
    patients = _make_patients(20)
    # Only Under-40 patients are scheduled — 0% rate vs 100% rate.
    scheduled_ids = {
        p['Patient_ID'] for p in patients if p['Age_Band'] == 'Under 40'
    }

    report = auditor.audit_schedule(patients, scheduled_ids, 'Age_Band')

    assert not report.passes_audit
    assert len(report.violations) >= 1
    # Auditor must have remembered the run
    assert len(auditor.audit_history) == 1


def test_full_audit_covers_all_attributes():
    auditor = FairnessAuditor()
    patients = _make_patients(15)
    scheduled_ids = {p['Patient_ID'] for p in patients if int(p['Patient_ID'][1:]) % 2 == 0}

    results = auditor.full_audit(patients, scheduled_ids)

    # full_audit keys
    for key in ('Age_Band', 'Gender', 'Distance_Group', 'Travel_Distance', 'Equal_Opportunity'):
        assert key in results
        assert isinstance(results[key], FairnessReport)


def test_compute_equal_opportunity_on_attended_only():
    auditor = FairnessAuditor()
    patients = _make_patients(15)
    # Schedule *all* attended patients so equal-opportunity holds across groups.
    scheduled_ids = {p['Patient_ID'] for p in patients if p['Attended_Status'] == 'Yes'}

    metrics = auditor.compute_equal_opportunity(
        patients, scheduled_ids,
        outcome_column='Showed_Up',
        group_column='Age_Band',
    )

    assert isinstance(metrics, list)
    for m in metrics:
        assert isinstance(m, FairnessMetric)
        # All attended patients are scheduled → rates equal → passes.
        assert m.passes_threshold
        assert m.difference <= auditor.EQUAL_OPPORTUNITY_THRESHOLD


def test_get_summary_reports_history_and_thresholds():
    auditor = FairnessAuditor()
    patients = _make_patients(10)
    scheduled_ids = {p['Patient_ID'] for p in patients}
    auditor.audit_schedule(patients, scheduled_ids, 'Age_Band')

    summary = auditor.get_summary()

    # Required keys
    for key in ('total_audits', 'thresholds', 'criteria',
                'protected_attributes', 'legal_basis',
                'last_audit', 'last_passed'):
        assert key in summary
    assert summary['total_audits'] == 1
    assert summary['thresholds']['demographic_parity'] == 0.15
    assert summary['thresholds']['disparate_impact'] == 0.80
    # Everyone scheduled → must pass
    assert summary['last_passed'] is True
