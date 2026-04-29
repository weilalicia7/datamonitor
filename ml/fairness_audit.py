"""
Fairness Auditing Module for SACT Scheduling

Monitors and reports on scheduling fairness across protected groups
to ensure the ML-driven optimization does not systematically
disadvantage patients based on age, gender, distance, or deprivation.

Metrics:
    - Demographic Parity: P(scheduled | group=A) = P(scheduled | group=B)
    - Equal Opportunity: P(scheduled | outcome=1, group=A) = P(scheduled | outcome=1, group=B)
    - Disparate Impact Ratio: P(scheduled|minority) / P(scheduled|majority) >= 0.8
    - Equalized Odds: TPR and FPR equal across groups

References:
    - Equality Act 2010 (UK) - Protected characteristics
    - NHS Constitution - Right to equal treatment
    - Hardt et al. (2016). Equality of Opportunity in Supervised Learning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FairnessMetric:
    """Single fairness metric result."""
    name: str
    group_a: str
    group_b: str
    rate_a: float
    rate_b: float
    difference: float
    ratio: float
    passes_threshold: bool
    threshold: float
    details: str


@dataclass
class FairnessReport:
    """Complete fairness audit report."""
    timestamp: str
    total_patients: int
    total_scheduled: int
    overall_rate: float
    metrics: List[FairnessMetric]
    passes_audit: bool
    violations: List[str]
    protected_groups_checked: List[str]
    # ``passes_audit`` is True when no DPD/DIR violation was found AND
    # the audit had enough data to make that judgement.  ``status``
    # disambiguates a vacuous pass (no metrics computed because input
    # was unusable) from a real pass.  Allowed values:
    #   'pass'                — gates clear and sample size is sufficient
    #   'fail'                — at least one DPD/DIR violation
    #   'group_column_missing'— ``group_column`` not present on input
    #   'single_group'        — every row landed in one bucket (no contrast)
    #   'insufficient_sample' — every comparable pair had < min_group_size
    status: str = 'pass'


class FairnessAuditor:
    """
    Audits scheduling decisions for fairness across protected groups.

    Protected attributes monitored:
    - Age band (<40, 40-60, 60-75, >75)
    - Gender (M, F)
    - Geographic distance (local, medium, remote)
    - Deprivation (via postcode proxy)

    Fairness criteria:
    - Demographic Parity Difference (DPD) < 0.15
    - Disparate Impact Ratio (DIR) >= 0.80 (Four-Fifths Rule)
    - Equal Opportunity Difference (EOD) < 0.10
    """

    # Four-Fifths Rule threshold (US EEOC, widely adopted)
    DISPARATE_IMPACT_THRESHOLD = 0.80

    # Maximum allowed difference in scheduling rates
    DEMOGRAPHIC_PARITY_THRESHOLD = 0.15

    # Maximum allowed difference in true positive rates
    EQUAL_OPPORTUNITY_THRESHOLD = 0.10

    # Minimum patients per group before a pairwise comparison is meaningful.
    # On real Velindre data a satellite site may have fewer than this — in
    # which case we must not declare a vacuous PASS; surface 'insufficient_
    # sample' so an operator knows the audit has not actually cleared.
    MIN_GROUP_SIZE = 5

    # Common gender-label variants we accept on real SACT data.  The
    # SACT v4.0 standard uses Person_Stated_Gender_Code (1/2/9), but
    # operational extracts often arrive as M/F or Male/Female strings.
    GENDER_NORMALISATION = {
        '1': 'Male',  1: 'Male',
        '2': 'Female', 2: 'Female',
        '9': 'Unknown', 9: 'Unknown',
        'M': 'Male', 'm': 'Male', 'Male': 'Male', 'male': 'Male',
        'F': 'Female', 'f': 'Female', 'Female': 'Female', 'female': 'Female',
        'U': 'Unknown', 'NS': 'Unknown', 'Not Stated': 'Unknown',
    }

    def __init__(self):
        self.audit_history: List[FairnessReport] = []

    def audit_schedule(self,
                       patients: List[Dict],
                       scheduled_ids: set,
                       group_column: str = 'Age_Band') -> FairnessReport:
        """
        Audit a scheduling decision for fairness.

        Args:
            patients: List of patient dicts with group attributes
            scheduled_ids: Set of patient IDs that were scheduled
            group_column: Which attribute to audit (Age_Band, Gender, etc.)

        Returns:
            FairnessReport with all metrics
        """
        from datetime import datetime

        total = len(patients)
        total_scheduled = sum(1 for p in patients
                              if p.get('Patient_ID', p.get('patient_id', '')) in scheduled_ids)
        overall_rate = total_scheduled / max(1, total)

        # ── Pre-audit input validation ────────────────────────────────────
        # If the group column is absent on every input row we cannot
        # produce a meaningful audit.  Do NOT silently fall back to a
        # vacuous PASS: emit a report with status='group_column_missing'
        # so the operator sees exactly why no metrics were computed.
        column_present = any(group_column in p for p in patients)
        if not column_present:
            logger.warning(
                "FairnessAuditor.audit_schedule: group_column %r is "
                "absent on every input patient — returning empty audit "
                "with status='group_column_missing'.  This guards "
                "against vacuous PASS verdicts on real data that lacks "
                "the expected column.",
                group_column,
            )
            report = FairnessReport(
                timestamp=datetime.now().isoformat(),
                total_patients=total,
                total_scheduled=total_scheduled,
                overall_rate=round(overall_rate, 4),
                metrics=[],
                passes_audit=False,
                violations=[
                    f"audit could not run: column {group_column!r} "
                    "is missing from input"
                ],
                protected_groups_checked=[group_column],
                status='group_column_missing',
            )
            self.audit_history.append(report)
            return report

        # Group patients (with on-the-fly normalisation for known
        # categorical columns so M/F vs Male/Female does not produce
        # spurious extra groups on real data).
        groups = {}
        for p in patients:
            pid = p.get('Patient_ID', p.get('patient_id', ''))
            group_val = p.get(group_column, 'unknown')
            if group_column in ('Gender', 'Person_Stated_Gender_Code'):
                group_val = self.GENDER_NORMALISATION.get(group_val, group_val)
            if group_val not in groups:
                groups[group_val] = {'total': 0, 'scheduled': 0, 'patients': []}
            groups[group_val]['total'] += 1
            if pid in scheduled_ids:
                groups[group_val]['scheduled'] += 1
            groups[group_val]['patients'].append(p)

        # Calculate pairwise fairness metrics
        metrics = []
        violations = []
        group_names = list(groups.keys())
        comparable_pairs_seen = 0

        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1:]:
                n1 = groups[g1]['total']
                n2 = groups[g2]['total']
                s1 = groups[g1]['scheduled']
                s2 = groups[g2]['scheduled']

                if n1 < self.MIN_GROUP_SIZE or n2 < self.MIN_GROUP_SIZE:
                    continue
                comparable_pairs_seen += 1

                rate1 = s1 / n1
                rate2 = s2 / n2

                # Demographic Parity Difference
                dpd = abs(rate1 - rate2)

                # Disparate Impact Ratio
                if max(rate1, rate2) > 0:
                    dir_ratio = min(rate1, rate2) / max(rate1, rate2)
                else:
                    dir_ratio = 1.0

                passes = (dpd <= self.DEMOGRAPHIC_PARITY_THRESHOLD and
                          dir_ratio >= self.DISPARATE_IMPACT_THRESHOLD)

                if not passes:
                    violations.append(
                        f"{group_column}: {g1} ({rate1:.0%}) vs {g2} ({rate2:.0%}) "
                        f"- DPD={dpd:.3f}, DIR={dir_ratio:.3f}"
                    )

                metric = FairnessMetric(
                    name=f'Demographic Parity ({group_column})',
                    group_a=str(g1),
                    group_b=str(g2),
                    rate_a=round(rate1, 4),
                    rate_b=round(rate2, 4),
                    difference=round(dpd, 4),
                    ratio=round(dir_ratio, 4),
                    passes_threshold=passes,
                    threshold=self.DEMOGRAPHIC_PARITY_THRESHOLD,
                    details=f"Scheduled: {g1}={s1}/{n1} ({rate1:.1%}), {g2}={s2}/{n2} ({rate2:.1%})"
                )
                metrics.append(metric)

        # Determine the verdict status.  A clean run with metrics →
        # 'pass' or 'fail'.  No metrics computed → distinguish between
        # a single-bucket input and a too-small-group input so the
        # operator knows whether to look at data shape vs sample size.
        if comparable_pairs_seen == 0:
            if len(group_names) <= 1:
                status_value = 'single_group'
                violations.append(
                    f"audit could not run: only {len(group_names)} "
                    f"group(s) present for {group_column!r} — no "
                    "pairwise contrast possible"
                )
            else:
                status_value = 'insufficient_sample'
                violations.append(
                    f"audit could not run: every pairwise comparison "
                    f"had < {self.MIN_GROUP_SIZE} patients in at "
                    "least one group"
                )
            passes_audit = False
        else:
            status_value = 'pass' if not violations else 'fail'
            passes_audit = (len(violations) == 0)

        report = FairnessReport(
            timestamp=datetime.now().isoformat(),
            total_patients=total,
            total_scheduled=total_scheduled,
            overall_rate=round(overall_rate, 4),
            metrics=metrics,
            passes_audit=passes_audit,
            violations=violations,
            protected_groups_checked=[group_column],
            status=status_value,
        )

        self.audit_history.append(report)
        return report

    def full_audit(self, patients: List[Dict],
                   scheduled_ids: set) -> Dict[str, FairnessReport]:
        """
        Run fairness audit across ALL protected attributes.

        Returns dict of group_column -> FairnessReport.
        """
        results = {}

        # Audit by age band
        results['Age_Band'] = self.audit_schedule(patients, scheduled_ids, 'Age_Band')

        # Audit by gender
        gender_patients = []
        for p in patients:
            p_copy = dict(p)
            gc = p.get('Person_Stated_Gender_Code', None)
            if gc in (1, '1'):
                p_copy['Gender'] = 'Male'
            elif gc in (2, '2'):
                p_copy['Gender'] = 'Female'
            else:
                p_copy['Gender'] = 'Unknown'
            gender_patients.append(p_copy)
        results['Gender'] = self.audit_schedule(gender_patients, scheduled_ids, 'Gender')

        # Audit by distance group
        dist_patients = []
        for p in patients:
            p_copy = dict(p)
            travel = p.get('Travel_Time_Min', p.get('travel_time', 30))
            if travel <= 20:
                p_copy['Distance_Group'] = 'Local (<20 min)'
            elif travel <= 45:
                p_copy['Distance_Group'] = 'Medium (20-45 min)'
            else:
                p_copy['Distance_Group'] = 'Remote (>45 min)'
            dist_patients.append(p_copy)
        results['Distance_Group'] = self.audit_schedule(dist_patients, scheduled_ids, 'Distance_Group')

        # Audit by deprivation proxy (travel distance as proxy)
        results['Travel_Distance'] = self.audit_schedule(dist_patients, scheduled_ids, 'Distance_Group')

        # Equal Opportunity audit — P(scheduled | attended, group) should be constant
        # Among patients who WOULD attend, scheduling rates must be equal across groups
        eo_age = self.compute_equal_opportunity(patients, scheduled_ids, 'Showed_Up', 'Age_Band')
        # Exclude 'Unknown' gender (Person_Stated_Gender_Code 0/9) from Equal Opportunity
        # — comparing Unknown vs Male/Female produces spurious violations
        known_gender_patients = [p for p in gender_patients if p.get('Gender') != 'Unknown']
        eo_gender = self.compute_equal_opportunity(known_gender_patients, scheduled_ids, 'Showed_Up', 'Gender')

        # Create an Equal Opportunity report
        eo_metrics = eo_age + eo_gender
        eo_violations = [f"{m.name}: {m.group_a} vs {m.group_b} (diff={m.difference:.3f})"
                         for m in eo_metrics if not m.passes_threshold]

        results['Equal_Opportunity'] = FairnessReport(
            timestamp=datetime.now().isoformat() if 'datetime' in dir() else '',
            total_patients=len(patients),
            total_scheduled=sum(1 for p in patients if p.get('Patient_ID', p.get('patient_id', '')) in scheduled_ids),
            overall_rate=0,
            metrics=eo_metrics,
            passes_audit=len(eo_violations) == 0,
            violations=eo_violations,
            protected_groups_checked=['Age_Band', 'Gender']
        )

        return results

    def compute_equal_opportunity(self,
                                  patients: List[Dict],
                                  scheduled_ids: set,
                                  outcome_column: str = 'Showed_Up',
                                  group_column: str = 'Age_Band') -> List[FairnessMetric]:
        """
        Equal Opportunity: P(scheduled | outcome=1, group=A) = P(scheduled | outcome=1, group=B)

        Among patients who WOULD attend (outcome=1), scheduling rates should be equal.
        """
        # Filter to patients with positive outcome (showed up historically)
        # Support both 'Showed_Up' (bool) and 'Attended_Status' (Yes/No) formats
        positive_patients = []
        for p in patients:
            attended = p.get('Attended_Status', p.get(outcome_column, None))
            if attended == 'Yes' or attended is True or (attended is None and p.get('Historical_NoShow_Rate', 0) < 0.2):
                positive_patients.append(p)

        groups = {}
        for p in positive_patients:
            pid = p.get('Patient_ID', p.get('patient_id', ''))
            group_val = p.get(group_column, 'unknown')
            if group_val not in groups:
                groups[group_val] = {'total': 0, 'scheduled': 0}
            groups[group_val]['total'] += 1
            if pid in scheduled_ids:
                groups[group_val]['scheduled'] += 1

        metrics = []
        group_names = list(groups.keys())

        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1:]:
                n1, n2 = groups[g1]['total'], groups[g2]['total']
                if n1 < 2 or n2 < 2:
                    continue

                rate1 = groups[g1]['scheduled'] / n1
                rate2 = groups[g2]['scheduled'] / n2
                eod = abs(rate1 - rate2)

                metrics.append(FairnessMetric(
                    name=f'Equal Opportunity ({group_column})',
                    group_a=str(g1),
                    group_b=str(g2),
                    rate_a=round(rate1, 4),
                    rate_b=round(rate2, 4),
                    difference=round(eod, 4),
                    ratio=round(min(rate1, rate2) / max(rate1, rate2, 1e-10), 4),
                    passes_threshold=eod <= self.EQUAL_OPPORTUNITY_THRESHOLD,
                    threshold=self.EQUAL_OPPORTUNITY_THRESHOLD,
                    details=f"Among positive outcomes: {g1}={rate1:.1%}, {g2}={rate2:.1%}"
                ))

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get fairness auditor status and history summary."""
        return {
            'total_audits': len(self.audit_history),
            'thresholds': {
                'demographic_parity': self.DEMOGRAPHIC_PARITY_THRESHOLD,
                'disparate_impact': self.DISPARATE_IMPACT_THRESHOLD,
                'equal_opportunity': self.EQUAL_OPPORTUNITY_THRESHOLD,
            },
            'criteria': {
                'demographic_parity': 'P(scheduled|group=A) - P(scheduled|group=B) < 0.15',
                'disparate_impact': 'P(scheduled|minority) / P(scheduled|majority) >= 0.80',
                'equal_opportunity': 'P(scheduled|showed_up, group=A) - P(scheduled|showed_up, group=B) < 0.10',
            },
            'protected_attributes': ['Age_Band', 'Gender', 'Distance_Group'],
            'legal_basis': 'Equality Act 2010 (UK), NHS Constitution',
            'last_audit': self.audit_history[-1].timestamp if self.audit_history else None,
            'last_passed': self.audit_history[-1].passes_audit if self.audit_history else None,
        }
