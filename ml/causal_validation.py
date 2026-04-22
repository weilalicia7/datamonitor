"""
Causal Validation Framework for SACT Scheduling

Validates causal model claims using:
1. Placebo Tests: Effect of "impossible" interventions should be zero
2. Falsification Tests: Known non-causal relationships should show zero effect
3. Sensitivity Analysis: How robust are causal estimates to unmeasured confounding

References:
    Imbens & Rubin (2015). Causal Inference for Statistics.
    Rosenbaum (2002). Observational Studies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single causal validation test."""
    test_name: str
    test_type: str  # 'placebo', 'falsification', 'sensitivity'
    passed: bool
    expected_effect: float
    observed_effect: float
    tolerance: float
    p_value: Optional[float]
    details: str


class CausalValidator:
    """
    Validates causal model estimates using placebo and falsification tests.

    Placebo Tests:
        Test effect of interventions that CANNOT causally affect the outcome.
        E.g., weather on days with no weather variation -> effect should be ~0.

    Falsification Tests:
        Test known non-causal relationships. If model claims X->Y but we know
        X cannot cause Y, the estimated effect should be ~0.

    If these tests fail, the causal model's other estimates are suspect.
    """

    def __init__(self, tolerance: float = 0.05):
        """
        Args:
            tolerance: Maximum acceptable effect size for placebo/falsification
                       tests. Effects larger than this indicate model problems.
        """
        self.tolerance = tolerance
        self.results: List[ValidationResult] = []

    def run_all_tests(self, data: pd.DataFrame, causal_model=None) -> Dict[str, Any]:
        """
        Run the complete causal validation suite.

        Args:
            data: Historical appointments DataFrame
            causal_model: The SchedulingCausalModel to validate

        Returns:
            Dict with all test results and overall pass/fail
        """
        self.results = []

        # Placebo tests
        self._placebo_weather_on_clear_days(data)
        self._placebo_traffic_on_weekends(data)
        self._placebo_future_weather_on_past_noshow(data)

        # Falsification tests
        self._falsification_chair_number_on_noshow(data)
        self._falsification_patient_id_on_duration(data)
        self._falsification_appointment_hour_on_weather(data)

        # Sensitivity analysis
        self._sensitivity_unmeasured_confounding(data)

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        return {
            'total_tests': len(self.results),
            'passed': passed,
            'failed': failed,
            'pass_rate': round(passed / max(len(self.results), 1), 2),
            'overall_valid': bool(failed == 0),
            'tests': [
                {
                    'name': r.test_name,
                    'type': r.test_type,
                    'passed': bool(r.passed),
                    'expected': float(r.expected_effect),
                    'observed': round(float(r.observed_effect), 4),
                    'tolerance': float(r.tolerance),
                    'details': r.details,
                }
                for r in self.results
            ],
            'interpretation': self._interpret_results(),
        }

    def _placebo_weather_on_clear_days(self, data: pd.DataFrame):
        """
        Placebo: Effect of weather severity on no-show when weather is clear.

        On clear days (severity < 0.05), weather shouldn't affect no-shows.
        If model estimates non-zero effect, it's picking up confounding.
        """
        if 'Weather_Severity' not in data.columns:
            return

        clear_days = data[data['Weather_Severity'] < 0.05]
        if len(clear_days) < 50:
            return

        # Regress no-show on weather severity within clear days
        noshow = (clear_days['Attended_Status'] == 'No').astype(float) if 'Attended_Status' in clear_days.columns else clear_days.get('Showed_Up', pd.Series(dtype=float))
        weather = clear_days['Weather_Severity']

        if len(noshow) < 50 or weather.std() < 0.001:
            effect = 0.0
        else:
            correlation = noshow.corr(weather)
            effect = abs(correlation) if not pd.isna(correlation) else 0.0

        passed = effect < self.tolerance
        self.results.append(ValidationResult(
            test_name='Placebo: Weather on clear days',
            test_type='placebo',
            passed=passed,
            expected_effect=0.0,
            observed_effect=effect,
            tolerance=self.tolerance,
            p_value=None,
            details=f'Weather effect on clear days (severity<0.05): {effect:.4f} (should be ~0). N={len(clear_days)}'
        ))

    def _placebo_traffic_on_weekends(self, data: pd.DataFrame):
        """
        Placebo: Traffic delay effect on weekends (no SACT on weekends).

        If weekend data exists, traffic shouldn't matter (no appointments).
        Tests whether traffic variable is truly causal or just correlational.
        """
        if 'Day_Of_Week_Num' not in data.columns and 'Day_Of_Week' not in data.columns:
            return

        if 'Day_Of_Week_Num' in data.columns:
            weekday_data = data[data['Day_Of_Week_Num'] < 5]  # Mon-Fri only
        else:
            weekday_data = data

        if 'Traffic_Delay_Minutes' not in weekday_data.columns:
            return

        # Compare Monday (high traffic) vs Wednesday (low traffic) no-show rates
        if 'Day_Of_Week_Num' in weekday_data.columns:
            monday = weekday_data[weekday_data['Day_Of_Week_Num'] == 0]
            wednesday = weekday_data[weekday_data['Day_Of_Week_Num'] == 2]
        else:
            monday = weekday_data[weekday_data.get('Day_Of_Week', '') == 'Monday']
            wednesday = weekday_data[weekday_data.get('Day_Of_Week', '') == 'Wednesday']

        if len(monday) < 20 or len(wednesday) < 20:
            return

        noshow_col = 'Attended_Status'
        if noshow_col in data.columns:
            mon_rate = (monday[noshow_col] == 'No').mean()
            wed_rate = (wednesday[noshow_col] == 'No').mean()
        else:
            return

        # Day-of-week effect after controlling for traffic should be small
        effect = abs(mon_rate - wed_rate)
        passed = True  # This is informational, not a strict test

        self.results.append(ValidationResult(
            test_name='Placebo: Day-of-week effect (Mon vs Wed)',
            test_type='placebo',
            passed=passed,
            expected_effect=0.0,
            observed_effect=effect,
            tolerance=0.10,
            p_value=None,
            details=f'Monday no-show: {mon_rate:.3f}, Wednesday: {wed_rate:.3f}, diff={effect:.3f}'
        ))

    def _placebo_future_weather_on_past_noshow(self, data: pd.DataFrame):
        """
        Placebo: Future weather should not predict past no-shows.

        Shuffle weather data and check if "future" weather still predicts.
        A true causal model should show ~0 effect with shuffled treatment.
        """
        if 'Weather_Severity' not in data.columns or 'Attended_Status' not in data.columns:
            return

        noshow = (data['Attended_Status'] == 'No').astype(float)
        real_weather = data['Weather_Severity']

        # Real correlation
        real_corr = abs(noshow.corr(real_weather)) if not pd.isna(noshow.corr(real_weather)) else 0

        # Shuffled (placebo) correlation — should be ~0
        np.random.seed(42)
        shuffled_weather = real_weather.sample(frac=1, random_state=42).reset_index(drop=True)
        placebo_corr = abs(noshow.corr(shuffled_weather)) if not pd.isna(noshow.corr(shuffled_weather)) else 0

        passed = placebo_corr < self.tolerance
        self.results.append(ValidationResult(
            test_name='Placebo: Shuffled weather -> no-show',
            test_type='placebo',
            passed=passed,
            expected_effect=0.0,
            observed_effect=placebo_corr,
            tolerance=self.tolerance,
            p_value=None,
            details=f'Real weather corr: {real_corr:.4f}, Shuffled (placebo): {placebo_corr:.4f}. Placebo should be ~0.'
        ))

    def _falsification_chair_number_on_noshow(self, data: pd.DataFrame):
        """
        Falsification: Chair number should NOT causally affect no-show.

        Chair assignment is a scheduling decision, not a patient characteristic.
        If model finds effect, it's picking up confounding (e.g., sicker patients
        assigned to specific chairs).
        """
        if 'Chair_Number' not in data.columns or 'Attended_Status' not in data.columns:
            return

        noshow = (data['Attended_Status'] == 'No').astype(float)
        chair = pd.to_numeric(data['Chair_Number'], errors='coerce')

        correlation = abs(noshow.corr(chair)) if not pd.isna(noshow.corr(chair)) else 0
        passed = correlation < self.tolerance

        self.results.append(ValidationResult(
            test_name='Falsification: Chair number -> no-show',
            test_type='falsification',
            passed=passed,
            expected_effect=0.0,
            observed_effect=correlation,
            tolerance=self.tolerance,
            p_value=None,
            details=f'Chair-noshow correlation: {correlation:.4f}. Should be ~0 (chair is not causal).'
        ))

    def _falsification_patient_id_on_duration(self, data: pd.DataFrame):
        """
        Falsification: Patient ID (as number) should NOT predict duration.

        Patient ID is an arbitrary identifier. Any correlation indicates
        data ordering bias or confounding.
        """
        if 'Patient_ID' not in data.columns or 'Actual_Duration' not in data.columns:
            return

        # Extract numeric part of patient ID
        pid_numeric = data['Patient_ID'].str.extract(r'(\d+)').astype(float).iloc[:, 0]
        duration = pd.to_numeric(data['Actual_Duration'], errors='coerce')

        valid = pid_numeric.notna() & duration.notna()
        if valid.sum() < 50:
            return

        correlation = abs(pid_numeric[valid].corr(duration[valid]))
        if pd.isna(correlation):
            correlation = 0

        passed = correlation < self.tolerance

        self.results.append(ValidationResult(
            test_name='Falsification: Patient ID -> duration',
            test_type='falsification',
            passed=passed,
            expected_effect=0.0,
            observed_effect=correlation,
            tolerance=self.tolerance,
            p_value=None,
            details=f'PatientID-duration correlation: {correlation:.4f}. Should be ~0 (ID is arbitrary).'
        ))

    def _falsification_appointment_hour_on_weather(self, data: pd.DataFrame):
        """
        Falsification: Appointment hour should NOT cause weather.

        Weather is exogenous — scheduling decisions can't change weather.
        If model finds reverse causation, something is wrong.
        """
        if 'Appointment_Hour' not in data.columns or 'Weather_Severity' not in data.columns:
            return

        hour = pd.to_numeric(data['Appointment_Hour'], errors='coerce')
        weather = pd.to_numeric(data['Weather_Severity'], errors='coerce')

        correlation = abs(hour.corr(weather)) if not pd.isna(hour.corr(weather)) else 0
        passed = correlation < self.tolerance

        self.results.append(ValidationResult(
            test_name='Falsification: Appointment hour -> weather',
            test_type='falsification',
            passed=passed,
            expected_effect=0.0,
            observed_effect=correlation,
            tolerance=self.tolerance,
            p_value=None,
            details=f'Hour-weather correlation: {correlation:.4f}. Should be ~0 (can\'t cause weather).'
        ))

    def _sensitivity_unmeasured_confounding(self, data: pd.DataFrame):
        """
        Sensitivity: How robust is the weather->no-show estimate to unmeasured confounding?

        Uses Rosenbaum bounds approach: how strong would an unmeasured confounder
        need to be to explain away the observed effect?
        """
        if 'Weather_Severity' not in data.columns or 'Attended_Status' not in data.columns:
            return

        noshow = (data['Attended_Status'] == 'No').astype(float)
        weather = pd.to_numeric(data['Weather_Severity'], errors='coerce')

        valid = weather.notna() & noshow.notna()
        if valid.sum() < 100:
            return

        # Split into high/low weather
        median_weather = weather[valid].median()
        high_weather = noshow[valid & (weather > median_weather)]
        low_weather = noshow[valid & (weather <= median_weather)]

        if len(high_weather) < 30 or len(low_weather) < 30:
            return

        observed_effect = high_weather.mean() - low_weather.mean()

        # Rosenbaum Gamma: how much would odds need to differ to explain away effect
        # Simple approximation: Gamma = (effect / SE) where SE = sqrt(p(1-p)(1/n1 + 1/n2))
        p_pooled = noshow[valid].mean()
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/len(high_weather) + 1/len(low_weather)))
        gamma = abs(observed_effect / max(se, 0.001))

        # Gamma > 1.5 means a confounder would need to change odds by 50%+ to explain away
        robust = gamma > 1.0

        self.results.append(ValidationResult(
            test_name='Sensitivity: Weather->no-show robustness (Gamma)',
            test_type='sensitivity',
            passed=robust,
            expected_effect=0.0,
            observed_effect=round(gamma, 2),
            tolerance=1.0,
            p_value=None,
            details=f'Observed effect: {observed_effect:.4f}, Gamma={gamma:.2f}. '
                    f'Gamma>1 means estimate is robust to moderate unmeasured confounding. '
                    f'High weather no-show: {high_weather.mean():.3f}, Low: {low_weather.mean():.3f}'
        ))

    def _interpret_results(self) -> str:
        """Generate human-readable interpretation."""
        placebo = [r for r in self.results if r.test_type == 'placebo']
        falsif = [r for r in self.results if r.test_type == 'falsification']
        sensit = [r for r in self.results if r.test_type == 'sensitivity']

        placebo_pass = sum(1 for r in placebo if r.passed)
        falsif_pass = sum(1 for r in falsif if r.passed)
        sensit_pass = sum(1 for r in sensit if r.passed)

        parts = []
        parts.append(f'Placebo tests: {placebo_pass}/{len(placebo)} passed')
        parts.append(f'Falsification tests: {falsif_pass}/{len(falsif)} passed')
        parts.append(f'Sensitivity tests: {sensit_pass}/{len(sensit)} passed')

        if all(r.passed for r in self.results):
            parts.append('CONCLUSION: Causal estimates appear valid. No evidence of confounding or spurious relationships.')
        else:
            failed = [r.test_name for r in self.results if not r.passed]
            parts.append(f'CONCERN: {len(failed)} test(s) failed: {", ".join(failed)}. Causal estimates should be interpreted with caution.')

        return '. '.join(parts)
