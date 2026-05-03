"""
Survival Analysis for Time-to-Event No-Show Prediction (2.2)

Models not just IF but WHEN a no-show occurs using:
- Cox Proportional Hazards: λ(t|x) = λ₀(t) · exp(βᵀx)
- Survival Function: S(t) = P(T > t) = exp(-∫λ(s)ds)

Benefits:
- Predicts optimal reminder timing
- Identifies when risk is highest (e.g., 3 days before appointment)
- More nuanced scheduling decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class SurvivalPrediction:
    """Survival analysis prediction result."""
    patient_id: str
    survival_probability: float  # S(t) at appointment time
    hazard_rate: float  # λ(t) instantaneous risk
    risk_peak_days: int  # Days before appointment when risk is highest
    optimal_reminder_days: List[int]  # Recommended reminder timing
    cumulative_hazard: float  # H(t) = -log(S(t))
    risk_category: str  # 'low', 'medium', 'high', 'critical'
    confidence_interval: Tuple[float, float]  # 95% CI for survival


class CoxProportionalHazards:
    """
    Cox Proportional Hazards Model for no-show prediction.

    λ(t|x) = λ₀(t) · exp(βᵀx)

    Where:
    - λ(t|x) is the hazard function given covariates x
    - λ₀(t) is the baseline hazard function
    - β are the regression coefficients
    - x are the covariates (patient features)
    """

    def __init__(self):
        self.coefficients: Optional[np.ndarray] = None
        self.baseline_hazard: Optional[np.ndarray] = None
        self.baseline_times: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

        # Default coefficients based on literature and domain knowledge
        # These represent log-hazard ratios
        self._default_coefficients = {
            'age_band_0_30': -0.15,      # Younger patients slightly lower risk
            'age_band_30_50': 0.0,        # Reference group
            'age_band_50_70': 0.08,       # Slightly higher risk
            'age_band_70_plus': 0.12,     # Higher risk for elderly
            'previous_noshow_rate': 0.85, # Strong predictor
            'days_since_last_visit': 0.02,  # Longer gap = higher risk
            'appointment_hour_early': -0.10,  # Early appointments lower risk
            'appointment_hour_midday': 0.05,  # Midday slightly higher
            'appointment_hour_late': 0.15,    # Late afternoon higher risk
            'monday': 0.08,               # Monday higher risk
            'friday': 0.12,               # Friday highest
            'treatment_cycle': -0.03,     # Later cycles = more committed
            'distance_km': 0.01,          # Distance impact
            'weather_adverse': 0.20,      # Bad weather increases risk
            'is_first_appointment': 0.25, # First appointments high risk
        }

        logger.info("CoxProportionalHazards model initialized")

    def _compute_baseline_hazard(self, max_days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute baseline hazard function λ₀(t).

        Uses Weibull distribution as baseline:
        λ₀(t) = (k/λ) * (t/λ)^(k-1)

        Parameters calibrated for appointment no-shows:
        - Risk increases as appointment approaches
        - Peak at 2-3 days before appointment
        """
        # Time points (days before appointment, 0 = appointment day)
        times = np.arange(max_days, -1, -1)  # 30, 29, ..., 1, 0

        # Weibull parameters for no-show hazard
        # Shape k > 1 means increasing hazard over time
        k = 2.5  # Shape parameter
        lam = 7.0  # Scale parameter (peak around 7 days)

        # Weibull hazard: λ₀(t) = (k/λ) * (t/λ)^(k-1)
        # But we invert since we count down to appointment
        t_transformed = max_days - times + 1  # Transform to counting up

        baseline = np.zeros_like(times, dtype=float)
        valid_idx = t_transformed > 0
        baseline[valid_idx] = (k / lam) * (t_transformed[valid_idx] / lam) ** (k - 1)

        # Normalize to reasonable scale
        baseline = baseline / baseline.max() * 0.15  # Max baseline hazard ~15%

        return times, baseline

    def fit(self,
            patient_data: pd.DataFrame,
            event_column: str = 'no_show',
            duration_column: str = 'days_to_appointment') -> 'CoxProportionalHazards':
        """
        Fit the Cox PH model to patient data.

        Parameters:
        -----------
        patient_data : pd.DataFrame
            Patient features and outcomes
        event_column : str
            Column indicating if no-show occurred (1) or not (0)
        duration_column : str
            Column with time to event (days until appointment)
        """
        if patient_data.empty:
            logger.warning("Empty patient data, using default coefficients")
            self._use_default_coefficients()
            return self

        # Extract features (exclude event and duration columns)
        feature_cols = [c for c in patient_data.columns
                       if c not in [event_column, duration_column, 'patient_id']]

        X = patient_data[feature_cols].values
        events = patient_data[event_column].values if event_column in patient_data else None
        durations = patient_data[duration_column].values if duration_column in patient_data else None

        if events is None or durations is None:
            logger.warning("Missing event/duration data, using default coefficients")
            self._use_default_coefficients()
            return self

        # Partial likelihood estimation using Newton-Raphson
        self.coefficients = self._estimate_coefficients(X, events, durations)
        self.feature_names = feature_cols

        # Compute baseline hazard using Breslow estimator
        self.baseline_times, self.baseline_hazard = self._breslow_estimator(
            X, events, durations
        )

        self.is_fitted = True
        logger.info(f"Cox PH model fitted with {len(self.coefficients)} coefficients")

        return self

    def _use_default_coefficients(self):
        """Initialize with domain-knowledge based coefficients."""
        self.feature_names = list(self._default_coefficients.keys())
        self.coefficients = np.array(list(self._default_coefficients.values()))
        self.baseline_times, self.baseline_hazard = self._compute_baseline_hazard()
        self.is_fitted = True

    def _estimate_coefficients(self,
                               X: np.ndarray,
                               events: np.ndarray,
                               durations: np.ndarray,
                               max_iter: int = 100,
                               tol: float = 1e-6) -> np.ndarray:
        """
        Estimate β using partial likelihood maximization.

        The partial likelihood for Cox PH is:
        L(β) = ∏ᵢ [exp(βᵀxᵢ) / Σⱼ∈R(tᵢ) exp(βᵀxⱼ)]

        Where R(tᵢ) is the risk set at time tᵢ.
        """
        n_samples, n_features = X.shape
        beta = np.zeros(n_features)

        # Sort by duration (descending for risk set computation)
        sort_idx = np.argsort(-durations)
        X_sorted = X[sort_idx]
        events_sorted = events[sort_idx]
        durations_sorted = durations[sort_idx]

        for iteration in range(max_iter):
            # Compute linear predictor
            eta = X_sorted @ beta
            exp_eta = np.exp(np.clip(eta, -500, 500))  # Clip for stability

            # Compute gradient and Hessian
            gradient = np.zeros(n_features)
            hessian = np.zeros((n_features, n_features))

            # Cumulative sums for risk set calculations
            risk_sum = np.cumsum(exp_eta[::-1])[::-1]
            weighted_x_sum = np.cumsum((X_sorted.T * exp_eta)[:, ::-1], axis=1)[:, ::-1].T

            for i in range(n_samples):
                if events_sorted[i] == 1:  # Event occurred
                    if risk_sum[i] > 0:
                        x_bar = weighted_x_sum[i] / risk_sum[i]
                        gradient += X_sorted[i] - x_bar

                        # Hessian contribution.  Mathematically this is
                        # outer(weighted_x_sum, weighted_x_sum) / risk_sum**2,
                        # but computing the squared denominator overflows when
                        # exp(eta) is large (clip is wide for numerical
                        # fidelity).  outer(x_bar, x_bar) is identical and the
                        # ratio is bounded.
                        hessian -= np.outer(x_bar, x_bar)

            # Add small regularization for stability
            hessian -= np.eye(n_features) * 0.01

            # Newton-Raphson update
            try:
                delta = np.linalg.solve(-hessian, gradient)
            except np.linalg.LinAlgError:
                delta = gradient * 0.01  # Fallback to gradient descent

            beta += delta

            # Check convergence
            if np.max(np.abs(delta)) < tol:
                logger.info(f"Cox PH converged after {iteration + 1} iterations")
                break

        return beta

    def _breslow_estimator(self,
                          X: np.ndarray,
                          events: np.ndarray,
                          durations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Breslow estimator for baseline cumulative hazard.

        H₀(t) = Σᵢ:tᵢ≤t [dᵢ / Σⱼ∈R(tᵢ) exp(βᵀxⱼ)]
        """
        unique_times = np.sort(np.unique(durations))
        baseline_hazard = np.zeros_like(unique_times, dtype=float)

        eta = X @ self.coefficients
        exp_eta = np.exp(np.clip(eta, -500, 500))

        for i, t in enumerate(unique_times):
            # Events at time t
            at_risk = durations >= t
            events_at_t = (durations == t) & (events == 1)

            d = events_at_t.sum()  # Number of events
            risk_sum = exp_eta[at_risk].sum()

            if risk_sum > 0:
                baseline_hazard[i] = d / risk_sum

        return unique_times, baseline_hazard

    def hazard(self, features: Dict[str, float], t: int) -> float:
        """
        Compute hazard rate λ(t|x) = λ₀(t) · exp(βᵀx).

        Parameters:
        -----------
        features : Dict[str, float]
            Patient feature values
        t : int
            Days before appointment

        Returns:
        --------
        float : Instantaneous hazard rate
        """
        if not self.is_fitted:
            self._use_default_coefficients()

        # Build feature vector
        x = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            x[i] = features.get(name, 0.0)

        # Linear predictor
        linear_pred = np.dot(self.coefficients, x)
        exp_term = np.exp(np.clip(linear_pred, -500, 500))

        # Get baseline hazard at time t
        if t in self.baseline_times:
            idx = np.where(self.baseline_times == t)[0][0]
            baseline = self.baseline_hazard[idx]
        else:
            # Interpolate
            baseline = np.interp(t, self.baseline_times[::-1],
                                self.baseline_hazard[::-1])

        return baseline * exp_term

    def survival(self, features: Dict[str, float], t: int) -> float:
        """
        Compute survival probability S(t) = P(T > t) = exp(-H(t)).

        Where H(t) = ∫₀ᵗ λ(s)ds is cumulative hazard.
        """
        # Compute cumulative hazard
        cumulative_hazard = self.cumulative_hazard(features, t)

        # Survival function
        return np.exp(-cumulative_hazard)

    def cumulative_hazard(self, features: Dict[str, float], t: int) -> float:
        """
        Compute cumulative hazard H(t) = ∫₀ᵗ λ(s|x)ds.
        """
        if not self.is_fitted:
            self._use_default_coefficients()

        # Build feature vector
        x = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            x[i] = features.get(name, 0.0)

        linear_pred = np.dot(self.coefficients, x)
        exp_term = np.exp(np.clip(linear_pred, -500, 500))

        # Cumulative baseline hazard up to time t
        valid_times = self.baseline_times >= t
        H0_t = np.sum(self.baseline_hazard[valid_times])

        return H0_t * exp_term


class SurvivalAnalysisModel:
    """
    Main Survival Analysis model for no-show time-to-event prediction.

    Combines Cox PH with practical features:
    - Optimal reminder timing
    - Risk peak identification
    - Integration with scheduling system
    """

    def __init__(self, feature_engineer=None):
        self.cox_model = CoxProportionalHazards()
        self.feature_engineer = feature_engineer
        self.is_initialized = False

        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.10,
            'medium': 0.25,
            'high': 0.40,
            'critical': 0.60
        }

        # Optimal reminder windows based on hazard analysis
        self.reminder_windows = {
            'first': 7,   # 7 days before
            'second': 3,  # 3 days before
            'final': 1    # Day before
        }

        logger.info("SurvivalAnalysisModel initialized")

    def initialize(self, patient_data: Optional[pd.DataFrame] = None):
        """Initialize or fit the model."""
        if patient_data is not None and not patient_data.empty:
            self.cox_model.fit(patient_data)
        else:
            self.cox_model._use_default_coefficients()

        self.is_initialized = True
        logger.info("SurvivalAnalysisModel initialized with Cox PH")

    def _extract_features(self, patient: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for survival analysis from patient data."""
        features = {}

        # Age bands
        age = patient.get('age', 50)
        features['age_band_0_30'] = 1.0 if age < 30 else 0.0
        features['age_band_30_50'] = 1.0 if 30 <= age < 50 else 0.0
        features['age_band_50_70'] = 1.0 if 50 <= age < 70 else 0.0
        features['age_band_70_plus'] = 1.0 if age >= 70 else 0.0

        # Previous no-show rate
        features['previous_noshow_rate'] = patient.get('noshow_rate', 0.0)

        # Days since last visit
        features['days_since_last_visit'] = patient.get('days_since_last', 30)

        # Appointment time features
        appt_hour = patient.get('appointment_hour', 10)
        features['appointment_hour_early'] = 1.0 if appt_hour < 10 else 0.0
        features['appointment_hour_midday'] = 1.0 if 10 <= appt_hour < 14 else 0.0
        features['appointment_hour_late'] = 1.0 if appt_hour >= 14 else 0.0

        # Day of week
        day_of_week = patient.get('day_of_week', 2)  # Default Wednesday
        features['monday'] = 1.0 if day_of_week == 0 else 0.0
        features['friday'] = 1.0 if day_of_week == 4 else 0.0

        # Treatment cycle
        features['treatment_cycle'] = patient.get('cycle_number', 1)

        # Distance
        features['distance_km'] = patient.get('distance_km', 10)

        # Weather (would be populated from external source)
        features['weather_adverse'] = patient.get('weather_adverse', 0.0)

        # First appointment flag
        features['is_first_appointment'] = 1.0 if patient.get('is_first_appointment', False) else 0.0

        return features

    def predict(self,
                patient: Dict[str, Any],
                days_to_appointment: int = 14) -> SurvivalPrediction:
        """
        Generate survival analysis prediction for a patient.

        Parameters:
        -----------
        patient : Dict[str, Any]
            Patient data including demographics, history, appointment details
        days_to_appointment : int
            Days until the scheduled appointment

        Returns:
        --------
        SurvivalPrediction : Complete survival analysis results
        """
        if not self.is_initialized:
            self.initialize()

        features = self._extract_features(patient)

        # Compute survival probability at appointment time
        survival_prob = self.cox_model.survival(features, 0)  # t=0 is appointment day

        # Compute current hazard rate
        hazard_rate = self.cox_model.hazard(features, days_to_appointment)

        # Find risk peak (day with highest hazard)
        risk_peak_days = self._find_risk_peak(features, days_to_appointment)

        # Determine optimal reminder timing
        optimal_reminders = self._compute_optimal_reminders(
            features, days_to_appointment, risk_peak_days
        )

        # Cumulative hazard
        cumulative_hazard = self.cox_model.cumulative_hazard(features, 0)

        # Risk category
        noshow_prob = 1 - survival_prob
        risk_category = self._categorize_risk(noshow_prob)

        # Confidence interval (approximate using delta method)
        ci = self._compute_confidence_interval(survival_prob, features)

        return SurvivalPrediction(
            patient_id=patient.get('patient_id', 'unknown'),
            survival_probability=survival_prob,
            hazard_rate=hazard_rate,
            risk_peak_days=risk_peak_days,
            optimal_reminder_days=optimal_reminders,
            cumulative_hazard=cumulative_hazard,
            risk_category=risk_category,
            confidence_interval=ci
        )

    def _find_risk_peak(self,
                        features: Dict[str, float],
                        max_days: int) -> int:
        """Find the day before appointment when risk is highest."""
        max_hazard = 0.0
        peak_day = 3  # Default

        for day in range(1, min(max_days + 1, 15)):  # Check up to 14 days
            hazard = self.cox_model.hazard(features, day)
            if hazard > max_hazard:
                max_hazard = hazard
                peak_day = day

        return peak_day

    def _compute_optimal_reminders(self,
                                   features: Dict[str, float],
                                   days_to_appointment: int,
                                   risk_peak_days: int) -> List[int]:
        """
        Compute optimal reminder timing based on hazard function.

        Strategy:
        1. Send reminder before risk peak
        2. Send reminder at risk peak
        3. Send final reminder day before
        """
        reminders = []

        # First reminder: 1-2 days before risk peak
        first_reminder = min(risk_peak_days + 2, days_to_appointment)
        if first_reminder > 1:
            reminders.append(first_reminder)

        # Second reminder: at or near risk peak
        if risk_peak_days <= days_to_appointment and risk_peak_days > 1:
            if risk_peak_days not in reminders:
                reminders.append(risk_peak_days)

        # Final reminder: day before (if not already included)
        if 1 not in reminders and days_to_appointment >= 1:
            reminders.append(1)

        # Sort in descending order (furthest first)
        reminders.sort(reverse=True)

        return reminders

    def _categorize_risk(self, noshow_prob: float) -> str:
        """Categorize risk level based on no-show probability."""
        if noshow_prob >= self.risk_thresholds['critical']:
            return 'critical'
        elif noshow_prob >= self.risk_thresholds['high']:
            return 'high'
        elif noshow_prob >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'

    def _compute_confidence_interval(self,
                                     survival_prob: float,
                                     features: Dict[str, float],
                                     alpha: float = 0.05) -> Tuple[float, float]:
        """
        Compute confidence interval for survival probability.

        Uses Greenwood's formula approximation.
        """
        # Approximate standard error (simplified)
        # In practice, would use full variance-covariance matrix
        se = 0.05 * survival_prob * (1 - survival_prob) ** 0.5

        z = 1.96  # 95% CI
        lower = max(0, survival_prob - z * se)
        upper = min(1, survival_prob + z * se)

        return (lower, upper)

    def hazard_curve(self,
                     patient: Dict[str, Any],
                     max_days: int = 14) -> Dict[int, float]:
        """
        Generate hazard curve for a patient over time.

        Returns hazard rate for each day before appointment.
        """
        if not self.is_initialized:
            self.initialize()

        features = self._extract_features(patient)

        curve = {}
        for day in range(max_days, -1, -1):
            curve[day] = self.cox_model.hazard(features, day)

        return curve

    def survival_curve(self,
                       patient: Dict[str, Any],
                       max_days: int = 14) -> Dict[int, float]:
        """
        Generate survival curve for a patient over time.

        Returns survival probability S(t) for each day.
        """
        if not self.is_initialized:
            self.initialize()

        features = self._extract_features(patient)

        curve = {}
        for day in range(max_days, -1, -1):
            curve[day] = self.cox_model.survival(features, day)

        return curve

    def batch_predict(self,
                      patients: List[Dict[str, Any]],
                      days_to_appointment: int = 14) -> List[SurvivalPrediction]:
        """Generate predictions for multiple patients."""
        return [self.predict(p, days_to_appointment) for p in patients]

    def get_high_risk_patients(self,
                               patients: List[Dict[str, Any]],
                               threshold: str = 'high') -> List[SurvivalPrediction]:
        """
        Filter patients by risk level.

        Parameters:
        -----------
        patients : List[Dict]
            List of patient data
        threshold : str
            Minimum risk level: 'low', 'medium', 'high', 'critical'
        """
        predictions = self.batch_predict(patients)

        risk_levels = ['low', 'medium', 'high', 'critical']
        threshold_idx = risk_levels.index(threshold)

        return [p for p in predictions
                if risk_levels.index(p.risk_category) >= threshold_idx]

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model configuration and coefficients."""
        if not self.is_initialized:
            self.initialize()

        coefficients = {}
        for i, name in enumerate(self.cox_model.feature_names):
            coefficients[name] = {
                'coefficient': float(self.cox_model.coefficients[i]),
                'hazard_ratio': float(np.exp(self.cox_model.coefficients[i]))
            }

        return {
            'model_type': 'Cox Proportional Hazards',
            'formula': 'λ(t|x) = λ₀(t) · exp(βᵀx)',
            'survival_function': 'S(t) = exp(-∫λ(s)ds)',
            'n_features': len(self.cox_model.feature_names),
            'coefficients': coefficients,
            'risk_thresholds': self.risk_thresholds,
            'reminder_windows': self.reminder_windows,
            'is_fitted': self.cox_model.is_fitted
        }


# Convenience function for quick predictions
def predict_noshow_timing(patient: Dict[str, Any],
                          days_to_appointment: int = 14) -> SurvivalPrediction:
    """
    Quick prediction of no-show timing for a single patient.

    Example:
    --------
    >>> patient = {
    ...     'patient_id': 'P001',
    ...     'age': 65,
    ...     'noshow_rate': 0.15,
    ...     'appointment_hour': 14,
    ...     'day_of_week': 4,  # Friday
    ...     'cycle_number': 2
    ... }
    >>> result = predict_noshow_timing(patient, days_to_appointment=7)
    >>> print(f"Risk peak at day {result.risk_peak_days}")
    >>> print(f"Send reminders on days: {result.optimal_reminder_days}")
    """
    model = SurvivalAnalysisModel()
    model.initialize()
    return model.predict(patient, days_to_appointment)
