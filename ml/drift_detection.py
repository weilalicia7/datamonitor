"""
Data and Model Drift Detection Module

Monitors for distribution shifts in incoming data that indicate
models need retraining.

Methods:
    - Population Stability Index (PSI) for feature drift
    - Kolmogorov-Smirnov test for distribution changes
    - CUSUM for gradual drift detection
    - Performance decay monitoring (accuracy/AUC over time)

Reference:
    PSI: Siddiqi (2006). Credit Risk Scorecards.
    CUSUM: Page (1954). Continuous Inspection Schemes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class DriftReport:
    """Result of drift detection analysis."""
    feature_name: str
    drift_score: float
    drift_detected: bool
    method: str
    threshold: float
    severity: str  # 'none', 'minor', 'moderate', 'severe'
    details: str
    recommendation: str


@dataclass
class DriftSummary:
    """Summary of drift detection across all features."""
    timestamp: str
    total_features_checked: int
    features_drifted: int
    max_drift_score: float
    overall_severity: str
    recommended_action: str  # 'none', 'recalibrate', 'retrain'
    reports: List[DriftReport]


class DriftDetector:
    """
    Detects data and model drift using multiple statistical methods.

    Usage:
        detector = DriftDetector()
        summary = detector.full_drift_check(reference_data, new_data)
        if summary.recommended_action == 'retrain':
            trigger_model_retrain()
    """

    # PSI thresholds
    PSI_NO_DRIFT = 0.1
    PSI_MODERATE = 0.25

    # Performance decay thresholds
    AUC_DECAY_THRESHOLD = 0.03  # 3% AUC drop triggers alert
    MAE_INCREASE_THRESHOLD = 0.15  # 15% MAE increase triggers alert

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reference_distributions: Dict[str, np.ndarray] = {}
        self.performance_history: List[Dict] = []

    def set_reference(self, data: Dict[str, np.ndarray]):
        """Set reference distributions from training data."""
        self.reference_distributions = {}
        for feature_name, values in data.items():
            values = np.array(values, dtype=float)
            values = values[~np.isnan(values)]
            if len(values) > 0:
                self.reference_distributions[feature_name] = values
        logger.info(f"Reference set with {len(self.reference_distributions)} features")

    def compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Compute Population Stability Index.

        PSI = sum( (P_i - Q_i) * ln(P_i / Q_i) )

        Where P_i = proportion in bin i for current data
              Q_i = proportion in bin i for reference data

        Interpretation:
            PSI < 0.1:  No significant shift
            0.1-0.25:   Moderate shift (investigate)
            > 0.25:     Significant shift (retrain)
        """
        # Create bins from reference distribution
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]

        if len(ref_clean) < 10 or len(cur_clean) < 10:
            return 0.0

        # Use percentile-based bins for robustness
        breakpoints = np.percentile(ref_clean, np.linspace(0, 100, self.n_bins + 1))
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 3:
            return 0.0

        # Count proportions in each bin
        ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
        cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

        # Convert to proportions with Laplace smoothing
        ref_props = (ref_counts + 1) / (len(ref_clean) + len(breakpoints) - 1)
        cur_props = (cur_counts + 1) / (len(cur_clean) + len(breakpoints) - 1)

        # Compute PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(psi)

    def ks_test(self, reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov two-sample test for distribution change.

        Returns (statistic, p_value).
        Reject H0 (same distribution) if p_value < 0.05.
        """
        if not SCIPY_AVAILABLE:
            return 0.0, 1.0

        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]

        if len(ref_clean) < 5 or len(cur_clean) < 5:
            return 0.0, 1.0

        stat, p_value = scipy_stats.ks_2samp(ref_clean, cur_clean)
        return float(stat), float(p_value)

    def cusum(self, values: np.ndarray, target: float = None,
              threshold: float = 5.0) -> Tuple[bool, int]:
        """
        CUSUM (Cumulative Sum) control chart for gradual drift detection.

        Detects when a process mean shifts from its target value.

        Returns (drift_detected, change_point_index).
        """
        if len(values) < 10:
            return False, -1

        if target is None:
            target = np.mean(values[:len(values) // 3])  # First third as baseline

        std = np.std(values[:len(values) // 3])
        if std < 1e-10:
            return False, -1

        # Standardized residuals
        residuals = (values - target) / std

        # CUSUM positive and negative
        s_pos = np.zeros(len(residuals))
        s_neg = np.zeros(len(residuals))

        for i in range(1, len(residuals)):
            s_pos[i] = max(0, s_pos[i - 1] + residuals[i] - 0.5)
            s_neg[i] = max(0, s_neg[i - 1] - residuals[i] - 0.5)

            if s_pos[i] > threshold or s_neg[i] > threshold:
                return True, i

        return False, -1

    def check_feature_drift(self, feature_name: str,
                            current_values: np.ndarray) -> DriftReport:
        """Check drift for a single feature."""
        if feature_name not in self.reference_distributions:
            return DriftReport(
                feature_name=feature_name,
                drift_score=0.0,
                drift_detected=False,
                method='N/A',
                threshold=0.0,
                severity='none',
                details='No reference distribution available',
                recommendation='Set reference data first'
            )

        reference = self.reference_distributions[feature_name]
        current = np.array(current_values, dtype=float)

        # Compute PSI
        psi = self.compute_psi(reference, current)

        # Compute KS test
        ks_stat, ks_p = self.ks_test(reference, current)

        # Determine severity
        if psi > self.PSI_MODERATE:
            severity = 'severe'
            drift_detected = True
            recommendation = 'Full model retrain recommended'
        elif psi > self.PSI_NO_DRIFT:
            severity = 'moderate'
            drift_detected = True
            recommendation = 'Feature weight recalibration recommended'
        elif ks_p < 0.01:
            severity = 'minor'
            drift_detected = True
            recommendation = 'Monitor closely, consider baseline update'
        else:
            severity = 'none'
            drift_detected = False
            recommendation = 'No action needed'

        details = (
            f"PSI={psi:.4f} (threshold={self.PSI_NO_DRIFT}), "
            f"KS statistic={ks_stat:.4f}, KS p-value={ks_p:.4f}"
        )

        return DriftReport(
            feature_name=feature_name,
            drift_score=psi,
            drift_detected=drift_detected,
            method='PSI + KS test',
            threshold=self.PSI_NO_DRIFT,
            severity=severity,
            details=details,
            recommendation=recommendation
        )

    def check_performance_drift(self, metric_name: str, current_value: float,
                                baseline_value: float) -> DriftReport:
        """Check if model performance has degraded."""
        if metric_name.lower() in ('auc', 'auc_roc', 'accuracy', 'f1'):
            # Higher is better
            decay = baseline_value - current_value
            threshold = self.AUC_DECAY_THRESHOLD
            drifted = decay > threshold
        else:
            # Lower is better (MAE, RMSE)
            if baseline_value > 0:
                increase = (current_value - baseline_value) / baseline_value
            else:
                increase = 0
            decay = increase
            threshold = self.MAE_INCREASE_THRESHOLD
            drifted = increase > threshold

        if drifted:
            severity = 'severe' if abs(decay) > threshold * 2 else 'moderate'
            recommendation = 'Model retrain recommended - performance degraded'
        else:
            severity = 'none'
            recommendation = 'Performance within acceptable range'

        return DriftReport(
            feature_name=f'performance_{metric_name}',
            drift_score=abs(decay),
            drift_detected=drifted,
            method='Performance monitoring',
            threshold=threshold,
            severity=severity,
            details=f"Baseline={baseline_value:.4f}, Current={current_value:.4f}, Change={decay:.4f}",
            recommendation=recommendation
        )

    def full_drift_check(self, new_data: Dict[str, np.ndarray]) -> DriftSummary:
        """
        Run full drift check across all features.

        Args:
            new_data: Dict mapping feature names to current value arrays.

        Returns:
            DriftSummary with overall assessment and per-feature reports.
        """
        reports = []

        for feature_name, values in new_data.items():
            report = self.check_feature_drift(feature_name, values)
            reports.append(report)

        # Summarize
        drifted = [r for r in reports if r.drift_detected]
        max_score = max((r.drift_score for r in reports), default=0.0)

        if any(r.severity == 'severe' for r in reports):
            overall = 'severe'
            action = 'retrain'
        elif any(r.severity == 'moderate' for r in reports):
            overall = 'moderate'
            action = 'recalibrate'
        elif any(r.severity == 'minor' for r in reports):
            overall = 'minor'
            action = 'none'
        else:
            overall = 'none'
            action = 'none'

        summary = DriftSummary(
            timestamp=datetime.now().isoformat() if 'datetime' in dir() else '',
            total_features_checked=len(reports),
            features_drifted=len(drifted),
            max_drift_score=max_score,
            overall_severity=overall,
            recommended_action=action,
            reports=reports
        )

        logger.info(
            f"Drift check: {len(drifted)}/{len(reports)} features drifted, "
            f"severity={overall}, action={action}"
        )

        return summary

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get drift detector status as dict."""
        return {
            'reference_features': list(self.reference_distributions.keys()),
            'n_features': len(self.reference_distributions),
            'n_bins': self.n_bins,
            'psi_thresholds': {
                'no_drift': self.PSI_NO_DRIFT,
                'moderate': self.PSI_MODERATE
            },
            'methods': ['PSI', 'Kolmogorov-Smirnov', 'CUSUM', 'Performance Monitoring']
        }


# Import datetime at module level for DriftSummary
from datetime import datetime
