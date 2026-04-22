"""
Conformal Prediction for Distribution-Free Uncertainty Quantification (5.1)

For distribution-free prediction intervals with guaranteed coverage:

C_n(X_{n+1}) = {y : s(X_{n+1}, y) <= q_{1-alpha}}

Where:
- s is the non-conformity score
- q_{1-alpha} is the quantile of calibration scores

Guarantee: P(Y_{n+1} in C_n(X_{n+1})) >= 1 - alpha

This provides valid prediction intervals WITHOUT distributional assumptions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NonConformityScore(Enum):
    """Types of non-conformity scores."""
    ABSOLUTE = "absolute"           # |y - y_hat|
    SCALED = "scaled"               # |y - y_hat| / sigma_hat
    QUANTILE = "quantile"           # For quantile regression
    CQR = "cqr"                     # Conformalized Quantile Regression


@dataclass
class ConformalPrediction:
    """Result of conformal prediction."""
    patient_id: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    coverage_level: float           # 1 - alpha
    non_conformity_score: float     # Score for this prediction
    calibration_quantile: float     # q_{1-alpha}


@dataclass
class ConformalCalibrationResult:
    """Result of conformal calibration."""
    n_calibration: int
    coverage_level: float
    calibration_quantile: float
    mean_score: float
    median_score: float
    score_std: float
    empirical_coverage: float       # On held-out data if available


class ConformalPredictor:
    """
    Split Conformal Prediction for distribution-free prediction intervals.

    Provides prediction intervals with guaranteed marginal coverage:
    P(Y_{n+1} in C_n(X_{n+1})) >= 1 - alpha

    Steps:
    1. Split data into training and calibration sets
    2. Train model on training set
    3. Compute non-conformity scores on calibration set
    4. Use quantile of calibration scores for prediction intervals

    This is the simplest form of conformal prediction with exact coverage.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score_type: NonConformityScore = NonConformityScore.ABSOLUTE,
        calibration_fraction: float = 0.2
    ):
        """
        Initialize conformal predictor.

        Parameters:
        -----------
        alpha : float
            Miscoverage level (default 0.1 for 90% coverage)
        score_type : NonConformityScore
            Type of non-conformity score to use
        calibration_fraction : float
            Fraction of data to use for calibration
        """
        self.alpha = alpha
        self.coverage_level = 1 - alpha
        self.score_type = score_type
        self.calibration_fraction = calibration_fraction

        self.model = None
        self.calibration_scores: Optional[np.ndarray] = None
        self.calibration_quantile: Optional[float] = None
        self.is_calibrated = False

        # For scaled scores
        self.sigma_model = None

        logger.info(f"ConformalPredictor initialized (alpha={alpha}, coverage={self.coverage_level:.0%})")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any = None
    ) -> 'ConformalPredictor':
        """
        Fit the conformal predictor.

        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
        y : np.ndarray
            Targets (n_samples,)
        model : Any, optional
            Pre-trained model. If None, trains a default model.

        Returns:
        --------
        self
        """
        X = np.atleast_2d(X)
        y = np.array(y).ravel()
        n = len(y)

        # Split into training and calibration
        n_cal = int(n * self.calibration_fraction)
        n_train = n - n_cal

        # Random permutation for splitting
        np.random.seed(42)
        perm = np.random.permutation(n)
        train_idx = perm[:n_train]
        cal_idx = perm[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]

        # Train model if not provided
        if model is None:
            self.model = self._create_default_model()
            self.model.fit(X_train, y_train)
        else:
            self.model = model

        # Calibrate on calibration set
        self._calibrate(X_cal, y_cal)

        return self

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        model: Any = None
    ) -> ConformalCalibrationResult:
        """
        Calibrate the conformal predictor on a calibration set.

        This is used when you have a pre-trained model and want to
        calibrate it separately.

        Parameters:
        -----------
        X_cal : np.ndarray
            Calibration features
        y_cal : np.ndarray
            Calibration targets
        model : Any, optional
            Pre-trained model

        Returns:
        --------
        ConformalCalibrationResult
        """
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("Model must be provided or fit() must be called first")

        return self._calibrate(X_cal, y_cal)

    def _calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> ConformalCalibrationResult:
        """Internal calibration method."""
        X_cal = np.atleast_2d(X_cal)
        y_cal = np.array(y_cal).ravel()

        # Get predictions
        y_pred = self.model.predict(X_cal)

        # Compute non-conformity scores
        self.calibration_scores = self._compute_scores(y_cal, y_pred, X_cal)

        # Compute calibration quantile
        # For exact coverage, use (n_cal + 1)(1 - alpha) / n_cal quantile
        n_cal = len(y_cal)
        adjusted_quantile = min(1.0, (n_cal + 1) * self.coverage_level / n_cal)
        self.calibration_quantile = np.quantile(self.calibration_scores, adjusted_quantile)

        self.is_calibrated = True

        result = ConformalCalibrationResult(
            n_calibration=n_cal,
            coverage_level=self.coverage_level,
            calibration_quantile=self.calibration_quantile,
            mean_score=float(np.mean(self.calibration_scores)),
            median_score=float(np.median(self.calibration_scores)),
            score_std=float(np.std(self.calibration_scores)),
            empirical_coverage=self._compute_empirical_coverage(y_cal, y_pred)
        )

        logger.info(f"Conformal calibration complete: q_{self.coverage_level:.2f} = {self.calibration_quantile:.4f}")

        return result

    def _compute_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute non-conformity scores."""
        if self.score_type == NonConformityScore.ABSOLUTE:
            # Simple absolute residual
            return np.abs(y_true - y_pred)

        elif self.score_type == NonConformityScore.SCALED:
            # Scaled by predicted variance (for heteroscedastic models)
            if self.sigma_model is not None and X is not None:
                sigma = self.sigma_model.predict(X)
                sigma = np.maximum(sigma, 1e-6)  # Avoid division by zero
                return np.abs(y_true - y_pred) / sigma
            else:
                return np.abs(y_true - y_pred)

        elif self.score_type == NonConformityScore.QUANTILE:
            # For quantile regression models
            return np.abs(y_true - y_pred)

        else:
            # Default to absolute
            return np.abs(y_true - y_pred)

    def _compute_empirical_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Compute empirical coverage on calibration set."""
        if self.calibration_quantile is None:
            return 0.0

        lower = y_pred - self.calibration_quantile
        upper = y_pred + self.calibration_quantile
        covered = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(covered))

    def predict(
        self,
        X: np.ndarray,
        patient_ids: Optional[List[str]] = None,
        alpha_adaptive: Optional[Sequence[float]] = None,
    ) -> List[ConformalPrediction]:
        """
        Make predictions with conformal prediction intervals.

        Parameters:
        -----------
        X : np.ndarray
            Features for prediction
        patient_ids : List[str], optional
            Patient IDs for predictions
        alpha_adaptive : sequence of per-row α values, optional (Dissertation §2.2)
            If provided (length matches X), each row's prediction interval
            is computed at its own miscoverage level α_i using a quantile
            lookup on the stored calibration scores.  When None, falls back
            to the legacy single-α behaviour using `self.calibration_quantile`
            — existing callers are unaffected.

        Returns:
        --------
        List[ConformalPrediction]
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated first. Call fit() or calibrate().")

        X = np.atleast_2d(X)
        n = X.shape[0]

        if patient_ids is None:
            patient_ids = [f"P{i:04d}" for i in range(n)]

        # Get point predictions
        y_pred = self.model.predict(X)

        # Resolve per-row quantile
        #   * no adaptive α  -> reuse the single calibration_quantile
        #   * adaptive α     -> per-row quantile from the sorted scores
        if alpha_adaptive is None:
            qs = np.full(n, self.calibration_quantile, dtype=float)
            coverage_levels = np.full(n, self.coverage_level, dtype=float)
        else:
            alphas = np.asarray(list(alpha_adaptive), dtype=float)
            if alphas.shape[0] != n:
                raise ValueError(
                    f"alpha_adaptive must have length {n}, got {alphas.shape[0]}"
                )
            coverage_levels = 1.0 - alphas
            n_cal = max(1, len(self.calibration_scores))
            adjusted = np.clip(
                (n_cal + 1) * coverage_levels / n_cal, 0.0, 1.0
            )
            qs = np.quantile(self.calibration_scores, adjusted)

        predictions = []
        for i in range(n):
            q_i = float(qs[i])
            pred = ConformalPrediction(
                patient_id=patient_ids[i] if i < len(patient_ids) else f"P{i:04d}",
                point_estimate=float(y_pred[i]),
                lower_bound=float(y_pred[i] - q_i),
                upper_bound=float(y_pred[i] + q_i),
                interval_width=float(2 * q_i),
                coverage_level=float(coverage_levels[i]),
                non_conformity_score=q_i,  # Would need y_true for actual score
                calibration_quantile=q_i,
            )
            predictions.append(pred)

        return predictions

    def predict_single(
        self,
        x: np.ndarray,
        patient_id: str = "unknown",
        alpha_adaptive: Optional[float] = None,
    ) -> ConformalPrediction:
        """Predict for a single sample. `alpha_adaptive` is a scalar α override."""
        alpha_list = None if alpha_adaptive is None else [float(alpha_adaptive)]
        return self.predict(x.reshape(1, -1), [patient_id], alpha_adaptive=alpha_list)[0]

    def _create_default_model(self):
        """Create a default regression model."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        except ImportError:
            return _NumpyLinearModel()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of conformal predictor."""
        return {
            'is_calibrated': self.is_calibrated,
            'coverage_level': self.coverage_level,
            'alpha': self.alpha,
            'score_type': self.score_type.value,
            'calibration_quantile': self.calibration_quantile,
            'calibration_scores': {
                'mean': float(np.mean(self.calibration_scores)) if self.calibration_scores is not None else None,
                'std': float(np.std(self.calibration_scores)) if self.calibration_scores is not None else None,
                'n': len(self.calibration_scores) if self.calibration_scores is not None else 0
            },
            'guarantee': f'P(Y in C(X)) >= {self.coverage_level:.0%}'
        }


class ConformizedQuantileRegression:
    """
    Conformalized Quantile Regression (CQR) for adaptive prediction intervals.

    Unlike standard conformal prediction, CQR uses quantile regression to
    produce adaptive intervals that can be wider or narrower depending on
    the predicted uncertainty.

    For a point x:
    - Train quantile regressors for alpha/2 and 1-alpha/2 quantiles
    - Non-conformity score: max(q_lo - y, y - q_hi)
    - Prediction interval: [q_lo(x) - q, q_hi(x) + q]

    This produces intervals that adapt to heteroscedasticity while
    maintaining marginal coverage.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.coverage_level = 1 - alpha
        self.quantile_lo = alpha / 2
        self.quantile_hi = 1 - alpha / 2

        self.model_lo = None
        self.model_hi = None
        self.calibration_quantile = None
        self.is_calibrated = False

        logger.info(f"CQR initialized (alpha={alpha}, quantiles=[{self.quantile_lo}, {self.quantile_hi}])")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        calibration_fraction: float = 0.2
    ) -> 'ConformizedQuantileRegression':
        """Fit CQR model."""
        X = np.atleast_2d(X)
        y = np.array(y).ravel()
        n = len(y)

        # Split data
        n_cal = int(n * calibration_fraction)
        n_train = n - n_cal

        np.random.seed(42)
        perm = np.random.permutation(n)
        train_idx = perm[:n_train]
        cal_idx = perm[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]

        # Train quantile regression models
        self.model_lo = self._create_quantile_model(self.quantile_lo)
        self.model_hi = self._create_quantile_model(self.quantile_hi)

        self.model_lo.fit(X_train, y_train)
        self.model_hi.fit(X_train, y_train)

        # Calibrate
        self._calibrate(X_cal, y_cal)

        return self

    def _calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Calibrate CQR."""
        pred_lo = self.model_lo.predict(X_cal)
        pred_hi = self.model_hi.predict(X_cal)

        # CQR non-conformity score: max(q_lo - y, y - q_hi)
        scores = np.maximum(pred_lo - y_cal, y_cal - pred_hi)
        # Retain raw scores for per-patient adaptive α lookup (§2.2).
        self.calibration_scores = scores
        self.n_cal = len(y_cal)

        n_cal = len(y_cal)
        adjusted_quantile = min(1.0, (n_cal + 1) * self.coverage_level / n_cal)
        self.calibration_quantile = np.quantile(scores, adjusted_quantile)

        self.is_calibrated = True
        logger.info(f"CQR calibrated: q = {self.calibration_quantile:.4f}")

    def predict(
        self,
        X: np.ndarray,
        patient_ids: Optional[List[str]] = None,
        alpha_adaptive: Optional[Sequence[float]] = None,
    ) -> List[ConformalPrediction]:
        """
        Predict with adaptive intervals.  `alpha_adaptive` (Dissertation §2.2)
        supplies a per-row miscoverage level; when None, falls back to the
        single calibration quantile computed at fit time.
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated first")

        X = np.atleast_2d(X)
        n = X.shape[0]

        if patient_ids is None:
            patient_ids = [f"P{i:04d}" for i in range(n)]

        pred_lo = self.model_lo.predict(X)
        pred_hi = self.model_hi.predict(X)

        if alpha_adaptive is None:
            qs = np.full(n, float(self.calibration_quantile))
            coverage_levels = np.full(n, float(self.coverage_level))
        else:
            alphas = np.asarray(list(alpha_adaptive), dtype=float)
            if alphas.shape[0] != n:
                raise ValueError(
                    f"alpha_adaptive must have length {n}, got {alphas.shape[0]}"
                )
            coverage_levels = 1.0 - alphas
            n_cal = max(1, getattr(self, 'n_cal', len(self.calibration_scores)))
            adjusted = np.clip((n_cal + 1) * coverage_levels / n_cal, 0.0, 1.0)
            qs = np.quantile(self.calibration_scores, adjusted)

        predictions = []
        for i in range(n):
            q_i = float(qs[i])
            lower = pred_lo[i] - q_i
            upper = pred_hi[i] + q_i
            point = (pred_lo[i] + pred_hi[i]) / 2

            predictions.append(ConformalPrediction(
                patient_id=patient_ids[i] if i < len(patient_ids) else f"P{i:04d}",
                point_estimate=float(point),
                lower_bound=float(lower),
                upper_bound=float(upper),
                interval_width=float(upper - lower),
                coverage_level=float(coverage_levels[i]),
                non_conformity_score=q_i,
                calibration_quantile=q_i,
            ))

        return predictions

    def _create_quantile_model(self, quantile: float):
        """Create a quantile regression model."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        except ImportError:
            return _NumpyQuantileModel(quantile)


class ConformalDurationPredictor:
    """
    Conformal prediction specifically for treatment duration prediction.

    Provides guaranteed coverage intervals for chemotherapy session durations.
    """

    def __init__(self, alpha: float = 0.1, use_cqr: bool = True):
        self.alpha = alpha
        self.use_cqr = use_cqr

        if use_cqr:
            self.predictor = ConformizedQuantileRegression(alpha=alpha)
        else:
            self.predictor = ConformalPredictor(alpha=alpha)

        self.is_fitted = False
        logger.info(f"ConformalDurationPredictor initialized (CQR={use_cqr})")

    def fit(
        self,
        patients: List[Dict[str, Any]],
        durations: np.ndarray
    ) -> 'ConformalDurationPredictor':
        """
        Fit the duration predictor.

        Parameters:
        -----------
        patients : List[Dict]
            Patient data dictionaries
        durations : np.ndarray
            Actual treatment durations in minutes
        """
        X = self._extract_features(patients)
        y = np.array(durations)

        self.predictor.fit(X, y)
        self.is_fitted = True

        logger.info(f"ConformalDurationPredictor fitted on {len(patients)} samples")
        return self

    def predict(
        self,
        patient: Dict[str, Any],
        alpha_adaptive: Optional[float] = None,
    ) -> ConformalPrediction:
        """
        Predict duration with guaranteed coverage interval.

        Parameters
        ----------
        patient : dict
            Patient feature dictionary.
        alpha_adaptive : float, optional
            Risk-adaptive α override (Dissertation §2.2).  When provided,
            the inner predictor uses a per-patient quantile so higher-risk
            patients receive wider intervals.  When None, falls back to
            the class-level `self.alpha`.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        x = self._extract_features([patient])
        patient_id = patient.get('patient_id', patient.get('Patient_ID', 'unknown'))

        alpha_list = [float(alpha_adaptive)] if alpha_adaptive is not None else None
        predictions = self.predictor.predict(
            x, [patient_id], alpha_adaptive=alpha_list,
        )
        return predictions[0]

    def _extract_features(self, patients: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for duration prediction."""
        features = []
        for p in patients:
            feat = [
                p.get('expected_duration', p.get('Expected_Duration', 120)),
                p.get('cycle_number', p.get('Cycle_Number', 1)),
                p.get('age', p.get('Age', 55)),
                p.get('complexity_factor', 0.5),
                p.get('weight', p.get('Weight', 70)),
            ]
            features.append(feat)
        return np.array(features)

    def get_summary(self) -> Dict[str, Any]:
        """Get model summary."""
        return {
            'method': 'Conformalized Quantile Regression' if self.use_cqr else 'Split Conformal',
            'is_fitted': self.is_fitted,
            'coverage_guarantee': f'>= {(1 - self.alpha) * 100:.0f}%',
            **self.predictor.get_summary()
        }


class ConformalNoShowPredictor:
    """
    Conformal prediction for no-show probability prediction.

    For classification, we use different non-conformity scores
    to produce prediction sets with guaranteed coverage.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.coverage_level = 1 - alpha
        self.model = None
        self.calibration_threshold = None
        self.is_calibrated = False

        logger.info(f"ConformalNoShowPredictor initialized (alpha={alpha})")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        calibration_fraction: float = 0.2
    ) -> 'ConformalNoShowPredictor':
        """Fit with split conformal calibration."""
        X = np.atleast_2d(X)
        y = np.array(y).ravel()
        n = len(y)

        # Split data
        n_cal = int(n * calibration_fraction)
        np.random.seed(42)
        perm = np.random.permutation(n)
        train_idx = perm[:-n_cal]
        cal_idx = perm[-n_cal:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]

        # Train classifier
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # Calibrate
        self._calibrate(X_cal, y_cal)

        return self

    def _calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Calibrate using probability scores."""
        # Non-conformity score for classification:
        # s(x, y) = 1 - P(Y=y|X=x)
        probs = self.model.predict_proba(X_cal)

        # For each sample, get the probability of the true class
        n_cal = len(y_cal)
        scores = np.array([1 - probs[i, int(y_cal[i])] for i in range(n_cal)])
        # Store raw scores so risk-adaptive α can recompute the threshold
        # per-patient at predict time (§2.2).  Back-compat default path
        # still uses the single self.calibration_threshold.
        self.calibration_scores = scores
        self.n_cal = n_cal

        # Calibration quantile at the baseline α
        adjusted_quantile = min(1.0, (n_cal + 1) * self.coverage_level / n_cal)
        self.calibration_threshold = np.quantile(scores, adjusted_quantile)

        self.is_calibrated = True
        logger.info(f"ConformalNoShow calibrated: threshold = {self.calibration_threshold:.4f}")

    def predict(
        self,
        X: np.ndarray,
        patient_ids: Optional[List[str]] = None,
        alpha_adaptive: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict with conformal prediction sets.

        Returns prediction sets that contain the true class with
        probability >= 1 - alpha.  When `alpha_adaptive` is provided
        (Dissertation §2.2) the per-row α is plugged into the
        calibration quantile so high-risk / high-occupancy patients
        receive wider sets (more often "don't know") while routine
        patients get sharper decisions.
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated first")

        X = np.atleast_2d(X)
        n = X.shape[0]

        if patient_ids is None:
            patient_ids = [f"P{i:04d}" for i in range(n)]

        probs = self.model.predict_proba(X)

        # Per-row threshold
        if alpha_adaptive is None:
            thresholds = np.full(n, float(self.calibration_threshold))
            coverage_levels = np.full(n, float(self.coverage_level))
        else:
            alphas = np.asarray(list(alpha_adaptive), dtype=float)
            if alphas.shape[0] != n:
                raise ValueError(
                    f"alpha_adaptive must have length {n}, got {alphas.shape[0]}"
                )
            coverage_levels = 1.0 - alphas
            n_cal = max(1, getattr(self, 'n_cal', len(self.calibration_scores)))
            adjusted = np.clip((n_cal + 1) * coverage_levels / n_cal, 0.0, 1.0)
            thresholds = np.quantile(self.calibration_scores, adjusted)

        predictions = []
        for i in range(n):
            threshold_i = float(thresholds[i])
            prediction_set = []
            if probs[i, 0] >= 1 - threshold_i:
                prediction_set.append(0)  # No show = False
            if probs[i, 1] >= 1 - threshold_i:
                prediction_set.append(1)  # No show = True

            predictions.append({
                'patient_id': patient_ids[i] if i < len(patient_ids) else f"P{i:04d}",
                'noshow_probability': float(probs[i, 1]),
                'prediction_set': prediction_set,
                'set_size': len(prediction_set),
                'coverage_level': float(coverage_levels[i]),
                'confident': len(prediction_set) == 1,
                'predicted_noshow': prediction_set[0] == 1 if len(prediction_set) == 1 else None,
                'alpha_used': float(1.0 - coverage_levels[i]),
            })

        return predictions

    def _create_model(self):
        """Create classification model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
        except ImportError:
            return _NumpyLogisticModel()


class _NumpyLinearModel:
    """Numpy fallback for linear regression."""
    def __init__(self):
        self.coef = None

    def fit(self, X, y):
        X = np.column_stack([np.ones(len(X)), X])
        self.coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        return X @ self.coef


class _NumpyQuantileModel:
    """Numpy fallback for quantile regression."""
    def __init__(self, quantile):
        self.quantile = quantile
        self.coef = None

    def fit(self, X, y):
        # Approximate with linear regression
        X = np.column_stack([np.ones(len(X)), X])
        self.coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        return X @ self.coef


class _NumpyLogisticModel:
    """Numpy fallback for logistic regression."""
    def __init__(self):
        self.coef = None

    def fit(self, X, y):
        X = np.column_stack([np.ones(len(X)), X])
        self.coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        logits = X @ self.coef
        return (logits > 0.5).astype(int)

    def predict_proba(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        logits = X @ self.coef
        probs = 1 / (1 + np.exp(-logits))
        probs = np.clip(probs, 0.01, 0.99)
        return np.column_stack([1 - probs, probs])


# Convenience functions
def create_conformal_duration_predictor(
    alpha: float = 0.1,
    use_cqr: bool = True
) -> ConformalDurationPredictor:
    """Create a conformal duration predictor."""
    return ConformalDurationPredictor(alpha=alpha, use_cqr=use_cqr)


def create_conformal_noshow_predictor(
    alpha: float = 0.1
) -> ConformalNoShowPredictor:
    """Create a conformal no-show predictor."""
    return ConformalNoShowPredictor(alpha=alpha)
