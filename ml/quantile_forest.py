"""
Quantile Regression Forests for Distribution-Free Confidence Intervals (3.2)

Instead of assuming normality for CIs, use distribution-free approach:

F(y|X=x) = Σᵢ wᵢ(x) · 𝟙(yᵢ≤y)

Where wᵢ(x) are tree-based weights.

Advantages:
- No normality assumption
- Automatically handles heteroscedasticity
- Captures asymmetric distributions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try importing sklearn
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using numpy fallback")


@dataclass
class QuantilePrediction:
    """Result of quantile regression prediction."""
    patient_id: str
    point_estimate: float       # Median prediction
    lower_bound: float          # Lower quantile (e.g., 2.5%)
    upper_bound: float          # Upper quantile (e.g., 97.5%)
    quantiles: Dict[float, float]  # Full quantile distribution
    prediction_interval_width: float
    heteroscedasticity_score: float  # Measure of variance heterogeneity
    distribution_shape: str     # 'symmetric', 'left_skewed', 'right_skewed'


class QuantileRegressionForest:
    """
    Quantile Regression Forest for distribution-free prediction intervals.

    Uses Random Forest structure but stores all training targets in leaves
    to compute arbitrary quantiles at prediction time.

    F(y|X=x) = Σᵢ wᵢ(x) · 𝟙(yᵢ≤y)

    Where wᵢ(x) = proportion of trees where sample i lands in same leaf as x.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = 10,
                 min_samples_leaf: int = 5,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.trees: List[Any] = []
        self.leaf_values: List[Dict[int, np.ndarray]] = []
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.is_fitted = False

        # Default quantiles for CIs
        self.default_quantiles = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]

        logger.info(f"QuantileRegressionForest initialized (n_trees={n_estimators})")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressionForest':
        """
        Fit the Quantile Regression Forest.

        Unlike standard RF, we store all training values in each leaf
        to enable quantile computation at prediction time.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        n_samples = len(y)

        if not SKLEARN_AVAILABLE:
            # Fallback: simple storage of training data
            self.is_fitted = True
            logger.info(f"QRF fitted with numpy fallback ({n_samples} samples)")
            return self

        np.random.seed(self.random_state)

        self.trees = []
        self.leaf_values = []

        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = self.X_train[indices]
            y_boot = self.y_train[indices]

            # Fit decision tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            # Store mapping from leaf ID to training values
            leaf_ids = tree.apply(self.X_train)
            leaf_value_map = {}
            for leaf_id in np.unique(leaf_ids):
                mask = leaf_ids == leaf_id
                leaf_value_map[leaf_id] = self.y_train[mask]

            self.leaf_values.append(leaf_value_map)

        self.is_fitted = True
        logger.info(f"QRF fitted with {n_samples} samples, {self.n_estimators} trees")
        return self

    def predict_quantiles(self,
                          X: np.ndarray,
                          quantiles: Optional[List[float]] = None) -> np.ndarray:
        """
        Predict specified quantiles for each sample.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        quantiles : List[float]
            Quantiles to predict (default: [0.025, 0.5, 0.975])

        Returns:
        --------
        np.ndarray : Shape (n_samples, n_quantiles)
        """
        if quantiles is None:
            quantiles = [0.025, 0.5, 0.975]

        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        n_quantiles = len(quantiles)

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if not SKLEARN_AVAILABLE or len(self.trees) == 0:
            # Fallback: use global quantiles
            return self._predict_fallback(X, quantiles)

        predictions = np.zeros((n_samples, n_quantiles))

        for i in range(n_samples):
            x = X[i:i+1]
            # Collect values from all trees
            all_values = []
            weights = []

            for tree, leaf_map in zip(self.trees, self.leaf_values):
                leaf_id = tree.apply(x)[0]
                if leaf_id in leaf_map:
                    values = leaf_map[leaf_id]
                    all_values.extend(values)
                    # Weight by 1/n_values (uniform within leaf)
                    weights.extend([1.0 / len(values)] * len(values))

            if all_values:
                all_values = np.array(all_values)
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize

                # Compute weighted quantiles
                for j, q in enumerate(quantiles):
                    predictions[i, j] = self._weighted_quantile(all_values, weights, q)
            else:
                # Fallback to training median
                predictions[i, :] = np.quantile(self.y_train, quantiles)

        return predictions

    def _weighted_quantile(self,
                           values: np.ndarray,
                           weights: np.ndarray,
                           quantile: float) -> float:
        """
        Compute weighted quantile.

        F(y|X=x) = Σᵢ wᵢ(x) · 𝟙(yᵢ≤y)
        """
        # Sort by values
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Cumulative weights
        cumsum = np.cumsum(sorted_weights)

        # Find quantile
        idx = np.searchsorted(cumsum, quantile)
        idx = min(idx, len(sorted_values) - 1)

        return sorted_values[idx]

    def _predict_fallback(self,
                          X: np.ndarray,
                          quantiles: List[float]) -> np.ndarray:
        """Fallback prediction using simple quantiles."""
        n_samples = X.shape[0]
        n_quantiles = len(quantiles)

        # Use training data quantiles with some feature-based adjustment
        base_quantiles = np.quantile(self.y_train, quantiles)

        predictions = np.zeros((n_samples, n_quantiles))
        for i in range(n_samples):
            # Simple adjustment based on features
            feature_factor = 1.0 + 0.1 * (np.mean(X[i]) - 0.5)
            predictions[i] = base_quantiles * feature_factor

        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict median (point estimate)."""
        return self.predict_quantiles(X, [0.5])[:, 0]

    def predict_interval(self,
                         X: np.ndarray,
                         confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence interval.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        confidence : float
            Confidence level (default: 0.95 for 95% CI)

        Returns:
        --------
        Tuple of (median, lower_bound, upper_bound)
        """
        alpha = 1 - confidence
        quantiles = [alpha / 2, 0.5, 1 - alpha / 2]

        preds = self.predict_quantiles(X, quantiles)
        return preds[:, 1], preds[:, 0], preds[:, 2]


class QuantileForestDurationModel:
    """
    Duration prediction model using Quantile Regression Forests.

    Provides distribution-free confidence intervals for treatment durations.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 10):
        self.qrf = QuantileRegressionForest(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        self.feature_names: List[str] = []
        self.is_fitted = False

        # Default feature means for standardization
        self._feature_means = {}
        self._feature_stds = {}

        logger.info("QuantileForestDurationModel initialized")

    def _extract_features(self, patient: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from patient data."""
        features = []

        # Base duration (normalized)
        base_duration = patient.get('expected_duration', patient.get('base_duration', 120))
        features.append(base_duration / 300.0)

        # Cycle number
        cycle = patient.get('cycle_number', 1)
        features.append(min(cycle / 10.0, 1.0))
        features.append(1.0 if cycle == 1 else 0.0)  # First cycle flag

        # Complexity
        complexity = patient.get('complexity_factor', patient.get('complexity', 0.5))
        features.append(complexity)

        # Age (normalized)
        age = patient.get('age', patient.get('Age', 55))
        features.append(age / 100.0)

        # Comorbidities
        comorbidities = patient.get('comorbidity_count', patient.get('comorbidities', 0))
        features.append(min(comorbidities / 5.0, 1.0))

        # Previous duration variance
        prev_variance = patient.get('duration_variance', patient.get('historical_variance', 0.2))
        features.append(prev_variance)

        # Protocol type encoding
        protocol = patient.get('protocol', patient.get('regimen_code', 'FOLFOX'))
        # Simple hash encoding
        protocol_hash = hash(protocol) % 10 / 10.0
        features.append(protocol_hash)

        # No-show rate (correlates with duration uncertainty)
        noshow_rate = patient.get('noshow_rate', patient.get('previous_noshow_rate', 0.1))
        features.append(noshow_rate)

        # Distance (may affect patient condition/stress)
        distance = patient.get('distance_km', patient.get('Distance', 10))
        features.append(min(distance / 50.0, 1.0))

        return np.array(features, dtype=np.float32)

    def fit(self,
            patients: List[Dict[str, Any]],
            durations: np.ndarray) -> 'QuantileForestDurationModel':
        """
        Fit the model on patient data and actual durations.

        Parameters:
        -----------
        patients : List[Dict]
            Patient feature dictionaries
        durations : np.ndarray
            Actual treatment durations in minutes
        """
        if len(patients) == 0:
            logger.warning("Empty training data")
            self.is_fitted = True
            return self

        X = np.array([self._extract_features(p) for p in patients])
        y = np.array(durations)

        self.qrf.fit(X, y)
        self.is_fitted = True

        logger.info(f"QuantileForestDurationModel fitted with {len(patients)} samples")
        return self

    def predict(self, patient: Dict[str, Any]) -> QuantilePrediction:
        """
        Predict duration with distribution-free confidence intervals.

        Returns full quantile distribution for asymmetric CIs.
        """
        # Use default prediction if not fitted or QRF has no training data
        if not self.is_fitted or self.qrf.y_train is None:
            return self._predict_default(patient)

        features = self._extract_features(patient)
        X = features.reshape(1, -1)

        # Get quantile predictions
        quantiles = self.qrf.default_quantiles
        preds = self.qrf.predict_quantiles(X, quantiles)[0]

        # Build quantile dictionary
        quantile_dict = {q: float(p) for q, p in zip(quantiles, preds)}

        # Key values
        lower = quantile_dict[0.025]
        median = quantile_dict[0.50]
        upper = quantile_dict[0.975]

        # Interval width
        interval_width = upper - lower

        # Heteroscedasticity score (variance relative to mean)
        hetero_score = interval_width / median if median > 0 else 0.0

        # Distribution shape
        lower_dist = median - lower
        upper_dist = upper - median
        if upper_dist > 1.2 * lower_dist:
            shape = 'right_skewed'
        elif lower_dist > 1.2 * upper_dist:
            shape = 'left_skewed'
        else:
            shape = 'symmetric'

        return QuantilePrediction(
            patient_id=patient.get('patient_id', patient.get('Patient_ID', 'unknown')),
            point_estimate=median,
            lower_bound=lower,
            upper_bound=upper,
            quantiles=quantile_dict,
            prediction_interval_width=interval_width,
            heteroscedasticity_score=hetero_score,
            distribution_shape=shape
        )

    def _predict_default(self, patient: Dict[str, Any]) -> QuantilePrediction:
        """Default prediction when not fitted."""
        base_duration = patient.get('expected_duration', patient.get('base_duration', 120))
        cycle = patient.get('cycle_number', 1)
        complexity = patient.get('complexity_factor', 0.5)

        # Adjust base duration
        median = base_duration
        if cycle == 1:
            median += 45  # First cycle longer
        median += complexity * 30

        # Asymmetric intervals (durations tend to be right-skewed)
        variance = patient.get('duration_variance', 0.2)
        lower = median * (1 - variance * 0.8)
        upper = median * (1 + variance * 1.2)  # Larger upper bound

        # Build quantiles
        quantile_dict = {
            0.025: lower,
            0.10: median * (1 - variance * 0.5),
            0.25: median * (1 - variance * 0.25),
            0.50: median,
            0.75: median * (1 + variance * 0.3),
            0.90: median * (1 + variance * 0.7),
            0.975: upper
        }

        return QuantilePrediction(
            patient_id=patient.get('patient_id', patient.get('Patient_ID', 'unknown')),
            point_estimate=median,
            lower_bound=lower,
            upper_bound=upper,
            quantiles=quantile_dict,
            prediction_interval_width=upper - lower,
            heteroscedasticity_score=variance,
            distribution_shape='right_skewed'
        )

    def batch_predict(self,
                      patients: List[Dict[str, Any]]) -> List[QuantilePrediction]:
        """Predict for multiple patients."""
        return [self.predict(p) for p in patients]

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model configuration summary."""
        return {
            'model_type': 'Quantile Regression Forest',
            'formula': 'F(y|X=x) = Σᵢ wᵢ(x) · 𝟙(yᵢ≤y)',
            'n_estimators': self.qrf.n_estimators,
            'max_depth': self.qrf.max_depth,
            'min_samples_leaf': self.qrf.min_samples_leaf,
            'default_quantiles': self.qrf.default_quantiles,
            'is_fitted': self.is_fitted,
            'sklearn_available': SKLEARN_AVAILABLE,
            'advantages': [
                'No normality assumption required',
                'Automatically handles heteroscedasticity',
                'Captures asymmetric distributions',
                'Distribution-free confidence intervals'
            ]
        }


class QuantileForestNoShowModel:
    """
    No-show probability prediction with uncertainty using Quantile Forests.

    Provides distribution over no-show probability estimates.
    """

    def __init__(self, n_estimators: int = 100):
        self.qrf = QuantileRegressionForest(
            n_estimators=n_estimators,
            max_depth=8
        )
        self.is_fitted = False

        logger.info("QuantileForestNoShowModel initialized")

    def _extract_features(self, patient: Dict[str, Any]) -> np.ndarray:
        """Extract features for no-show prediction."""
        features = []

        # Previous no-show rate
        features.append(patient.get('noshow_rate', patient.get('previous_noshow_rate', 0.1)))

        # Age
        age = patient.get('age', patient.get('Age', 55))
        features.append(age / 100.0)

        # Distance
        distance = patient.get('distance_km', patient.get('Distance', 10))
        features.append(min(distance / 50.0, 1.0))

        # Cycle number
        cycle = patient.get('cycle_number', 1)
        features.append(min(cycle / 10.0, 1.0))

        # First appointment
        features.append(1.0 if patient.get('is_first_appointment', cycle == 1) else 0.0)

        # Day of week
        dow = patient.get('day_of_week', 2)
        features.append(1.0 if dow in [0, 4] else 0.0)  # Monday/Friday risk

        # Appointment hour
        hour = patient.get('appointment_hour', 10)
        features.append(1.0 if hour >= 14 else 0.0)  # Late afternoon risk

        return np.array(features, dtype=np.float32)

    def fit(self,
            patients: List[Dict[str, Any]],
            noshow_labels: np.ndarray) -> 'QuantileForestNoShowModel':
        """Fit on patient data and no-show outcomes."""
        if len(patients) == 0:
            self.is_fitted = True
            return self

        X = np.array([self._extract_features(p) for p in patients])

        # For binary classification, we fit on probabilities derived from local averages
        # This gives us a distribution over probability estimates
        self.qrf.fit(X, noshow_labels.astype(float))
        self.is_fitted = True

        logger.info(f"QuantileForestNoShowModel fitted with {len(patients)} samples")
        return self

    def predict(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict no-show probability with uncertainty bounds.

        Returns distribution over probability estimates.
        """
        features = self._extract_features(patient)
        X = features.reshape(1, -1)

        if self.is_fitted and SKLEARN_AVAILABLE:
            quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
            preds = self.qrf.predict_quantiles(X, quantiles)[0]
            preds = np.clip(preds, 0, 1)  # Ensure valid probabilities
        else:
            # Fallback based on features
            base_prob = features[0]  # Previous no-show rate
            adjustment = features[4] * 0.2 + features[5] * 0.1 + features[6] * 0.1
            prob = np.clip(base_prob + adjustment, 0.01, 0.99)
            preds = [prob * 0.5, prob * 0.8, prob, prob * 1.2, prob * 1.5]
            preds = np.clip(preds, 0, 1)

        return {
            'patient_id': patient.get('patient_id', patient.get('Patient_ID', 'unknown')),
            'probability': float(preds[2]),  # Median
            'lower_bound': float(preds[0]),
            'upper_bound': float(preds[4]),
            'interquartile_range': [float(preds[1]), float(preds[3])],
            'uncertainty': float(preds[4] - preds[0])
        }


# Convenience function
def predict_duration_quantiles(patient: Dict[str, Any]) -> QuantilePrediction:
    """
    Quick quantile-based duration prediction.

    Example:
    --------
    >>> patient = {
    ...     'patient_id': 'P001',
    ...     'expected_duration': 180,
    ...     'cycle_number': 2,
    ...     'complexity_factor': 0.6
    ... }
    >>> result = predict_duration_quantiles(patient)
    >>> print(f"Duration: {result.point_estimate:.0f} min")
    >>> print(f"95% CI: [{result.lower_bound:.0f}, {result.upper_bound:.0f}]")
    """
    model = QuantileForestDurationModel()
    model.is_fitted = True
    return model.predict(patient)
