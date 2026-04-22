"""
Duration Prediction Model
=========================

Predicts treatment duration using regression models.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle

# ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODEL_SAVE_DIR,
    get_logger
)
from .feature_engineering import FeatureEngineer

logger = get_logger(__name__)


@dataclass
class DurationPrediction:
    """Result of duration prediction"""
    patient_id: str
    predicted_duration: int  # minutes
    confidence_interval: Tuple[int, int]  # (lower, upper)
    expected_variance: str  # low, medium, high
    adjustment_factors: List[Tuple[str, int]]  # factor, minutes


class DurationModel:
    """
    Predicts treatment duration.

    Uses ensemble regression to predict actual treatment duration
    based on protocol, patient history, and external factors.
    """

    # Standard treatment durations by type (minutes)
    STANDARD_DURATIONS = {
        'simple': 30,
        'standard': 60,
        'complex': 120,
        'very_complex': 180
    }

    # Protocol duration adjustments (C1 durations from REGIMENS table)
    # These match the generate_sample_data.py REGIMENS definitions
    # Includes both REG codes AND regimen names for flexible lookup
    PROTOCOL_DURATIONS = {
        # === SACT PROTOCOL CODES ===
        'FOLFOX': 180,  # FOLFOX
        'FOLFIRI': 180,  # FOLFIRI
        'RCHOP': 360,  # R-CHOP
        'DOCE': 150,  # Docetaxel
        'PACW': 180,  # Paclitaxel Weekly
        'CARBPAC': 240,  # Carboplatin/Paclitaxel
        'PEMBRO': 60,   # Pembrolizumab
        'NIVO': 60,   # Nivolumab
        'TRAS': 120,  # Trastuzumab
        'FECT': 120,  # FEC-D
        'GEM': 60,   # Gemcitabine
        'CAPOX': 180,  # CAPOX
        'RITUX': 300,  # Rituximab Maintenance
        'CISE': 300,  # Cisplatin/Etoposide
        'IPNIVO': 120,  # Ipilimumab/Nivolumab
        'ZOLE': 30,   # Zoledronic Acid
        'AC': 90,   # AC (Doxorubicin/Cyclophosphamide)
        'BEVA': 90,   # Bevacizumab Maintenance
        'PEME': 120,  # Pemetrexed/Carboplatin
        'VINO': 30,   # Vinorelbine
        # === REGIMEN NAMES (for backwards compatibility) ===
        'FOLFOX': 180,
        'FOLFIRI': 180,
        'R-CHOP': 360,
        'Docetaxel': 150,
        'Paclitaxel': 180,
        'Paclitaxel Weekly': 180,
        'Carboplatin': 240,
        'Carboplatin/Paclitaxel': 240,
        'Pembrolizumab': 60,
        'Nivolumab': 60,
        'Trastuzumab': 120,
        'Herceptin': 120,  # Trastuzumab alias
        'FEC': 120,
        'FEC-D': 120,
        'Gemcitabine': 60,
        'CAPOX': 180,
        'Rituximab': 300,
        'Rituximab Maintenance': 300,
        'Cisplatin/Etoposide': 300,
        'Ipilimumab/Nivolumab': 120,
        'Zoledronic Acid': 30,
        'AC': 90,
        'AC (Doxorubicin/Cyclophosphamide)': 90,
        'Bevacizumab': 90,
        'Bevacizumab Maintenance': 90,
        'Pemetrexed/Carboplatin': 120,
        'Vinorelbine': 30,
        'default': 90
    }

    # Protocol-specific historical standard deviations (from data analysis)
    # Used for confidence interval calculation: σ = sqrt(σ_model² + σ_protocol²)
    PROTOCOL_VARIANCE = {
        # SACT codes with historical std dev (minutes)
        'FOLFOX': 23,   # FOLFOX
        'FOLFIRI': 28,   # FOLFIRI
        'RCHOP': 63,   # R-CHOP (high variance due to long duration)
        'DOCE': 33,   # Docetaxel
        'PACW': 29,   # Paclitaxel Weekly
        'CARBPAC': 32,   # Carboplatin/Paclitaxel
        'PEMBRO': 13,   # Pembrolizumab (short, low variance)
        'NIVO': 13,   # Nivolumab
        'TRAS': 24,   # Trastuzumab
        'FECT': 17,   # FEC-D
        'GEM': 13,   # Gemcitabine
        'CAPOX': 30,   # CAPOX
        'RITUX': 48,   # Rituximab Maintenance
        'CISE': 45,   # Cisplatin/Etoposide
        'IPNIVO': 24,   # Ipilimumab/Nivolumab
        'ZOLE': 6,    # Zoledronic Acid (very short)
        'AC': 14,   # AC
        'BEVA': 24,   # Bevacizumab Maintenance
        'PEME': 17,   # Pemetrexed/Carboplatin
        'VINO': 6,    # Vinorelbine (very short)
        'default': 25   # Mean protocol variance
    }

    def __init__(self, model_path: str = None):
        """
        Initialize duration model.

        Args:
            model_path: Path to saved model (optional)
        """
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.models = {}
        self.feature_names = []
        self.is_trained = False

        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            self._initialize_models()

        logger.info("Duration prediction model initialized")

    def _initialize_models(self):
        """Initialize ensemble models"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, model training disabled")
            return

        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }

        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )

    def train(self, X: pd.DataFrame, y: pd.Series,
              test_size: float = 0.2) -> Dict:
        """
        Train the ensemble model.

        Args:
            X: Feature DataFrame
            y: Target Series (duration in minutes)
            test_size: Fraction for test split

        Returns:
            Dict with training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for training")

        logger.info(f"Training duration model on {len(X)} samples")

        # Store feature names
        self.feature_names = list(X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        metrics = {}

        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)

            metrics[name] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }

            logger.info(f"{name} MAE: {metrics[name]['mae']:.1f} minutes")

        # Cross-validation for primary model
        cv_scores = cross_val_score(
            self.models['random_forest'],
            X_train_scaled, y_train,
            cv=5, scoring='neg_mean_absolute_error'
        )
        metrics['cv_mean_mae'] = -cv_scores.mean()
        metrics['cv_std_mae'] = cv_scores.std()

        self.is_trained = True
        logger.info(f"Training complete. CV MAE: {metrics['cv_mean_mae']:.1f} ± {metrics['cv_std_mae']:.1f}")

        return metrics

    def predict(self, patient_data: Dict, appointment_data: Dict,
                external_data: Dict = None) -> DurationPrediction:
        """
        Predict treatment duration for a single patient.

        Args:
            patient_data: Patient demographics and history
            appointment_data: Appointment details including protocol
            external_data: External factors

        Returns:
            DurationPrediction object
        """
        # Create features
        features = self.feature_engineer.create_patient_features(
            patient_data, appointment_data, external_data
        )

        # Convert to DataFrame
        X = pd.DataFrame([features.features])

        # Ensure correct feature order
        if self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            for col in missing:
                X[col] = 0.0
            X = X[self.feature_names]

        # Get protocol for variance lookup
        protocol = appointment_data.get('protocol') or \
                  appointment_data.get('Regimen_Code') or 'default'

        # Get prediction
        if self.is_trained and SKLEARN_AVAILABLE:
            duration, std = self._ensemble_predict(X, protocol)
        else:
            # Fallback to rule-based prediction
            duration, std = self._rule_based_predict(
                features.features, appointment_data
            )

        # Calculate confidence interval
        lower = max(15, int(duration - 1.96 * std))
        upper = int(duration + 1.96 * std)

        # Determine variance level (calibrated to actual data variance)
        # Data shows: short treatments ~15 std, medium ~25-40, long ~50-70
        if std < 20:
            variance = 'low'
        elif std < 45:
            variance = 'medium'
        else:
            variance = 'high'

        # Get adjustment factors
        adjustments = self._get_adjustment_factors(
            features.features, appointment_data, external_data
        )

        return DurationPrediction(
            patient_id=features.patient_id,
            predicted_duration=int(duration),
            confidence_interval=(lower, upper),
            expected_variance=variance,
            adjustment_factors=adjustments
        )

    def _ensemble_predict(self, X: pd.DataFrame, protocol: str = 'default') -> Tuple[float, float]:
        """Get ensemble prediction with uncertainty using combined variance formula"""
        X_scaled = self.scaler.transform(X)

        predictions = []
        weights = {'random_forest': 0.4, 'gradient_boosting': 0.35, 'xgboost': 0.25}

        for name, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            weight = weights.get(name, 0.33)
            predictions.append((pred, weight))

        # Weighted average
        total_weight = sum(w for _, w in predictions)
        mean_duration = sum(p * w for p, w in predictions) / total_weight

        # Combined variance formula: σ = sqrt(σ_model² + σ_protocol²)
        # σ_model: from ensemble disagreement
        variance = sum(w * (p - mean_duration)**2 for p, w in predictions) / total_weight
        sigma_model = np.sqrt(variance)

        # σ_protocol: from historical protocol-specific variance
        sigma_protocol = self.PROTOCOL_VARIANCE.get(protocol, self.PROTOCOL_VARIANCE['default'])

        # Combined standard deviation (independent variance sources)
        std = np.sqrt(sigma_model**2 + sigma_protocol**2)

        # Ensure minimum reasonable std (at least 10% of duration)
        std = max(std, mean_duration * 0.10)

        return mean_duration, std

    def _rule_based_predict(self, features: Dict,
                           appointment_data: Dict) -> Tuple[float, float]:
        """Fallback rule-based prediction"""
        # PRIORITY 1: Use cycle-adjusted planned duration if provided
        # This is the most accurate as it's already cycle-specific
        base_duration = appointment_data.get('expected_duration') or \
                       appointment_data.get('planned_duration') or \
                       appointment_data.get('Planned_Duration')

        cycle_adjusted = base_duration is not None

        if not base_duration:
            # PRIORITY 2: Look up from protocol (C1 durations)
            protocol = appointment_data.get('protocol', 'default')
            base_duration = self.PROTOCOL_DURATIONS.get(
                protocol, self.PROTOCOL_DURATIONS['default']
            )

            # PRIORITY 3: From treatment type
            if base_duration == 90:  # default
                treatment_type = appointment_data.get('treatment_type', 'standard')
                base_duration = self.STANDARD_DURATIONS.get(
                    treatment_type.lower(), 60
                )

        adjustments = 0

        # First cycle adjustment ONLY if duration wasn't already cycle-adjusted
        # Data shows C1 is ~37 min longer than C2+ on average
        if not cycle_adjusted and features.get('is_first_cycle', 0):
            adjustments += 37

        # Complex treatments
        complexity = features.get('treatment_complexity', 1.5)
        if complexity > 2:
            adjustments += 20

        # Patient history adjustments
        if features.get('total_appointments', 0) < 3:
            adjustments += 15  # New patients need more time

        # External factors can add delays
        weather_severity = features.get('weather_severity', 0)
        if weather_severity > 0.3:
            adjustments += 10

        traffic_delay = features.get('expected_delay_min', 0)
        if traffic_delay > 15:
            adjustments += 5

        # Time of day
        hour = features.get('hour', 10)
        if hour < 9 or hour > 16:  # Early or late
            adjustments += 5

        total_duration = base_duration + adjustments

        # Combined variance formula: σ = sqrt(σ_protocol² + σ_residual²)
        # Get protocol for variance lookup
        protocol = appointment_data.get('protocol') or \
                  appointment_data.get('Regimen_Code') or 'default'

        # σ_protocol: from historical protocol-specific variance
        sigma_protocol = self.PROTOCOL_VARIANCE.get(protocol, self.PROTOCOL_VARIANCE['default'])

        # σ_residual: residual variance after protocol adjustment (~21 min from data)
        # Plus complexity adjustment
        sigma_residual = 21.0
        complexity_factor = 1.0 + (complexity - 1) * 0.25  # Complexity increases variance
        sigma_residual *= complexity_factor

        # Combined standard deviation
        std = np.sqrt(sigma_protocol**2 + sigma_residual**2)

        # Ensure minimum reasonable std (at least 10% of duration)
        std = max(std, total_duration * 0.10)

        return total_duration, std

    def _get_adjustment_factors(self, features: Dict,
                                appointment_data: Dict,
                                external_data: Dict = None) -> List[Tuple[str, int]]:
        """Get factors contributing to duration adjustment"""
        adjustments = []
        external_data = external_data or {}

        # Protocol-based
        protocol = appointment_data.get('protocol', '')
        if protocol in self.PROTOCOL_DURATIONS:
            adjustments.append(('Protocol type', 0))  # Base, not adjustment

        # First cycle (when not using cycle-adjusted duration)
        if features.get('is_first_cycle', 0):
            adjustments.append(('First cycle', 37))

        # Complexity
        complexity = features.get('treatment_complexity', 1.5)
        if complexity > 2:
            adjustments.append(('Treatment complexity', 20))
        elif complexity > 1.5:
            adjustments.append(('Treatment complexity', 10))

        # New patient
        if features.get('total_appointments', 0) < 3:
            adjustments.append(('New patient setup', 15))

        # Weather
        weather = external_data.get('weather', {})
        if weather.get('severity', 0) > 0.3:
            adjustments.append(('Weather conditions', 10))

        # Traffic
        traffic = external_data.get('traffic', {})
        if traffic.get('delay_minutes', 0) > 15:
            adjustments.append(('Traffic delays', 5))

        # Sort by impact
        adjustments.sort(key=lambda x: abs(x[1]), reverse=True)
        return adjustments[:5]

    def predict_batch(self, patients: List[Dict], appointments: List[Dict],
                     external_data: Dict = None) -> List[DurationPrediction]:
        """
        Predict duration for multiple patients.

        Args:
            patients: List of patient data dicts
            appointments: List of appointment data dicts
            external_data: Shared external data

        Returns:
            List of DurationPrediction objects
        """
        return [
            self.predict(patient, appointment, external_data)
            for patient, appointment in zip(patients, appointments)
        ]

    def save(self, path: str = None):
        """Save model to disk"""
        path = path or MODEL_SAVE_DIR / "duration_model.pkl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }

        # T2.3: SHA-256-verified pickle write — sidecar lets load() reject
        # tampered weights at boot.
        from safe_loader import safe_save
        safe_save(model_data, path)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        # T2.3: verify SHA-256 sidecar before unpickling.
        from safe_loader import safe_load
        model_data = safe_load(path)

        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or 'random_forest' not in self.models:
            return {}

        importance = self.models['random_forest'].feature_importances_
        return {
            name: round(imp, 4)
            for name, imp in sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            )
        }

    def get_protocol_durations(self) -> Dict[str, int]:
        """Get standard protocol durations"""
        return self.PROTOCOL_DURATIONS.copy()


# Example usage
if __name__ == "__main__":
    model = DurationModel()

    # Sample prediction
    patient = {
        'patient_id': 'P001',
        'postcode': 'CF14 4XW',
        'total_appointments': 2
    }

    appointment = {
        'appointment_time': datetime.now(),
        'site_code': 'WC',
        'priority': 'P2',
        'protocol': 'R-CHOP',
        'treatment_type': 'complex',
        'expected_duration': 180
    }

    external = {
        'weather': {'severity': 0.2},
        'traffic': {'delay_minutes': 20}
    }

    result = model.predict(patient, appointment, external)

    print("Duration Prediction:")
    print("=" * 50)
    print(f"Patient ID: {result.patient_id}")
    print(f"Predicted Duration: {result.predicted_duration} minutes")
    print(f"Confidence Interval: {result.confidence_interval[0]}-{result.confidence_interval[1]} min")
    print(f"Expected Variance: {result.expected_variance}")
    print("\nAdjustment Factors:")
    for factor, minutes in result.adjustment_factors:
        if minutes != 0:
            print(f"  - {factor}: +{minutes} min")

    print("\nStandard Protocol Durations:")
    for protocol, duration in model.get_protocol_durations().items():
        if protocol != 'default':
            print(f"  {protocol}: {duration} min")
