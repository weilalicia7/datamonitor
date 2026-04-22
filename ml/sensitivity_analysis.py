"""
Sensitivity Analysis for SACT Scheduling ML Models

Computes local and global sensitivity indices to understand
how each input feature affects model predictions.

Local Sensitivity (per-sample):
    S_i = dy_hat / dx_i  (numerical partial derivative)

Global Importance (across population):
    I_j = (1/n) * sum_i |S_ij|  (mean absolute sensitivity)

References:
    Saltelli et al. (2008). Global Sensitivity Analysis.
    Sobol' (1993). Sensitivity estimates for nonlinear models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LocalSensitivity:
    """Local sensitivity result for a single patient."""
    patient_id: str
    model_type: str  # 'noshow' or 'duration'
    base_prediction: float
    sensitivities: Dict[str, float]  # feature_name -> S_i
    top_positive: List[Tuple[str, float]]  # features that increase prediction
    top_negative: List[Tuple[str, float]]  # features that decrease prediction
    perturbation_size: float


@dataclass
class GlobalImportance:
    """Global importance across all samples."""
    model_type: str
    n_samples: int
    importance: Dict[str, float]  # feature_name -> I_j (mean |S_ij|)
    importance_std: Dict[str, float]  # standard deviation of |S_ij|
    importance_rank: List[Tuple[str, float]]  # sorted by importance
    feature_categories: Dict[str, List[Tuple[str, float]]]  # grouped by category


class SensitivityAnalyzer:
    """
    Computes local and global sensitivity indices for scheduling ML models.

    Uses finite-difference numerical differentiation:
        S_i = [f(x + h*e_i) - f(x - h*e_i)] / (2h)

    where h = perturbation_size * max(|x_i|, 1) for numerical stability.
    """

    # Feature categories for grouped analysis
    FEATURE_CATEGORIES = {
        'Patient History': [
            'prev_noshow_rate', 'prev_noshow_count', 'total_appointments',
            'is_new_patient', 'prev_cancel_rate', 'prev_late_rate',
            'days_since_last_visit', 'treatment_cycle_number', 'is_first_cycle'
        ],
        'Demographics': [
            'age', 'age_band_under40', 'age_band_40_60', 'age_band_60_75',
            'age_band_over75'
        ],
        'Temporal': [
            'hour', 'day_of_week', 'is_weekend', 'month',
            'days_until_appointment', 'booking_lead_days',
            'slot_early_morning', 'slot_mid_morning', 'slot_late_morning',
            'is_mon', 'is_tue', 'is_wed', 'is_thu', 'is_fri',
            'is_winter', 'is_spring', 'is_summer', 'is_autumn'
        ],
        'Geographic': [
            'distance_km', 'travel_time_min', 'is_local',
            'is_medium_distance', 'is_long_distance', 'is_very_long_distance',
            'travel_under_15min', 'travel_15_30min', 'travel_30_60min',
            'travel_over_60min', 'is_cardiff', 'is_valleys'
        ],
        'Treatment': [
            'priority_level', 'expected_duration_min', 'is_short_treatment',
            'is_long_treatment', 'is_very_long_treatment',
            'is_immunotherapy', 'is_chemo_combo', 'cycle_number',
            'long_infusion', 'is_main_site'
        ],
        'External': [
            'weather_severity', 'precipitation_prob', 'is_bad_weather',
            'traffic_severity', 'expected_delay_min', 'is_rush_hour',
            'event_severity', 'combined_external_severity'
        ],
    }

    def __init__(self, perturbation: float = 0.01):
        """
        Args:
            perturbation: Relative perturbation size for numerical gradient.
                         0.01 = 1% perturbation (default).
        """
        self.perturbation = perturbation

    def local_sensitivity(
        self,
        model,
        feature_vector: np.ndarray,
        feature_names: List[str],
        model_type: str = 'noshow',
        patient_id: str = 'unknown',
        features_subset: List[str] = None
    ) -> LocalSensitivity:
        """
        Compute local sensitivity S_i = dy_hat/dx_i for each feature.

        Uses central finite differences for numerical stability:
            S_i = [f(x + h*e_i) - f(x - h*e_i)] / (2h)

        Args:
            model: Trained sklearn-compatible model with predict/predict_proba
            feature_vector: 1D numpy array of feature values
            feature_names: List of feature names matching vector
            model_type: 'noshow' (probability) or 'duration' (minutes)
            patient_id: Patient identifier for results
            features_subset: Optional subset of features to analyze

        Returns:
            LocalSensitivity object
        """
        x = feature_vector.copy().astype(float)
        n_features = len(x)

        # Get base prediction
        base_pred = self._get_prediction(model, x.reshape(1, -1), model_type)

        sensitivities = {}
        analyze_indices = range(n_features)

        if features_subset:
            analyze_indices = [
                i for i, name in enumerate(feature_names)
                if name in features_subset
            ]

        for i in analyze_indices:
            fname = feature_names[i] if i < len(feature_names) else f'feature_{i}'

            # Adaptive step size: h = perturbation * max(|x_i|, 1)
            h = self.perturbation * max(abs(x[i]), 1.0)

            # Central difference: [f(x+h) - f(x-h)] / 2h
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h

            pred_plus = self._get_prediction(model, x_plus.reshape(1, -1), model_type)
            pred_minus = self._get_prediction(model, x_minus.reshape(1, -1), model_type)

            s_i = (pred_plus - pred_minus) / (2 * h)
            sensitivities[fname] = float(s_i)

        # Sort by absolute value
        sorted_sens = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [(k, v) for k, v in sorted_sens if v > 0][:5]
        top_negative = [(k, v) for k, v in sorted_sens if v < 0][:5]

        return LocalSensitivity(
            patient_id=patient_id,
            model_type=model_type,
            base_prediction=float(base_pred),
            sensitivities=sensitivities,
            top_positive=top_positive,
            top_negative=top_negative,
            perturbation_size=self.perturbation
        )

    def global_importance(
        self,
        model,
        feature_matrix: np.ndarray,
        feature_names: List[str],
        model_type: str = 'noshow',
        max_samples: int = 200,
        features_subset: List[str] = None
    ) -> GlobalImportance:
        """
        Compute global importance I_j = (1/n) * sum|S_ij| across samples.

        Args:
            model: Trained sklearn-compatible model
            feature_matrix: 2D numpy array (n_samples, n_features)
            feature_names: List of feature names
            model_type: 'noshow' or 'duration'
            max_samples: Maximum samples to use (for speed)
            features_subset: Optional subset of features

        Returns:
            GlobalImportance object
        """
        X = feature_matrix.copy().astype(float)
        n_samples = min(len(X), max_samples)

        # Subsample if needed
        if len(X) > max_samples:
            indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
            X = X[indices]

        # Collect all local sensitivities
        all_sensitivities = []
        for i in range(n_samples):
            local = self.local_sensitivity(
                model, X[i], feature_names, model_type,
                patient_id=f'sample_{i}',
                features_subset=features_subset
            )
            all_sensitivities.append(local.sensitivities)

        # Compute global importance: I_j = (1/n) * sum|S_ij|
        importance = {}
        importance_std = {}

        feature_set = features_subset or list(all_sensitivities[0].keys())
        for fname in feature_set:
            values = [abs(s.get(fname, 0)) for s in all_sensitivities]
            importance[fname] = float(np.mean(values))
            importance_std[fname] = float(np.std(values))

        # Rank by importance
        importance_rank = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Group by category
        feature_categories = {}
        for cat_name, cat_features in self.FEATURE_CATEGORIES.items():
            cat_results = [
                (f, importance.get(f, 0))
                for f in cat_features
                if f in importance
            ]
            if cat_results:
                cat_results.sort(key=lambda x: x[1], reverse=True)
                feature_categories[cat_name] = cat_results

        return GlobalImportance(
            model_type=model_type,
            n_samples=n_samples,
            importance=importance,
            importance_std=importance_std,
            importance_rank=importance_rank,
            feature_categories=feature_categories
        )

    def elasticity(
        self,
        model,
        feature_vector: np.ndarray,
        feature_names: List[str],
        model_type: str = 'noshow'
    ) -> Dict[str, float]:
        """
        Compute elasticity: percentage change in output per percentage change in input.

        E_i = (dy/dx_i) * (x_i / y)

        Useful for comparing sensitivity across features with different scales.

        Returns:
            Dict of feature_name -> elasticity value
        """
        x = feature_vector.copy().astype(float)
        base_pred = self._get_prediction(model, x.reshape(1, -1), model_type)

        if abs(base_pred) < 1e-10:
            return {name: 0.0 for name in feature_names}

        local = self.local_sensitivity(
            model, x, feature_names, model_type
        )

        elasticities = {}
        for i, fname in enumerate(feature_names):
            s_i = local.sensitivities.get(fname, 0)
            # E_i = S_i * (x_i / y_hat)
            if abs(base_pred) > 1e-10 and abs(x[i]) > 1e-10:
                elasticities[fname] = float(s_i * x[i] / base_pred)
            else:
                elasticities[fname] = 0.0

        return elasticities

    def one_at_a_time(
        self,
        model,
        feature_vector: np.ndarray,
        feature_names: List[str],
        model_type: str = 'noshow',
        n_steps: int = 10
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        One-At-a-Time (OAT) sensitivity: vary each feature across its range.

        For each feature, sweep from x_i - 2*std to x_i + 2*std in n_steps
        and record [f(x_modified)] at each step.

        Returns:
            Dict of feature_name -> [(x_value, prediction), ...]
        """
        x = feature_vector.copy().astype(float)
        results = {}

        for i, fname in enumerate(feature_names):
            sweep_range = max(abs(x[i]) * 0.5, 1.0)
            steps = np.linspace(x[i] - sweep_range, x[i] + sweep_range, n_steps)

            curve = []
            for val in steps:
                x_mod = x.copy()
                x_mod[i] = val
                pred = self._get_prediction(model, x_mod.reshape(1, -1), model_type)
                curve.append((float(val), float(pred)))

            results[fname] = curve

        return results

    def sklearn_importance(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_type: str = 'noshow',
        n_repeats: int = 10
    ) -> Dict[str, Any]:
        """
        Use sklearn's built-in tools for feature importance.

        Combines:
        1. model.feature_importances_ (tree-based impurity importance)
        2. sklearn.inspection.permutation_importance (model-agnostic)

        Args:
            model: Trained sklearn model
            X: Feature matrix
            y: Target vector
            feature_names: Feature names
            model_type: 'noshow' or 'duration'
            n_repeats: Number of permutation repeats

        Returns:
            Dict with both importance methods
        """
        from sklearn.inspection import permutation_importance

        result = {}

        # 1. Tree-based feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            tree_imp = sorted(
                zip(feature_names, imp),
                key=lambda x: x[1], reverse=True
            )
            result['tree_importance'] = [
                {'feature': f, 'importance': round(float(v), 6)}
                for f, v in tree_imp[:20]
            ]

        # 2. Permutation importance (model-agnostic)
        try:
            scoring = 'roc_auc' if model_type == 'noshow' else 'neg_mean_squared_error'
            perm_result = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=42,
                scoring=scoring,
                n_jobs=-1
            )
            perm_imp = sorted(
                zip(feature_names, perm_result.importances_mean, perm_result.importances_std),
                key=lambda x: x[1], reverse=True
            )
            result['permutation_importance'] = [
                {'feature': f, 'importance': round(float(v), 6), 'std': round(float(s), 6)}
                for f, v, s in perm_imp[:20]
            ]
        except Exception as e:
            result['permutation_importance'] = [{'error': str(e)}]

        return result

    def _get_prediction(self, model, X: np.ndarray, model_type: str) -> float:
        """Get scalar prediction from model."""
        try:
            if model_type == 'noshow' and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                return float(proba[0, 1]) if proba.ndim > 1 else float(proba[0])
            else:
                pred = model.predict(X)
                return float(pred[0])
        except Exception:
            # Fallback: try predict
            try:
                pred = model.predict(X)
                return float(pred[0])
            except Exception:
                return 0.0

    def format_results(self, local: LocalSensitivity = None,
                       glob: GlobalImportance = None) -> Dict[str, Any]:
        """Format results for API response."""
        result = {
            'formulas': {
                'local_sensitivity': 'S_i = d(y_hat)/d(x_i) [central finite difference]',
                'global_importance': 'I_j = (1/n) * sum_i |S_ij| [mean absolute sensitivity]',
                'elasticity': 'E_i = S_i * (x_i / y_hat) [% change output per % change input]',
            }
        }

        if local:
            # Sort sensitivities by absolute value for display
            sorted_local = sorted(
                local.sensitivities.items(),
                key=lambda x: abs(x[1]), reverse=True
            )
            result['local'] = {
                'patient_id': local.patient_id,
                'model': local.model_type,
                'base_prediction': round(local.base_prediction, 4),
                'perturbation': local.perturbation_size,
                'sensitivities': {k: round(v, 6) for k, v in sorted_local[:20]},
                'top_increasing': [
                    {'feature': k, 'sensitivity': round(v, 6)}
                    for k, v in local.top_positive
                ],
                'top_decreasing': [
                    {'feature': k, 'sensitivity': round(v, 6)}
                    for k, v in local.top_negative
                ],
                'interpretation': (
                    f"For patient {local.patient_id}, "
                    f"{'no-show probability' if local.model_type == 'noshow' else 'predicted duration'} "
                    f"= {local.base_prediction:.4f}. "
                    f"Most influential: {local.top_positive[0][0] if local.top_positive else 'N/A'} "
                    f"(+{local.top_positive[0][1]:.4f})" if local.top_positive else ""
                ),
            }

        if glob:
            result['global'] = {
                'model': glob.model_type,
                'n_samples': glob.n_samples,
                'top_20_features': [
                    {
                        'feature': fname,
                        'importance': round(imp, 6),
                        'std': round(glob.importance_std.get(fname, 0), 6),
                    }
                    for fname, imp in glob.importance_rank[:20]
                ],
                'by_category': {
                    cat: [
                        {'feature': f, 'importance': round(v, 6)}
                        for f, v in features
                    ]
                    for cat, features in glob.feature_categories.items()
                },
                'interpretation': (
                    f"Analyzed {glob.n_samples} samples. "
                    f"Top feature: {glob.importance_rank[0][0]} "
                    f"(I={glob.importance_rank[0][1]:.4f}). "
                    f"{'Patient history features dominate.' if any(f[0] in self.FEATURE_CATEGORIES.get('Patient History',[]) for f in glob.importance_rank[:3]) else 'External/temporal features are influential.'}"
                ) if glob.importance_rank else "No results.",
            }

        return result
