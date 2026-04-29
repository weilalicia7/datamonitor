"""
Online Learning Module for SACT Scheduling

Implements real-time model updates as new data arrives without full retraining.

Methods:
    1. Stochastic Gradient Descent (SGD) for ensemble models
       θ_{t+1} = θ_t - η_t * ∇L(θ_t; x_t, y_t)

    2. Bayesian Online Updating for hierarchical model
       P(θ|D_{1:t}) ∝ P(D_t|θ) · P(θ|D_{1:t-1})

    3. Exponential Moving Average (EMA) for baseline rates
       μ_{t+1} = α · x_t + (1-α) · μ_t

References:
    Bottou, L. (2010). Large-Scale Machine Learning with SGD.
    Murphy, K. (2012). Machine Learning: A Probabilistic Perspective.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import SGDClassifier, SGDRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class OnlineUpdateResult:
    """Result of an online learning update."""
    model_name: str
    method: str  # 'sgd', 'bayesian', 'ema'
    samples_used: int
    timestamp: str
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    improvement: float
    details: str


class OnlineLearner:
    """
    Online learning manager for real-time model updates.

    Supports three update strategies:
    1. SGD: Incremental gradient updates for classification/regression
    2. Bayesian: Posterior updating for hierarchical models
    3. EMA: Exponential moving average for simple baselines

    Usage:
        learner = OnlineLearner()

        # When new appointment outcome arrives:
        result = learner.update_on_new_observation(
            patient_data={'age': 65, 'distance_km': 20, ...},
            outcome={'attended': True, 'duration': 145}
        )
    """

    def __init__(self, learning_rate: float = 0.01, ema_alpha: float = 0.1):
        """
        Args:
            learning_rate: η for SGD updates (default 0.01)
            ema_alpha: α for EMA updates (default 0.1)
        """
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha
        self.update_count = 0
        self.update_history: List[OnlineUpdateResult] = []

        # Feature-schema guard.  Set via ``set_feature_schema`` during
        # offline training.  When set, every SGD update is checked
        # against this expected dimensionality (and feature names, if
        # provided) before partial_fit runs.  Mismatches are logged
        # and skipped — they would otherwise corrupt the SGD weight
        # vector with the wrong feature ordering on Channel 2 data
        # whose schema differs from synthetic.
        self.expected_feature_dim: Optional[int] = None
        self.expected_feature_names: Optional[List[str]] = None
        self.skipped_due_to_schema_mismatch = 0

        # SGD models for online classification/regression
        if SKLEARN_AVAILABLE:
            self.sgd_classifier = SGDClassifier(
                loss='log_loss',
                learning_rate='invscaling',
                eta0=learning_rate,
                warm_start=True,
                random_state=42
            )
            self.sgd_regressor = SGDRegressor(
                loss='squared_error',
                learning_rate='invscaling',
                eta0=learning_rate,
                warm_start=True,
                random_state=42
            )
            self._sgd_fitted = False
        else:
            self.sgd_classifier = None
            self.sgd_regressor = None
            self._sgd_fitted = False

        # Bayesian posterior parameters
        self.posterior = {
            'noshow_rate': {'alpha': 2.0, 'beta': 10.0},  # Beta prior for no-show rate
            'duration_mean': {'mu': 120.0, 'sigma2': 400.0, 'n': 10},  # Normal prior
            'event_impact': {'mu': 0.05, 'sigma2': 0.01, 'n': 5},  # Event impact prior
        }

        # EMA baselines
        self.ema_baselines = {
            'noshow_rate': 0.12,
            'avg_duration': 120.0,
            'utilization': 0.75,
            'weather_impact': 0.05,
        }

        logger.info(f"OnlineLearner initialized (eta={learning_rate}, alpha={ema_alpha})")

    def set_feature_schema(self, dim: int,
                           names: Optional[List[str]] = None) -> None:
        """Lock in the feature dimensionality (and, optionally, the
        feature names) the offline training cohort used.  Subsequent
        SGD updates are validated against this schema; mismatches are
        skipped with a warning rather than silently corrupting the
        weight vector.

        Call this once after offline training, before promoting to
        Channel 2 (real Velindre data).  If never called, online
        updates run unguarded — the legacy behaviour.
        """
        self.expected_feature_dim = int(dim)
        self.expected_feature_names = list(names) if names is not None else None
        logger.info(
            "OnlineLearner feature schema locked: dim=%d names=%s",
            self.expected_feature_dim,
            f"({len(self.expected_feature_names)} provided)"
            if self.expected_feature_names else "not provided",
        )

    def _features_match_schema(self, features) -> bool:
        """Return True if the supplied feature vector matches the locked
        schema (or no schema is locked).  Updates a counter on
        mismatch so the operator can spot a pattern of skipped
        updates after Channel 2 promotion."""
        if self.expected_feature_dim is None:
            return True
        try:
            arr = np.asarray(features, dtype=float).reshape(1, -1)
        except Exception:
            self.skipped_due_to_schema_mismatch += 1
            logger.warning(
                "OnlineLearner: feature vector could not be cast to "
                "float — skipping SGD update."
            )
            return False
        if arr.shape[1] != self.expected_feature_dim:
            self.skipped_due_to_schema_mismatch += 1
            logger.warning(
                "OnlineLearner: feature dim mismatch (got %d, expected "
                "%d).  Skipping SGD update; total skipped=%d.  This "
                "guards against silent SGD corruption on Channel 2 "
                "data whose feature set differs from offline training.",
                arr.shape[1],
                self.expected_feature_dim,
                self.skipped_due_to_schema_mismatch,
            )
            return False
        return True

    def update_on_new_observation(self,
                                   patient_features: np.ndarray,
                                   attended: bool,
                                   actual_duration: float = None,
                                   weather_severity: float = 0.0) -> List[OnlineUpdateResult]:
        """
        Update all models when a new appointment outcome is observed.

        This is the main entry point — called after each appointment completes.

        Args:
            patient_features: Feature vector for the patient
            attended: Whether patient attended (True) or no-showed (False)
            actual_duration: Actual treatment duration in minutes (if attended)
            weather_severity: Weather severity at time of appointment

        Returns:
            List of OnlineUpdateResult for each model updated
        """
        results = []
        self.update_count += 1

        # 1. SGD update for no-show classifier
        sgd_result = self.sgd_update_noshow(patient_features, attended)
        if sgd_result:
            results.append(sgd_result)

        # 2. SGD update for duration regressor (only if attended)
        if attended and actual_duration:
            sgd_dur_result = self.sgd_update_duration(patient_features, actual_duration)
            if sgd_dur_result:
                results.append(sgd_dur_result)

        # 3. Bayesian update for no-show rate posterior
        bayes_result = self.bayesian_update_noshow(attended)
        results.append(bayes_result)

        # 4. Bayesian update for duration posterior (if attended)
        if attended and actual_duration:
            bayes_dur = self.bayesian_update_duration(actual_duration)
            results.append(bayes_dur)

        # 5. EMA update for baselines
        ema_result = self.ema_update(attended, actual_duration, weather_severity)
        results.append(ema_result)

        self.update_history.extend(results)

        return results

    def sgd_update_noshow(self, features: np.ndarray, attended: bool) -> Optional[OnlineUpdateResult]:
        """
        SGD update for no-show prediction.

        θ_{t+1} = θ_t - η_t * ∇L(θ_t; x_t, y_t)

        Uses sklearn's SGDClassifier with partial_fit for incremental learning.
        """
        if not SKLEARN_AVAILABLE or self.sgd_classifier is None:
            return None
        if not self._features_match_schema(features):
            return None

        X = np.array(features).reshape(1, -1)
        y = np.array([0 if attended else 1])  # 1 = no-show

        try:
            if not self._sgd_fitted:
                self.sgd_classifier.partial_fit(X, y, classes=[0, 1])
                self._sgd_fitted = True
                prob_before = 0.5
            else:
                prob_before = float(self.sgd_classifier.predict_proba(X)[0, 1]) if hasattr(self.sgd_classifier, 'predict_proba') else 0.5
                self.sgd_classifier.partial_fit(X, y)

            prob_after = float(self.sgd_classifier.predict_proba(X)[0, 1]) if hasattr(self.sgd_classifier, 'predict_proba') else 0.5

            return OnlineUpdateResult(
                model_name='SGD No-Show Classifier',
                method='sgd',
                samples_used=self.update_count,
                timestamp=datetime.now().isoformat(),
                metrics_before={'noshow_prob': round(prob_before, 4)},
                metrics_after={'noshow_prob': round(prob_after, 4)},
                improvement=round(abs(prob_after - prob_before), 4),
                details=f"θ updated via SGD (η={self.learning_rate}), attended={attended}"
            )
        except Exception as e:
            logger.warning(f"SGD no-show update failed: {e}")
            return None

    def sgd_update_duration(self, features: np.ndarray, actual_duration: float) -> Optional[OnlineUpdateResult]:
        """
        SGD update for duration prediction.

        θ_{t+1} = θ_t - η_t * ∇L(θ_t; x_t, y_t)
        where L = (y - θ^T x)^2 (squared error)
        """
        if not SKLEARN_AVAILABLE or self.sgd_regressor is None:
            return None
        if not self._features_match_schema(features):
            return None

        X = np.array(features).reshape(1, -1)
        y = np.array([actual_duration])

        try:
            if not hasattr(self.sgd_regressor, 'coef_') or self.sgd_regressor.coef_ is None:
                self.sgd_regressor.partial_fit(X, y)
                pred_before = actual_duration
            else:
                pred_before = float(self.sgd_regressor.predict(X)[0])
                self.sgd_regressor.partial_fit(X, y)

            pred_after = float(self.sgd_regressor.predict(X)[0])

            return OnlineUpdateResult(
                model_name='SGD Duration Regressor',
                method='sgd',
                samples_used=self.update_count,
                timestamp=datetime.now().isoformat(),
                metrics_before={'predicted_duration': round(pred_before, 1)},
                metrics_after={'predicted_duration': round(pred_after, 1)},
                improvement=round(abs(actual_duration - pred_after) - abs(actual_duration - pred_before), 1),
                details=f"θ updated via SGD, actual={actual_duration}min"
            )
        except Exception as e:
            logger.warning(f"SGD duration update failed: {e}")
            return None

    def bayesian_update_noshow(self, attended: bool) -> OnlineUpdateResult:
        """
        Bayesian online update for no-show rate.

        Prior: θ ~ Beta(α, β)
        Likelihood: x ~ Bernoulli(θ)
        Posterior: θ|x ~ Beta(α + x, β + 1 - x)

        P(θ|D_{1:t}) ∝ P(D_t|θ) · P(θ|D_{1:t-1})
        """
        prior = self.posterior['noshow_rate']
        alpha_before = prior['alpha']
        beta_before = prior['beta']
        mean_before = alpha_before / (alpha_before + beta_before)

        # Bayesian update: conjugate Beta-Bernoulli
        noshow = 0 if attended else 1
        prior['alpha'] += noshow      # α += x (1 if no-show)
        prior['beta'] += 1 - noshow   # β += (1-x) (1 if attended)

        mean_after = prior['alpha'] / (prior['alpha'] + prior['beta'])

        return OnlineUpdateResult(
            model_name='Bayesian No-Show Rate',
            method='bayesian',
            samples_used=int(prior['alpha'] + prior['beta'] - 12),  # Subtract prior
            timestamp=datetime.now().isoformat(),
            metrics_before={'rate': round(mean_before, 4), 'alpha': round(alpha_before, 1), 'beta': round(beta_before, 1)},
            metrics_after={'rate': round(mean_after, 4), 'alpha': round(prior['alpha'], 1), 'beta': round(prior['beta'], 1)},
            improvement=round(abs(mean_after - mean_before), 4),
            details=f"Beta({prior['alpha']:.1f}, {prior['beta']:.1f}) → mean={mean_after:.4f}"
        )

    def bayesian_update_duration(self, actual_duration: float) -> OnlineUpdateResult:
        """
        Bayesian online update for duration mean.

        Prior: μ ~ N(μ_0, σ²_0/n_0)
        Likelihood: x ~ N(μ, σ²)
        Posterior: μ|x ~ N(μ_n, σ²_n)

        μ_n = (n_0·μ_0 + x) / (n_0 + 1)
        σ²_n = σ²_0 / (n_0 + 1)
        """
        prior = self.posterior['duration_mean']
        mu_before = prior['mu']
        n = prior['n']

        # Bayesian normal-normal conjugate update
        prior['mu'] = (n * prior['mu'] + actual_duration) / (n + 1)
        prior['sigma2'] = prior['sigma2'] * n / (n + 1)
        prior['n'] = n + 1

        return OnlineUpdateResult(
            model_name='Bayesian Duration Mean',
            method='bayesian',
            samples_used=prior['n'],
            timestamp=datetime.now().isoformat(),
            metrics_before={'mean': round(mu_before, 1), 'n': n},
            metrics_after={'mean': round(prior['mu'], 1), 'n': prior['n']},
            improvement=round(abs(prior['mu'] - mu_before), 1),
            details=f"N({prior['mu']:.1f}, {prior['sigma2']:.1f}/{prior['n']}) after x={actual_duration}"
        )

    def ema_update(self, attended: bool, duration: float = None, weather: float = 0.0) -> OnlineUpdateResult:
        """
        Exponential Moving Average update for baselines.

        μ_{t+1} = α · x_t + (1-α) · μ_t

        Fast, simple, no model needed. Good for tracking drifting baselines.
        """
        alpha = self.ema_alpha
        before = dict(self.ema_baselines)

        # No-show rate
        noshow_val = 0.0 if attended else 1.0
        self.ema_baselines['noshow_rate'] = alpha * noshow_val + (1 - alpha) * self.ema_baselines['noshow_rate']

        # Duration (if attended)
        if attended and duration:
            self.ema_baselines['avg_duration'] = alpha * duration + (1 - alpha) * self.ema_baselines['avg_duration']

        # Weather impact
        if weather > 0.3 and not attended:
            self.ema_baselines['weather_impact'] = alpha * weather + (1 - alpha) * self.ema_baselines['weather_impact']

        return OnlineUpdateResult(
            model_name='EMA Baselines',
            method='ema',
            samples_used=self.update_count,
            timestamp=datetime.now().isoformat(),
            metrics_before=before,
            metrics_after=dict(self.ema_baselines),
            improvement=round(abs(self.ema_baselines['noshow_rate'] - before['noshow_rate']), 4),
            details=f"α={alpha}, noshow_rate={self.ema_baselines['noshow_rate']:.4f}, avg_dur={self.ema_baselines['avg_duration']:.1f}"
        )

    def get_posterior_summary(self) -> Dict[str, Any]:
        """Get current posterior parameters."""
        ns = self.posterior['noshow_rate']
        dur = self.posterior['duration_mean']
        return {
            'noshow_rate': {
                'distribution': 'Beta',
                'alpha': round(ns['alpha'], 2),
                'beta': round(ns['beta'], 2),
                'mean': round(ns['alpha'] / (ns['alpha'] + ns['beta']), 4),
                'var': round(ns['alpha'] * ns['beta'] / ((ns['alpha'] + ns['beta'])**2 * (ns['alpha'] + ns['beta'] + 1)), 6),
            },
            'duration_mean': {
                'distribution': 'Normal',
                'mu': round(dur['mu'], 1),
                'sigma2': round(dur['sigma2'], 1),
                'n_observations': dur['n'],
            },
            'ema_baselines': dict(self.ema_baselines),
            'total_updates': self.update_count,
            'sgd_fitted': self._sgd_fitted,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get online learning status."""
        return {
            'method': 'Online Learning (SGD + Bayesian + EMA)',
            'learning_rate': self.learning_rate,
            'ema_alpha': self.ema_alpha,
            'total_updates': self.update_count,
            'sgd_available': SKLEARN_AVAILABLE,
            'sgd_fitted': self._sgd_fitted,
            'posterior': self.get_posterior_summary(),
            'formulas': {
                'sgd': 'θ_{t+1} = θ_t - η_t · ∇L(θ_t; x_t, y_t)',
                'bayesian': 'P(θ|D_{1:t}) ∝ P(D_t|θ) · P(θ|D_{1:t-1})',
                'ema': 'μ_{t+1} = α · x_t + (1-α) · μ_t',
            },
            'recent_updates': [
                {
                    'model': r.model_name,
                    'method': r.method,
                    'improvement': r.improvement,
                    'details': r.details,
                }
                for r in self.update_history[-5:]
            ]
        }
