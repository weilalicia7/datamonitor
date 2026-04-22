"""
Hierarchical Bayesian Model for Duration Prediction (3.3)

Models treatment durations with patient-random effects:
y_ij ~ N(α_i + β^T x_ij, σ²)
α_i ~ N(0, τ²)

Where:
- i indexes patients
- j indexes appointments
- α_i is the patient-specific random effect
- β are fixed effects (shared across patients)
- τ² is the between-patient variance
- σ² is the within-patient (residual) variance

Benefits:
- Personalizes predictions per patient
- Quantifies uncertainty naturally
- Handles small sample sizes via shrinkage
- Borrows strength across patients with partial pooling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for PyMC availability
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
    logger.info("PyMC available for Hierarchical Bayesian Model")
except ImportError:
    PYMC_AVAILABLE = False
    logger.warning("PyMC not available, using fallback implementation")

# NumPy-based fallback
try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class HierarchicalPrediction:
    """Result of hierarchical Bayesian prediction."""
    patient_id: str
    predicted_duration: float  # Posterior mean
    credible_interval: Tuple[float, float]  # 95% credible interval
    patient_effect: float  # α_i estimate
    uncertainty: float  # Posterior standard deviation
    shrinkage_factor: float  # How much pooling toward population mean
    prediction_type: str  # 'posterior' or 'prior_predictive'


@dataclass
class HierarchicalModelSummary:
    """Summary of fitted hierarchical model."""
    n_patients: int
    n_observations: int
    population_mean: float  # μ (intercept)
    between_patient_std: float  # τ
    within_patient_std: float  # σ
    fixed_effects: Dict[str, float]  # β coefficients
    patient_effects: Dict[str, float]  # α_i estimates
    r_hat_max: float  # Convergence diagnostic
    ess_min: float  # Effective sample size


class HierarchicalBayesianModel:
    """
    Hierarchical Bayesian Model for personalized duration prediction.

    Uses partial pooling to balance:
    - Complete pooling (ignore patient differences)
    - No pooling (treat each patient independently)

    The hierarchical structure provides:
    - Better predictions for patients with few observations
    - Proper uncertainty quantification
    - Shrinkage toward population mean for extreme cases
    """

    # Feature names for fixed effects
    FEATURE_NAMES = [
        'cycle_number',
        'expected_duration',
        'complexity_factor',
        'has_comorbidities',
        'is_first_cycle',
        'hour_of_day',
        'is_monday',
        'is_friday',
        'weather_severity',
        'travel_distance_km'
    ]

    def __init__(self,
                 n_samples: int = 2000,
                 n_chains: int = 2,
                 use_pymc: bool = True):
        """
        Initialize Hierarchical Bayesian Model.

        Parameters:
        -----------
        n_samples : int
            Number of posterior samples per chain
        n_chains : int
            Number of MCMC chains
        use_pymc : bool
            Whether to use PyMC (if available)
        """
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.use_pymc = use_pymc and PYMC_AVAILABLE

        # Model state
        self.trace = None
        self.model = None
        self.is_fitted = False
        self.is_initialized = False

        # Patient mapping
        self.patient_ids: List[str] = []
        self.patient_to_idx: Dict[str, int] = {}

        # Fixed effects estimates
        self.beta_mean: Optional[np.ndarray] = None
        self.beta_std: Optional[np.ndarray] = None

        # Random effects estimates
        self.alpha_mean: Optional[np.ndarray] = None
        self.alpha_std: Optional[np.ndarray] = None

        # Variance components
        self.tau: float = 15.0  # Default between-patient std
        self.sigma: float = 20.0  # Default within-patient std
        self.mu: float = 100.0  # Default population mean

        # Prior specifications (weakly informative)
        self.priors = {
            'mu_mean': 100.0,  # Prior mean for population intercept
            'mu_std': 50.0,    # Prior std for population intercept
            'tau_std': 20.0,   # Prior for between-patient std (HalfNormal)
            'sigma_std': 30.0, # Prior for residual std (HalfNormal)
            'beta_std': 10.0   # Prior std for fixed effects
        }

        logger.info(f"HierarchicalBayesianModel initialized (PyMC: {self.use_pymc})")

    def fit(self,
            patient_data: List[Dict[str, Any]],
            durations: np.ndarray,
            patient_ids: Optional[List[str]] = None) -> 'HierarchicalBayesianModel':
        """
        Fit the hierarchical model to training data.

        Parameters:
        -----------
        patient_data : List[Dict]
            List of patient feature dictionaries (one per observation)
        durations : np.ndarray
            Observed treatment durations
        patient_ids : List[str], optional
            Patient IDs for each observation (for grouping)

        Returns:
        --------
        self : Fitted model
        """
        n_obs = len(patient_data)
        if n_obs < 10:
            logger.warning("Insufficient data for hierarchical model, using defaults")
            self._use_default_parameters()
            return self

        # Extract patient IDs if not provided
        if patient_ids is None:
            patient_ids = [d.get('patient_id', f'P{i}') for i, d in enumerate(patient_data)]

        # Build patient index mapping
        unique_patients = list(set(patient_ids))
        self.patient_ids = unique_patients
        self.patient_to_idx = {pid: i for i, pid in enumerate(unique_patients)}
        n_patients = len(unique_patients)

        # Convert patient IDs to indices
        patient_idx = np.array([self.patient_to_idx[pid] for pid in patient_ids])

        # Build feature matrix
        X = self._build_feature_matrix(patient_data)
        n_features = X.shape[1]

        # Fit model
        if self.use_pymc:
            self._fit_pymc(X, durations, patient_idx, n_patients, n_features)
        else:
            self._fit_empirical_bayes(X, durations, patient_idx, n_patients, n_features)

        self.is_fitted = True
        self.is_initialized = True
        logger.info(f"Hierarchical model fitted: {n_patients} patients, {n_obs} observations")

        return self

    def _build_feature_matrix(self, patient_data: List[Dict]) -> np.ndarray:
        """Build standardized feature matrix from patient data."""
        features = []

        for data in patient_data:
            row = []
            for name in self.FEATURE_NAMES:
                # Map various possible key names
                if name == 'cycle_number':
                    val = data.get('cycle_number', data.get('Cycle_Number', 1))
                elif name == 'expected_duration':
                    val = data.get('expected_duration', data.get('Planned_Duration', 120))
                elif name == 'complexity_factor':
                    val = data.get('complexity_factor', data.get('Complexity_Factor', 0.5))
                elif name == 'has_comorbidities':
                    val = 1.0 if data.get('has_comorbidities', data.get('Has_Comorbidities', False)) else 0.0
                elif name == 'is_first_cycle':
                    cycle = data.get('cycle_number', data.get('Cycle_Number', 1))
                    val = 1.0 if cycle == 1 else 0.0
                elif name == 'hour_of_day':
                    val = data.get('appointment_hour', data.get('Appointment_Hour', 10))
                elif name == 'is_monday':
                    dow = data.get('day_of_week', data.get('Day_Of_Week_Num', 2))
                    val = 1.0 if dow == 0 else 0.0
                elif name == 'is_friday':
                    dow = data.get('day_of_week', data.get('Day_Of_Week_Num', 2))
                    val = 1.0 if dow == 4 else 0.0
                elif name == 'weather_severity':
                    val = data.get('weather_severity', data.get('Weather_Severity', 0.0))
                elif name == 'travel_distance_km':
                    val = data.get('distance_km', data.get('Travel_Distance_KM', 10))
                else:
                    val = data.get(name, 0.0)

                # Handle NaN
                if pd.isna(val):
                    val = 0.0

                row.append(float(val))

            features.append(row)

        X = np.array(features)

        # Standardize features (except binary)
        self._feature_means = np.mean(X, axis=0)
        self._feature_stds = np.std(X, axis=0) + 1e-8  # Avoid division by zero

        # Don't standardize binary features
        binary_cols = [3, 4, 6, 7]  # has_comorbidities, is_first_cycle, is_monday, is_friday
        for col in binary_cols:
            self._feature_means[col] = 0.0
            self._feature_stds[col] = 1.0

        X_standardized = (X - self._feature_means) / self._feature_stds

        return X_standardized

    def _fit_pymc(self, X: np.ndarray, y: np.ndarray,
                  patient_idx: np.ndarray, n_patients: int, n_features: int):
        """Fit model using PyMC MCMC sampling."""
        logger.info("Fitting hierarchical model with PyMC...")

        with pm.Model() as self.model:
            # Hyperpriors
            # Population mean (intercept)
            mu = pm.Normal('mu', mu=self.priors['mu_mean'], sigma=self.priors['mu_std'])

            # Between-patient standard deviation
            tau = pm.HalfNormal('tau', sigma=self.priors['tau_std'])

            # Within-patient (residual) standard deviation
            sigma = pm.HalfNormal('sigma', sigma=self.priors['sigma_std'])

            # Patient random effects: α_i ~ N(0, τ²)
            alpha = pm.Normal('alpha', mu=0, sigma=tau, shape=n_patients)

            # Fixed effects: β ~ N(0, β_std²)
            beta = pm.Normal('beta', mu=0, sigma=self.priors['beta_std'], shape=n_features)

            # Linear predictor: μ + α_i + β^T x_ij
            linear_pred = mu + alpha[patient_idx] + pm.math.dot(X, beta)

            # Likelihood: y_ij ~ N(linear_pred, σ²)
            y_obs = pm.Normal('y', mu=linear_pred, sigma=sigma, observed=y)

            # Sample from posterior
            self.trace = pm.sample(
                draws=self.n_samples,
                chains=self.n_chains,
                cores=1,  # Single core for compatibility
                return_inferencedata=True,
                progressbar=False
            )

        # Extract posterior summaries
        self._extract_posterior_summaries()

    def _extract_posterior_summaries(self):
        """Extract summary statistics from PyMC trace."""
        if self.trace is None:
            return

        # Population parameters
        self.mu = float(self.trace.posterior['mu'].mean())
        self.tau = float(self.trace.posterior['tau'].mean())
        self.sigma = float(self.trace.posterior['sigma'].mean())

        # Fixed effects
        self.beta_mean = self.trace.posterior['beta'].mean(dim=['chain', 'draw']).values
        self.beta_std = self.trace.posterior['beta'].std(dim=['chain', 'draw']).values

        # Random effects
        self.alpha_mean = self.trace.posterior['alpha'].mean(dim=['chain', 'draw']).values
        self.alpha_std = self.trace.posterior['alpha'].std(dim=['chain', 'draw']).values

        logger.info(f"Posterior: mu={self.mu:.1f}, tau={self.tau:.1f}, sigma={self.sigma:.1f}")

    def _fit_empirical_bayes(self, X: np.ndarray, y: np.ndarray,
                             patient_idx: np.ndarray, n_patients: int, n_features: int):
        """
        Fit model using Empirical Bayes (fallback without PyMC).

        Uses restricted maximum likelihood (REML) approximation:
        1. Estimate fixed effects with OLS
        2. Estimate variance components from residuals
        3. Compute BLUPs for random effects
        """
        logger.info("Fitting hierarchical model with Empirical Bayes...")

        n_obs = len(y)

        # Step 1: OLS for initial fixed effects
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(n_obs), X])

        # OLS: β = (X'X)^(-1) X'y
        XtX = X_with_intercept.T @ X_with_intercept
        XtX_inv = np.linalg.pinv(XtX + 0.01 * np.eye(XtX.shape[0]))  # Ridge regularization
        beta_ols = XtX_inv @ X_with_intercept.T @ y

        self.mu = beta_ols[0]
        self.beta_mean = beta_ols[1:]

        # Step 2: Estimate variance components from residuals
        residuals = y - X_with_intercept @ beta_ols

        # Between-patient variance (using patient means)
        patient_means = np.zeros(n_patients)
        patient_counts = np.zeros(n_patients)
        for i, pid_idx in enumerate(patient_idx):
            patient_means[pid_idx] += residuals[i]
            patient_counts[pid_idx] += 1
        patient_means /= np.maximum(patient_counts, 1)

        # τ² estimate (between-patient variance)
        tau_sq = np.var(patient_means)
        self.tau = np.sqrt(max(tau_sq, 1.0))

        # σ² estimate (within-patient variance)
        within_residuals = residuals - patient_means[patient_idx]
        sigma_sq = np.var(within_residuals)
        self.sigma = np.sqrt(max(sigma_sq, 1.0))

        # Step 3: Compute BLUPs for random effects
        # α_i = (n_i * τ²) / (n_i * τ² + σ²) * (ȳ_i - μ - β'x̄_i)
        self.alpha_mean = np.zeros(n_patients)
        self.alpha_std = np.zeros(n_patients)

        for i in range(n_patients):
            mask = patient_idx == i
            n_i = np.sum(mask)
            if n_i > 0:
                y_bar_i = np.mean(y[mask])
                x_bar_i = np.mean(X[mask], axis=0)

                # Shrinkage factor
                shrinkage = (n_i * self.tau**2) / (n_i * self.tau**2 + self.sigma**2)

                # BLUP
                predicted_mean = self.mu + np.dot(x_bar_i, self.beta_mean)
                self.alpha_mean[i] = shrinkage * (y_bar_i - predicted_mean)

                # Posterior std (approximate)
                self.alpha_std[i] = self.tau * np.sqrt(1 - shrinkage)

        # Fixed effects uncertainty (approximate from OLS)
        mse = sigma_sq
        self.beta_std = np.sqrt(np.diag(XtX_inv[1:, 1:]) * mse)

        logger.info(f"Empirical Bayes: μ={self.mu:.1f}, τ={self.tau:.1f}, σ={self.sigma:.1f}")

    def _use_default_parameters(self):
        """Set default parameters when fitting is not possible."""
        self.mu = 100.0  # 100 minute default
        self.tau = 15.0  # Between-patient std
        self.sigma = 20.0  # Within-patient std

        n_features = len(self.FEATURE_NAMES)
        self.beta_mean = np.zeros(n_features)
        self.beta_std = np.ones(n_features) * self.priors['beta_std']

        # Informative priors for key features
        feature_priors = {
            'expected_duration': 0.8,  # Strong positive effect
            'is_first_cycle': 15.0,    # First cycle takes longer
            'complexity_factor': 20.0,  # Complexity adds time
            'cycle_number': -2.0       # Later cycles slightly shorter
        }

        for i, name in enumerate(self.FEATURE_NAMES):
            if name in feature_priors:
                self.beta_mean[i] = feature_priors[name]

        self.alpha_mean = np.array([])
        self.alpha_std = np.array([])
        self.patient_ids = []
        self.patient_to_idx = {}

        self.is_fitted = True
        self.is_initialized = True

        logger.info("Using default hierarchical model parameters")

    def predict(self,
                patient_data: Dict[str, Any],
                patient_id: Optional[str] = None) -> HierarchicalPrediction:
        """
        Predict duration for a patient with uncertainty quantification.

        Parameters:
        -----------
        patient_data : Dict
            Patient features for prediction
        patient_id : str, optional
            Patient ID (for personalized prediction)

        Returns:
        --------
        HierarchicalPrediction with posterior mean and credible interval
        """
        if not self.is_initialized:
            self._use_default_parameters()

        # Extract patient ID
        if patient_id is None:
            patient_id = patient_data.get('patient_id', patient_data.get('Patient_ID', 'unknown'))

        # Build features
        X = self._build_feature_matrix([patient_data])
        x = X[0]  # Single observation

        # Fixed effects contribution
        fixed_effect = self.mu + np.dot(x, self.beta_mean)

        # Random effect for this patient
        if patient_id in self.patient_to_idx:
            # Known patient - use estimated random effect
            idx = self.patient_to_idx[patient_id]
            patient_effect = self.alpha_mean[idx]
            patient_std = self.alpha_std[idx]

            # Shrinkage factor (how much we pool toward population)
            shrinkage = 1.0 - (patient_std**2 / (self.tau**2 + 1e-8))
            prediction_type = 'posterior'
        else:
            # New patient - use population prior
            patient_effect = 0.0
            patient_std = self.tau
            shrinkage = 0.0  # No pooling (use population mean)
            prediction_type = 'prior_predictive'

        # Posterior mean
        predicted_duration = fixed_effect + patient_effect

        # Posterior uncertainty
        # Var(y_new) = Var(fixed) + Var(random) + Var(residual)
        fixed_var = np.sum((x * self.beta_std)**2)  # Uncertainty in fixed effects
        random_var = patient_std**2  # Uncertainty in random effect
        residual_var = self.sigma**2  # Residual variance

        posterior_var = fixed_var + random_var + residual_var
        posterior_std = np.sqrt(posterior_var)

        # 95% credible interval
        lower = predicted_duration - 1.96 * posterior_std
        upper = predicted_duration + 1.96 * posterior_std

        # Ensure reasonable bounds
        lower = max(15, lower)  # Minimum 15 minutes
        predicted_duration = max(15, predicted_duration)

        return HierarchicalPrediction(
            patient_id=patient_id,
            predicted_duration=float(predicted_duration),
            credible_interval=(float(lower), float(upper)),
            patient_effect=float(patient_effect),
            uncertainty=float(posterior_std),
            shrinkage_factor=float(shrinkage),
            prediction_type=prediction_type
        )

    def predict_batch(self,
                      patient_data: List[Dict[str, Any]],
                      patient_ids: Optional[List[str]] = None) -> List[HierarchicalPrediction]:
        """Predict for multiple observations."""
        if patient_ids is None:
            patient_ids = [d.get('patient_id', d.get('Patient_ID')) for d in patient_data]

        return [
            self.predict(data, pid)
            for data, pid in zip(patient_data, patient_ids)
        ]

    def get_patient_effect(self, patient_id: str) -> Tuple[float, float]:
        """
        Get estimated random effect for a patient.

        Returns:
        --------
        Tuple[float, float]: (mean, std) of random effect
        """
        if patient_id in self.patient_to_idx:
            idx = self.patient_to_idx[patient_id]
            return (float(self.alpha_mean[idx]), float(self.alpha_std[idx]))
        else:
            # Unknown patient - return prior
            return (0.0, float(self.tau))

    def get_model_summary(self) -> HierarchicalModelSummary:
        """Get comprehensive model summary."""
        if not self.is_fitted:
            self._use_default_parameters()

        # Build fixed effects dict
        fixed_effects = {
            name: float(self.beta_mean[i])
            for i, name in enumerate(self.FEATURE_NAMES)
        }

        # Build patient effects dict
        patient_effects = {
            pid: float(self.alpha_mean[self.patient_to_idx[pid]])
            for pid in self.patient_ids
        } if self.patient_ids else {}

        # Convergence diagnostics (if PyMC trace available)
        r_hat_max = 1.0
        ess_min = self.n_samples * self.n_chains

        if self.trace is not None:
            try:
                summary = az.summary(self.trace)
                r_hat_max = float(summary['r_hat'].max())
                ess_min = float(summary['ess_bulk'].min())
            except Exception:
                pass

        return HierarchicalModelSummary(
            n_patients=len(self.patient_ids),
            n_observations=sum(1 for _ in self.patient_to_idx) if self.patient_to_idx else 0,
            population_mean=float(self.mu),
            between_patient_std=float(self.tau),
            within_patient_std=float(self.sigma),
            fixed_effects=fixed_effects,
            patient_effects=patient_effects,
            r_hat_max=r_hat_max,
            ess_min=ess_min
        )

    def get_variance_decomposition(self) -> Dict[str, float]:
        """
        Decompose total variance into components.

        Returns:
        --------
        Dict with variance components and ICC
        """
        tau_sq = self.tau**2
        sigma_sq = self.sigma**2
        total_var = tau_sq + sigma_sq

        # Intraclass Correlation Coefficient
        # ICC = τ² / (τ² + σ²)
        # Proportion of variance explained by patient differences
        icc = tau_sq / total_var if total_var > 0 else 0.0

        return {
            'between_patient_variance': tau_sq,
            'within_patient_variance': sigma_sq,
            'total_variance': total_var,
            'intraclass_correlation': icc,
            'interpretation': f"{icc*100:.1f}% of variance is between patients"
        }

    def compare_patients(self, patient_id_1: str, patient_id_2: str) -> Dict[str, Any]:
        """
        Compare two patients' predicted durations.

        Returns probability that patient_1 has longer duration than patient_2.
        """
        effect_1, std_1 = self.get_patient_effect(patient_id_1)
        effect_2, std_2 = self.get_patient_effect(patient_id_2)

        # Difference distribution: N(μ1 - μ2, σ1² + σ2²)
        diff_mean = effect_1 - effect_2
        diff_std = np.sqrt(std_1**2 + std_2**2)

        # P(patient_1 > patient_2)
        if SCIPY_AVAILABLE:
            prob_1_greater = 1 - stats.norm.cdf(0, loc=diff_mean, scale=diff_std)
        else:
            # Approximate with normal CDF
            z = -diff_mean / diff_std if diff_std > 0 else 0
            prob_1_greater = 0.5 * (1 + np.tanh(z * 0.6))  # Logistic approximation

        return {
            'patient_1_effect': effect_1,
            'patient_2_effect': effect_2,
            'difference_mean': diff_mean,
            'difference_std': diff_std,
            'prob_1_longer': prob_1_greater,
            'interpretation': f"{patient_id_1} is expected to take {abs(diff_mean):.1f} min {'longer' if diff_mean > 0 else 'shorter'}"
        }


# Convenience function for quick predictions
def predict_hierarchical(patient_data: Dict[str, Any],
                         patient_id: Optional[str] = None) -> HierarchicalPrediction:
    """
    Quick hierarchical prediction for a single patient.

    Uses default model parameters (no fitting required).

    Example:
    --------
    >>> patient = {
    ...     'patient_id': 'P001',
    ...     'cycle_number': 3,
    ...     'expected_duration': 120,
    ...     'complexity_factor': 0.7
    ... }
    >>> result = predict_hierarchical(patient)
    >>> print(f"Predicted: {result.predicted_duration:.0f} min")
    >>> print(f"95% CI: {result.credible_interval}")
    """
    model = HierarchicalBayesianModel()
    model._use_default_parameters()
    return model.predict(patient_data, patient_id)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)

    # Generate hierarchical data
    n_patients = 20
    obs_per_patient = 10

    # True parameters
    true_mu = 100
    true_tau = 15
    true_sigma = 20
    true_beta = np.array([0.5, 0.8, 10, 5, 15, 0.5, 3, 2, 5, 0.2])

    # Generate patient effects
    true_alpha = np.random.normal(0, true_tau, n_patients)

    patient_data = []
    durations = []
    patient_ids = []

    for i in range(n_patients):
        for j in range(obs_per_patient):
            # Random features
            features = {
                'patient_id': f'P{i:03d}',
                'cycle_number': j + 1,
                'expected_duration': 90 + np.random.normal(0, 30),
                'complexity_factor': np.random.uniform(0.3, 0.9),
                'has_comorbidities': np.random.random() < 0.3,
                'appointment_hour': np.random.randint(8, 17),
                'day_of_week': np.random.randint(0, 5),
                'weather_severity': np.random.uniform(0, 0.5),
                'distance_km': np.random.uniform(5, 50)
            }

            patient_data.append(features)
            patient_ids.append(f'P{i:03d}')

            # Generate duration with hierarchical structure
            duration = (true_mu +
                       true_alpha[i] +
                       np.random.normal(0, true_sigma))
            durations.append(max(30, duration))

    durations = np.array(durations)

    # Fit model
    print("=" * 60)
    print("Hierarchical Bayesian Model (3.3)")
    print("=" * 60)

    model = HierarchicalBayesianModel(n_samples=500, n_chains=2)
    model.fit(patient_data, durations, patient_ids)

    # Get summary
    summary = model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Patients: {summary.n_patients}")
    print(f"  Population mean (μ): {summary.population_mean:.1f}")
    print(f"  Between-patient std (τ): {summary.between_patient_std:.1f}")
    print(f"  Within-patient std (σ): {summary.within_patient_std:.1f}")

    # Variance decomposition
    var_decomp = model.get_variance_decomposition()
    print(f"\nVariance Decomposition:")
    print(f"  ICC: {var_decomp['intraclass_correlation']:.3f}")
    print(f"  {var_decomp['interpretation']}")

    # Predict for known patient
    print(f"\nPrediction for known patient (P005):")
    pred = model.predict({'patient_id': 'P005', 'cycle_number': 5}, 'P005')
    print(f"  Predicted: {pred.predicted_duration:.1f} min")
    print(f"  95% CI: ({pred.credible_interval[0]:.1f}, {pred.credible_interval[1]:.1f})")
    print(f"  Patient effect: {pred.patient_effect:.1f} min")
    print(f"  Shrinkage: {pred.shrinkage_factor:.2f}")

    # Predict for new patient
    print(f"\nPrediction for new patient:")
    pred_new = model.predict({'patient_id': 'P_NEW', 'cycle_number': 1}, 'P_NEW')
    print(f"  Predicted: {pred_new.predicted_duration:.1f} min")
    print(f"  95% CI: ({pred_new.credible_interval[0]:.1f}, {pred_new.credible_interval[1]:.1f})")
    print(f"  Type: {pred_new.prediction_type}")
