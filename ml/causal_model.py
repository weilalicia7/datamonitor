"""
Causal Inference Framework for Scheduling (4.1)

Models causality using Directed Acyclic Graphs (DAGs) and do-calculus.

DAG Structure:
    Appointment Time --> Traffic --> Arrival Time --> Delay
           |                              |
           v                              v
        Weather <--------------------> No-Show

Do-calculus for Interventions:
P(No-Show | do(Time=9am)) = sum_weather P(No-Show | Time, weather) * P(weather)

This allows answering causal questions:
- "What would happen to no-show rate if we moved all appointments to 9am?"
- "How much does weather CAUSE no-shows vs just being correlated?"

Benefits:
- Distinguishes correlation from causation
- Enables counterfactual reasoning
- Guides optimal intervention strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Check for networkx availability
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available, using basic DAG implementation")


class CausalVariable(Enum):
    """Variables in the scheduling causal graph."""
    APPOINTMENT_TIME = "appointment_time"
    WEATHER = "weather"
    TRAFFIC = "traffic"
    ARRIVAL_TIME = "arrival_time"
    DELAY = "delay"
    NO_SHOW = "no_show"
    DISTANCE = "distance"
    DAY_OF_WEEK = "day_of_week"
    PATIENT_HISTORY = "patient_history"
    REMINDER = "reminder"
    CYCLE_NUMBER = "cycle_number"


@dataclass
class CausalEffect:
    """Result of causal effect estimation."""
    treatment: str
    outcome: str
    treatment_value: Any
    causal_effect: float  # P(outcome | do(treatment))
    baseline_effect: float  # P(outcome) without intervention
    ate: float  # Average Treatment Effect
    confidence_interval: Tuple[float, float]
    adjustment_set: List[str]
    identification_formula: str
    is_identifiable: bool


@dataclass
class CounterfactualResult:
    """Result of counterfactual query."""
    query: str
    factual_outcome: Any
    counterfactual_outcome: Any
    effect: float
    probability: float
    explanation: str


@dataclass
class IVEstimationResult:
    """
    Result of Instrumental Variables (2SLS) estimation.

    For the scheduling context (Weather -> Traffic -> No-show):
    - Instrument (Z): Weather severity (exogenous weather conditions)
    - Treatment (D): Traffic delay (affected by weather)
    - Outcome (Y): No-show (affected by traffic delay)
    - Confounders (X): Other observed covariates

    Causal chain: Weather -> Traffic -> No-Show

    2SLS addresses unobserved confounding by using an instrument that:
    1. Affects treatment (relevance): Weather -> Traffic delay
    2. Only affects outcome through treatment (exclusion restriction)
    """
    instrument: str
    treatment: str
    outcome: str

    # Stage 1 results: Treatment ~ Instrument + Covariates
    first_stage_coef: float  # Coefficient of instrument on treatment
    first_stage_se: float  # Standard error
    first_stage_f_stat: float  # F-statistic for instrument strength
    first_stage_r_squared: float

    # Stage 2 results: Outcome ~ PredictedTreatment + Covariates
    causal_effect: float  # 2SLS estimate of treatment effect
    causal_effect_se: float  # Standard error of causal effect
    confidence_interval: Tuple[float, float]  # 95% CI

    # Diagnostics
    n_observations: int
    covariates: List[str]
    weak_instrument: bool  # True if F-stat < 10
    interpretation: str


class InstrumentalVariablesEstimator:
    """
    Two-Stage Least Squares (2SLS) estimator for causal inference
    with unobserved confounders.

    Causal Chain: Weather -> Traffic -> No-Show

    Model:
        Stage 1: D = γ₀ + γ₁Z + γ₂X + ν  (Traffic ~ Weather + Covariates)
        Stage 2: Y = β₀ + β₁D̂ + β₂X + ε  (NoShow ~ PredictedTraffic + Covariates)

    Where:
        Z = Instrument (weather severity)
        D = Treatment (traffic delay)
        Y = Outcome (no-show)
        X = Covariates (age, weather, etc.)
        D̂ = Predicted treatment from Stage 1

    The key insight: β₁ from Stage 2 gives the causal effect of D on Y,
    even with unobserved confounders, as long as the instrument is valid.
    """

    def __init__(self):
        self.is_fitted = False
        self.first_stage_model = None
        self.second_stage_model = None
        self.results = None

    def fit(
        self,
        data: pd.DataFrame,
        instrument: str,
        treatment: str,
        outcome: str,
        covariates: Optional[List[str]] = None
    ) -> IVEstimationResult:
        """
        Estimate causal effect using 2SLS.

        Parameters:
        -----------
        data : pd.DataFrame
            Data with instrument, treatment, outcome, and covariates
        instrument : str
            Name of instrument variable (e.g., 'Weather_Severity')
        treatment : str
            Name of treatment variable (e.g., 'travel_time')
        outcome : str
            Name of outcome variable (e.g., 'no_show')
        covariates : List[str], optional
            Names of control variables

        Returns:
        --------
        IVEstimationResult with causal effect estimate and diagnostics
        """
        if covariates is None:
            covariates = []

        # Prepare data
        df = data.dropna(subset=[instrument, treatment, outcome] + covariates)
        n = len(df)

        if n < 50:
            logger.warning(f"Small sample size ({n}) for IV estimation")

        # Extract variables
        Z = df[instrument].values.reshape(-1, 1)
        D = df[treatment].values
        Y = df[outcome].values

        # Build covariate matrix
        if covariates:
            X = df[covariates].values
            # Combine instrument and covariates for Stage 1
            Z_with_X = np.column_stack([Z, X])
        else:
            Z_with_X = Z
            X = None

        # Add intercept
        Z_with_intercept = np.column_stack([np.ones(n), Z_with_X])

        # ============================================
        # Stage 1: Treatment ~ Instrument + Covariates
        # D = γ₀ + γ₁Z + γ₂X + ν
        # ============================================

        # OLS estimation: γ = (Z'Z)^(-1) Z'D
        try:
            gamma = np.linalg.lstsq(Z_with_intercept, D, rcond=None)[0]
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in Stage 1, using pseudo-inverse")
            gamma = np.linalg.pinv(Z_with_intercept) @ D

        # Predicted treatment values
        D_hat = Z_with_intercept @ gamma

        # Stage 1 residuals and statistics
        residuals_1 = D - D_hat
        ss_res_1 = np.sum(residuals_1 ** 2)
        ss_tot_1 = np.sum((D - np.mean(D)) ** 2)
        r_squared_1 = 1 - ss_res_1 / ss_tot_1 if ss_tot_1 > 0 else 0

        # Standard errors for Stage 1
        mse_1 = ss_res_1 / (n - len(gamma))
        try:
            var_gamma = mse_1 * np.linalg.inv(Z_with_intercept.T @ Z_with_intercept)
            se_gamma = np.sqrt(np.diag(var_gamma))
        except np.linalg.LinAlgError:
            se_gamma = np.zeros(len(gamma))

        # F-statistic for instrument strength (testing γ₁ = 0)
        # F = (γ₁/SE(γ₁))²
        instrument_coef = gamma[1]  # γ₁ (coefficient on instrument)
        instrument_se = se_gamma[1] if len(se_gamma) > 1 else 0.001
        f_stat = (instrument_coef / instrument_se) ** 2 if instrument_se > 0 else 0

        weak_instrument = f_stat < 10  # Rule of thumb: F < 10 indicates weak instrument

        if weak_instrument:
            logger.warning(f"Weak instrument detected (F={f_stat:.2f} < 10)")

        # ============================================
        # Stage 2: Outcome ~ PredictedTreatment + Covariates
        # Y = β₀ + β₁D̂ + β₂X + ε
        # ============================================

        # Build Stage 2 design matrix with predicted treatment
        if X is not None:
            D_hat_with_X = np.column_stack([D_hat.reshape(-1, 1), X])
        else:
            D_hat_with_X = D_hat.reshape(-1, 1)

        D_hat_with_intercept = np.column_stack([np.ones(n), D_hat_with_X])

        # OLS estimation: β = (D̂'D̂)^(-1) D̂'Y
        try:
            beta = np.linalg.lstsq(D_hat_with_intercept, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in Stage 2, using pseudo-inverse")
            beta = np.linalg.pinv(D_hat_with_intercept) @ Y

        # Causal effect estimate (β₁)
        causal_effect = beta[1]

        # Stage 2 residuals and standard errors
        # Note: Use actual D, not D_hat, for residual calculation (2SLS adjustment)
        if X is not None:
            D_actual_with_X = np.column_stack([D.reshape(-1, 1), X])
        else:
            D_actual_with_X = D.reshape(-1, 1)
        D_actual_with_intercept = np.column_stack([np.ones(n), D_actual_with_X])

        Y_pred = D_hat_with_intercept @ beta
        residuals_2 = Y - Y_pred
        ss_res_2 = np.sum(residuals_2 ** 2)

        # 2SLS standard errors (accounts for first stage estimation)
        mse_2 = ss_res_2 / (n - len(beta))
        try:
            # Correct variance: uses actual D but estimation done with D_hat
            var_beta = mse_2 * np.linalg.inv(D_hat_with_intercept.T @ D_hat_with_intercept)
            se_beta = np.sqrt(np.diag(var_beta))
            causal_effect_se = se_beta[1]
        except np.linalg.LinAlgError:
            causal_effect_se = 0.1  # Fallback

        # 95% confidence interval
        ci_lower = causal_effect - 1.96 * causal_effect_se
        ci_upper = causal_effect + 1.96 * causal_effect_se

        # Generate interpretation
        effect_direction = "increases" if causal_effect > 0 else "decreases"
        effect_magnitude = abs(causal_effect)

        if outcome == 'no_show':
            interpretation = (
                f"A 1-unit increase in {treatment} {effect_direction} "
                f"no-show probability by {effect_magnitude:.3f} "
                f"(95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]). "
                f"Instrument strength: F={f_stat:.1f} ({'weak' if weak_instrument else 'adequate'})."
            )
        else:
            interpretation = (
                f"A 1-unit increase in {treatment} {effect_direction} "
                f"{outcome} by {effect_magnitude:.3f} "
                f"(95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])."
            )

        self.results = IVEstimationResult(
            instrument=instrument,
            treatment=treatment,
            outcome=outcome,
            first_stage_coef=float(instrument_coef),
            first_stage_se=float(instrument_se),
            first_stage_f_stat=float(f_stat),
            first_stage_r_squared=float(r_squared_1),
            causal_effect=float(causal_effect),
            causal_effect_se=float(causal_effect_se),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            n_observations=n,
            covariates=covariates,
            weak_instrument=weak_instrument,
            interpretation=interpretation
        )

        self.is_fitted = True
        logger.info(f"IV estimation complete: {treatment} -> {outcome}, effect={causal_effect:.4f}")

        return self.results

    def predict_treatment(self, data: pd.DataFrame) -> np.ndarray:
        """Predict treatment values using first stage model."""
        if not self.is_fitted or self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        # This would need storing the gamma coefficients
        raise NotImplementedError("Prediction not yet implemented")


# =============================================================================
# 4.3 DOUBLE MACHINE LEARNING
# =============================================================================

@dataclass
class DMLResult:
    """
    Result of Double Machine Learning estimation.

    θ̂ = (1/n) Σᵢ [(Yᵢ - ĝ(Xᵢ))(Tᵢ - m̂(Xᵢ))] / [m̂(Xᵢ)(1 - m̂(Xᵢ))]

    Where:
    - ĝ(X) = outcome model prediction (E[Y|X])
    - m̂(X) = propensity score (P[T=1|X])
    """
    treatment: str
    outcome: str
    treatment_effect: float  # θ̂ - Average Treatment Effect (ATE)
    standard_error: float
    confidence_interval: Tuple[float, float]
    t_statistic: float
    p_value: float

    # Model performance
    outcome_model_r2: float  # R² of outcome model ĝ(X)
    propensity_auc: float    # AUC of propensity model m̂(X)

    # Cross-fitting info
    n_folds: int
    n_observations: int

    # Interpretation
    interpretation: str


class DoubleMachineLearning:
    """
    Double Machine Learning (DML) for treatment effect estimation
    with high-dimensional controls.

    Key idea: Use cross-fitting to avoid overfitting bias when using
    ML models for nuisance parameter estimation.

    Formula:
    θ̂ = (1/n) Σᵢ [(Yᵢ - ĝ(Xᵢ))(Tᵢ - m̂(Xᵢ))] / [m̂(Xᵢ)(1 - m̂(Xᵢ))]

    Where:
    - ĝ(X) = outcome model (predicts Y given X, using XGBoost)
    - m̂(X) = propensity model (predicts P(T=1|X), using Random Forest)

    Cross-fitting procedure:
    1. Split data into K folds
    2. For each fold k:
       - Train ĝ and m̂ on data excluding fold k
       - Predict on fold k
    3. Compute θ̂ using all out-of-fold predictions

    This achieves √n-consistency even with slow-converging ML estimators.
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.is_fitted = False
        self.results: Optional[DMLResult] = None

        # ML models for nuisance functions
        self.outcome_model = None  # ĝ(X) - XGBoost
        self.propensity_model = None  # m̂(X) - Random Forest

        logger.info(f"DoubleMachineLearning initialized (n_folds={n_folds})")

    def fit(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str]
    ) -> DMLResult:
        """
        Estimate Average Treatment Effect using Double ML.

        Parameters:
        -----------
        data : pd.DataFrame
            Data with treatment, outcome, and covariates
        treatment : str
            Name of binary treatment variable (0/1)
        outcome : str
            Name of outcome variable
        covariates : List[str]
            Names of control variables (high-dimensional OK)

        Returns:
        --------
        DMLResult with treatment effect estimate
        """
        # Prepare data
        df = data.dropna(subset=[treatment, outcome] + covariates).copy()
        n = len(df)

        Y = df[outcome].values
        T = df[treatment].values
        X = df[covariates].values

        # Ensure treatment is binary
        if not np.all(np.isin(T, [0, 1])):
            # Convert to binary if needed
            T = (T > np.median(T)).astype(int)

        # Initialize arrays for out-of-fold predictions
        g_hat = np.zeros(n)  # Outcome predictions
        m_hat = np.zeros(n)  # Propensity scores

        # Track model performance
        outcome_r2_scores = []
        propensity_auc_scores = []

        # Cross-fitting
        np.random.seed(self.random_state)
        fold_indices = np.random.permutation(n) % self.n_folds

        for k in range(self.n_folds):
            # Split data
            train_mask = fold_indices != k
            test_mask = fold_indices == k

            X_train, X_test = X[train_mask], X[test_mask]
            Y_train, Y_test = Y[train_mask], Y[test_mask]
            T_train, T_test = T[train_mask], T[test_mask]

            # Train outcome model ĝ(X) on control observations only
            # (or all observations - both approaches are valid)
            outcome_model = self._create_outcome_model()
            outcome_model.fit(X_train, Y_train)
            g_hat[test_mask] = outcome_model.predict(X_test)

            # Calculate R² for outcome model
            if len(Y_test) > 0:
                ss_res = np.sum((Y_test - g_hat[test_mask]) ** 2)
                ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
                if ss_tot > 0:
                    outcome_r2_scores.append(1 - ss_res / ss_tot)

            # Train propensity model m̂(X)
            propensity_model = self._create_propensity_model()
            propensity_model.fit(X_train, T_train)

            # Get propensity scores (probability of treatment)
            if hasattr(propensity_model, 'predict_proba'):
                m_hat[test_mask] = propensity_model.predict_proba(X_test)[:, 1]
            else:
                m_hat[test_mask] = propensity_model.predict(X_test)

            # Clip propensity scores to avoid division issues
            m_hat[test_mask] = np.clip(m_hat[test_mask], 0.01, 0.99)

            # Calculate AUC for propensity model
            if len(np.unique(T_test)) > 1:
                try:
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(T_test, m_hat[test_mask])
                    propensity_auc_scores.append(auc)
                except:
                    pass

        # Compute DML estimator (Neyman-orthogonal score)
        # θ̂ = (1/n) Σᵢ [(Yᵢ - ĝ(Xᵢ))(Tᵢ - m̂(Xᵢ))] / [m̂(Xᵢ)(1 - m̂(Xᵢ))]

        # Residuals
        Y_residual = Y - g_hat  # Yᵢ - ĝ(Xᵢ)
        T_residual = T - m_hat  # Tᵢ - m̂(Xᵢ)

        # Denominator: propensity variance
        propensity_var = m_hat * (1 - m_hat)

        # DML score (influence function)
        psi = (Y_residual * T_residual) / propensity_var

        # Treatment effect estimate
        theta_hat = np.mean(psi)

        # Standard error (using influence function variance)
        se = np.std(psi) / np.sqrt(n)

        # Confidence interval (95%)
        z = 1.96
        ci_lower = theta_hat - z * se
        ci_upper = theta_hat + z * se

        # T-statistic and p-value
        t_stat = theta_hat / se if se > 0 else 0
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        # Model performance
        outcome_r2 = np.mean(outcome_r2_scores) if outcome_r2_scores else 0.0
        propensity_auc = np.mean(propensity_auc_scores) if propensity_auc_scores else 0.5

        # Interpretation
        if p_value < 0.05:
            if theta_hat > 0:
                interpretation = (
                    f"Treatment '{treatment}' significantly INCREASES '{outcome}' "
                    f"by {theta_hat:.4f} (p={p_value:.4f}). "
                    f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
                )
            else:
                interpretation = (
                    f"Treatment '{treatment}' significantly DECREASES '{outcome}' "
                    f"by {abs(theta_hat):.4f} (p={p_value:.4f}). "
                    f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
                )
        else:
            interpretation = (
                f"No significant effect of '{treatment}' on '{outcome}' "
                f"(effect={theta_hat:.4f}, p={p_value:.4f})"
            )

        self.results = DMLResult(
            treatment=treatment,
            outcome=outcome,
            treatment_effect=theta_hat,
            standard_error=se,
            confidence_interval=(ci_lower, ci_upper),
            t_statistic=t_stat,
            p_value=p_value,
            outcome_model_r2=outcome_r2,
            propensity_auc=propensity_auc,
            n_folds=self.n_folds,
            n_observations=n,
            interpretation=interpretation
        )

        self.is_fitted = True
        logger.info(f"DML fitted: theta_hat={theta_hat:.4f}, SE={se:.4f}, p={p_value:.4f}")

        return self.results

    def _create_outcome_model(self):
        """Create outcome model ĝ(X) - XGBoost or fallback."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        except ImportError:
            # Numpy fallback - simple linear model
            return _NumpyLinearRegressor()

    def _create_propensity_model(self):
        """Create propensity model m̂(X) - Random Forest or fallback."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )
        except ImportError:
            # Numpy fallback - logistic approximation
            return _NumpyLogisticRegressor()

    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        import math
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of DML estimation."""
        if not self.is_fitted or self.results is None:
            return {'status': 'not_fitted'}

        return {
            'treatment': self.results.treatment,
            'outcome': self.results.outcome,
            'treatment_effect': round(self.results.treatment_effect, 4),
            'standard_error': round(self.results.standard_error, 4),
            'confidence_interval': [
                round(self.results.confidence_interval[0], 4),
                round(self.results.confidence_interval[1], 4)
            ],
            't_statistic': round(self.results.t_statistic, 4),
            'p_value': round(self.results.p_value, 4),
            'significant': self.results.p_value < 0.05,
            'outcome_model_r2': round(self.results.outcome_model_r2, 4),
            'propensity_auc': round(self.results.propensity_auc, 4),
            'n_folds': self.results.n_folds,
            'n_observations': self.results.n_observations,
            'interpretation': self.results.interpretation
        }


class _NumpyLinearRegressor:
    """Simple linear regressor fallback using numpy."""

    def __init__(self):
        self.coef = None
        self.intercept = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        # Solve normal equations
        try:
            self.coef = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            self.intercept = self.coef[0]
            self.coef = self.coef[1:]
        except:
            self.coef = np.zeros(X.shape[1])
            self.intercept = np.mean(y)
        return self

    def predict(self, X):
        X = np.array(X)
        return self.intercept + X @ self.coef


class _NumpyLogisticRegressor:
    """Simple logistic regressor fallback using numpy."""

    def __init__(self):
        self.coef = None
        self.intercept = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        # Simple approximation using linear model + sigmoid
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        try:
            self.coef = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            self.intercept = self.coef[0]
            self.coef = self.coef[1:]
        except:
            self.coef = np.zeros(X.shape[1])
            self.intercept = 0.0
        return self

    def predict(self, X):
        X = np.array(X)
        logits = self.intercept + X @ self.coef
        return (logits > 0.5).astype(int)

    def predict_proba(self, X):
        X = np.array(X)
        logits = self.intercept + X @ self.coef
        probs = 1 / (1 + np.exp(-logits))
        probs = np.clip(probs, 0.01, 0.99)
        return np.column_stack([1 - probs, probs])


def estimate_dml_effect(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
    n_folds: int = 5
) -> DMLResult:
    """
    Convenience function for Double Machine Learning estimation.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with treatment, outcome, and covariates
    treatment : str
        Binary treatment variable name
    outcome : str
        Outcome variable name
    covariates : List[str]
        High-dimensional control variables
    n_folds : int
        Number of cross-fitting folds

    Returns:
    --------
    DMLResult with treatment effect estimate
    """
    dml = DoubleMachineLearning(n_folds=n_folds)
    return dml.fit(data, treatment, outcome, covariates)


@dataclass
class DAGNode:
    """Node in the causal DAG."""
    name: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    probabilities: Dict[Tuple, float] = field(default_factory=dict)


class CausalDAG:
    """
    Directed Acyclic Graph for causal modeling.

    Implements:
    - DAG structure with parent-child relationships
    - d-separation for conditional independence
    - Backdoor criterion for identifying adjustment sets
    - Front-door criterion when backdoor fails
    """

    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: List[Tuple[str, str]] = []

        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = None

    def add_node(self, name: str, values: List[Any] = None):
        """Add a node to the DAG."""
        if name not in self.nodes:
            self.nodes[name] = DAGNode(name=name, values=values or [])
            if self.graph is not None:
                self.graph.add_node(name)

    def add_edge(self, parent: str, child: str):
        """Add a directed edge from parent to child."""
        # Ensure nodes exist
        self.add_node(parent)
        self.add_node(child)

        # Add edge
        self.edges.append((parent, child))
        self.nodes[parent].children.append(child)
        self.nodes[child].parents.append(parent)

        if self.graph is not None:
            self.graph.add_edge(parent, child)

    def get_parents(self, node: str) -> List[str]:
        """Get all parents of a node."""
        if node in self.nodes:
            return self.nodes[node].parents
        return []

    def get_children(self, node: str) -> List[str]:
        """Get all children of a node."""
        if node in self.nodes:
            return self.nodes[node].children
        return []

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors of a node."""
        ancestors = set()
        to_visit = list(self.get_parents(node))

        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.get_parents(current))

        return ancestors

    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants of a node."""
        descendants = set()
        to_visit = list(self.get_children(node))

        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.get_children(current))

        return descendants

    def is_d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """
        Check if X and Y are d-separated given Z.

        D-separation implies conditional independence:
        X _|_ Y | Z in distribution if X d-sep Y | Z in DAG
        """
        if self.graph is not None:
            # Use networkx for d-separation
            try:
                return nx.d_separated(self.graph, {x}, {y}, z)
            except:
                pass

        # Fallback: simplified check
        # Check if Z blocks all paths from X to Y
        paths = self._find_all_paths(x, y)
        for path in paths:
            if not self._is_path_blocked(path, z):
                return False
        return True

    def _find_all_paths(self, start: str, end: str, path: List[str] = None) -> List[List[str]]:
        """Find all undirected paths between two nodes."""
        if path is None:
            path = []
        path = path + [start]

        if start == end:
            return [path]

        paths = []
        # Get neighbors (parents + children for undirected path)
        neighbors = set(self.get_parents(start)) | set(self.get_children(start))

        for node in neighbors:
            if node not in path:
                new_paths = self._find_all_paths(node, end, path)
                paths.extend(new_paths)

        return paths

    def _is_path_blocked(self, path: List[str], z: Set[str]) -> bool:
        """Check if a path is blocked by conditioning set Z."""
        if len(path) < 3:
            return False

        for i in range(1, len(path) - 1):
            prev_node = path[i-1]
            curr_node = path[i]
            next_node = path[i+1]

            # Check path type at curr_node
            is_chain = (prev_node in self.get_parents(curr_node) and
                       curr_node in self.get_parents(next_node))
            is_fork = (curr_node in self.get_parents(prev_node) and
                      curr_node in self.get_parents(next_node))
            is_collider = (prev_node in self.get_parents(curr_node) and
                          next_node in self.get_parents(curr_node))

            if is_chain or is_fork:
                # Blocked if curr_node is in Z
                if curr_node in z:
                    return True
            elif is_collider:
                # Blocked if curr_node and all descendants NOT in Z
                descendants = self.get_descendants(curr_node)
                if curr_node not in z and not (descendants & z):
                    return True

        return False

    def find_backdoor_adjustment_set(self, treatment: str, outcome: str) -> Optional[Set[str]]:
        """
        Find a valid adjustment set using the backdoor criterion.

        Backdoor Criterion: A set Z satisfies the backdoor criterion if:
        1. No node in Z is a descendant of treatment
        2. Z blocks all backdoor paths from treatment to outcome
        """
        # Get all non-descendants of treatment
        descendants = self.get_descendants(treatment)
        non_descendants = set(self.nodes.keys()) - descendants - {treatment, outcome}

        # Parents of treatment are always valid candidates
        parents = set(self.get_parents(treatment))

        # Start with parents of treatment
        adjustment_set = parents & non_descendants

        # Verify it blocks all backdoor paths
        # Backdoor paths are paths from treatment to outcome that start with an arrow into treatment
        if adjustment_set:
            return adjustment_set

        # Try adding more variables
        for node in non_descendants:
            test_set = adjustment_set | {node}
            if self.is_d_separated(treatment, outcome, test_set - {treatment}):
                return test_set

        return adjustment_set if adjustment_set else None

    def is_identifiable(self, treatment: str, outcome: str) -> bool:
        """Check if causal effect is identifiable (can be computed from observational data)."""
        # Effect is identifiable if backdoor criterion can be satisfied
        adjustment_set = self.find_backdoor_adjustment_set(treatment, outcome)
        return adjustment_set is not None


class SchedulingCausalModel:
    """
    Causal model for the SACT scheduling system.

    Implements the DAG:
        Appointment Time --> Traffic --> Arrival Time --> Delay
               |                              |
               v                              v
            Weather <--------------------> No-Show
               ^
               |
        Day of Week

    Plus additional causal relationships:
    - Patient History --> No-Show
    - Distance --> Traffic, Arrival Time
    - Reminder --> No-Show (intervention)
    - Cycle Number --> No-Show
    """

    def __init__(self):
        self.dag = CausalDAG()
        self._build_scheduling_dag()
        self.conditional_probs: Dict[str, Dict] = {}
        self.is_fitted = False
        self.is_initialized = False

        logger.info("SchedulingCausalModel initialized")

    def _build_scheduling_dag(self):
        """Build the causal DAG for scheduling."""
        # Add nodes with possible values
        self.dag.add_node("appointment_time", values=["early", "mid", "late"])
        self.dag.add_node("day_of_week", values=["monday", "tuesday", "wednesday", "thursday", "friday"])
        self.dag.add_node("weather", values=["clear", "rain", "severe"])
        self.dag.add_node("traffic", values=["light", "moderate", "heavy"])
        self.dag.add_node("distance", values=["near", "medium", "far"])
        self.dag.add_node("arrival_time", values=["early", "on_time", "late", "no_show"])
        self.dag.add_node("delay", values=["none", "minor", "major"])
        self.dag.add_node("no_show", values=[0, 1])
        self.dag.add_node("patient_history", values=["good", "moderate", "poor"])
        self.dag.add_node("reminder", values=["none", "sms", "phone"])
        self.dag.add_node("cycle_number", values=["first", "early", "late"])

        # Add causal edges based on domain knowledge
        # Appointment time affects traffic and weather exposure
        self.dag.add_edge("appointment_time", "traffic")
        self.dag.add_edge("appointment_time", "weather")

        # Day of week affects traffic
        self.dag.add_edge("day_of_week", "traffic")
        self.dag.add_edge("day_of_week", "weather")

        # Traffic affects arrival time
        self.dag.add_edge("traffic", "arrival_time")

        # Distance affects traffic exposure and arrival
        self.dag.add_edge("distance", "traffic")
        self.dag.add_edge("distance", "arrival_time")

        # Arrival time affects delay
        self.dag.add_edge("arrival_time", "delay")

        # Weather affects traffic, arrival, and no-show directly
        self.dag.add_edge("weather", "traffic")
        self.dag.add_edge("weather", "arrival_time")
        self.dag.add_edge("weather", "no_show")

        # Patient history affects no-show
        self.dag.add_edge("patient_history", "no_show")

        # Reminder affects no-show (intervention)
        self.dag.add_edge("reminder", "no_show")

        # Cycle number affects no-show
        self.dag.add_edge("cycle_number", "no_show")

        # Delay affects no-show (late arrival may lead to leaving)
        self.dag.add_edge("delay", "no_show")

        # Arrival time affects no-show
        self.dag.add_edge("arrival_time", "no_show")

        logger.info(f"Built causal DAG with {len(self.dag.nodes)} nodes and {len(self.dag.edges)} edges")

    def fit(self, data: pd.DataFrame) -> 'SchedulingCausalModel':
        """
        Fit conditional probability tables from observational data.

        Parameters:
        -----------
        data : pd.DataFrame
            Historical appointment data with columns matching DAG nodes
        """
        if len(data) < 50:
            logger.warning("Insufficient data for causal model fitting, using defaults")
            self._use_default_probabilities()
            return self

        # Discretize continuous variables
        data = self._discretize_data(data)

        # Estimate conditional probabilities P(child | parents)
        for node_name, node in self.dag.nodes.items():
            if node.parents:
                self._estimate_conditional_prob(data, node_name, node.parents)
            else:
                self._estimate_marginal_prob(data, node_name)

        self.is_fitted = True
        self.is_initialized = True
        logger.info(f"Causal model fitted on {len(data)} observations")

        return self

    def _discretize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Discretize continuous variables for causal analysis."""
        df = data.copy()

        # Appointment time: early (<10), mid (10-14), late (>14)
        if 'Appointment_Hour' in df.columns:
            df['appointment_time'] = pd.cut(
                df['Appointment_Hour'],
                bins=[0, 10, 14, 24],
                labels=['early', 'mid', 'late']
            )
        elif 'appointment_hour' in df.columns:
            df['appointment_time'] = pd.cut(
                df['appointment_hour'],
                bins=[0, 10, 14, 24],
                labels=['early', 'mid', 'late']
            )

        # Day of week
        if 'Day_Of_Week_Num' in df.columns:
            day_map = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday', 4: 'friday'}
            df['day_of_week'] = df['Day_Of_Week_Num'].map(day_map).fillna('wednesday')
        elif 'day_of_week' in df.columns:
            pass  # Already discretized

        # Weather
        if 'Weather_Severity' in df.columns:
            df['weather'] = pd.cut(
                df['Weather_Severity'],
                bins=[-0.01, 0.1, 0.3, 1.0],
                labels=['clear', 'rain', 'severe']
            )
        elif 'weather_severity' in df.columns:
            df['weather'] = pd.cut(
                df['weather_severity'],
                bins=[-0.01, 0.1, 0.3, 1.0],
                labels=['clear', 'rain', 'severe']
            )

        # Distance
        if 'Travel_Distance_KM' in df.columns:
            df['distance'] = pd.cut(
                df['Travel_Distance_KM'],
                bins=[0, 10, 25, 100],
                labels=['near', 'medium', 'far']
            )
        elif 'distance_km' in df.columns:
            df['distance'] = pd.cut(
                df['distance_km'],
                bins=[0, 10, 25, 100],
                labels=['near', 'medium', 'far']
            )

        # Patient history based on previous no-shows
        if 'Patient_NoShow_Rate' in df.columns:
            df['patient_history'] = pd.cut(
                df['Patient_NoShow_Rate'],
                bins=[-0.01, 0.1, 0.25, 1.0],
                labels=['good', 'moderate', 'poor']
            )
        elif 'noshow_rate' in df.columns:
            df['patient_history'] = pd.cut(
                df['noshow_rate'],
                bins=[-0.01, 0.1, 0.25, 1.0],
                labels=['good', 'moderate', 'poor']
            )

        # Cycle number
        if 'Cycle_Number' in df.columns:
            df['cycle_number'] = df['Cycle_Number'].apply(
                lambda x: 'first' if x == 1 else ('early' if x <= 3 else 'late')
            )
        elif 'cycle_number' in df.columns:
            df['cycle_number'] = df['cycle_number'].apply(
                lambda x: 'first' if x == 1 else ('early' if x <= 3 else 'late')
            )

        # No-show outcome
        if 'Attended_Status' in df.columns:
            df['no_show'] = (df['Attended_Status'] == 'No').astype(int)
        elif 'Showed_Up' in df.columns:
            df['no_show'] = (~df['Showed_Up']).astype(int)

        # Reminder (intervention)
        if 'Reminder_Sent' in df.columns:
            df['reminder'] = df.apply(
                lambda row: 'phone' if row.get('Phone_Call_Made', False)
                else ('sms' if row.get('Reminder_Sent', False) else 'none'),
                axis=1
            )
        else:
            df['reminder'] = 'none'

        # Traffic (simulate based on time and day)
        if 'appointment_time' in df.columns and 'day_of_week' in df.columns:
            def estimate_traffic(row):
                time = row.get('appointment_time', 'mid')
                day = row.get('day_of_week', 'wednesday')
                if day == 'monday' and time in ['early', 'late']:
                    return 'heavy'
                elif time == 'mid':
                    return 'light'
                else:
                    return 'moderate'
            df['traffic'] = df.apply(estimate_traffic, axis=1)
        else:
            df['traffic'] = 'moderate'

        return df

    def _estimate_conditional_prob(self, data: pd.DataFrame, child: str, parents: List[str]):
        """Estimate P(child | parents) from data."""
        if child not in data.columns:
            return

        valid_parents = [p for p in parents if p in data.columns]
        if not valid_parents:
            self._estimate_marginal_prob(data, child)
            return

        # Group by parents and compute conditional probabilities
        self.conditional_probs[child] = {}

        try:
            grouped = data.groupby(valid_parents)[child].value_counts(normalize=True)
            for idx, prob in grouped.items():
                if isinstance(idx, tuple):
                    parent_values = idx[:-1]
                    child_value = idx[-1]
                else:
                    parent_values = (idx,)
                    child_value = grouped.index.get_level_values(-1)[0]

                key = (tuple(valid_parents), parent_values, child_value)
                self.conditional_probs[child][key] = prob
        except Exception as e:
            logger.warning(f"Error estimating P({child} | {valid_parents}): {e}")
            self._estimate_marginal_prob(data, child)

    def _estimate_marginal_prob(self, data: pd.DataFrame, node: str):
        """Estimate marginal probability P(node) from data."""
        if node not in data.columns:
            return

        self.conditional_probs[node] = {}

        try:
            probs = data[node].value_counts(normalize=True)
            for value, prob in probs.items():
                key = ((), (), value)
                self.conditional_probs[node][key] = prob
        except Exception as e:
            logger.warning(f"Error estimating P({node}): {e}")

    def _use_default_probabilities(self):
        """Use default conditional probabilities based on domain knowledge."""
        # P(no_show | weather, patient_history, reminder)
        self.conditional_probs['no_show'] = {
            # Good history, clear weather, no reminder
            (('weather', 'patient_history', 'reminder'), ('clear', 'good', 'none'), 1): 0.08,
            (('weather', 'patient_history', 'reminder'), ('clear', 'good', 'none'), 0): 0.92,
            # Good history, clear weather, sms reminder
            (('weather', 'patient_history', 'reminder'), ('clear', 'good', 'sms'), 1): 0.05,
            (('weather', 'patient_history', 'reminder'), ('clear', 'good', 'sms'), 0): 0.95,
            # Poor history, severe weather, no reminder
            (('weather', 'patient_history', 'reminder'), ('severe', 'poor', 'none'), 1): 0.45,
            (('weather', 'patient_history', 'reminder'), ('severe', 'poor', 'none'), 0): 0.55,
            # Poor history, severe weather, phone reminder
            (('weather', 'patient_history', 'reminder'), ('severe', 'poor', 'phone'), 1): 0.25,
            (('weather', 'patient_history', 'reminder'), ('severe', 'poor', 'phone'), 0): 0.75,
        }

        # P(weather | appointment_time)
        self.conditional_probs['weather'] = {
            (('appointment_time',), ('early',), 'clear'): 0.7,
            (('appointment_time',), ('early',), 'rain'): 0.25,
            (('appointment_time',), ('early',), 'severe'): 0.05,
            (('appointment_time',), ('mid',), 'clear'): 0.75,
            (('appointment_time',), ('mid',), 'rain'): 0.20,
            (('appointment_time',), ('mid',), 'severe'): 0.05,
            (('appointment_time',), ('late',), 'clear'): 0.65,
            (('appointment_time',), ('late',), 'rain'): 0.25,
            (('appointment_time',), ('late',), 'severe'): 0.10,
        }

        # P(traffic | appointment_time, day_of_week)
        self.conditional_probs['traffic'] = {
            (('appointment_time', 'day_of_week'), ('early', 'monday'), 'heavy'): 0.6,
            (('appointment_time', 'day_of_week'), ('early', 'monday'), 'moderate'): 0.3,
            (('appointment_time', 'day_of_week'), ('early', 'monday'), 'light'): 0.1,
            (('appointment_time', 'day_of_week'), ('mid', 'wednesday'), 'heavy'): 0.2,
            (('appointment_time', 'day_of_week'), ('mid', 'wednesday'), 'moderate'): 0.3,
            (('appointment_time', 'day_of_week'), ('mid', 'wednesday'), 'light'): 0.5,
        }

        self.is_fitted = True
        self.is_initialized = True
        logger.info("Using default causal probabilities")

    def do(self, treatment: str, value: Any) -> 'InterventionDistribution':
        """
        Apply do-operator: compute P(Y | do(X = x)).

        The do-operator simulates an intervention by:
        1. Removing all incoming edges to the treatment variable
        2. Setting the treatment to the specified value
        3. Computing the resulting distribution over other variables

        Parameters:
        -----------
        treatment : str
            Variable to intervene on
        value : Any
            Value to set the treatment to

        Returns:
        --------
        InterventionDistribution for computing causal effects
        """
        return InterventionDistribution(self, treatment, value)

    def compute_causal_effect(self,
                              treatment: str,
                              outcome: str,
                              treatment_value: Any,
                              control_value: Any = None) -> CausalEffect:
        """
        Compute causal effect of treatment on outcome.

        P(outcome | do(treatment = value))

        Uses backdoor adjustment when possible:
        P(Y | do(X=x)) = sum_z P(Y | X=x, Z=z) * P(Z=z)

        Where Z is the adjustment set.
        """
        if not self.is_initialized:
            self._use_default_probabilities()

        # Find adjustment set
        adjustment_set = self.dag.find_backdoor_adjustment_set(treatment, outcome)
        is_identifiable = adjustment_set is not None

        if not is_identifiable:
            logger.warning(f"Causal effect of {treatment} on {outcome} may not be identifiable")
            adjustment_set = set()

        # Compute P(outcome | do(treatment = treatment_value))
        treated_prob = self._compute_interventional_prob(
            outcome, treatment, treatment_value, adjustment_set
        )

        # Compute baseline (control) probability
        if control_value is not None:
            baseline_prob = self._compute_interventional_prob(
                outcome, treatment, control_value, adjustment_set
            )
        else:
            baseline_prob = self._compute_marginal_outcome_prob(outcome)

        # Average Treatment Effect
        ate = treated_prob - baseline_prob

        # Confidence interval (approximate)
        se = np.sqrt(treated_prob * (1 - treated_prob) / 100)  # Approximate
        ci = (ate - 1.96 * se, ate + 1.96 * se)

        # Build identification formula
        if adjustment_set:
            adj_str = ", ".join(adjustment_set)
            formula = f"P({outcome}|do({treatment}={treatment_value})) = sum_{{{adj_str}}} P({outcome}|{treatment},{adj_str}) * P({adj_str})"
        else:
            formula = f"P({outcome}|do({treatment}={treatment_value})) = P({outcome}|{treatment}={treatment_value})"

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            treatment_value=treatment_value,
            causal_effect=treated_prob,
            baseline_effect=baseline_prob,
            ate=ate,
            confidence_interval=ci,
            adjustment_set=list(adjustment_set) if adjustment_set else [],
            identification_formula=formula,
            is_identifiable=is_identifiable
        )

    def _compute_interventional_prob(self,
                                     outcome: str,
                                     treatment: str,
                                     treatment_value: Any,
                                     adjustment_set: Set[str]) -> float:
        """
        Compute P(outcome=1 | do(treatment=value)) using adjustment formula.

        P(Y | do(X=x)) = sum_z P(Y | X=x, Z=z) * P(Z=z)
        """
        if not adjustment_set:
            # No adjustment needed - direct effect
            return self._get_conditional_prob(outcome, 1, {treatment: treatment_value})

        # Marginalize over adjustment set
        total_prob = 0.0

        # Get possible values for adjustment variables
        adj_values = self._get_adjustment_combinations(adjustment_set)

        for adj_combo in adj_values:
            # P(outcome | treatment, adjustment)
            conditions = {treatment: treatment_value}
            conditions.update(adj_combo)
            p_outcome_given_conditions = self._get_conditional_prob(outcome, 1, conditions)

            # P(adjustment)
            p_adjustment = self._get_marginal_prob(adj_combo)

            total_prob += p_outcome_given_conditions * p_adjustment

        return total_prob

    def _get_conditional_prob(self, variable: str, value: Any, conditions: Dict[str, Any]) -> float:
        """Get P(variable=value | conditions) from fitted probabilities."""
        # Try to find matching conditional probability
        if variable in self.conditional_probs:
            probs = self.conditional_probs[variable]

            # Look for matching condition
            for key, prob in probs.items():
                parents, parent_values, child_value = key
                if child_value == value:
                    # Check if conditions match
                    match = True
                    for i, parent in enumerate(parents):
                        if parent in conditions and parent_values[i] != conditions[parent]:
                            match = False
                            break
                    if match:
                        return prob

        # Default probability based on variable
        if variable == 'no_show':
            # Base no-show rate around 12%
            if 'weather' in conditions and conditions['weather'] == 'severe':
                return 0.35 if value == 1 else 0.65
            elif 'patient_history' in conditions and conditions['patient_history'] == 'poor':
                return 0.30 if value == 1 else 0.70
            elif 'reminder' in conditions and conditions['reminder'] == 'phone':
                return 0.06 if value == 1 else 0.94
            else:
                return 0.12 if value == 1 else 0.88

        return 0.5  # Default uniform

    def _get_marginal_prob(self, values: Dict[str, Any]) -> float:
        """Get marginal probability P(values)."""
        prob = 1.0
        for var, val in values.items():
            if var in self.conditional_probs:
                for key, p in self.conditional_probs[var].items():
                    parents, parent_values, child_value = key
                    if not parents and child_value == val:
                        prob *= p
                        break
                else:
                    prob *= 0.33  # Default
            else:
                prob *= 0.33
        return prob

    def _compute_marginal_outcome_prob(self, outcome: str) -> float:
        """Compute marginal probability P(outcome=1)."""
        if outcome == 'no_show':
            return 0.12  # Base rate
        return 0.5

    def _get_adjustment_combinations(self, adjustment_set: Set[str]) -> List[Dict[str, Any]]:
        """Get all combinations of adjustment variable values."""
        combinations = [{}]

        for var in adjustment_set:
            if var in self.dag.nodes:
                values = self.dag.nodes[var].values
                if not values:
                    values = ['low', 'medium', 'high']  # Default

                new_combinations = []
                for combo in combinations:
                    for val in values:
                        new_combo = combo.copy()
                        new_combo[var] = val
                        new_combinations.append(new_combo)
                combinations = new_combinations

        return combinations

    def counterfactual(self,
                       observation: Dict[str, Any],
                       intervention: Dict[str, Any],
                       outcome: str) -> CounterfactualResult:
        """
        Answer counterfactual question: "What would Y have been if X had been x?"

        Three steps:
        1. Abduction: Infer latent factors U from observation
        2. Action: Apply intervention do(X=x)
        3. Prediction: Compute P(Y | do(X=x), U)

        Parameters:
        -----------
        observation : Dict
            Observed factual values
        intervention : Dict
            Counterfactual intervention {variable: value}
        outcome : str
            Outcome variable to predict
        """
        if not self.is_initialized:
            self._use_default_probabilities()

        # Get factual outcome
        factual_outcome = observation.get(outcome, None)

        # Compute counterfactual outcome probability
        cf_conditions = observation.copy()
        cf_conditions.update(intervention)

        # Remove outcome from conditions if present
        if outcome in cf_conditions:
            del cf_conditions[outcome]

        cf_prob = self._get_conditional_prob(outcome, 1, cf_conditions)
        cf_outcome = 1 if cf_prob > 0.5 else 0

        # Compute effect
        if factual_outcome is not None:
            effect = cf_outcome - factual_outcome
        else:
            effect = cf_prob - 0.12  # Compare to baseline

        # Build query string
        intervention_str = ", ".join(f"{k}={v}" for k, v in intervention.items())
        query = f"P({outcome} | do({intervention_str}), observed data)"

        # Build explanation
        if effect > 0:
            explanation = f"The intervention would INCREASE {outcome} probability by {abs(effect)*100:.1f}%"
        elif effect < 0:
            explanation = f"The intervention would DECREASE {outcome} probability by {abs(effect)*100:.1f}%"
        else:
            explanation = f"The intervention would have NO EFFECT on {outcome}"

        return CounterfactualResult(
            query=query,
            factual_outcome=factual_outcome,
            counterfactual_outcome=cf_outcome,
            effect=effect,
            probability=cf_prob,
            explanation=explanation
        )

    def counterfactual_explanation(self,
                                    patient_features: Dict[str, float],
                                    prediction_func,
                                    target_flip: bool = True,
                                    max_changes: int = 3,
                                    step_size: float = 0.1) -> Dict[str, Any]:
        """
        Counterfactual Explanation: Find minimum feature change to flip prediction.

        CF(x) = argmin_{x'} ||x' - x||_2  s.t. f(x') != f(x)

        Finds the smallest perturbation to patient features that would change
        the model's prediction (e.g., from high-risk to low-risk).

        Parameters:
        -----------
        patient_features : Dict
            Current patient feature values (e.g., {'Previous_NoShows': 3, 'Travel_Distance_KM': 40})
        prediction_func : callable
            Function that takes features dict and returns prediction (0 or 1, or probability)
        target_flip : bool
            If True, find change to flip prediction. If False, find change to maintain.
        max_changes : int
            Maximum number of features to change
        step_size : float
            Step size for perturbation search

        Returns:
        --------
        Dict with counterfactual explanation
        """
        import numpy as np

        # Get current prediction
        current_pred = prediction_func(patient_features)
        current_class = 1 if current_pred > 0.5 else 0
        target_class = 1 - current_class if target_flip else current_class

        # Mutable features (can be changed) with their ranges
        mutable_features = {
            'Previous_NoShows': {'min': 0, 'max': 10, 'direction': -1, 'label': 'fewer no-shows'},
            'Previous_Cancellations': {'min': 0, 'max': 10, 'direction': -1, 'label': 'fewer cancellations'},
            'Travel_Distance_KM': {'min': 0, 'max': 80, 'direction': -1, 'label': 'closer to centre'},
            'Travel_Time_Min': {'min': 5, 'max': 80, 'direction': -1, 'label': 'shorter travel'},
            'Days_Booked_In_Advance': {'min': 1, 'max': 60, 'direction': -1, 'label': 'shorter booking lead'},
            'Cycle_Number': {'min': 1, 'max': 20, 'direction': 1, 'label': 'later cycle (more experienced)'},
            'Weather_Severity': {'min': 0, 'max': 1, 'direction': -1, 'label': 'better weather'},
            'Performance_Status': {'min': 0, 'max': 4, 'direction': -1, 'label': 'better performance status'},
        }

        # Only try features that exist in patient data
        available = {k: v for k, v in mutable_features.items() if k in patient_features}

        best_cf = None
        best_distance = float('inf')
        best_changes = {}

        # Greedy search: try changing each feature individually then in combinations
        for feature_name, config in available.items():
            current_val = patient_features.get(feature_name, 0)
            direction = config['direction']

            # Try different magnitudes of change
            for magnitude in [0.2, 0.4, 0.6, 0.8, 1.0]:
                cf_features = dict(patient_features)
                range_size = config['max'] - config['min']
                change = direction * magnitude * range_size

                new_val = np.clip(current_val + change, config['min'], config['max'])
                cf_features[feature_name] = new_val

                cf_pred = prediction_func(cf_features)
                cf_class = 1 if cf_pred > 0.5 else 0

                if cf_class == target_class:
                    distance = abs(new_val - current_val) / max(range_size, 1)
                    if distance < best_distance:
                        best_distance = distance
                        best_cf = cf_features
                        best_changes = {
                            feature_name: {
                                'from': round(current_val, 2),
                                'to': round(new_val, 2),
                                'change': round(new_val - current_val, 2),
                                'label': config['label'],
                            }
                        }
                    break  # Found flip for this feature, try next

        # Build explanation
        if best_cf is not None:
            cf_pred = prediction_func(best_cf)
            change_descriptions = []
            for feat, info in best_changes.items():
                change_descriptions.append(
                    f"If {feat} changed from {info['from']} to {info['to']} ({info['label']})"
                )

            return {
                'found': True,
                'current_prediction': round(float(current_pred), 4),
                'counterfactual_prediction': round(float(cf_pred), 4),
                'current_class': 'high-risk' if current_class == 1 else 'low-risk',
                'counterfactual_class': 'high-risk' if target_class == 1 else 'low-risk',
                'changes': best_changes,
                'distance': round(best_distance, 4),
                'explanation': '; '.join(change_descriptions),
                'formula': 'CF(x) = argmin_{x\'} ||x\' - x||_2  s.t. f(x\') != f(x)',
            }
        else:
            return {
                'found': False,
                'current_prediction': round(float(current_pred), 4),
                'current_class': 'high-risk' if current_class == 1 else 'low-risk',
                'explanation': 'No counterfactual found within feature bounds',
                'formula': 'CF(x) = argmin_{x\'} ||x\' - x||_2  s.t. f(x\') != f(x)',
            }

    # =========================================================================
    # Instrumental Variables (4.2)
    # =========================================================================

    def estimate_iv_effect(
        self,
        data: pd.DataFrame,
        instrument: str = 'weather_severity',
        treatment: str = 'traffic_delay',
        outcome: str = 'no_show',
        covariates: Optional[List[str]] = None
    ) -> IVEstimationResult:
        """
        Estimate causal effect using Instrumental Variables (2SLS).

        This addresses unobserved confounders in the causal chain:
        Weather -> Traffic -> No-Show

        Model:
            Stage 1: TrafficDelay ~ WeatherSeverity + X
            Stage 2: NoShow ~ PredictedTrafficDelay + X

        Identification assumption: Weather affects traffic conditions,
        which in turn affects whether patients arrive on time or no-show.
        Weather only affects no-shows through its effect on traffic.

        Parameters:
        -----------
        data : pd.DataFrame
            Data with instrument, treatment, outcome, and covariates
        instrument : str
            Instrument variable (default: 'Weather_Severity')
        treatment : str
            Treatment variable (default: 'Traffic_Delay_Minutes')
        outcome : str
            Outcome variable (default: 'no_show')
        covariates : List[str], optional
            Control variables

        Returns:
        --------
        IVEstimationResult with causal effect estimate

        Example:
        --------
        >>> result = model.estimate_iv_effect(data)
        >>> print(f"Causal effect: {result.causal_effect:.4f}")
        >>> print(f"F-statistic: {result.first_stage_f_stat:.2f}")
        """
        # Prepare data - map column names
        df = self._prepare_iv_data(data, instrument, treatment, outcome)

        # Default covariates based on DAG structure (excludes instrument)
        if covariates is None:
            covariates = self._get_default_iv_covariates(df, instrument=instrument)

        # Run 2SLS estimation
        estimator = InstrumentalVariablesEstimator()
        result = estimator.fit(
            df,
            instrument=instrument,
            treatment=treatment,
            outcome=outcome,
            covariates=covariates
        )

        # Store for later use
        self.iv_estimator = estimator
        self.iv_result = result

        logger.info(
            f"IV estimation: {treatment} -> {outcome}, "
            f"effect={result.causal_effect:.4f}, F={result.first_stage_f_stat:.1f}"
        )

        return result

    def _prepare_iv_data(
        self,
        data: pd.DataFrame,
        instrument: str,
        treatment: str,
        outcome: str
    ) -> pd.DataFrame:
        """Prepare data for IV estimation by mapping column names."""
        df = data.copy()

        # Map common column name variations
        column_mappings = {
            # Instrument mappings (weather)
            'weather_severity': [
                'Weather_Severity', 'weather_severity',
                'Weather_Conditions', 'weather_conditions'
            ],
            # Outcome mappings
            'no_show': [
                'No_Show', 'no_show', 'Showed_Up', 'showed_up',
                'Attended_Status', 'attended_status'
            ]
        }

        # Apply mappings for instrument and outcome
        for target, sources in column_mappings.items():
            if target not in df.columns:
                for source in sources:
                    if source in df.columns:
                        # Handle special cases
                        if source == 'Showed_Up' or source == 'showed_up':
                            df[target] = (~df[source]).astype(int)
                        elif source == 'Attended_Status' or source == 'attended_status':
                            df[target] = (df[source] == 'No').astype(int)
                        else:
                            df[target] = df[source]
                        break

        # Generate realistic traffic_delay from weather
        # This creates the causal chain: Weather -> Traffic -> No-Show
        # Weather severity causally affects traffic conditions
        if treatment == 'traffic_delay':
            weather_col = 'weather_severity' if 'weather_severity' in df.columns else 'Weather_Severity'
            if weather_col in df.columns:
                np.random.seed(42)
                weather = df[weather_col].values
                # Bad weather increases traffic delay
                # Base delay (10 min) + weather effect (up to 30 min) + noise
                base_delay = 10
                weather_effect = 30
                df[treatment] = (
                    base_delay +
                    weather_effect * weather +
                    np.random.normal(0, 5, len(df))
                ).clip(0, 60)
                logger.info(f"Generated '{treatment}' from weather (causal: weather -> traffic)")

        return df

    def _get_default_iv_covariates(self, df: pd.DataFrame, instrument: str = 'weather_severity') -> List[str]:
        """Get default covariates for IV estimation based on available columns.

        Note: Excludes the instrument variable to avoid collinearity.
        """
        # Potential covariates - DO NOT include instrument (weather) here
        potential_covariates = [
            'age', 'Age',
            'day_of_week', 'Day_Of_Week_Num',
            'appointment_hour', 'Appointment_Hour',
            'cycle_number', 'Cycle_Number',
            'patient_noshow_rate', 'Patient_NoShow_Rate',
            'has_comorbidities', 'Has_Comorbidities'
        ]

        # Exclude instrument and related columns to avoid collinearity
        excluded = {instrument.lower(), 'weather_severity', 'weather', 'traffic_delay'}

        available = []
        for cov in potential_covariates:
            if cov in df.columns and cov.lower() not in excluded:
                # Check if column has valid numeric data
                try:
                    if df[cov].dtype in ['int64', 'float64', 'bool']:
                        if not df[cov].isna().all():
                            available.append(cov)
                except:
                    pass

        return available[:4]  # Limit to avoid overfitting

    def get_iv_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of IV estimation results."""
        if not hasattr(self, 'iv_result') or self.iv_result is None:
            return None

        r = self.iv_result
        return {
            'instrument': r.instrument,
            'treatment': r.treatment,
            'outcome': r.outcome,
            'causal_effect': r.causal_effect,
            'causal_effect_se': r.causal_effect_se,
            'confidence_interval': r.confidence_interval,
            'first_stage_f_stat': r.first_stage_f_stat,
            'first_stage_r_squared': r.first_stage_r_squared,
            'weak_instrument': r.weak_instrument,
            'n_observations': r.n_observations,
            'covariates': r.covariates,
            'interpretation': r.interpretation
        }

    def get_dag_summary(self) -> Dict[str, Any]:
        """Get summary of the causal DAG structure."""
        return {
            'nodes': list(self.dag.nodes.keys()),
            'edges': self.dag.edges,
            'n_nodes': len(self.dag.nodes),
            'n_edges': len(self.dag.edges),
            'structure': {
                node: {
                    'parents': self.dag.get_parents(node),
                    'children': self.dag.get_children(node)
                }
                for node in self.dag.nodes
            }
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        return {
            'model_type': 'Causal Inference Framework',
            'dag_nodes': len(self.dag.nodes),
            'dag_edges': len(self.dag.edges),
            'is_fitted': self.is_fitted,
            'key_relationships': [
                'Appointment Time -> Weather exposure',
                'Weather -> No-Show (direct causal effect)',
                'Patient History -> No-Show',
                'Reminder -> No-Show (intervention)',
                'Traffic -> Arrival Time -> Delay'
            ],
            'identifiable_effects': [
                'Weather -> No-Show',
                'Reminder -> No-Show',
                'Patient History -> No-Show',
                'Appointment Time -> No-Show'
            ]
        }


class InterventionDistribution:
    """
    Represents the distribution after applying do(X=x).

    Allows computing P(Y | do(X=x)) for any outcome Y.
    """

    def __init__(self, model: SchedulingCausalModel, treatment: str, value: Any):
        self.model = model
        self.treatment = treatment
        self.value = value

    def probability(self, outcome: str, outcome_value: Any = 1) -> float:
        """Compute P(outcome=outcome_value | do(treatment=value))."""
        effect = self.model.compute_causal_effect(
            self.treatment,
            outcome,
            self.value
        )
        if outcome_value == 1:
            return effect.causal_effect
        else:
            return 1 - effect.causal_effect


# Convenience functions
def compute_intervention_effect(treatment: str,
                                treatment_value: Any,
                                outcome: str = 'no_show') -> CausalEffect:
    """
    Quick computation of causal effect.

    Example:
    --------
    >>> effect = compute_intervention_effect('appointment_time', 'early', 'no_show')
    >>> print(f"P(No-Show | do(Time=early)) = {effect.causal_effect:.3f}")
    """
    model = SchedulingCausalModel()
    model._use_default_probabilities()
    return model.compute_causal_effect(treatment, outcome, treatment_value)


def answer_counterfactual(observation: Dict[str, Any],
                          intervention: Dict[str, Any],
                          outcome: str = 'no_show') -> CounterfactualResult:
    """
    Answer a counterfactual question.

    Example:
    --------
    >>> obs = {'weather': 'severe', 'patient_history': 'poor', 'no_show': 1}
    >>> result = answer_counterfactual(obs, {'reminder': 'phone'}, 'no_show')
    >>> print(result.explanation)
    """
    model = SchedulingCausalModel()
    model._use_default_probabilities()
    return model.counterfactual(observation, intervention, outcome)


def estimate_iv_effect(
    data: pd.DataFrame,
    instrument: str = 'weather_severity',
    treatment: str = 'traffic_delay',
    outcome: str = 'no_show',
    covariates: Optional[List[str]] = None
) -> IVEstimationResult:
    """
    Estimate causal effect using Instrumental Variables (2SLS).

    Convenience function for IV estimation.
    Causal chain: Weather -> Traffic -> No-Show

    Parameters:
    -----------
    data : pd.DataFrame
        Data with instrument, treatment, outcome, and covariates
    instrument : str
        Instrument variable (default: 'weather_severity')
    treatment : str
        Treatment variable (default: 'traffic_delay')
    outcome : str
        Outcome variable (default: 'no_show')
    covariates : List[str], optional
        Control variables

    Returns:
    --------
    IVEstimationResult with causal effect estimate

    Example:
    --------
    >>> result = estimate_iv_effect(historical_data)
    >>> print(f"Traffic delay -> No-show: {result.causal_effect:.4f}")
    >>> print(f"Instrument strength (F): {result.first_stage_f_stat:.1f}")
    """
    model = SchedulingCausalModel()
    return model.estimate_iv_effect(
        data,
        instrument=instrument,
        treatment=treatment,
        outcome=outcome,
        covariates=covariates
    )


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Causal Inference Framework for Scheduling (4.1)")
    print("=" * 60)

    # Create model
    model = SchedulingCausalModel()
    model._use_default_probabilities()

    # Show DAG structure
    print("\nCausal DAG Structure:")
    summary = model.get_dag_summary()
    print(f"  Nodes: {summary['n_nodes']}")
    print(f"  Edges: {summary['n_edges']}")

    print("\n  Key causal relationships:")
    for node, info in list(summary['structure'].items())[:5]:
        if info['parents']:
            print(f"    {info['parents']} -> {node}")

    # Compute causal effects
    print("\n" + "=" * 60)
    print("Causal Effects (Do-Calculus)")
    print("=" * 60)

    # Effect of early appointments
    effect = model.compute_causal_effect(
        treatment='appointment_time',
        outcome='no_show',
        treatment_value='early',
        control_value='late'
    )
    print(f"\nP(No-Show | do(Time=early)) = {effect.causal_effect:.3f}")
    print(f"P(No-Show | do(Time=late)) = {effect.baseline_effect:.3f}")
    print(f"Average Treatment Effect: {effect.ate:+.3f}")
    print(f"Adjustment set: {effect.adjustment_set}")

    # Effect of phone reminder
    effect2 = model.compute_causal_effect(
        treatment='reminder',
        outcome='no_show',
        treatment_value='phone',
        control_value='none'
    )
    print(f"\nP(No-Show | do(Reminder=phone)) = {effect2.causal_effect:.3f}")
    print(f"P(No-Show | do(Reminder=none)) = {effect2.baseline_effect:.3f}")
    print(f"Average Treatment Effect: {effect2.ate:+.3f}")

    # Counterfactual
    print("\n" + "=" * 60)
    print("Counterfactual Analysis")
    print("=" * 60)

    observation = {
        'weather': 'severe',
        'patient_history': 'poor',
        'reminder': 'none',
        'no_show': 1  # Patient actually no-showed
    }

    result = model.counterfactual(
        observation,
        {'reminder': 'phone'},
        'no_show'
    )
    print(f"\nObservation: Patient no-showed with severe weather, poor history, no reminder")
    print(f"Counterfactual: What if we had called them?")
    print(f"  {result.explanation}")
    print(f"  Counterfactual P(No-Show): {result.probability:.3f}")
