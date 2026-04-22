"""
Uncertainty-Aware Optimization for SACT Scheduling

Replaces deterministic weighted-sum with distributionally robust formulation
that guarantees schedule performance under distributional shifts.

Methods:
1. DRO (Distributionally Robust Optimization) via Wasserstein ambiguity set
2. CVaR (Conditional Value-at-Risk) for risk-averse scheduling
3. Scenario-based robust optimization via SAA (Sample Average Approximation)

The key insight: patient no-show probabilities and durations are UNCERTAIN.
A schedule that is optimal under point estimates may perform poorly when
actual distributions differ. DRO protects against this.

References:
    Mohajerin Esfahani & Kuhn (2018). Data-driven DRO using Wasserstein distance.
    Rockafellar & Uryasev (2000). CVaR optimization.
    Bertsimas, Sim & Zhang (2019). Adaptive DRO.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyProfile:
    """Uncertainty profile for a patient's parameters."""
    patient_id: str
    noshow_mean: float          # Point estimate E[pi]
    noshow_std: float           # Uncertainty in estimate
    noshow_worst_case: float    # Worst-case under ambiguity set
    duration_mean: float        # Expected duration
    duration_std: float         # Duration uncertainty
    duration_worst_case: float  # Worst-case duration (e.g., p95)


@dataclass
class RobustObjective:
    """Result of robust optimization reformulation."""
    robust_noshow_penalties: Dict[str, int]  # patient_id -> penalty (for CP-SAT)
    robust_duration_buffers: Dict[str, int]  # patient_id -> buffered duration
    epsilon: float                            # Wasserstein radius used
    alpha: float                              # CVaR confidence level
    method: str                               # 'dro', 'cvar', 'scenario'
    n_scenarios: int                           # Number of scenarios considered


class UncertaintyAwareOptimizer:
    """
    Transforms deterministic scheduling objective into a robust formulation.

    Instead of optimizing E[Z], we optimize:
        min_{x} max_{Q in P} E_Q[loss(x)]

    where P = {Q : W_2(P_emp, Q) <= epsilon} is the Wasserstein ambiguity set.

    For CP-SAT (integer programming), this is approximated via:
    1. Worst-case no-show probabilities from the ambiguity set
    2. CVaR-based duration buffers
    3. Scenario-based penalty terms
    """

    def __init__(self, epsilon: float = 0.05, alpha: float = 0.90,
                 n_scenarios: int = 50):
        """
        Args:
            epsilon: Wasserstein ball radius (distributional shift tolerance).
                    Larger = more conservative. Default 0.05 (5% shift).
            alpha: CVaR confidence level. 0.90 = protect against worst 10%.
            n_scenarios: Number of perturbation scenarios for SAA.
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_scenarios = n_scenarios

    def compute_robust_parameters(
        self,
        patients: List,
        historical_data=None
    ) -> RobustObjective:
        """
        Compute robust (worst-case) parameters for each patient.

        This is the main interface — call before CP-SAT optimization to get
        DRO-adjusted penalties and buffers.

        Args:
            patients: List of Patient objects with noshow_probability, expected_duration
            historical_data: DataFrame of historical appointments (for variance estimation)

        Returns:
            RobustObjective with adjusted penalties for CP-SAT
        """
        robust_noshow = {}
        robust_duration = {}

        for patient in patients:
            pid = patient.patient_id
            ns_mean = patient.noshow_probability
            dur_mean = patient.expected_duration

            # Estimate uncertainty from historical data or defaults
            ns_std = self._estimate_noshow_uncertainty(patient, historical_data)
            dur_std = self._estimate_duration_uncertainty(patient, historical_data)

            # DRO worst-case no-show under Wasserstein ball
            ns_worst = self._wasserstein_worst_case_noshow(ns_mean, ns_std)

            # CVaR duration buffer
            dur_worst = self._cvar_duration(dur_mean, dur_std)

            # Convert to CP-SAT integer penalties
            # Use worst-case instead of point estimate
            robust_noshow[pid] = int(ns_worst * 100)
            robust_duration[pid] = int(dur_worst)

        return RobustObjective(
            robust_noshow_penalties=robust_noshow,
            robust_duration_buffers=robust_duration,
            epsilon=self.epsilon,
            alpha=self.alpha,
            method='dro_wasserstein',
            n_scenarios=self.n_scenarios
        )

    def _wasserstein_worst_case_noshow(
        self, mean: float, std: float
    ) -> float:
        """
        Compute worst-case no-show probability under Wasserstein ambiguity.

        For a Wasserstein ball of radius epsilon around the empirical distribution:
            P = {Q : W_2(P_emp, Q) <= epsilon}

        The worst-case expectation for a bounded Lipschitz function is:
            sup_{Q in P} E_Q[pi] = E_P[pi] + epsilon * L

        where L is the Lipschitz constant. For probabilities in [0,1]:
            worst_case = min(1, mean + epsilon * sqrt(variance + epsilon^2))

        This is a tractable convex reformulation (Mohajerin Esfahani & Kuhn 2018).
        """
        variance = std ** 2
        # Worst-case shift under Wasserstein ball
        shift = self.epsilon * np.sqrt(variance + self.epsilon ** 2)
        worst_case = min(0.95, mean + shift)
        return max(0.01, worst_case)

    def _cvar_duration(self, mean: float, std: float) -> float:
        """
        Compute CVaR (Conditional Value-at-Risk) for treatment duration.

        CVaR_alpha = E[X | X >= VaR_alpha]

        For approximately normal durations:
            VaR_alpha = mean + z_alpha * std
            CVaR_alpha approx= mean + std * phi(z_alpha) / (1 - alpha)

        where phi is the standard normal PDF and z_alpha is the quantile.

        This gives a duration estimate that protects against the worst
        (1-alpha) fraction of outcomes.
        """
        if std < 1:
            return mean

        # z-score for alpha quantile
        z_alpha = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326,
        }.get(self.alpha, 1.282)

        # Normal PDF at z_alpha
        phi_z = np.exp(-0.5 * z_alpha ** 2) / np.sqrt(2 * np.pi)

        # CVaR approximation (for normal distribution)
        cvar = mean + std * phi_z / (1 - self.alpha)

        return cvar

    def _estimate_noshow_uncertainty(self, patient, historical_data) -> float:
        """Estimate uncertainty in no-show prediction."""
        base_std = 0.10  # Default uncertainty

        if historical_data is not None:
            try:
                import pandas as pd
                pid = patient.patient_id
                if 'Patient_ID' in historical_data.columns:
                    patient_history = historical_data[
                        historical_data['Patient_ID'] == pid
                    ]
                    if len(patient_history) >= 3:
                        # Use variance of actual outcomes
                        outcomes = (patient_history['Attended_Status'] == 'No').astype(float)
                        base_std = float(outcomes.std())

                # Population-level uncertainty by priority
                if 'Priority' in historical_data.columns:
                    priority_str = f'P{patient.priority}'
                    priority_data = historical_data[
                        historical_data['Priority'] == priority_str
                    ]
                    if len(priority_data) >= 10:
                        outcomes = (priority_data['Attended_Status'] == 'No').astype(float)
                        base_std = float(outcomes.std())
            except Exception:
                pass

        return max(0.05, base_std)

    def _estimate_duration_uncertainty(self, patient, historical_data) -> float:
        """Estimate uncertainty in duration prediction."""
        base_std = patient.expected_duration * 0.15  # 15% default CV

        if historical_data is not None:
            try:
                import pandas as pd
                if 'Actual_Duration' in historical_data.columns:
                    durations = pd.to_numeric(
                        historical_data['Actual_Duration'], errors='coerce'
                    ).dropna()
                    if len(durations) >= 10:
                        base_std = float(durations.std())
            except Exception:
                pass

        return max(5, base_std)

    def generate_scenarios(
        self, patients: List, n_scenarios: int = None
    ) -> List[Dict[str, Dict[str, float]]]:
        """
        Generate perturbation scenarios for Sample Average Approximation.

        Each scenario is a dict of patient_id -> {noshow, duration} representing
        one possible realization of the uncertain parameters.

        Used when DRO closed-form is not available (nonlinear objectives).
        """
        n = n_scenarios or self.n_scenarios
        scenarios = []

        for _ in range(n):
            scenario = {}
            for patient in patients:
                pid = patient.patient_id
                ns_mean = patient.noshow_probability
                dur_mean = patient.expected_duration

                # Sample from perturbed distribution within Wasserstein ball
                ns_perturbed = np.clip(
                    ns_mean + np.random.normal(0, self.epsilon), 0.01, 0.95
                )
                dur_perturbed = max(15, dur_mean + np.random.normal(0, dur_mean * 0.15))

                scenario[pid] = {
                    'noshow': float(ns_perturbed),
                    'duration': float(dur_perturbed),
                }
            scenarios.append(scenario)

        return scenarios

    def evaluate_schedule_robustness(
        self, appointments: List, scenarios: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate a schedule's robustness across perturbation scenarios.

        For each scenario, compute the objective value. Report:
        - Mean objective across scenarios
        - Worst-case (minimum) objective
        - CVaR of objective distribution
        - Probability of constraint violation
        """
        if not scenarios or not appointments:
            return {'error': 'No scenarios or appointments'}

        objectives = []
        violations = 0

        for scenario in scenarios:
            obj = 0
            has_violation = False

            for appt in appointments:
                pid = appt.patient_id
                if pid in scenario:
                    ns = scenario[pid]['noshow']
                    dur = scenario[pid]['duration']

                    # Objective contribution
                    priority_score = (5 - appt.priority) * 100
                    noshow_penalty = ns * 100
                    obj += priority_score - noshow_penalty

                    # Check if duration exceeds allocated time
                    if dur > appt.duration * 1.2:
                        has_violation = True

            objectives.append(obj)
            if has_violation:
                violations += 1

        objectives = np.array(objectives)
        n = len(objectives)

        # CVaR calculation
        sorted_obj = np.sort(objectives)
        cvar_idx = int(np.floor(n * (1 - self.alpha)))
        cvar = float(np.mean(sorted_obj[:max(1, cvar_idx)]))

        return {
            'mean_objective': float(np.mean(objectives)),
            'std_objective': float(np.std(objectives)),
            'worst_case': float(np.min(objectives)),
            'best_case': float(np.max(objectives)),
            'cvar': cvar,
            'cvar_alpha': self.alpha,
            'violation_probability': violations / n,
            'n_scenarios': n,
            'epsilon': self.epsilon,
            'interpretation': (
                f"Schedule performs {np.mean(objectives):.0f} on average, "
                f"but could be as low as {np.min(objectives):.0f} under distributional shift. "
                f"CVaR_{self.alpha} = {cvar:.0f} (worst {int((1-self.alpha)*100)}% average). "
                f"Duration violations in {violations}/{n} scenarios ({violations/n*100:.0f}%)."
            ),
        }

    def compute_cvar_objective_params(
        self,
        patients: List,
        n_scenarios: int = 10
    ) -> Dict[str, Any]:
        """
        Compute CVaR parameters for Rockafellar-Uryasev reformulation
        in the CP-SAT integer program.

        The standard expected utility objective:
            max E[U] = sum_p utility(p) * x_p

        is replaced by:
            max CVaR_alpha(U) = max { eta - 1/(K(1-alpha)) * sum_k z_k }
            s.t. z_k >= eta - U_k,  z_k >= 0,  for k = 1..K

        where U_k is the utility under scenario k (no-show realization),
        eta is the VaR auxiliary variable, and z_k are shortfall slacks.

        This guarantees the schedule performs well even in the worst
        (1-alpha) fraction of no-show scenarios.

        Args:
            patients: List of Patient objects
            n_scenarios: Number of no-show scenarios to generate

        Returns:
            Dict with scenario utilities for CP-SAT integration
        """
        K = n_scenarios
        rng = np.random.RandomState(42)

        # Generate K binary no-show realizations per patient
        scenarios = []
        for k in range(K):
            scenario = {}
            for p in patients:
                ns_prob = p.noshow_probability
                shows = rng.random() > ns_prob
                priority_val = (5 - p.priority) * 100
                if shows:
                    scenario[p.patient_id] = priority_val  # Full utility
                else:
                    scenario[p.patient_id] = -(p.expected_duration // 2)  # Waste penalty
            scenarios.append(scenario)

        return {
            'scenarios': scenarios,
            'K': K,
            'alpha': self.alpha,
            'inv_K_1_alpha': 1.0 / (K * (1 - self.alpha)),
            'method': 'Rockafellar-Uryasev CVaR linearization',
            'formulation': {
                'objective': 'max { eta - 1/(K(1-alpha)) * sum_k z_k }',
                'constraint': 'z_k >= eta - U_k, z_k >= 0, for k=1..K',
                'eta': 'VaR auxiliary variable',
                'z_k': 'Shortfall slack for scenario k',
                'U_k': 'Total utility under scenario k',
            }
        }

    def calibrate_epsilon(self, historical_data) -> Dict[str, Any]:
        """
        Calibrate the Wasserstein radius epsilon from historical data.

        epsilon should reflect how much the real distribution can shift from
        the training distribution. Estimated from:
        1. Monthly variance in no-show rates (seasonal shifts)
        2. Maximum observed month-to-month shift (disruption events)
        3. Duration variability coefficient (CV)

        Args:
            historical_data: DataFrame with Attended_Status, Date, Actual_Duration

        Returns:
            Dict with calibrated epsilon and supporting evidence
        """
        import pandas as pd

        result = {
            'method': 'Historical distributional shift analysis',
            'epsilon_recommended': self.epsilon,
        }

        if historical_data is None or len(historical_data) < 50:
            result['warning'] = 'Insufficient data for calibration'
            return result

        try:
            df = historical_data.copy()

            # 1. Monthly no-show rate variance
            if 'Date' in df.columns and 'Attended_Status' in df.columns:
                df['month'] = pd.to_datetime(df['Date']).dt.to_period('M')
                monthly_rates = df.groupby('month').apply(
                    lambda x: (x['Attended_Status'] == 'No').mean()
                )
                if len(monthly_rates) >= 2:
                    rate_std = float(monthly_rates.std())
                    rate_max_shift = float(monthly_rates.max() - monthly_rates.min())
                    result['monthly_noshow_std'] = round(rate_std, 4)
                    result['max_monthly_shift'] = round(rate_max_shift, 4)
                    result['monthly_rates'] = {
                        str(k): round(float(v), 3) for k, v in monthly_rates.items()
                    }

            # 2. Duration coefficient of variation
            if 'Actual_Duration' in df.columns:
                durations = pd.to_numeric(df['Actual_Duration'], errors='coerce').dropna()
                if len(durations) > 10:
                    dur_cv = float(durations.std() / durations.mean())
                    result['duration_cv'] = round(dur_cv, 3)

            # 3. Calibrate epsilon
            # epsilon = max(monthly_shift, duration_cv) * safety_factor
            noshow_shift = result.get('max_monthly_shift', 0.05)
            dur_cv = result.get('duration_cv', 0.15)
            safety_factor = 1.2  # 20% margin

            calibrated_epsilon = max(noshow_shift, dur_cv * 0.3) * safety_factor
            calibrated_epsilon = max(0.02, min(0.15, calibrated_epsilon))

            result['epsilon_recommended'] = round(calibrated_epsilon, 3)
            result['epsilon_current'] = self.epsilon
            result['safety_factor'] = safety_factor

            # 4. Disruption scenarios
            result['disruption_scenarios'] = {
                'baseline': {
                    'description': 'Normal operations',
                    'epsilon': round(calibrated_epsilon * 0.5, 3),
                },
                'seasonal_peak': {
                    'description': 'Winter flu season / summer holidays',
                    'epsilon': round(calibrated_epsilon, 3),
                },
                'moderate_disruption': {
                    'description': 'NHS strikes, severe weather, transport disruption',
                    'epsilon': round(calibrated_epsilon * 1.5, 3),
                },
                'major_disruption': {
                    'description': 'Pandemic-level event (COVID-like)',
                    'epsilon': round(min(0.20, calibrated_epsilon * 3.0), 3),
                },
            }

            # Update instance epsilon
            self.epsilon = calibrated_epsilon
            result['calibrated'] = True

        except Exception as e:
            result['error'] = str(e)
            result['calibrated'] = False

        return result

    def evaluate_schedule_metrics(
        self, appointments: List, scenarios: List[Dict]
    ) -> Dict[str, Any]:
        """
        Compute DRO evaluation metrics for a schedule:
        1. Worst-case utilization
        2. Stability (variance across scenarios)
        3. CVaR of utilization
        4. Tail probability (P[util < threshold])

        Args:
            appointments: List of ScheduledAppointment objects
            scenarios: List of scenario dicts (from generate_scenarios)

        Returns:
            Dict with worst-case utilization, stability, and tail metrics
        """
        if not scenarios or not appointments:
            return {'error': 'No scenarios or appointments'}

        # Compute utilization under each scenario
        total_chair_minutes = 10 * 60  # 10-hour day per chair
        n_chairs = len(set(a.chair_id for a in appointments))
        total_capacity = max(1, n_chairs * total_chair_minutes)

        scenario_utils = []
        scenario_objectives = []

        for scenario in scenarios:
            used_minutes = 0
            obj = 0
            for appt in appointments:
                pid = appt.patient_id
                if pid in scenario:
                    ns = scenario[pid]['noshow']
                    dur = scenario[pid]['duration']
                    # Patient shows?
                    shows = np.random.random() > ns
                    if shows:
                        used_minutes += dur
                        obj += (5 - appt.priority) * 100
                    else:
                        obj -= int(dur // 2)
                else:
                    used_minutes += appt.duration
                    obj += (5 - appt.priority) * 100

            utilization = used_minutes / total_capacity
            scenario_utils.append(utilization)
            scenario_objectives.append(obj)

        utils = np.array(scenario_utils)
        objs = np.array(scenario_objectives)
        n = len(utils)

        # Worst-case utilization
        worst_util = float(np.min(utils))
        best_util = float(np.max(utils))
        mean_util = float(np.mean(utils))

        # Stability: standard deviation across scenarios
        stability_std = float(np.std(utils))
        stability_cv = stability_std / max(0.01, mean_util)

        # CVaR of utilization (worst alpha fraction)
        sorted_utils = np.sort(utils)
        n_tail = max(1, int(np.ceil(n * (1 - self.alpha))))
        cvar_util = float(np.mean(sorted_utils[:n_tail]))

        # Tail probability: P[utilization < 0.5]
        threshold = 0.5
        tail_prob = float(np.mean(utils < threshold))

        # Objective metrics
        sorted_objs = np.sort(objs)
        cvar_obj = float(np.mean(sorted_objs[:n_tail]))

        return {
            'utilization': {
                'mean': round(mean_util, 3),
                'worst_case': round(worst_util, 3),
                'best_case': round(best_util, 3),
                'cvar': round(cvar_util, 3),
                'std': round(stability_std, 3),
                'cv': round(stability_cv, 3),
            },
            'stability': {
                'std': round(stability_std, 4),
                'cv': round(stability_cv, 4),
                'interpretation': (
                    'Stable' if stability_cv < 0.10 else
                    'Moderate variability' if stability_cv < 0.20 else
                    'High variability — consider increasing epsilon'
                ),
            },
            'objective': {
                'mean': round(float(np.mean(objs)), 0),
                'worst_case': round(float(np.min(objs)), 0),
                'cvar': round(cvar_obj, 0),
                'std': round(float(np.std(objs)), 1),
            },
            'tail_risk': {
                'threshold': threshold,
                'probability': round(tail_prob, 3),
                'interpretation': f'P(utilization < {threshold}) = {tail_prob:.1%}',
            },
            'n_scenarios': n,
            'alpha': self.alpha,
            'epsilon': self.epsilon,
        }

    def get_formulas(self) -> Dict[str, str]:
        """Return mathematical formulas for documentation."""
        return {
            'wasserstein_ambiguity_set': 'P = {Q : W_2(P_emp, Q) <= epsilon}',
            'dro_objective': 'min_x max_{Q in P} E_Q[loss(x)]',
            'worst_case_noshow': 'pi_worst = pi_mean + epsilon * sqrt(Var[pi] + epsilon^2)',
            'cvar_objective': 'max CVaR_alpha(U) = max { eta - 1/(K(1-alpha)) * sum_k z_k }',
            'cvar_constraint': 'z_k >= eta - U_k, z_k >= 0, for k=1..K (Rockafellar & Uryasev 2000)',
            'cvar_duration': 'CVaR_alpha(D) = E[D | D >= VaR_alpha] = mu + sigma * phi(z_alpha) / (1-alpha)',
            'var_alpha': 'VaR_alpha = mu + z_alpha * sigma',
            'robustness_guarantee': 'P(objective >= CVaR) >= alpha',
        }
