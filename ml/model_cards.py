"""
Model Cards for ML Transparency — SACT Scheduling System

Each ML model has a Model Card documenting:
1. Intended Use — what the model does and who it's for
2. Performance Metrics — accuracy by subgroup (age, gender, site)
3. Known Limitations — where the model may fail
4. Ethical Considerations — fairness, bias, patient safety

References:
    Mitchell et al. (2019). "Model Cards for Model Reporting." FAT* Conference.
    NHS England (2023). "A Buyer's Guide to AI in Health and Care."
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SubgroupMetric:
    """Performance metric for a specific subgroup."""
    subgroup_name: str
    subgroup_value: str
    n_samples: int
    metric_name: str
    metric_value: float


@dataclass
class ModelCard:
    """Complete Model Card for an ML model."""
    model_name: str
    model_version: str
    model_type: str
    date_created: str

    # Intended Use
    intended_use: str
    intended_users: List[str]
    out_of_scope_uses: List[str]

    # Training Data
    training_data_description: str
    training_data_size: int
    training_data_date_range: str

    # Performance
    overall_metrics: Dict[str, float]
    subgroup_metrics: List[SubgroupMetric]

    # Limitations
    known_limitations: List[str]
    failure_modes: List[str]

    # Ethical Considerations
    ethical_considerations: List[str]
    fairness_assessment: Dict[str, Any]

    # Maintenance
    update_frequency: str
    monitoring_plan: str
    contact: str


class ModelCardGenerator:
    """
    Generates Model Cards from trained models and historical data.

    Model Cards provide transparency about:
    - What the model does and doesn't do
    - How it performs across different patient groups
    - Where it might fail or produce biased results
    """

    # Model definitions — static metadata
    MODEL_DEFINITIONS = {
        'noshow_ensemble': {
            'model_name': 'No-Show Prediction Ensemble',
            'model_version': '2.0',
            'model_type': 'Stacked Ensemble (RF + GB + XGBoost + Meta-learner)',
            'intended_use': (
                'Predict probability of patient no-show for SACT chemotherapy appointments. '
                'Used by the CP-SAT optimizer to adjust scheduling decisions: high no-show '
                'probability patients may be double-booked or given reminders.'
            ),
            'intended_users': [
                'SACT scheduling system (automated)',
                'Chemotherapy unit coordinators',
                'Clinical nurse specialists',
            ],
            'out_of_scope_uses': [
                'Denying treatment based on no-show prediction',
                'Replacing clinical judgment about patient care',
                'Predicting outcomes for non-SACT appointments',
                'Use without human oversight for patient-facing decisions',
            ],
            'known_limitations': [
                'Trained on synthetic data — real-world performance may differ',
                'Limited to Velindre Cancer Centre patient patterns',
                'Weather and traffic effects are estimates, not from live feeds',
                'Cannot capture sudden life events (bereavement, acute illness)',
                'First-time patients have limited history, higher uncertainty',
                'Ensemble not trained until sufficient historical data available',
            ],
            'failure_modes': [
                'Novel patient demographics not in training data',
                'Extreme weather events beyond training distribution',
                'Public holidays or NHS industrial action (rare events)',
                'Data entry errors in patient history',
            ],
            'ethical_considerations': [
                'Must not disadvantage patients from deprived areas (higher no-show rates due to transport barriers)',
                'Age-based predictions must not lead to reduced access for elderly patients',
                'Predictions should trigger supportive interventions (reminders, transport), not punitive action',
                'Model outputs are advisory — clinical staff retain final scheduling authority',
                'Regular fairness audits required across protected characteristics',
                'Compliant with NHS AI Ethics Framework and Equality Act 2010',
            ],
            'update_frequency': 'Re-trained when new data channel activated or manual recalibration triggered',
            'monitoring_plan': 'Drift detection (PSI, KS-test) runs every 24h. Fairness audit on each optimization run.',
            'contact': 'SACT Scheduling System — Velindre Cancer Centre',
        },
        'duration_model': {
            'model_name': 'Treatment Duration Prediction',
            'model_version': '1.0',
            'model_type': 'Ensemble (RF + GB) with protocol-specific variance',
            'intended_use': (
                'Predict actual treatment duration for SACT regimens. Used to allocate '
                'chair time in the optimizer and calculate schedule robustness.'
            ),
            'intended_users': [
                'SACT scheduling optimizer (automated)',
                'Pharmacy preparation planning',
            ],
            'out_of_scope_uses': [
                'Clinical dosing decisions',
                'Predicting treatment efficacy or toxicity',
            ],
            'known_limitations': [
                'Duration varies significantly by regimen (FOLFOX: 180-360min, PEMBRO: 30-60min)',
                'First cycle durations are typically 20-50% longer',
                'Does not account for pharmacy preparation delays',
                'Reaction monitoring may extend sessions unpredictably',
            ],
            'failure_modes': [
                'Novel regimens not in training data',
                'Patients requiring dose modifications mid-treatment',
                'Equipment failures or IV access difficulties',
            ],
            'ethical_considerations': [
                'Underestimating duration could lead to rushed treatments',
                'Overestimating could reduce access for other patients',
                'Must not pressure clinical staff to shorten treatments',
            ],
            'update_frequency': 'Continuous via online learning (Bayesian EMA updates)',
            'monitoring_plan': 'Conformal prediction intervals checked for coverage guarantee',
            'contact': 'SACT Scheduling System — Velindre Cancer Centre',
        },
        'causal_model': {
            'model_name': 'Causal Inference Model',
            'model_version': '1.0',
            'model_type': 'DAG + do-calculus + IV (2SLS) + DML',
            'intended_use': (
                'Estimate causal effects of interventions (reminders, transport offers) '
                'on no-show rates. Identifies which interventions are most effective '
                'for specific patient subgroups.'
            ),
            'intended_users': [
                'Service improvement teams',
                'Clinical leads reviewing intervention policies',
            ],
            'out_of_scope_uses': [
                'Real-time clinical decisions without domain expert review',
                'Causal claims from observational data without validation',
            ],
            'known_limitations': [
                'Causal estimates from observational data — unmeasured confounders possible',
                'IV estimates require valid instrument assumption (weather as instrument)',
                'Treatment assignment in synthetic data is derived, not randomized',
            ],
            'failure_modes': [
                'Violated instrument assumptions',
                'Insufficient overlap in treatment/control groups',
                'Structural breaks in patient behaviour patterns',
            ],
            'ethical_considerations': [
                'Causal claims must be validated before policy changes',
                'Interventions should benefit all patient groups equitably',
                'Must not use causal findings to justify reduced support for any group',
            ],
            'update_frequency': 'Re-estimated when data source changes',
            'monitoring_plan': 'Causal validation suite (7 tests: placebo + falsification + sensitivity)',
            'contact': 'SACT Scheduling System — Velindre Cancer Centre',
        },
        'rl_scheduler': {
            'model_name': 'RL Scheduling Agent',
            'model_version': '1.0',
            'model_type': 'Q-Learning with epsilon-greedy policy',
            'intended_use': (
                'Learn optimal scheduling actions (assign, delay, double-book, redistribute) '
                'from historical outcomes. Provides pre-optimization recommendations.'
            ),
            'intended_users': [
                'SACT scheduling optimizer (advisory input)',
            ],
            'out_of_scope_uses': [
                'Autonomous scheduling without human approval',
                'Learning from individual patient outcomes (too sparse)',
            ],
            'known_limitations': [
                'Q-table discretizes continuous state space — resolution trade-off',
                'Epsilon-greedy exploration may recommend suboptimal actions early on',
                'Reward function is proxy (utilization-based), not clinical outcome',
            ],
            'failure_modes': [
                'State space regions with insufficient exploration',
                'Reward hacking if utilization metric misaligned with quality',
            ],
            'ethical_considerations': [
                'RL recommendations are advisory only — human schedulers decide',
                'Must not learn to deprioritize specific patient groups',
                'Exploration must not expose patients to unnecessary risk',
            ],
            'update_frequency': 'Learns from each optimization run (epsilon decay)',
            'monitoring_plan': 'Q-value convergence tracked, policy reviewed quarterly',
            'contact': 'SACT Scheduling System — Velindre Cancer Centre',
        },
    }

    def __init__(self):
        self.cards: Dict[str, ModelCard] = {}

    def generate_card(
        self,
        model_key: str,
        model=None,
        historical_data: pd.DataFrame = None,
        is_trained: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a Model Card for a specific model.

        Args:
            model_key: Key from MODEL_DEFINITIONS
            model: The trained model instance (optional, for metrics)
            historical_data: Historical data for subgroup analysis
            is_trained: Whether the model is actually trained

        Returns:
            Dict with complete Model Card
        """
        defn = self.MODEL_DEFINITIONS.get(model_key, {})
        if not defn:
            return {'error': f'Unknown model: {model_key}'}

        # Compute subgroup metrics if data available
        subgroup_metrics = []
        overall_metrics = {}
        fairness_assessment = {}

        if historical_data is not None and len(historical_data) > 50:
            metrics_result = self._compute_subgroup_metrics(
                model_key, model, historical_data, is_trained
            )
            subgroup_metrics = metrics_result.get('subgroup_metrics', [])
            overall_metrics = metrics_result.get('overall_metrics', {})
            fairness_assessment = metrics_result.get('fairness', {})

        card = {
            'model_name': defn['model_name'],
            'model_version': defn['model_version'],
            'model_type': defn['model_type'],
            'date_created': datetime.now().strftime('%Y-%m-%d'),
            'is_trained': is_trained,

            'intended_use': {
                'primary_use': defn['intended_use'],
                'intended_users': defn['intended_users'],
                'out_of_scope': defn['out_of_scope_uses'],
            },

            'training_data': {
                'description': 'SACT v4.0 compliant appointment records with outcome labels',
                'size': len(historical_data) if historical_data is not None else 0,
                'features': model.feature_names if model and hasattr(model, 'feature_names') else [],
                'label_distribution': self._get_label_distribution(historical_data),
            },

            'performance': {
                'overall': overall_metrics,
                'by_subgroup': [
                    {
                        'group': m['group'],
                        'value': m['value'],
                        'n': m['n'],
                        'metric': m['metric'],
                        'score': m['score'],
                    }
                    for m in subgroup_metrics
                ],
            },

            'limitations': {
                'known_limitations': defn['known_limitations'],
                'failure_modes': defn['failure_modes'],
            },

            'ethical_considerations': defn['ethical_considerations'],
            'fairness_assessment': fairness_assessment,

            'maintenance': {
                'update_frequency': defn['update_frequency'],
                'monitoring_plan': defn['monitoring_plan'],
                'contact': defn['contact'],
            },
        }

        return card

    def generate_all_cards(
        self,
        models: Dict[str, Any] = None,
        historical_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Generate Model Cards for all registered models."""
        models = models or {}
        cards = {}

        for model_key in self.MODEL_DEFINITIONS:
            model = models.get(model_key)
            is_trained = getattr(model, 'is_trained', False) if model else False
            cards[model_key] = self.generate_card(
                model_key, model, historical_data, is_trained
            )

        return {
            'cards': cards,
            'total_models': len(cards),
            'trained_models': sum(1 for c in cards.values() if c.get('is_trained')),
            'generated_at': datetime.now().isoformat(),
        }

    def _compute_subgroup_metrics(
        self, model_key: str, model, data: pd.DataFrame, is_trained: bool
    ) -> Dict:
        """Compute performance metrics broken down by subgroup."""
        results = {'subgroup_metrics': [], 'overall_metrics': {}, 'fairness': {}}

        if 'Attended_Status' not in data.columns:
            return results

        noshow = (data['Attended_Status'] == 'No').astype(int)
        overall_rate = float(noshow.mean())
        results['overall_metrics'] = {
            'noshow_rate': round(overall_rate, 3),
            'n_total': len(data),
            'n_noshow': int(noshow.sum()),
            'n_attended': int((1 - noshow).sum()),
        }

        # Subgroup analysis
        subgroups = {
            'Age_Band': data.get('Age_Band', pd.Series(dtype=str)),
            'Site_Code': data.get('Site_Code', pd.Series(dtype=str)),
            'Priority': data.get('Priority', pd.Series(dtype=str)),
        }

        if 'Person_Stated_Gender_Code' in data.columns:
            subgroups['Gender'] = data['Person_Stated_Gender_Code']

        for group_name, group_col in subgroups.items():
            if group_col.empty or group_col.isna().all():
                continue
            for value in group_col.dropna().unique():
                mask = group_col == value
                n = int(mask.sum())
                if n < 10:
                    continue
                rate = float(noshow[mask].mean())
                results['subgroup_metrics'].append({
                    'group': group_name,
                    'value': str(value),
                    'n': n,
                    'metric': 'noshow_rate',
                    'score': round(rate, 3),
                })

        # Fairness: max disparity across subgroups
        for group_name in subgroups:
            group_rates = [
                m['score'] for m in results['subgroup_metrics']
                if m['group'] == group_name
            ]
            if len(group_rates) >= 2:
                disparity = max(group_rates) - min(group_rates)
                ratio = min(group_rates) / max(group_rates) if max(group_rates) > 0 else 1.0
                results['fairness'][group_name] = {
                    'max_disparity': round(disparity, 3),
                    'min_max_ratio': round(ratio, 3),
                    'four_fifths_rule': ratio >= 0.8,
                    'groups_analyzed': len(group_rates),
                }

        return results

    def _get_label_distribution(self, data: pd.DataFrame) -> Dict:
        """Get target label distribution."""
        if data is None or 'Attended_Status' not in data.columns:
            return {}
        counts = data['Attended_Status'].value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}
