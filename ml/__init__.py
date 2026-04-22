"""
SACT Scheduling System - Machine Learning Module
================================================

ML models for predicting no-shows, treatment durations, and event impacts.
"""

from .noshow_model import NoShowModel, PredictionResult
from .duration_model import DurationModel, DurationPrediction
from .feature_engineering import FeatureEngineer, PatientFeatures
from .sequence_model import SequenceNoShowModel
from .survival_model import (
    SurvivalAnalysisModel,
    SurvivalPrediction,
    CoxProportionalHazards,
    predict_noshow_timing
)
from .uplift_model import (
    UpliftModel,
    UpliftPrediction,
    InterventionRecommendation,
    InterventionType,
    recommend_intervention,
    estimate_intervention_impact
)
from .multitask_model import (
    MultiTaskModel,
    MultiTaskPrediction,
    predict_joint
)
from .quantile_forest import (
    QuantileRegressionForest,
    QuantileForestDurationModel,
    QuantileForestNoShowModel,
    QuantilePrediction,
    predict_duration_quantiles
)
from .hierarchical_model import (
    HierarchicalBayesianModel,
    HierarchicalPrediction,
    HierarchicalModelSummary,
    predict_hierarchical
)
from .causal_model import (
    SchedulingCausalModel,
    CausalEffect,
    CounterfactualResult,
    CausalDAG,
    compute_intervention_effect,
    answer_counterfactual,
    # 4.2 Instrumental Variables
    InstrumentalVariablesEstimator,
    IVEstimationResult,
    estimate_iv_effect,
    # 4.3 Double Machine Learning
    DoubleMachineLearning,
    DMLResult,
    estimate_dml_effect
)
from .conformal_prediction import (
    # 5.1 Conformal Prediction
    ConformalPredictor,
    ConformalPrediction,
    ConformalCalibrationResult,
    ConformizedQuantileRegression,
    ConformalDurationPredictor,
    ConformalNoShowPredictor,
    NonConformityScore,
    create_conformal_duration_predictor,
    create_conformal_noshow_predictor
)
from .mc_dropout import (
    # 5.2 Monte Carlo Dropout
    MonteCarloDropout,
    MCDropoutResult,
    predict_with_mc_dropout
)
from .event_impact_model import (
    # 4.4 Event Impact Model
    EventImpactModel,
    EventImpactPrediction,
    Event,
    EventType,
    EventSeverity,
    SentimentAnalyzer,
    analyze_event_impact,
    estimate_event_severity
)
from .inverse_rl_preferences import (
    # 1.4 Inverse RL Preference Learner
    InverseRLPreferenceLearner,
    ObjectiveFeatures,
    OverrideRecord,
    IRLFitResult,
    compute_objective_features,
    OBJECTIVE_KEYS,
)
from .decision_focused_learning import (
    # 2.1 Decision-Focused Learning (Smart Predict-then-Optimise)
    DFLCalibrator,
    DFLFitResult,
    smooth_decision_cost,
    regret as dfl_regret,
)
from .adaptive_alpha import (
    # 2.2 Risk-Adaptive Conformal α
    AdaptiveAlphaPolicy,
    get_policy as get_adaptive_alpha_policy,
    set_policy as set_adaptive_alpha_policy,
    reload_policy as reload_adaptive_alpha_policy,
)
from .temporal_fusion_transformer import (
    # 2.3 Temporal Fusion Transformer (Lite)
    TFTTrainer,
    TFTFitResult,
    TORCH_AVAILABLE as TFT_TORCH_AVAILABLE,
)
from .rct_randomization import (
    # 2.4 Causal ML with Real RCT Data
    TrialArm,
    TrialAssigner,
    TrialAssignment,
    TrialOutcome,
    ATEResult,
    apply_rct_prior,
)
from .feature_store import (
    # 3.1 Online Feature Store (streaming ML)
    FeatureStore,
    Entity as FSEntity,
    FeatureView,
    MaterialisationInfo,
    get_store as get_feature_store,
    SCHEMA_VERSION as FEATURE_STORE_SCHEMA_VERSION,
)
from .micro_batch_optimizer import (
    # 3.2 Micro-batch optimizer (fast / slow / RL background)
    MicroBatchOptimizer,
    MicroBatchResult,
    MicroBatchStatus,
    ChangeRecord,
)

__all__ = [
    'NoShowModel',
    'PredictionResult',
    'DurationModel',
    'DurationPrediction',
    'FeatureEngineer',
    'PatientFeatures',
    'SequenceNoShowModel',
    'SurvivalAnalysisModel',
    'SurvivalPrediction',
    'CoxProportionalHazards',
    'predict_noshow_timing',
    'UpliftModel',
    'UpliftPrediction',
    'InterventionRecommendation',
    'InterventionType',
    'recommend_intervention',
    'estimate_intervention_impact',
    'MultiTaskModel',
    'MultiTaskPrediction',
    'predict_joint',
    'QuantileRegressionForest',
    'QuantileForestDurationModel',
    'QuantileForestNoShowModel',
    'QuantilePrediction',
    'predict_duration_quantiles',
    'HierarchicalBayesianModel',
    'HierarchicalPrediction',
    'HierarchicalModelSummary',
    'predict_hierarchical',
    'SchedulingCausalModel',
    'CausalEffect',
    'CounterfactualResult',
    'CausalDAG',
    'compute_intervention_effect',
    'answer_counterfactual',
    # 4.2 Instrumental Variables
    'InstrumentalVariablesEstimator',
    'IVEstimationResult',
    'estimate_iv_effect',
    # 4.3 Double Machine Learning
    'DoubleMachineLearning',
    'DMLResult',
    'estimate_dml_effect',
    # 5.2 Monte Carlo Dropout
    'MonteCarloDropout',
    'MCDropoutResult',
    'predict_with_mc_dropout',
    # 4.4 Event Impact Model
    'EventImpactModel',
    'EventImpactPrediction',
    'Event',
    'EventType',
    'EventSeverity',
    'SentimentAnalyzer',
    'analyze_event_impact',
    'estimate_event_severity',
    # 5.1 Conformal Prediction
    'ConformalPredictor',
    'ConformalPrediction',
    'ConformalCalibrationResult',
    'ConformizedQuantileRegression',
    'ConformalDurationPredictor',
    'ConformalNoShowPredictor',
    'NonConformityScore',
    'create_conformal_duration_predictor',
    'create_conformal_noshow_predictor',
    # 1.4 Inverse RL Preference Learner
    'InverseRLPreferenceLearner',
    'ObjectiveFeatures',
    'OverrideRecord',
    'IRLFitResult',
    'compute_objective_features',
    'OBJECTIVE_KEYS',
    # 2.1 Decision-Focused Learning
    'DFLCalibrator',
    'DFLFitResult',
    'smooth_decision_cost',
    'dfl_regret',
    # 2.2 Risk-Adaptive Conformal α
    'AdaptiveAlphaPolicy',
    'get_adaptive_alpha_policy',
    'set_adaptive_alpha_policy',
    'reload_adaptive_alpha_policy',
    # 2.3 Temporal Fusion Transformer (Lite)
    'TFTTrainer',
    'TFTFitResult',
    'TFT_TORCH_AVAILABLE',
    # 2.4 RCT Randomisation
    'TrialArm',
    'TrialAssigner',
    'TrialAssignment',
    'TrialOutcome',
    'ATEResult',
    'apply_rct_prior',
    # 3.1 Feature Store (streaming)
    'FeatureStore',
    'FSEntity',
    'FeatureView',
    'MaterialisationInfo',
    'get_feature_store',
    'FEATURE_STORE_SCHEMA_VERSION',
    # 3.2 Micro-batch optimizer
    'MicroBatchOptimizer',
    'MicroBatchResult',
    'MicroBatchStatus',
    'ChangeRecord',
]
