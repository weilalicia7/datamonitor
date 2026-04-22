"""
Monte Carlo Dropout for Bayesian Uncertainty Estimation (5.2)

Approximate Bayesian inference by keeping dropout active at inference time
and running T stochastic forward passes:

    p(y|x, D) ~ (1/T) * sum_{t=1}^{T} p(y|x, w_t)

where w_t are sampled by applying dropout masks at each forward pass.

Predictive uncertainty decomposes into:
    Var[y] = E[Var[y|w]] + Var[E[y|w]]
           = Aleatoric      + Epistemic

References:
    Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning. ICML.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available for MC Dropout")


@dataclass
class MCDropoutResult:
    """Result of MC Dropout prediction."""
    patient_id: str
    # No-show predictions
    noshow_mean: float          # Mean predicted probability
    noshow_std: float           # Epistemic uncertainty
    noshow_ci_lower: float      # 95% credible interval lower
    noshow_ci_upper: float      # 95% credible interval upper
    # Duration predictions
    duration_mean: float        # Mean predicted duration
    duration_std: float         # Epistemic uncertainty
    duration_ci_lower: float    # 95% credible interval lower
    duration_ci_upper: float    # 95% credible interval upper
    # Uncertainty decomposition
    epistemic_uncertainty: float   # Model uncertainty (reducible with more data)
    aleatoric_uncertainty: float   # Data uncertainty (irreducible)
    total_uncertainty: float       # Combined
    # Metadata
    n_forward_passes: int
    dropout_rate: float
    interpretation: str


class MonteCarloDropout:
    """
    Monte Carlo Dropout for Bayesian uncertainty estimation.

    Wraps existing PyTorch models (MultiTaskNetwork, GRU/LSTM) and
    performs T stochastic forward passes with dropout active to
    approximate the predictive posterior distribution.

    Mathematical formulation:
        Predictive mean:  E[y] = (1/T) * sum_t f(x; w_t)
        Predictive var:   Var[y] = (1/T) * sum_t f(x; w_t)^2 - E[y]^2

    Uncertainty decomposition:
        Epistemic = Var_w[E[y|w]]    (model uncertainty)
        Aleatoric = E_w[Var[y|w]]    (data noise)
    """

    def __init__(self,
                 n_forward_passes: int = 100,
                 confidence_level: float = 0.95):
        """
        Parameters
        ----------
        n_forward_passes : int
            Number of stochastic forward passes (T). More passes = better
            approximation but slower. 50-200 recommended.
        confidence_level : float
            Confidence level for credible intervals (default 95%).
        """
        self.n_forward_passes = n_forward_passes
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - MC Dropout will use fallback")

    @staticmethod
    def _enable_dropout(model: 'nn.Module'):
        """
        Enable dropout layers during inference.

        Standard PyTorch model.eval() disables dropout. MC Dropout requires
        dropout to remain active to sample from the approximate posterior.
        """
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @staticmethod
    def _get_dropout_rate(model: 'nn.Module') -> float:
        """Extract dropout rate from model."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                return module.p
        return 0.0

    def predict_multitask(self,
                          network: 'nn.Module',
                          patient_features: np.ndarray,
                          patient_id: str = 'unknown') -> MCDropoutResult:
        """
        MC Dropout prediction using MultiTaskNetwork.

        Runs T forward passes with dropout active and aggregates results.

        Parameters
        ----------
        network : MultiTaskNetwork
            Trained PyTorch multi-task network with dropout layers.
        patient_features : np.ndarray
            Feature vector for the patient.
        patient_id : str
            Patient identifier.

        Returns
        -------
        MCDropoutResult
            Prediction with uncertainty estimates.
        """
        if not TORCH_AVAILABLE:
            return self._predict_fallback(patient_features, patient_id)

        # Set model to eval mode first (disables batch norm training etc.)
        network.eval()
        # Then re-enable dropout for MC sampling
        self._enable_dropout(network)

        dropout_rate = self._get_dropout_rate(network)

        x = torch.FloatTensor(patient_features).unsqueeze(0)

        noshow_samples = []
        duration_samples = []

        # T stochastic forward passes
        with torch.no_grad():
            for _ in range(self.n_forward_passes):
                outputs = network(x)

                noshow_logit = outputs['noshow_prob']
                noshow_prob = torch.sigmoid(noshow_logit).item()
                noshow_samples.append(noshow_prob)

                duration_val = outputs['duration_mean'].item()
                duration_samples.append(max(30, min(500, duration_val * 300)))

        noshow_samples = np.array(noshow_samples)
        duration_samples = np.array(duration_samples)

        # Restore standard eval mode
        network.eval()

        return self._compute_result(
            noshow_samples, duration_samples,
            patient_id, dropout_rate
        )

    def predict_sequence(self,
                         network: 'nn.Module',
                         sequence_input: 'torch.Tensor',
                         patient_id: str = 'unknown') -> MCDropoutResult:
        """
        MC Dropout prediction using Sequence model (GRU/LSTM).

        Parameters
        ----------
        network : PatientGRU or PatientLSTM
            Trained sequence model with dropout layers.
        sequence_input : torch.Tensor
            Sequence input tensor (batch=1, seq_len, features).
        patient_id : str
            Patient identifier.
        """
        if not TORCH_AVAILABLE:
            return self._predict_fallback(None, patient_id)

        network.eval()
        self._enable_dropout(network)

        dropout_rate = self._get_dropout_rate(network)

        noshow_samples = []

        with torch.no_grad():
            for _ in range(self.n_forward_passes):
                output = network(sequence_input)
                if isinstance(output, tuple):
                    output = output[0]
                prob = torch.sigmoid(output[:, -1]).item()
                noshow_samples.append(prob)

        noshow_samples = np.array(noshow_samples)

        network.eval()

        # Sequence model only predicts no-show
        noshow_mean = float(np.mean(noshow_samples))
        noshow_std = float(np.std(noshow_samples))

        ci_lower_idx = int(self.alpha / 2 * self.n_forward_passes)
        ci_upper_idx = int((1 - self.alpha / 2) * self.n_forward_passes)
        sorted_noshow = np.sort(noshow_samples)

        epistemic = noshow_std ** 2
        aleatoric = noshow_mean * (1 - noshow_mean)
        total = epistemic + aleatoric

        interpretation = self._interpret(noshow_mean, noshow_std, None, None)

        return MCDropoutResult(
            patient_id=patient_id,
            noshow_mean=round(noshow_mean, 4),
            noshow_std=round(noshow_std, 4),
            noshow_ci_lower=round(float(sorted_noshow[max(0, ci_lower_idx)]), 4),
            noshow_ci_upper=round(float(sorted_noshow[min(len(sorted_noshow)-1, ci_upper_idx)]), 4),
            duration_mean=0.0,
            duration_std=0.0,
            duration_ci_lower=0.0,
            duration_ci_upper=0.0,
            epistemic_uncertainty=round(float(epistemic), 4),
            aleatoric_uncertainty=round(float(aleatoric), 4),
            total_uncertainty=round(float(total), 4),
            n_forward_passes=self.n_forward_passes,
            dropout_rate=dropout_rate,
            interpretation=interpretation
        )

    def _compute_result(self,
                        noshow_samples: np.ndarray,
                        duration_samples: np.ndarray,
                        patient_id: str,
                        dropout_rate: float) -> MCDropoutResult:
        """Compute MC Dropout statistics from samples."""

        # No-show statistics
        noshow_mean = float(np.mean(noshow_samples))
        noshow_std = float(np.std(noshow_samples))

        # Duration statistics
        duration_mean = float(np.mean(duration_samples))
        duration_std = float(np.std(duration_samples))

        # Credible intervals (percentile method)
        ci_lower_pct = self.alpha / 2 * 100
        ci_upper_pct = (1 - self.alpha / 2) * 100

        noshow_ci = np.percentile(noshow_samples, [ci_lower_pct, ci_upper_pct])
        duration_ci = np.percentile(duration_samples, [ci_lower_pct, ci_upper_pct])

        # Uncertainty decomposition
        # Epistemic: variance of the means across forward passes
        epistemic_noshow = noshow_std ** 2
        epistemic_duration = duration_std ** 2

        # Aleatoric: mean of the variances (for Bernoulli: p*(1-p))
        aleatoric_noshow = noshow_mean * (1 - noshow_mean)
        aleatoric_duration = float(np.mean([
            (d - duration_mean) ** 2 for d in duration_samples
        ]))

        # Combined (normalized)
        epistemic = float(np.sqrt(epistemic_noshow + epistemic_duration / (duration_mean + 1e-8) ** 2))
        aleatoric = float(np.sqrt(aleatoric_noshow + aleatoric_duration / (duration_mean + 1e-8) ** 2))
        total = float(np.sqrt(epistemic ** 2 + aleatoric ** 2))

        interpretation = self._interpret(noshow_mean, noshow_std,
                                         duration_mean, duration_std)

        return MCDropoutResult(
            patient_id=patient_id,
            noshow_mean=round(noshow_mean, 4),
            noshow_std=round(noshow_std, 4),
            noshow_ci_lower=round(float(noshow_ci[0]), 4),
            noshow_ci_upper=round(float(noshow_ci[1]), 4),
            duration_mean=round(duration_mean, 1),
            duration_std=round(duration_std, 1),
            duration_ci_lower=round(float(duration_ci[0]), 1),
            duration_ci_upper=round(float(duration_ci[1]), 1),
            epistemic_uncertainty=round(epistemic, 4),
            aleatoric_uncertainty=round(aleatoric, 4),
            total_uncertainty=round(total, 4),
            n_forward_passes=self.n_forward_passes,
            dropout_rate=dropout_rate,
            interpretation=interpretation
        )

    def _interpret(self,
                   noshow_mean: float,
                   noshow_std: float,
                   duration_mean: Optional[float],
                   duration_std: Optional[float]) -> str:
        """Generate human-readable interpretation."""
        parts = []

        # No-show confidence
        if noshow_std < 0.05:
            parts.append(f"High confidence in no-show prediction ({noshow_mean:.0%})")
        elif noshow_std < 0.10:
            parts.append(f"Moderate confidence in no-show prediction ({noshow_mean:.0%} +/- {noshow_std:.0%})")
        else:
            parts.append(f"Low confidence - model uncertain about no-show ({noshow_mean:.0%} +/- {noshow_std:.0%})")

        # Duration confidence
        if duration_mean and duration_std:
            cv = duration_std / (duration_mean + 1e-8)
            if cv < 0.05:
                parts.append(f"Duration estimate is precise ({duration_mean:.0f} min)")
            elif cv < 0.15:
                parts.append(f"Duration has moderate uncertainty ({duration_mean:.0f} +/- {duration_std:.0f} min)")
            else:
                parts.append(f"Duration is highly uncertain ({duration_mean:.0f} +/- {duration_std:.0f} min)")

        return ". ".join(parts)

    def _predict_fallback(self,
                          features: Optional[np.ndarray],
                          patient_id: str) -> MCDropoutResult:
        """Fallback when PyTorch is not available."""
        # Simple bootstrap-like approximation
        noshow_base = 0.15
        duration_base = 120.0

        noshow_samples = np.random.beta(2, 10, self.n_forward_passes)
        duration_samples = np.random.normal(duration_base, 15, self.n_forward_passes)

        return self._compute_result(
            noshow_samples, duration_samples,
            patient_id, 0.3
        )

    def batch_predict(self,
                      network: 'nn.Module',
                      patients_features: List[np.ndarray],
                      patient_ids: List[str]) -> List[MCDropoutResult]:
        """Run MC Dropout for multiple patients."""
        results = []
        for features, pid in zip(patients_features, patient_ids):
            result = self.predict_multitask(network, features, pid)
            results.append(result)
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get MC Dropout configuration summary."""
        return {
            'method': 'Monte Carlo Dropout',
            'reference': 'Gal & Ghahramani (2016)',
            'n_forward_passes': self.n_forward_passes,
            'confidence_level': self.confidence_level,
            'formula': 'Var[y] = E[Var[y|w]] + Var[E[y|w]]',
            'uncertainty_decomposition': {
                'epistemic': 'Model uncertainty (Var_w[E[y|w]]) - reducible with more data',
                'aleatoric': 'Data noise (E_w[Var[y|w]]) - irreducible'
            },
            'advantages': [
                'No architecture change needed (uses existing dropout)',
                'Principled Bayesian uncertainty from frequentist model',
                'Decomposes uncertainty into epistemic vs aleatoric',
                'Works with any model containing dropout layers'
            ]
        }


def predict_with_mc_dropout(model, patient_features, patient_id='unknown',
                            n_iter=100, confidence=0.95):
    """
    Convenience function for MC Dropout prediction.

    Parameters
    ----------
    model : nn.Module
        PyTorch model with dropout layers.
    patient_features : np.ndarray
        Feature vector.
    patient_id : str
        Patient ID.
    n_iter : int
        Number of forward passes.
    confidence : float
        Confidence level for intervals.

    Returns
    -------
    MCDropoutResult
    """
    mc = MonteCarloDropout(n_forward_passes=n_iter, confidence_level=confidence)
    return mc.predict_multitask(model, patient_features, patient_id)
