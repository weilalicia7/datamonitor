"""
Multi-Task Learning for Joint No-Show and Duration Prediction (3.1)

Train a single neural network to predict both no-show AND duration:
L = L_no_show + λ · L_duration

Architecture:
Input → Shared Layers → Task-Specific Heads
                           → No-Show (binary cross-entropy)
                           → Duration (MSE + quantile loss)

Benefits:
- Learns correlations between outcomes
- More efficient with limited data
- Consistent uncertainty estimates
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using numpy fallback for MultiTask model")


@dataclass
class MultiTaskPrediction:
    """Result of multi-task prediction."""
    patient_id: str
    noshow_probability: float
    predicted_duration: float
    duration_lower: float  # Lower quantile (10th percentile)
    duration_upper: float  # Upper quantile (90th percentile)
    uncertainty: float     # Combined uncertainty estimate
    correlation_factor: float  # Learned correlation between tasks


if TORCH_AVAILABLE:
    class SharedEncoder(nn.Module):
        """
        Shared feature encoder for multi-task learning.

        Learns joint representations useful for both tasks.
        """

        def __init__(self,
                     input_dim: int,
                     hidden_dims: List[int] = [128, 64, 32],
                     dropout: float = 0.3):
            super().__init__()

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            self.encoder = nn.Sequential(*layers)
            self.output_dim = hidden_dims[-1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)


    class NoShowHead(nn.Module):
        """
        Task-specific head for no-show prediction.

        Binary classification - outputs logits (use BCEWithLogitsLoss for training).
        """

        def __init__(self, input_dim: int, hidden_dim: int = 16):
            super().__init__()

            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1)
                # No sigmoid here - use BCEWithLogitsLoss for numerical stability
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(x).squeeze(-1)


    class DurationHead(nn.Module):
        """
        Task-specific head for duration prediction.

        Outputs mean and quantiles for uncertainty estimation.
        """

        def __init__(self, input_dim: int, hidden_dim: int = 16):
            super().__init__()

            # Mean prediction
            self.mean_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # Ensure positive duration
            )

            # Quantile predictions (10th and 90th percentiles)
            self.quantile_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # Lower and upper quantiles
                nn.Softplus()
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            mean = self.mean_head(x).squeeze(-1)
            quantiles = self.quantile_head(x)
            return mean, quantiles


    class MultiTaskNetwork(nn.Module):
        """
        Multi-Task Neural Network for joint prediction.

        Architecture:
        Input → SharedEncoder → [NoShowHead, DurationHead]
        """

        def __init__(self,
                     input_dim: int,
                     hidden_dims: List[int] = [128, 64, 32],
                     dropout: float = 0.3):
            super().__init__()

            self.shared_encoder = SharedEncoder(input_dim, hidden_dims, dropout)
            self.noshow_head = NoShowHead(self.shared_encoder.output_dim)
            self.duration_head = DurationHead(self.shared_encoder.output_dim)

            # Learnable task correlation parameter
            self.task_correlation = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            # Shared representation
            shared_features = self.shared_encoder(x)

            # Task-specific predictions
            noshow_prob = self.noshow_head(shared_features)
            duration_mean, duration_quantiles = self.duration_head(shared_features)

            return {
                'noshow_prob': noshow_prob,
                'duration_mean': duration_mean,
                'duration_quantiles': duration_quantiles,
                'correlation': torch.sigmoid(self.task_correlation)
            }


    class MultiTaskLoss(nn.Module):
        """
        Combined loss function for multi-task learning.

        L = L_no_show + λ · L_duration

        Where:
        - L_no_show: Binary cross-entropy
        - L_duration: MSE + quantile loss
        """

        def __init__(self,
                     lambda_duration: float = 0.5,
                     quantile_weight: float = 0.3):
            super().__init__()
            self.lambda_duration = lambda_duration
            self.quantile_weight = quantile_weight
            # Use BCEWithLogitsLoss for numerical stability (expects logits, not probabilities)
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.mse_loss = nn.MSELoss()

        def quantile_loss(self,
                         predictions: torch.Tensor,
                         targets: torch.Tensor,
                         quantiles: torch.Tensor) -> torch.Tensor:
            """
            Pinball loss for quantile regression.

            L_q(y, ŷ) = q(y - ŷ)⁺ + (1-q)(ŷ - y)⁺
            """
            q_low, q_high = 0.1, 0.9
            lower, upper = quantiles[:, 0], quantiles[:, 1]

            # Ensure predictions are within quantiles
            # Penalize if actual is outside predicted interval
            lower_loss = torch.mean(torch.relu(lower - targets) * q_low +
                                   torch.relu(targets - lower) * (1 - q_low))
            upper_loss = torch.mean(torch.relu(upper - targets) * (1 - q_high) +
                                   torch.relu(targets - upper) * q_high)

            return lower_loss + upper_loss

        def forward(self,
                    outputs: Dict[str, torch.Tensor],
                    noshow_targets: torch.Tensor,
                    duration_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
            # No-show loss (binary cross-entropy)
            noshow_loss = self.bce_loss(outputs['noshow_prob'], noshow_targets)

            # Duration loss (MSE + quantile)
            mse_loss = self.mse_loss(outputs['duration_mean'], duration_targets)
            q_loss = self.quantile_loss(
                outputs['duration_mean'],
                duration_targets,
                outputs['duration_quantiles']
            )
            duration_loss = mse_loss + self.quantile_weight * q_loss

            # Combined loss
            total_loss = noshow_loss + self.lambda_duration * duration_loss

            return {
                'total': total_loss,
                'noshow': noshow_loss,
                'duration': duration_loss,
                'mse': mse_loss,
                'quantile': q_loss
            }


class MultiTaskModel:
    """
    Multi-Task Learning model for joint No-Show and Duration prediction.

    Uses shared neural network architecture to learn correlations
    between no-show probability and treatment duration.
    """

    def __init__(self,
                 input_dim: int = 20,
                 hidden_dims: List[int] = [128, 64, 32],
                 lambda_duration: float = 0.5,
                 learning_rate: float = 0.001,
                 use_torch: bool = True):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lambda_duration = lambda_duration
        self.learning_rate = learning_rate
        self.use_torch = use_torch and TORCH_AVAILABLE

        self.is_fitted = False
        self.feature_names: List[str] = []

        # Default feature weights for fallback
        self._default_noshow_weights = {
            'previous_noshow_rate': 2.5,
            'age_normalized': 0.3,
            'distance_normalized': 0.15,
            'is_first_appointment': 0.8,
            'cycle_number': -0.1,
            'days_since_last': 0.02,
            'appointment_hour_late': 0.4,
            'friday': 0.3
        }

        self._default_duration_weights = {
            'base_duration': 1.0,
            'complexity_factor': 30.0,
            'is_first_cycle': 45.0,
            'age_factor': 0.5,
            'comorbidity_count': 15.0
        }

        if self.use_torch:
            self.network = MultiTaskNetwork(input_dim, hidden_dims)
            self.loss_fn = MultiTaskLoss(lambda_duration)
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
            logger.info(f"MultiTaskModel initialized with PyTorch (input_dim={input_dim})")
        else:
            self.network = None
            logger.info("MultiTaskModel initialized with numpy fallback")

    def _extract_features(self, patient: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from patient data."""
        features = []

        # Previous no-show rate
        features.append(patient.get('noshow_rate', patient.get('previous_noshow_rate', 0.1)))

        # Age (normalized)
        age = patient.get('age', patient.get('Age', 55))
        features.append(age / 100.0)

        # Distance (normalized)
        distance = patient.get('distance_km', patient.get('Distance', 10))
        features.append(min(distance / 50.0, 1.0))

        # First appointment flag
        features.append(1.0 if patient.get('is_first_appointment', False) else 0.0)

        # Cycle number (normalized)
        cycle = patient.get('cycle_number', 1)
        features.append(min(cycle / 10.0, 1.0))

        # Days since last visit (normalized)
        days_since = patient.get('days_since_last', 30)
        features.append(min(days_since / 60.0, 1.0))

        # Appointment hour features
        hour = patient.get('appointment_hour', 10)
        features.append(1.0 if hour < 10 else 0.0)  # Early
        features.append(1.0 if 10 <= hour < 14 else 0.0)  # Midday
        features.append(1.0 if hour >= 14 else 0.0)  # Late

        # Day of week
        dow = patient.get('day_of_week', 2)
        features.append(1.0 if dow == 0 else 0.0)  # Monday
        features.append(1.0 if dow == 4 else 0.0)  # Friday

        # Protocol/treatment features
        base_duration = patient.get('expected_duration', patient.get('base_duration', 120))
        features.append(base_duration / 300.0)  # Normalized

        complexity = patient.get('complexity_factor', patient.get('complexity', 0.5))
        features.append(complexity)

        is_first_cycle = patient.get('is_first_cycle', cycle == 1)
        features.append(1.0 if is_first_cycle else 0.0)

        # Comorbidities
        comorbidities = patient.get('comorbidity_count', patient.get('comorbidities', 0))
        features.append(min(comorbidities / 5.0, 1.0))

        # Historical duration variance
        hist_variance = patient.get('duration_variance', 0.2)
        features.append(hist_variance)

        # Pad to input_dim
        while len(features) < self.input_dim:
            features.append(0.0)

        return np.array(features[:self.input_dim], dtype=np.float32)

    def fit(self,
            patients: List[Dict[str, Any]],
            noshow_labels: np.ndarray,
            duration_labels: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32) -> 'MultiTaskModel':
        """
        Fit the multi-task model.

        Parameters:
        -----------
        patients : List[Dict]
            Patient feature dictionaries
        noshow_labels : np.ndarray
            Binary no-show labels (0 or 1)
        duration_labels : np.ndarray
            Actual treatment durations in minutes
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
        """
        if len(patients) == 0:
            logger.warning("Empty training data, using default weights")
            self.is_fitted = True
            return self

        # Extract features
        X = np.array([self._extract_features(p) for p in patients])

        if self.use_torch:
            self._fit_torch(X, noshow_labels, duration_labels, epochs, batch_size)
        else:
            # Fallback: just mark as fitted (uses default weights)
            pass

        self.is_fitted = True
        logger.info(f"MultiTaskModel fitted with {len(patients)} samples")
        return self

    def _fit_torch(self,
                   X: np.ndarray,
                   noshow_labels: np.ndarray,
                   duration_labels: np.ndarray,
                   epochs: int,
                   batch_size: int):
        """Train using PyTorch."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        noshow_tensor = torch.FloatTensor(noshow_labels)
        duration_tensor = torch.FloatTensor(duration_labels)

        # Create dataloader
        dataset = TensorDataset(X_tensor, noshow_tensor, duration_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Training loop
        self.network.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_noshow, batch_duration in dataloader:
                self.optimizer.zero_grad()

                outputs = self.network(batch_X)
                losses = self.loss_fn(outputs, batch_noshow, batch_duration)

                losses['total'].backward()
                self.optimizer.step()

                epoch_loss += losses['total'].item()

            if (epoch + 1) % 20 == 0:
                logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    def predict(self, patient: Dict[str, Any]) -> MultiTaskPrediction:
        """
        Make joint prediction for a patient.

        Returns both no-show probability and duration prediction.
        """
        features = self._extract_features(patient)

        if self.use_torch and self.network is not None:
            return self._predict_torch(patient, features)
        else:
            return self._predict_fallback(patient, features)

    def _predict_torch(self,
                       patient: Dict[str, Any],
                       features: np.ndarray) -> MultiTaskPrediction:
        """Predict using PyTorch model."""
        self.network.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            outputs = self.network(x)

            # Apply sigmoid to convert logits to probability
            noshow_logit = outputs['noshow_prob']
            noshow_prob = torch.sigmoid(noshow_logit).item()
            duration_mean = outputs['duration_mean'].item()
            quantiles = outputs['duration_quantiles'].squeeze().numpy()
            correlation = outputs['correlation'].item()

        # Ensure reasonable duration values
        duration_mean = max(30, min(500, duration_mean * 300))  # Denormalize
        duration_lower = max(15, duration_mean - quantiles[0] * 60)
        duration_upper = duration_mean + quantiles[1] * 60

        # Combined uncertainty
        noshow_uncertainty = noshow_prob * (1 - noshow_prob)
        duration_uncertainty = (duration_upper - duration_lower) / duration_mean
        uncertainty = 0.5 * noshow_uncertainty + 0.5 * duration_uncertainty

        return MultiTaskPrediction(
            patient_id=patient.get('patient_id', patient.get('Patient_ID', 'unknown')),
            noshow_probability=noshow_prob,
            predicted_duration=duration_mean,
            duration_lower=duration_lower,
            duration_upper=duration_upper,
            uncertainty=uncertainty,
            correlation_factor=correlation
        )

    def _predict_fallback(self,
                          patient: Dict[str, Any],
                          features: np.ndarray) -> MultiTaskPrediction:
        """Predict using simple weighted model (numpy fallback)."""
        # No-show prediction
        noshow_score = -1.5  # Base log-odds
        for name, weight in self._default_noshow_weights.items():
            value = patient.get(name, 0.0)
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            noshow_score += weight * value

        noshow_prob = 1 / (1 + np.exp(-noshow_score))
        noshow_prob = np.clip(noshow_prob, 0.01, 0.99)

        # Duration prediction
        base_duration = patient.get('expected_duration', patient.get('base_duration', 120))
        complexity = patient.get('complexity_factor', 0.5)
        is_first_cycle = patient.get('is_first_cycle', patient.get('cycle_number', 1) == 1)
        comorbidities = patient.get('comorbidity_count', 0)

        duration_mean = (
            base_duration +
            complexity * 30 +
            (45 if is_first_cycle else 0) +
            comorbidities * 15
        )

        # Quantiles based on variance
        variance_factor = patient.get('duration_variance', 0.2)
        duration_lower = duration_mean * (1 - variance_factor)
        duration_upper = duration_mean * (1 + variance_factor)

        # Uncertainty
        noshow_uncertainty = noshow_prob * (1 - noshow_prob)
        duration_uncertainty = variance_factor
        uncertainty = 0.5 * noshow_uncertainty + 0.5 * duration_uncertainty

        # Correlation (default moderate positive correlation)
        correlation = 0.3  # Higher no-show risk often correlates with longer durations

        return MultiTaskPrediction(
            patient_id=patient.get('patient_id', patient.get('Patient_ID', 'unknown')),
            noshow_probability=noshow_prob,
            predicted_duration=duration_mean,
            duration_lower=duration_lower,
            duration_upper=duration_upper,
            uncertainty=uncertainty,
            correlation_factor=correlation
        )

    def batch_predict(self,
                      patients: List[Dict[str, Any]]) -> List[MultiTaskPrediction]:
        """Make predictions for multiple patients."""
        return [self.predict(p) for p in patients]

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model configuration."""
        summary = {
            'model_type': 'Multi-Task Learning Network',
            'architecture': 'Input → Shared Layers → Task-Specific Heads',
            'loss_function': 'L = L_no_show + λ · L_duration',
            'tasks': {
                'no_show': {
                    'type': 'Binary Classification',
                    'loss': 'Binary Cross-Entropy',
                    'output': 'Probability [0, 1]'
                },
                'duration': {
                    'type': 'Regression with Quantiles',
                    'loss': 'MSE + Quantile Loss',
                    'output': 'Mean + [10th, 90th] percentiles'
                }
            },
            'lambda_duration': self.lambda_duration,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'using_torch': self.use_torch,
            'is_fitted': self.is_fitted,
            'benefits': [
                'Learns correlations between no-show and duration',
                'More efficient with limited data (shared representations)',
                'Consistent uncertainty estimates across tasks'
            ]
        }

        if self.use_torch and self.network is not None:
            summary['network_params'] = sum(p.numel() for p in self.network.parameters())

        return summary


# Convenience function
def predict_joint(patient: Dict[str, Any]) -> MultiTaskPrediction:
    """
    Quick joint prediction for a single patient.

    Example:
    --------
    >>> patient = {
    ...     'patient_id': 'P001',
    ...     'age': 65,
    ...     'noshow_rate': 0.15,
    ...     'expected_duration': 180,
    ...     'cycle_number': 2
    ... }
    >>> result = predict_joint(patient)
    >>> print(f"No-show: {result.noshow_probability:.1%}")
    >>> print(f"Duration: {result.predicted_duration:.0f} min")
    """
    model = MultiTaskModel(use_torch=TORCH_AVAILABLE)
    model.is_fitted = True  # Use default weights
    return model.predict(patient)
