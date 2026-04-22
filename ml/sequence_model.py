"""
Sequence-Based No-Show Prediction Model
========================================

Uses LSTM/GRU to model patient appointment history as sequences.
Captures temporal patterns in patient behavior that independent
models miss.

Mathematical formulation:
    h_t = GRU(x_t, h_{t-1})
    y_t = sigmoid(W * h_t + b)

Where x_t includes appointment features and h_t captures patient trajectory.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle
import json

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Try importing sklearn for preprocessing
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_logger, MODEL_SAVE_DIR

logger = get_logger(__name__)


@dataclass
class SequenceFeatures:
    """Features for a patient's appointment sequence"""
    patient_id: str
    sequence_length: int
    features: np.ndarray  # Shape: (seq_len, n_features)
    labels: np.ndarray    # Shape: (seq_len,) - attendance for each appointment
    timestamps: List[datetime]


class PatientSequenceDataset(Dataset):
    """PyTorch Dataset for patient appointment sequences"""

    def __init__(self, sequences: List[SequenceFeatures]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'features': torch.FloatTensor(seq.features),
            'labels': torch.FloatTensor(seq.labels),
            'length': seq.sequence_length,
            'patient_id': seq.patient_id
        }


def collate_sequences(batch):
    """Custom collate function for variable-length sequences"""
    # Sort by sequence length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)

    features = [item['features'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = [item['length'] for item in batch]
    patient_ids = [item['patient_id'] for item in batch]

    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)  # -1 for masked

    return {
        'features': features_padded,
        'labels': labels_padded,
        'lengths': torch.LongTensor(lengths),
        'patient_ids': patient_ids
    }


class PatientGRU(nn.Module):
    """
    GRU-based model for patient no-show prediction.

    Architecture:
        Input -> Embedding (optional) -> GRU -> Dropout -> Linear -> Sigmoid

    The GRU captures temporal dependencies in patient appointment history,
    learning patterns like:
    - Patients who miss one appointment are more likely to miss the next
    - Seasonal attendance patterns
    - Recovery from no-show streaks
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False):
        super(PatientGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            lengths: Actual sequence lengths for packing

        Returns:
            Predictions of shape (batch, seq_len) - probability for each timestep
        """
        batch_size, seq_len, input_size = x.shape

        # Apply batch normalization to each timestep
        x_reshaped = x.view(-1, input_size)
        x_normed = self.input_bn(x_reshaped)
        x = x_normed.view(batch_size, seq_len, input_size)

        # Pack sequences if lengths provided
        if lengths is not None:
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
            gru_out, _ = self.gru(x_packed)
            gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, _ = self.gru(x)

        # Apply dropout and linear layer to each timestep
        out = self.dropout(gru_out)
        out = self.fc(out).squeeze(-1)
        out = self.sigmoid(out)

        return out


class PatientLSTM(nn.Module):
    """
    LSTM-based model (alternative to GRU).

    LSTMs have separate cell state and hidden state, which can capture
    longer-term dependencies but are more computationally expensive.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False):
        super(PatientLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        batch_size, seq_len, input_size = x.shape

        x_reshaped = x.view(-1, input_size)
        x_normed = self.input_bn(x_reshaped)
        x = x_normed.view(batch_size, seq_len, input_size)

        if lengths is not None:
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
            lstm_out, _ = self.lstm(x_packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        out = self.dropout(lstm_out)
        out = self.fc(out).squeeze(-1)
        out = self.sigmoid(out)

        return out


class SequenceFeatureEngineer:
    """
    Creates sequence features from patient appointment history.

    Features per timestep:
    - Time gap since last appointment (days)
    - Day of week (one-hot)
    - Time of day (normalized)
    - Cumulative no-show rate up to this point
    - Streak of attended/missed appointments
    - Seasonal indicators
    - Previous attendance (lag features)
    """

    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names = []
        self.is_fitted = False

    def create_sequence_features(self, appointments_df: pd.DataFrame,
                                   patient_id: str) -> Optional[SequenceFeatures]:
        """
        Create sequence features for a single patient.

        Args:
            appointments_df: DataFrame with appointment history
            patient_id: Patient ID to extract

        Returns:
            SequenceFeatures object or None if insufficient data
        """
        # Filter to this patient and sort by date
        patient_appts = appointments_df[
            appointments_df['Patient_ID'] == patient_id
        ].sort_values('Appointment_Date')

        if len(patient_appts) < 2:
            return None  # Need at least 2 appointments for sequence

        features_list = []
        labels = []
        timestamps = []

        cumulative_attended = 0
        cumulative_total = 0
        streak = 0  # Positive for attended streak, negative for no-show streak

        for i, (_, row) in enumerate(patient_appts.iterrows()):
            features = {}

            # Time features
            appt_date = pd.to_datetime(row['Appointment_Date'])
            timestamps.append(appt_date)

            features['day_of_week_sin'] = np.sin(2 * np.pi * appt_date.dayofweek / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * appt_date.dayofweek / 7)
            features['month_sin'] = np.sin(2 * np.pi * appt_date.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * appt_date.month / 12)
            features['is_monday'] = 1 if appt_date.dayofweek == 0 else 0
            features['is_friday'] = 1 if appt_date.dayofweek == 4 else 0

            # Gap since last appointment
            if i > 0:
                prev_date = pd.to_datetime(patient_appts.iloc[i-1]['Appointment_Date'])
                gap_days = (appt_date - prev_date).days
                features['gap_days'] = min(gap_days / 30.0, 6.0)  # Normalize, cap at 6 months
                features['gap_weeks'] = min(gap_days / 7.0, 26.0)
            else:
                features['gap_days'] = 0
                features['gap_weeks'] = 0

            # Cumulative attendance rate (up to but not including current)
            if cumulative_total > 0:
                features['cumulative_attend_rate'] = cumulative_attended / cumulative_total
            else:
                features['cumulative_attend_rate'] = 0.5  # Prior

            features['cumulative_noshow_rate'] = 1 - features['cumulative_attend_rate']
            features['total_appointments'] = min(cumulative_total / 20.0, 1.0)  # Normalized

            # Streak features
            features['attendance_streak'] = max(0, streak) / 10.0  # Positive streaks
            features['noshow_streak'] = max(0, -streak) / 5.0  # Negative streaks (capped)

            # Previous attendance (lag features)
            for lag in [1, 2, 3]:
                if i >= lag:
                    prev_attended = patient_appts.iloc[i-lag]['Attended_Status']
                    features[f'prev_attended_{lag}'] = 1 if prev_attended == 'Yes' else 0
                else:
                    features[f'prev_attended_{lag}'] = 0.5  # Unknown

            # Treatment features (if available)
            if 'Cycle_Number' in row:
                features['cycle_number'] = min(int(row.get('Cycle_Number', 1)) / 10.0, 1.0)
                features['is_first_cycle'] = 1 if row.get('Cycle_Number', 1) == 1 else 0
            else:
                features['cycle_number'] = 0
                features['is_first_cycle'] = 0

            if 'Planned_Duration' in row:
                features['duration_hours'] = float(row.get('Planned_Duration', 60)) / 240.0
            else:
                features['duration_hours'] = 0.5

            if 'Clinical_Priority' in row:
                priority_map = {'P1': 0.25, 'P2': 0.5, 'P3': 0.75, 'P4': 1.0}
                features['priority'] = priority_map.get(str(row.get('Clinical_Priority', 'P3')), 0.75)
            else:
                features['priority'] = 0.75

            # Travel time
            if 'Travel_Time_Min' in row:
                features['travel_time'] = min(float(row.get('Travel_Time_Min', 30)) / 120.0, 1.0)
            else:
                features['travel_time'] = 0.25

            features_list.append(features)

            # Label (current appointment attended?)
            attended = row.get('Attended_Status', 'Yes')
            label = 0 if attended == 'Yes' else 1  # 1 = no-show
            labels.append(label)

            # Update cumulative stats
            cumulative_total += 1
            if attended == 'Yes':
                cumulative_attended += 1
                streak = streak + 1 if streak >= 0 else 1
            else:
                streak = streak - 1 if streak <= 0 else -1

        # Convert to numpy array
        self.feature_names = list(features_list[0].keys())
        features_array = np.array([[f[k] for k in self.feature_names] for f in features_list])
        labels_array = np.array(labels)

        return SequenceFeatures(
            patient_id=patient_id,
            sequence_length=len(features_list),
            features=features_array,
            labels=labels_array,
            timestamps=timestamps
        )

    def create_all_sequences(self, appointments_df: pd.DataFrame,
                              min_sequence_length: int = 3) -> List[SequenceFeatures]:
        """
        Create sequences for all patients.

        Args:
            appointments_df: DataFrame with all appointment history
            min_sequence_length: Minimum appointments to include patient

        Returns:
            List of SequenceFeatures objects
        """
        sequences = []
        patient_ids = appointments_df['Patient_ID'].unique()

        for patient_id in patient_ids:
            seq = self.create_sequence_features(appointments_df, patient_id)
            if seq is not None and seq.sequence_length >= min_sequence_length:
                sequences.append(seq)

        logger.info(f"Created {len(sequences)} sequences from {len(patient_ids)} patients")
        logger.info(f"Sequence lengths: min={min(s.sequence_length for s in sequences)}, "
                   f"max={max(s.sequence_length for s in sequences)}, "
                   f"mean={np.mean([s.sequence_length for s in sequences]):.1f}")

        return sequences


class SequenceNoShowModel:
    """
    Main class for sequence-based no-show prediction.

    Combines:
    - Sequence feature engineering
    - GRU/LSTM neural network
    - Training with proper masking
    - Integration with existing NoShowModel ensemble
    """

    def __init__(self, model_type: str = 'gru', hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False, learning_rate: float = 0.001):
        """
        Initialize sequence model.

        Args:
            model_type: 'gru' or 'lstm'
            hidden_size: Hidden state size
            num_layers: Number of RNN layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional RNN
            learning_rate: Learning rate for optimizer
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for sequence model. Install with: pip install torch")

        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate

        self.feature_engineer = SequenceFeatureEngineer()
        self.model = None
        self.optimizer = None
        self.is_trained = False
        self.input_size = None
        self.training_history = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"SequenceNoShowModel initialized ({model_type.upper()}, device={self.device})")

    def _build_model(self, input_size: int):
        """Build the neural network model"""
        self.input_size = input_size

        if self.model_type == 'lstm':
            self.model = PatientLSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )
        else:  # Default to GRU
            self.model = PatientGRU(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model built with {n_params:,} trainable parameters")

    def train(self, appointments_df: pd.DataFrame,
              epochs: int = 50, batch_size: int = 32,
              validation_split: float = 0.2,
              early_stopping_patience: int = 10,
              min_sequence_length: int = 3) -> Dict:
        """
        Train the sequence model.

        Args:
            appointments_df: DataFrame with appointment history
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            early_stopping_patience: Stop if no improvement for this many epochs
            min_sequence_length: Minimum appointments per patient

        Returns:
            Dict with training metrics
        """
        # Create sequences
        logger.info("Creating patient sequences...")
        sequences = self.feature_engineer.create_all_sequences(
            appointments_df, min_sequence_length
        )

        if len(sequences) < 10:
            raise ValueError(f"Not enough sequences ({len(sequences)}). Need at least 10.")

        # Split into train/validation
        np.random.seed(42)
        indices = np.random.permutation(len(sequences))
        val_size = int(len(sequences) * validation_split)

        train_seqs = [sequences[i] for i in indices[val_size:]]
        val_seqs = [sequences[i] for i in indices[:val_size]]

        logger.info(f"Train: {len(train_seqs)} sequences, Val: {len(val_seqs)} sequences")

        # Create data loaders
        train_dataset = PatientSequenceDataset(train_seqs)
        val_dataset = PatientSequenceDataset(val_seqs)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences
        )

        # Build model
        input_size = sequences[0].features.shape[1]
        self._build_model(input_size)

        # Loss function (with masking for padded positions)
        criterion = nn.BCELoss(reduction='none')

        # Training loop
        best_val_auc = 0
        patience_counter = 0
        self.training_history = []

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_preds, train_labels = [], []

            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths']

                self.optimizer.zero_grad()

                outputs = self.model(features, lengths)

                # Create mask for valid positions
                mask = (labels != -1).float()

                # Compute masked loss
                loss = criterion(outputs, labels.clamp(0, 1))
                loss = (loss * mask).sum() / mask.sum()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()

                # Collect predictions for metrics
                for i, length in enumerate(lengths):
                    train_preds.extend(outputs[i, :length].detach().cpu().numpy())
                    train_labels.extend(labels[i, :length].detach().cpu().numpy())

            train_loss /= len(train_loader)
            train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0.5

            # Validation
            self.model.eval()
            val_loss = 0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    lengths = batch['lengths']

                    outputs = self.model(features, lengths)

                    mask = (labels != -1).float()
                    loss = criterion(outputs, labels.clamp(0, 1))
                    loss = (loss * mask).sum() / mask.sum()

                    val_loss += loss.item()

                    for i, length in enumerate(lengths):
                        val_preds.extend(outputs[i, :length].cpu().numpy())
                        val_labels.extend(labels[i, :length].cpu().numpy())

            val_loss /= len(val_loader)
            val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5

            # Record history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_loss': val_loss,
                'val_auc': val_auc
            })

            # Early stopping check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"Train Loss={train_loss:.4f}, AUC={train_auc:.3f} | "
                           f"Val Loss={val_loss:.4f}, AUC={val_auc:.3f}")

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        self.is_trained = True

        # Final metrics
        metrics = {
            'best_val_auc': best_val_auc,
            'final_train_auc': train_auc,
            'epochs_trained': epoch + 1,
            'train_sequences': len(train_seqs),
            'val_sequences': len(val_seqs),
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }

        logger.info(f"Training complete. Best Val AUC: {best_val_auc:.3f}")

        return metrics

    def predict_patient(self, appointments_df: pd.DataFrame,
                        patient_id: str) -> Optional[Dict]:
        """
        Predict no-show probability for a patient's next appointment.

        Args:
            appointments_df: Historical appointments
            patient_id: Patient ID

        Returns:
            Dict with prediction and confidence, or None if insufficient history
        """
        if not self.is_trained:
            logger.warning("Model not trained")
            return None

        # Create sequence for this patient
        seq = self.feature_engineer.create_sequence_features(appointments_df, patient_id)

        if seq is None:
            return None

        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(seq.features).unsqueeze(0).to(self.device)
            lengths = torch.LongTensor([seq.sequence_length])

            outputs = self.model(features, lengths)

            # Get prediction for last (most recent) appointment pattern
            # This represents the model's learned "trajectory" for this patient
            last_prob = outputs[0, -1].item()

        return {
            'patient_id': patient_id,
            'noshow_probability': last_prob,
            'risk_level': 'high' if last_prob >= 0.3 else 'medium' if last_prob >= 0.15 else 'low',
            'sequence_length': seq.sequence_length,
            'model_type': f'sequence_{self.model_type}'
        }

    def save(self, path: str = None):
        """Save model to disk"""
        path = path or str(MODEL_SAVE_DIR / "sequence_noshow_model.pkl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'input_size': self.input_size,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_names': self.feature_engineer.feature_names
        }

        # T2.3: SHA-256-verified pickle write — sidecar lets load() reject
        # tampered weights at boot.
        from safe_loader import safe_save
        safe_save(model_data, path)

        logger.info(f"Sequence model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        # T2.3: verify SHA-256 sidecar before unpickling.
        from safe_loader import safe_load
        model_data = safe_load(path)

        self.model_type = model_data['model_type']
        self.hidden_size = model_data['hidden_size']
        self.num_layers = model_data['num_layers']
        self.dropout = model_data['dropout']
        self.bidirectional = model_data['bidirectional']
        self.input_size = model_data['input_size']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', [])
        self.feature_engineer.feature_names = model_data.get('feature_names', [])

        if model_data['model_state_dict'] and self.input_size:
            self._build_model(self.input_size)
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.eval()

        logger.info(f"Sequence model loaded from {path}")


# Example usage
if __name__ == "__main__":
    print("Sequence-Based No-Show Prediction Model")
    print("=" * 50)

    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
    else:
        print(f"PyTorch available, using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

        # Example with synthetic data
        print("\nCreating synthetic appointment data...")
        np.random.seed(42)

        n_patients = 50
        appointments = []

        for p in range(n_patients):
            n_appts = np.random.randint(5, 15)
            base_noshow_prob = np.random.uniform(0.05, 0.4)

            for a in range(n_appts):
                noshow_prob = base_noshow_prob + 0.05 * np.random.randn()
                noshow_prob = np.clip(noshow_prob, 0, 1)

                appointments.append({
                    'Patient_ID': f'P{p:03d}',
                    'Appointment_Date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=a*21 + np.random.randint(-7, 7)),
                    'Attended_Status': 'No' if np.random.random() < noshow_prob else 'Yes',
                    'Cycle_Number': a + 1,
                    'Planned_Duration': 60 + np.random.randint(0, 120),
                    'Clinical_Priority': np.random.choice(['P1', 'P2', 'P3', 'P4']),
                    'Travel_Time_Min': 15 + np.random.randint(0, 60)
                })

        df = pd.DataFrame(appointments)
        print(f"Created {len(df)} appointments for {n_patients} patients")

        # Train model
        print("\nTraining sequence model...")
        model = SequenceNoShowModel(model_type='gru', hidden_size=32, num_layers=1)
        metrics = model.train(df, epochs=20, batch_size=16)

        print(f"\nTraining Results:")
        print(f"  Best Validation AUC: {metrics['best_val_auc']:.3f}")
        print(f"  Epochs trained: {metrics['epochs_trained']}")

        # Test prediction
        print("\nTesting prediction for patient P000...")
        result = model.predict_patient(df, 'P000')
        if result:
            print(f"  No-show probability: {result['noshow_probability']:.3f}")
            print(f"  Risk level: {result['risk_level']}")
            print(f"  Based on {result['sequence_length']} appointments")
