"""
Model Training Script
=====================

Train ML models on historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import argparse

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR,
    MODEL_SAVE_DIR,
    get_logger
)
from .feature_engineering import FeatureEngineer
from .noshow_model import NoShowModel
from .duration_model import DurationModel

logger = get_logger(__name__)


def load_training_data(data_path: str = None) -> pd.DataFrame:
    """
    Load historical appointment data for training.

    Args:
        data_path: Path to data file (Excel or CSV)

    Returns:
        DataFrame with training data
    """
    if data_path is None:
        data_path = DATA_DIR / "sample_data" / "historical_appointments.xlsx"

    if not Path(data_path).exists():
        logger.warning(f"Training data not found at {data_path}")
        logger.info("Generating synthetic training data for demonstration")
        return generate_synthetic_data()

    # Load data based on file type
    if str(data_path).endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    logger.info(f"Loaded {len(df)} records from {data_path}")
    return df


def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic training data using SACT v4.0 compliant field names.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with synthetic data matching our dataset column names
    """
    np.random.seed(42)

    # Patient data — using our SACT v4.0 aligned column names
    data = {
        'Patient_ID': [f'P{i:05d}' for i in range(n_samples)],
        'Postcode_District': np.random.choice(
            ['CF14', 'CF15', 'CF23', 'CF24', 'CF10', 'NP20', 'CF37', 'CF5'],
            n_samples,
            p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]
        ),
        'Total_Appointments_Before': np.random.poisson(8, n_samples),
        'Previous_NoShows': np.random.binomial(3, 0.1, n_samples),
        'Previous_Cancellations': np.random.binomial(3, 0.15, n_samples),
    }

    # Appointment data
    base_date = datetime.now() - timedelta(days=365)
    data['Date'] = [
        (base_date + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
        for _ in range(n_samples)
    ]
    data['Appointment_Hour'] = np.random.choice(
        range(8, 17), n_samples,
        p=[0.05, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]
    )

    data['Priority'] = np.random.choice(
        ['P1', 'P2', 'P3', 'P4'],
        n_samples,
        p=[0.1, 0.4, 0.35, 0.15]
    )

    # SACT protocol codes (matching our regimens)
    data['Regimen_Code'] = np.random.choice(
        ['RCHOP', 'FECT', 'PACW', 'CARBPAC', 'FOLFOX', 'PEMBRO', 'DOCE', 'GEM'],
        n_samples
    )

    data['Planned_Duration'] = np.random.choice(
        [60, 90, 120, 180, 240],
        n_samples,
        p=[0.1, 0.3, 0.3, 0.2, 0.1]
    )

    data['Days_Booked_In_Advance'] = np.random.exponential(14, n_samples).astype(int)
    data['Cycle_Number'] = np.random.randint(1, 12, n_samples)
    data['Is_First_Cycle'] = (data['Cycle_Number'] == 1)
    data['Age'] = np.random.randint(30, 85, n_samples)
    data['Has_Comorbidities'] = np.random.binomial(1, 0.35, n_samples).astype(bool)
    data['IV_Access_Difficulty'] = np.random.binomial(1, 0.15, n_samples).astype(bool)
    data['Travel_Distance_KM'] = np.random.choice([5, 10, 15, 20, 30, 50], n_samples)
    data['Performance_Status'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.25, 0.35, 0.25, 0.12, 0.03])

    # External conditions
    data['Weather_Severity'] = np.random.beta(1, 5, n_samples)
    data['Traffic_Delay_Minutes'] = np.maximum(0, data['Weather_Severity'] * 15 + np.random.normal(0, 5, n_samples))

    # Outcomes
    noshow_prob = (
        0.05 +
        data['Previous_NoShows'] / 10 * 0.3 +
        (data['Priority'] == 'P4').astype(float) * 0.05 +
        data['Weather_Severity'] * 0.1 +
        (data['Days_Booked_In_Advance'] > 30).astype(float) * 0.05
    )
    noshow_prob = np.clip(noshow_prob, 0.01, 0.5)

    data['Attended_Status'] = np.where(
        np.random.binomial(1, noshow_prob), 'No', 'Yes'
    )
    data['Patient_NoShow_Rate'] = data['Previous_NoShows'] / np.maximum(data['Total_Appointments_Before'], 1)

    # Actual duration
    base_duration = np.array(data['Planned_Duration'])
    duration_noise = np.random.normal(0, 15, n_samples)
    complexity_factor = np.random.uniform(0.8, 1.3, n_samples)

    data['Actual_Duration'] = np.clip(
        base_duration * complexity_factor + duration_noise,
        30, 360
    ).astype(int)

    df = pd.DataFrame(data)

    logger.info(f"Generated {n_samples} synthetic training samples")
    logger.info(f"No-show rate: {(df['Attended_Status'] == 'No').mean():.1%}")

    return df


def prepare_features(df: pd.DataFrame, engineer: FeatureEngineer) -> Tuple:
    """
    Prepare features for model training using SACT v4.0 column names.

    Args:
        df: Raw data DataFrame
        engineer: FeatureEngineer instance

    Returns:
        Tuple of (X, y_noshow, y_duration)
    """
    patients = []
    appointments = []

    for _, row in df.iterrows():
        patient = {
            'patient_id': row['Patient_ID'],
            'postcode': row.get('Postcode_District', 'CF14'),
            'total_appointments': row.get('Total_Appointments_Before', 5),
            'no_shows': row.get('Previous_NoShows', 0),
            'cancellations': row.get('Previous_Cancellations', 0),
        }
        patients.append(patient)

        apt_date = row.get('Date', datetime.now().strftime('%Y-%m-%d'))
        if isinstance(apt_date, str):
            apt_time = datetime.fromisoformat(apt_date)
        else:
            apt_time = apt_date
        apt_time = apt_time.replace(hour=int(row.get('Appointment_Hour', 10)))

        appointment = {
            'appointment_time': apt_time,
            'site_code': 'WC',
            'priority': row.get('Priority', 'P3'),
            'protocol': row.get('Regimen_Code', 'FOLFOX'),
            'expected_duration': row.get('Planned_Duration', 120),
            'days_until': row.get('Days_Booked_In_Advance', 7)
        }
        appointments.append(appointment)

    # External data
    external = {
        'weather': {'severity': 0.1},
        'traffic': {'severity': 0.1, 'delay_minutes': 0}
    }

    # Create features DataFrame
    X = engineer.create_features_dataframe(patients, appointments, external)

    # Add external features from data
    if 'Weather_Severity' in df.columns:
        X['Weather_Severity'] = df['Weather_Severity'].values
    if 'Traffic_Delay_Minutes' in df.columns:
        X['Traffic_Delay_Minutes'] = df['Traffic_Delay_Minutes'].values

    # Targets
    y_noshow = (df['Attended_Status'] == 'No').astype(int).values
    y_duration = df['Actual_Duration'].values

    # Remove non-feature columns
    feature_cols = [c for c in X.columns if c not in ['patient_id', 'Patient_ID']]
    X = X[feature_cols]

    return X, y_noshow, y_duration


def train_models(data_path: str = None, save: bool = True) -> Dict:
    """
    Train all ML models.

    Args:
        data_path: Path to training data
        save: Whether to save trained models

    Returns:
        Dict with training results
    """
    logger.info("Starting model training pipeline")

    # Load data
    df = load_training_data(data_path)

    # Initialize components
    engineer = FeatureEngineer()
    noshow_model = NoShowModel()
    duration_model = DurationModel()

    # Prepare features
    logger.info("Preparing features...")
    X, y_noshow, y_duration = prepare_features(df, engineer)
    logger.info(f"Created {len(X.columns)} features for {len(X)} samples")

    results = {}

    # Train no-show model
    logger.info("\n" + "=" * 50)
    logger.info("Training No-Show Prediction Model")
    logger.info("=" * 50)

    noshow_metrics = noshow_model.train(X, pd.Series(y_noshow))
    results['noshow'] = noshow_metrics

    print("\nNo-Show Model Results:")
    for model_name, metrics in noshow_metrics.items():
        if isinstance(metrics, dict):
            print(f"  {model_name}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.3f}")

    # Train duration model
    logger.info("\n" + "=" * 50)
    logger.info("Training Duration Prediction Model")
    logger.info("=" * 50)

    # Filter out no-shows for duration training
    duration_mask = y_noshow == 0
    X_duration = X[duration_mask]
    y_dur = y_duration[duration_mask]

    duration_metrics = duration_model.train(
        X_duration,
        pd.Series(y_dur)
    )
    results['duration'] = duration_metrics

    print("\nDuration Model Results:")
    for model_name, metrics in duration_metrics.items():
        if isinstance(metrics, dict):
            print(f"  {model_name}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.1f}")

    # Save models
    if save:
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        noshow_model.save(MODEL_SAVE_DIR / "noshow_model.pkl")
        duration_model.save(MODEL_SAVE_DIR / "duration_model.pkl")

        # Save feature importance
        importance = {
            'noshow': noshow_model.get_feature_importance(),
            'duration': duration_model.get_feature_importance()
        }
        with open(MODEL_SAVE_DIR / "feature_importance.json", 'w') as f:
            json.dump(importance, f, indent=2)

        # Save training results
        with open(MODEL_SAVE_DIR / "training_results.json", 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'samples': len(df),
                'features': len(X.columns),
                'results': {
                    k: {
                        kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                        for kk, vv in v.items()
                    } if isinstance(v, dict) else float(v)
                    for k, v in results.items()
                }
            }, f, indent=2)

        logger.info(f"Models saved to {MODEL_SAVE_DIR}")

    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train SACT scheduling ML models")
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to training data file (Excel or CSV)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained models'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for training'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=5000,
        help='Number of synthetic samples to generate'
    )

    args = parser.parse_args()

    if args.synthetic:
        df = generate_synthetic_data(args.samples)
        # Save synthetic data
        synthetic_path = DATA_DIR / "synthetic_training_data.csv"
        df.to_csv(synthetic_path, index=False)
        logger.info(f"Synthetic data saved to {synthetic_path}")
        data_path = synthetic_path
    else:
        data_path = args.data

    results = train_models(data_path=data_path, save=not args.no_save)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


# Support for Tuple type hint
from typing import Tuple


if __name__ == "__main__":
    main()
