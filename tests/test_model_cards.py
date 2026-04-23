"""
Tests for ml.model_cards — ModelCardGenerator.

Covers:
    * generate_card for a known model key returns all Model Card sections.
    * Unknown key returns an {'error': ...} dict.
    * generate_all_cards summarises every known model and counts trained ones.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.model_cards import ModelCardGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TrainedStub:
    """Stub model with is_trained=True and a feature_names list."""
    is_trained = True
    feature_names = ['age', 'distance_km', 'priority']


def _make_historical_data(n: int = 80) -> pd.DataFrame:
    """Synthetic appointments frame with enough rows (>= 50) for subgroup metrics."""
    rows = []
    for i in range(n):
        rows.append({
            'Patient_ID': f'P{i:03d}',
            'Age_Band': 'Under 40' if i % 2 == 0 else 'Over 75',
            'Site_Code': 'VCC' if i % 3 == 0 else 'NVCC',
            'Priority': 'P1' if i % 4 == 0 else 'P3',
            'Attended_Status': 'Yes' if i % 5 != 0 else 'No',
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_card_returns_all_sections_for_known_model():
    gen = ModelCardGenerator()
    card = gen.generate_card(
        'noshow_ensemble',
        model=TrainedStub(),
        historical_data=_make_historical_data(),
        is_trained=True,
    )

    # All top-level sections defined in the dataclass / generator
    for section in ('model_name', 'model_version', 'model_type',
                    'date_created', 'is_trained',
                    'intended_use', 'training_data',
                    'performance', 'limitations',
                    'ethical_considerations', 'fairness_assessment',
                    'maintenance'):
        assert section in card

    assert card['is_trained'] is True
    # Subgroup metrics must be populated given our 80-row frame.
    assert len(card['performance']['by_subgroup']) > 0
    # Intended-use structure
    assert 'primary_use' in card['intended_use']
    assert isinstance(card['intended_use']['intended_users'], list)


def test_generate_card_unknown_model_returns_error():
    gen = ModelCardGenerator()
    card = gen.generate_card('totally_nonexistent_model')
    assert 'error' in card
    assert 'Unknown' in card['error']


def test_generate_all_cards_covers_every_known_key():
    gen = ModelCardGenerator()
    models = {'noshow_ensemble': TrainedStub()}  # only one trained
    bundle = gen.generate_all_cards(
        models=models,
        historical_data=_make_historical_data(),
    )

    # All keys from MODEL_DEFINITIONS must be generated
    expected_keys = set(ModelCardGenerator.MODEL_DEFINITIONS.keys())
    assert set(bundle['cards'].keys()) == expected_keys

    # Top-level summary
    assert bundle['total_models'] == len(expected_keys)
    assert bundle['trained_models'] == 1
    assert 'generated_at' in bundle
