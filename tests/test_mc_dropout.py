"""
Tests for ml/mc_dropout.py — Wave 3.2 (T3 coverage).

Covers:
- MonteCarloDropout construction defaults
- _predict_fallback path (returns MCDropoutResult with uncertainty fields)
- Summary dict shape
- predict_multitask via a tiny real torch net (skipped if torch missing)
- _enable_dropout correctly activates Dropout modules
- predict_sequence via tiny GRU wrapper (skipped if torch missing)
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.mc_dropout import MCDropoutResult, MonteCarloDropout, TORCH_AVAILABLE

torch = pytest.importorskip("torch") if TORCH_AVAILABLE else None


# --------------------------------------------------------------------------- #
# Construction / fallback
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_default_state(self):
        mc = MonteCarloDropout()
        assert mc.n_forward_passes == 100
        assert mc.confidence_level == 0.95
        assert mc.alpha == pytest.approx(0.05)

    def test_summary_keys(self):
        mc = MonteCarloDropout(n_forward_passes=5)
        s = mc.get_summary()
        assert s["method"] == "Monte Carlo Dropout"
        assert s["n_forward_passes"] == 5
        assert "uncertainty_decomposition" in s


# --------------------------------------------------------------------------- #
# Fallback path (no torch needed)
# --------------------------------------------------------------------------- #


class TestFallback:
    def test_fallback_returns_mcdropoutresult(self):
        mc = MonteCarloDropout(n_forward_passes=30)
        # Use the internal fallback directly — works regardless of torch.
        result = mc._predict_fallback(np.zeros(5), "P999")
        assert isinstance(result, MCDropoutResult)
        assert result.patient_id == "P999"
        # Epistemic + aleatoric are non-negative.
        assert result.epistemic_uncertainty >= 0.0
        assert result.aleatoric_uncertainty >= 0.0
        assert result.total_uncertainty >= 0.0
        assert 0.0 <= result.noshow_mean <= 1.0
        assert result.noshow_ci_lower <= result.noshow_ci_upper


# --------------------------------------------------------------------------- #
# Torch-backed tests (skipped when torch missing)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestTorchBacked:
    def _tiny_multitask_net(self, input_dim=5):
        """Minimal module emulating the MultiTaskNetwork interface MC Dropout expects."""
        import torch.nn as nn

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 8),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(8, 4),
                )
                self.noshow_head = nn.Linear(4, 1)
                self.duration_head = nn.Linear(4, 1)

            def forward(self, x):
                h = self.encoder(x)
                return {
                    "noshow_prob": self.noshow_head(h).squeeze(-1),
                    "duration_mean": self.duration_head(h).squeeze(-1),
                }

        return TinyNet()

    def test_enable_dropout_sets_train_mode(self):
        import torch.nn as nn

        dropout = nn.Dropout(0.3)
        module = nn.Sequential(nn.Linear(4, 4), dropout)
        module.eval()
        assert dropout.training is False
        MonteCarloDropout._enable_dropout(module)
        assert dropout.training is True

    def test_predict_multitask_end_to_end(self):
        net = self._tiny_multitask_net(input_dim=5)
        mc = MonteCarloDropout(n_forward_passes=10, confidence_level=0.9)
        features = np.random.default_rng(0).normal(0, 1, size=5).astype(np.float32)
        result = mc.predict_multitask(net, features, patient_id="P42")
        assert isinstance(result, MCDropoutResult)
        assert result.patient_id == "P42"
        assert result.n_forward_passes == 10
        # Dropout rate detected from the module.
        assert 0.0 < result.dropout_rate < 1.0

    def test_predict_sequence_end_to_end(self):
        import torch.nn as nn

        class TinyGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(3, 4, batch_first=True)
                self.dropout = nn.Dropout(0.25)
                self.head = nn.Linear(4, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                out = self.dropout(out)
                return self.head(out).squeeze(-1)

        net = TinyGRU()
        mc = MonteCarloDropout(n_forward_passes=5, confidence_level=0.95)
        seq = torch.randn(1, 4, 3)  # batch=1, seq_len=4, features=3
        result = mc.predict_sequence(net, seq, patient_id="Pseq")
        assert isinstance(result, MCDropoutResult)
        assert result.patient_id == "Pseq"
        assert result.duration_mean == 0.0
