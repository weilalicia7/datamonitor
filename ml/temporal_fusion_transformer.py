"""
Temporal Fusion Transformer (Lite) — Dissertation §2.3
======================================================

A pragmatic reimplementation of the Temporal Fusion Transformer
(Lim et al. 2021, *Temporal Fusion Transformers for interpretable
multi-horizon time series forecasting*) for the SACT scheduling
context.  The design keeps the three innovations that matter most
for a clinical decision-support tool:

  1. **Gated Residual Networks (GRN).**  Small MLP blocks with ELU
     activation, dropout, and a Gated Linear Unit skip connection.
     Each GRN is also gated by a context vector so static features
     (patient) can modulate every dynamic feature's representation.
  2. **Variable Selection Networks (VSN).**  At each time step (and
     for the static slice) a softmax over feature embeddings picks
     which inputs matter, with the weights exposed for
     interpretability.
  3. **Interpretable multi-head attention.**  The horizon-1 prediction
     attends over the past window; the attention weights are
     returned alongside the prediction so operators can ask
     "*why* is this patient flagged?".

The outputs are joint — one model, three heads:

  * `p_noshow`  — probability the patient no-shows the NEXT
                  scheduled appointment.
  * `duration_q[10/50/90]` — native quantile regression, i.e. the
                  predicted treatment duration with its own
                  10/50/90 prediction interval.
  * `p_cancel`  — probability the next appointment is cancelled
                  (distinct from no-show: cancellation is typically
                  patient-initiated in advance, no-show is same-day).

The model is trained jointly with

    L = BCE(p̂_ns, y_ns) + λ_q · Σ_τ pinball(q̂_τ, y_dur, τ)
                         + λ_c · BCE(p̂_cx, y_cx)

with λ_q = 0.01 (scaling for minute-valued durations) and λ_c = 0.5.

The module deliberately runs **invisibly** in the prediction
pipeline: when fitted, a hook in `flask_app.run_ml_predictions()`
routes every patient through the TFT and overrides
`patient.noshow_probability` / `patient.expected_duration` with the
joint-model outputs; when unfitted the legacy ensemble path runs
unchanged.  Only a diagnostic/status Flask endpoint is exposed; no
dedicated UI tab is added.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math
import pickle

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, MODELS_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hyper-parameters (kept deliberately small so CPU-only training is feasible
# in the dissertation setting).  The architecture is faithful to Lim et al.
# 2021 but every "d_model = 160 / 8-head" in the paper is scaled down by an
# order of magnitude here because we have ~1,900 historical records rather
# than the millions the original paper targets.
# ---------------------------------------------------------------------------
DEFAULT_PAST_WINDOW: int = 10          # number of past appointments consumed
DEFAULT_STATIC_DIM: int = 5            # age, gender, priority, postcode_band, baseline_rate
DEFAULT_OBS_DIM: int = 6               # past observed features per step
DEFAULT_D_MODEL: int = 32
DEFAULT_N_HEADS: int = 4
DEFAULT_DROPOUT: float = 0.1
DEFAULT_LR: float = 1e-3
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_EPOCHS: int = 60
DEFAULT_LAMBDA_Q: float = 0.01
DEFAULT_LAMBDA_C: float = 0.5
DEFAULT_QUANTILES: Tuple[float, ...] = (0.10, 0.50, 0.90)

TFT_MODEL_FILE: Path = MODELS_DIR / 'tft_lite.pt'
TFT_META_FILE: Path = MODELS_DIR / 'tft_lite_meta.pkl'
TFT_HISTORY_FILE: Path = DATA_CACHE_DIR / 'tft_history.jsonl'


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


if TORCH_AVAILABLE:

    class GatedLinearUnit(nn.Module):
        """GLU: y = x · σ(gate(x))."""

        def __init__(self, d_in: int, d_out: int):
            super().__init__()
            self.fc = nn.Linear(d_in, d_out)
            self.gate = nn.Linear(d_in, d_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x) * torch.sigmoid(self.gate(x))


    class GatedResidualNetwork(nn.Module):
        """
        GRN from TFT paper.  Non-linear transform + GLU + residual skip
        + LayerNorm.  Optional context vector injected additively before
        the non-linearity (used by Variable Selection + static enrichment).
        """

        def __init__(
            self,
            d_in: int,
            d_hidden: int,
            d_out: int,
            d_context: int = 0,
            dropout: float = DEFAULT_DROPOUT,
        ):
            super().__init__()
            self.proj_in = nn.Linear(d_in, d_hidden)
            self.proj_ctx = nn.Linear(d_context, d_hidden, bias=False) if d_context > 0 else None
            self.fc = nn.Linear(d_hidden, d_hidden)
            self.drop = nn.Dropout(dropout)
            self.glu = GatedLinearUnit(d_hidden, d_out)
            self.norm = nn.LayerNorm(d_out)
            self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

        def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
            h = self.proj_in(x)
            if self.proj_ctx is not None and context is not None:
                if context.dim() < h.dim():
                    context = context.unsqueeze(1).expand(-1, h.shape[1], -1) if h.dim() == 3 else context
                h = h + self.proj_ctx(context)
            h = F.elu(h)
            h = self.fc(h)
            h = self.drop(h)
            h = self.glu(h)
            return self.norm(h + self.skip(x))


    class VariableSelectionNetwork(nn.Module):
        """
        Selects a soft-max-weighted combination of K feature embeddings.
        Exposes the per-step selection weights for interpretability.
        """

        def __init__(self, d_in_per_feature: int, n_features: int, d_model: int, dropout: float = DEFAULT_DROPOUT):
            super().__init__()
            self.n_features = n_features
            self.weights_grn = GatedResidualNetwork(
                d_in=d_in_per_feature * n_features,
                d_hidden=d_model,
                d_out=n_features,
                dropout=dropout,
            )
            # Per-feature transforms
            self.feature_grns = nn.ModuleList(
                [GatedResidualNetwork(d_in_per_feature, d_model, d_model, dropout=dropout)
                 for _ in range(n_features)]
            )

        def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            features shape: (B, n_features, d_in_per_feature) for static
                         or (B, T, n_features, d_in_per_feature) for temporal
            Returns (combined [B,(T,)d_model], weights [B,(T,)n_features]).
            """
            if features.dim() == 3:  # static
                B, F_, D = features.shape
                flat = features.reshape(B, F_ * D)
                w = torch.softmax(self.weights_grn(flat), dim=-1)  # (B, F)
                transformed = torch.stack(
                    [self.feature_grns[i](features[:, i, :]) for i in range(F_)], dim=1
                )  # (B, F, d_model)
                combined = (w.unsqueeze(-1) * transformed).sum(dim=1)
                return combined, w
            # temporal: (B, T, F, D)
            B, T, F_, D = features.shape
            flat = features.reshape(B, T, F_ * D)
            w = torch.softmax(self.weights_grn(flat), dim=-1)  # (B, T, F)
            transformed = torch.stack(
                [self.feature_grns[i](features[:, :, i, :]) for i in range(F_)], dim=2
            )  # (B, T, F, d_model)
            combined = (w.unsqueeze(-1) * transformed).sum(dim=2)
            return combined, w


    class TFTLite(nn.Module):
        """
        Scaled-down Temporal Fusion Transformer for scheduling.

        Inputs
        ------
        static_x : (B, static_dim)           — patient-level features
        past_x   : (B, past_window, obs_dim) — past appointments

        Outputs
        -------
        dict with keys:
            'p_noshow'        : (B,)
            'duration_q'      : (B, 3)         quantiles in minutes
            'p_cancel'        : (B,)
            'attention'       : (B, past_window)   attention over past
            'vsn_weights_past': (B, past_window, obs_dim)
            'vsn_weights_static': (B, static_dim)
        """

        def __init__(
            self,
            static_dim: int = DEFAULT_STATIC_DIM,
            obs_dim: int = DEFAULT_OBS_DIM,
            past_window: int = DEFAULT_PAST_WINDOW,
            d_model: int = DEFAULT_D_MODEL,
            n_heads: int = DEFAULT_N_HEADS,
            dropout: float = DEFAULT_DROPOUT,
            n_quantiles: int = len(DEFAULT_QUANTILES),
        ):
            super().__init__()
            self.static_dim = static_dim
            self.obs_dim = obs_dim
            self.past_window = past_window
            self.d_model = d_model

            # Per-feature linear embeddings so VSN sees a 3-D tensor.
            self.static_embed = nn.ModuleList(
                [nn.Linear(1, d_model) for _ in range(static_dim)]
            )
            self.obs_embed = nn.ModuleList(
                [nn.Linear(1, d_model) for _ in range(obs_dim)]
            )

            self.static_vsn = VariableSelectionNetwork(
                d_in_per_feature=d_model, n_features=static_dim,
                d_model=d_model, dropout=dropout,
            )
            self.past_vsn = VariableSelectionNetwork(
                d_in_per_feature=d_model, n_features=obs_dim,
                d_model=d_model, dropout=dropout,
            )

            # Static enrichment: GRN context for LSTM input
            self.static_enrichment = GatedResidualNetwork(
                d_in=d_model, d_hidden=d_model, d_out=d_model,
                d_context=d_model, dropout=dropout,
            )

            self.lstm = nn.LSTM(
                input_size=d_model, hidden_size=d_model,
                num_layers=1, batch_first=True, dropout=0.0,
            )

            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads, dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(d_model)
            self.post_attn_grn = GatedResidualNetwork(
                d_model, d_model, d_model, dropout=dropout,
            )

            # Output heads — sit on top of the last-timestep representation
            self.head_noshow = nn.Linear(d_model, 1)
            self.head_cancel = nn.Linear(d_model, 1)
            self.head_quantile = nn.Linear(d_model, n_quantiles)

        def _embed_static(self, static_x: torch.Tensor) -> torch.Tensor:
            """(B, static_dim) → (B, static_dim, d_model)."""
            parts = [self.static_embed[i](static_x[:, i:i+1]) for i in range(self.static_dim)]
            return torch.stack(parts, dim=1)

        def _embed_past(self, past_x: torch.Tensor) -> torch.Tensor:
            """(B, T, obs_dim) → (B, T, obs_dim, d_model)."""
            parts = [self.obs_embed[i](past_x[:, :, i:i+1]) for i in range(self.obs_dim)]
            return torch.stack(parts, dim=2)

        def forward(self, static_x: torch.Tensor, past_x: torch.Tensor) -> Dict[str, torch.Tensor]:
            # Embeddings
            static_emb = self._embed_static(static_x)                  # (B, S, d)
            past_emb = self._embed_past(past_x)                        # (B, T, O, d)

            # Variable selection
            static_vec, static_w = self.static_vsn(static_emb)         # (B, d), (B, S)
            past_vec, past_w = self.past_vsn(past_emb)                 # (B, T, d), (B, T, O)

            # Static enrichment of past sequence
            past_enriched = self.static_enrichment(past_vec, context=static_vec)

            # Temporal processing
            lstm_out, _ = self.lstm(past_enriched)                     # (B, T, d)

            # Self-attention on LSTM outputs (interpretable attention)
            attn_out, attn_w = self.attn(lstm_out, lstm_out, lstm_out, need_weights=True,
                                         average_attn_weights=True)    # (B, T, d), (B, T, T)
            attn_out = self.attn_norm(attn_out + lstm_out)
            attn_out = self.post_attn_grn(attn_out)

            # Horizon-1 = the representation at the most recent step
            # (last position of the past window).  This is the cleanest
            # "current prediction" slot.
            final_repr = attn_out[:, -1, :]                            # (B, d)

            # Attention weights at the last query step → interpretability
            attention_scalar = attn_w[:, -1, :]                        # (B, T)

            # Output heads — NaN-safe.  If any upstream feature comes in
            # as NaN/inf (eg. a historical row missing Actual_Duration) we
            # clip logits and clamp sigmoids to (0, 1) exclusive so BCE
            # stays finite and differentiable.
            noshow_logit = self.head_noshow(final_repr).squeeze(-1)
            cancel_logit = self.head_cancel(final_repr).squeeze(-1)
            noshow_logit = torch.nan_to_num(noshow_logit, nan=0.0, posinf=20.0, neginf=-20.0)
            cancel_logit = torch.nan_to_num(cancel_logit, nan=0.0, posinf=20.0, neginf=-20.0)
            p_noshow = torch.sigmoid(noshow_logit).clamp(1e-6, 1 - 1e-6)
            p_cancel = torch.sigmoid(cancel_logit).clamp(1e-6, 1 - 1e-6)
            duration_q = self.head_quantile(final_repr)                # (B, Q)
            duration_q = torch.nan_to_num(duration_q, nan=0.0, posinf=5.0, neginf=-5.0)

            return {
                'p_noshow': p_noshow,
                'p_cancel': p_cancel,
                'duration_q': duration_q,
                'attention': attention_scalar,
                'vsn_weights_past': past_w,
                'vsn_weights_static': static_w,
            }


    def pinball_loss(
        q_pred: torch.Tensor, y_true: torch.Tensor, quantiles: Tuple[float, ...] = DEFAULT_QUANTILES
    ) -> torch.Tensor:
        """Vectorised pinball loss summed over the quantile list."""
        total = 0.0
        for i, tau in enumerate(quantiles):
            e = y_true - q_pred[:, i]
            total = total + torch.max(tau * e, (tau - 1) * e).mean()
        return total


# ---------------------------------------------------------------------------
# Training wrapper — owns preprocessing, fit, predict, persistence
# ---------------------------------------------------------------------------


@dataclass
class TFTFitResult:
    """Summary of a single TFT fit."""
    n_samples: int
    epochs: int
    final_loss: float
    noshow_auc: Optional[float]
    cancel_auc: Optional[float]
    duration_q50_mae: Optional[float]
    duration_q90_coverage: Optional[float]
    past_window: int
    d_model: int
    n_heads: int
    converged: bool
    fit_ts: str


class TFTTrainer:
    """
    Non-PyTorch-aware façade so call sites don't need to import torch.
    Manages: feature extraction from historical appointments, training,
    single-patient inference, saving/loading, and a fitted-flag.
    """

    def __init__(
        self,
        past_window: int = DEFAULT_PAST_WINDOW,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        model_path: Path = TFT_MODEL_FILE,
        meta_path: Path = TFT_META_FILE,
        history_path: Path = TFT_HISTORY_FILE,
    ):
        self.past_window = past_window
        self.d_model = d_model
        self.n_heads = n_heads
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.history_path = Path(history_path)
        self.model: Optional['TFTLite'] = None
        self.last_fit: Optional[TFTFitResult] = None
        # Input normalisation statistics — set by fit(), required by predict()
        self.static_mean: Optional[np.ndarray] = None
        self.static_std: Optional[np.ndarray] = None
        self.obs_mean: Optional[np.ndarray] = None
        self.obs_std: Optional[np.ndarray] = None
        self.duration_mean: float = 90.0
        self.duration_std: float = 30.0

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        if TORCH_AVAILABLE:
            self._try_load_state()

    def is_fitted(self) -> bool:
        return TORCH_AVAILABLE and self.model is not None and self.last_fit is not None

    # -------- preprocessing --------

    @staticmethod
    def _extract_static(row: Dict) -> np.ndarray:
        """Pull the 5-dim static vector from a historical row or patient dict."""
        age = float(row.get('Age', row.get('age', 60)) or 60)
        gender_raw = row.get('Person_Stated_Gender_Code', row.get('gender_code',
                              row.get('Gender', 9)))
        try:
            gender = float(gender_raw) if gender_raw not in (None, '') else 9.0
        except (TypeError, ValueError):
            gender = 9.0
        priority_raw = row.get('Priority', row.get('priority', 'P3'))
        if isinstance(priority_raw, str) and priority_raw.startswith('P'):
            try:
                priority = float(priority_raw[1:])
            except ValueError:
                priority = 3.0
        else:
            priority = float(priority_raw or 3)
        postcode_band = 1.0  # near/medium/remote — collapsed to 1 for simplicity
        if row.get('Patient_Postcode', ''):
            pc = str(row.get('Patient_Postcode', ''))
            if pc.startswith('SA') or pc.startswith('CF81') or pc.startswith('CF82'):
                postcode_band = 2.0  # remote
            elif pc.startswith('CF10') or pc.startswith('CF11') or pc.startswith('CF14'):
                postcode_band = 0.0  # near
        baseline_rate = float(row.get('Patient_NoShow_Rate',
                                      row.get('noshow_probability', 0.15)) or 0.15)
        return np.array([age, gender, priority, postcode_band, baseline_rate], dtype=float)

    def _build_dataset(
        self, df
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct (static_X, past_X, y_noshow, y_cancel, y_duration) arrays.

        For each historical appointment we collect the `past_window` prior
        appointments of the same patient; if fewer than 3 priors exist the
        sample is dropped.  The remaining slots are zero-padded.
        """
        import pandas as pd
        if 'Patient_ID' not in df.columns:
            raise ValueError("TFTTrainer requires Patient_ID column.")

        df = df.sort_values(['Patient_ID', 'Date']) if 'Date' in df.columns else df
        static_list, past_list, y_ns, y_cx, y_dur = [], [], [], [], []

        for pid, group in df.groupby('Patient_ID'):
            rows = group.to_dict('records')
            for i in range(len(rows)):
                if i < 3:
                    continue  # need at least 3 priors
                cur = rows[i]
                # targets
                attended = str(cur.get('Attended_Status', 'Yes')).strip().lower()
                y_ns.append(0 if attended in ('yes', 'attended') else 1 if attended in ('no',) else 0)
                y_cx.append(1 if attended in ('cancelled', 'cancel', 'cancellation') else 0)
                dur_val = cur.get('Actual_Duration', cur.get('Planned_Duration', 90))
                try:
                    dur = float(dur_val) if dur_val not in (None, '') else 90.0
                except (TypeError, ValueError):
                    dur = 90.0
                y_dur.append(dur)

                # static
                static_list.append(self._extract_static(cur))

                # past window
                past_rows = rows[max(0, i - self.past_window):i]
                past_arr = np.zeros((self.past_window, DEFAULT_OBS_DIM), dtype=float)
                for j, pr in enumerate(past_rows[-self.past_window:]):
                    slot = self.past_window - len(past_rows) + j
                    if slot < 0:
                        slot = j
                    past_slot = min(slot, self.past_window - 1)
                    p_attended = str(pr.get('Attended_Status', 'Yes')).strip().lower()
                    past_arr[past_slot, 0] = 0 if p_attended in ('yes', 'attended') else 1
                    past_arr[past_slot, 1] = float(pr.get('Actual_Duration', pr.get('Planned_Duration', 90)) or 90)
                    past_arr[past_slot, 2] = float(pr.get('Cycle_Number', 1) or 1)
                    past_arr[past_slot, 3] = float(pr.get('Weather_Severity', 0.0) or 0.0)
                    past_arr[past_slot, 4] = float(pr.get('Day_Of_Week_Num', pr.get('Day_Of_Week', 2)) or 2)
                    past_arr[past_slot, 5] = float(pr.get('Travel_Distance_KM', 15) or 15)
                past_list.append(past_arr)

        if not static_list:
            raise ValueError("No training samples could be built (need ≥3 prior appts per patient).")

        static_X = np.vstack(static_list).astype(np.float32)
        past_X = np.stack(past_list, axis=0).astype(np.float32)
        y_ns_arr = np.array(y_ns, dtype=np.float32)
        y_cx_arr = np.array(y_cx, dtype=np.float32)
        y_dur_arr = np.array(y_dur, dtype=np.float32)
        return static_X, past_X, y_ns_arr, y_cx_arr, y_dur_arr

    # -------- fit / predict --------

    def fit(
        self,
        historical_df,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        lr: float = DEFAULT_LR,
    ) -> TFTFitResult:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for TFT training.")

        static_X, past_X, y_ns, y_cx, y_dur = self._build_dataset(historical_df)
        # Defensive: drop rows whose targets are NaN (cancelled rows without
        # a recorded actual_duration sometimes leak in) and replace any NaN
        # in inputs with 0.0 before computing normalisation statistics.
        mask = np.isfinite(y_ns) & np.isfinite(y_cx) & np.isfinite(y_dur)
        if not mask.all():
            static_X = static_X[mask]; past_X = past_X[mask]
            y_ns = y_ns[mask]; y_cx = y_cx[mask]; y_dur = y_dur[mask]
        static_X = np.nan_to_num(static_X, nan=0.0, posinf=0.0, neginf=0.0)
        past_X = np.nan_to_num(past_X, nan=0.0, posinf=0.0, neginf=0.0)
        n = static_X.shape[0]
        if n < 10:
            raise ValueError(f"Not enough clean samples for TFT training (got {n}).")

        # Normalisation
        self.static_mean = static_X.mean(axis=0)
        self.static_std = static_X.std(axis=0) + 1e-6
        self.obs_mean = past_X.reshape(-1, past_X.shape[-1]).mean(axis=0)
        self.obs_std = past_X.reshape(-1, past_X.shape[-1]).std(axis=0) + 1e-6
        self.duration_mean = float(y_dur.mean())
        self.duration_std = float(y_dur.std() + 1e-6)

        Xs = (static_X - self.static_mean) / self.static_std
        Xp = (past_X - self.obs_mean) / self.obs_std
        yd = (y_dur - self.duration_mean) / self.duration_std

        self.model = TFTLite(
            static_dim=static_X.shape[1],
            obs_dim=past_X.shape[-1],
            past_window=self.past_window,
            d_model=self.d_model,
            n_heads=self.n_heads,
        )
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        Xs_t = torch.tensor(Xs, dtype=torch.float32)
        Xp_t = torch.tensor(Xp, dtype=torch.float32)
        y_ns_t = torch.tensor(y_ns, dtype=torch.float32)
        y_cx_t = torch.tensor(y_cx, dtype=torch.float32)
        yd_t = torch.tensor(yd, dtype=torch.float32)

        last_loss = float('inf')
        converged = False
        for ep in range(epochs):
            perm = torch.randperm(n)
            ep_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                out = self.model(Xs_t[idx], Xp_t[idx])
                l_ns = F.binary_cross_entropy(out['p_noshow'], y_ns_t[idx])
                l_cx = F.binary_cross_entropy(out['p_cancel'], y_cx_t[idx])
                l_q = pinball_loss(out['duration_q'], yd_t[idx])
                loss = l_ns + DEFAULT_LAMBDA_C * l_cx + DEFAULT_LAMBDA_Q * l_q
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * idx.numel()
            ep_loss /= max(1, n)
            if abs(last_loss - ep_loss) < 1e-5 and ep > 10:
                converged = True
                break
            last_loss = ep_loss
            if ep % max(1, epochs // 5) == 0:
                logger.info(f"TFT ep {ep+1}/{epochs}  loss={ep_loss:.4f}")

        # Eval
        self.model.eval()
        with torch.no_grad():
            out = self.model(Xs_t, Xp_t)
        noshow_auc = _auc_safe(y_ns, out['p_noshow'].cpu().numpy())
        cancel_auc = _auc_safe(y_cx, out['p_cancel'].cpu().numpy())
        q50_pred = out['duration_q'][:, 1].cpu().numpy() * self.duration_std + self.duration_mean
        q90_pred = out['duration_q'][:, 2].cpu().numpy() * self.duration_std + self.duration_mean
        mae = float(np.mean(np.abs(q50_pred - y_dur)))
        cov_90 = float(np.mean(y_dur <= q90_pred))

        fit = TFTFitResult(
            n_samples=n,
            epochs=ep + 1,
            final_loss=float(last_loss),
            noshow_auc=noshow_auc,
            cancel_auc=cancel_auc,
            duration_q50_mae=mae,
            duration_q90_coverage=cov_90,
            past_window=self.past_window,
            d_model=self.d_model,
            n_heads=self.n_heads,
            converged=converged,
            fit_ts=datetime.utcnow().isoformat(timespec='seconds'),
        )
        self.last_fit = fit
        self._save_state()
        self._append_history(fit)
        logger.info(
            f"TFT fit complete: N={n} loss={last_loss:.4f} "
            f"noshow_auc={noshow_auc} q50_mae={mae:.1f}"
        )
        return fit

    def predict_single(self, patient: Dict, past_rows: List[Dict]) -> Dict:
        """Inference for one patient given their past appointment history."""
        if not self.is_fitted():
            raise RuntimeError("TFT is not fitted; predict_single unavailable.")

        static_vec = self._extract_static(patient)
        past_arr = np.zeros((self.past_window, DEFAULT_OBS_DIM), dtype=float)
        past_rows = past_rows[-self.past_window:]
        for j, pr in enumerate(past_rows):
            slot = self.past_window - len(past_rows) + j
            p_attended = str(pr.get('Attended_Status', 'Yes')).strip().lower()
            past_arr[slot, 0] = 0 if p_attended in ('yes', 'attended') else 1
            past_arr[slot, 1] = float(pr.get('Actual_Duration', pr.get('Planned_Duration', 90)) or 90)
            past_arr[slot, 2] = float(pr.get('Cycle_Number', 1) or 1)
            past_arr[slot, 3] = float(pr.get('Weather_Severity', 0.0) or 0.0)
            past_arr[slot, 4] = float(pr.get('Day_Of_Week_Num', pr.get('Day_Of_Week', 2)) or 2)
            past_arr[slot, 5] = float(pr.get('Travel_Distance_KM', 15) or 15)

        # Normalise
        Xs = (static_vec - self.static_mean) / self.static_std
        Xp = (past_arr - self.obs_mean) / self.obs_std
        Xs_t = torch.tensor(Xs.astype(np.float32)).unsqueeze(0)
        Xp_t = torch.tensor(Xp.astype(np.float32)).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            out = self.model(Xs_t, Xp_t)
        q = out['duration_q'].squeeze(0).cpu().numpy() * self.duration_std + self.duration_mean
        return {
            'p_noshow': float(out['p_noshow'].item()),
            'p_cancel': float(out['p_cancel'].item()),
            'duration_q10': float(q[0]),
            'duration_q50': float(q[1]),
            'duration_q90': float(q[2]),
            'attention': out['attention'].squeeze(0).cpu().numpy().tolist(),
            'vsn_weights_past': out['vsn_weights_past'].squeeze(0).mean(dim=0).cpu().numpy().tolist(),
            'vsn_weights_static': out['vsn_weights_static'].squeeze(0).cpu().numpy().tolist(),
        }

    def reset(self) -> None:
        self.model = None
        self.last_fit = None
        for p in (self.model_path, self.meta_path):
            try:
                p.unlink(missing_ok=True)
            except TypeError:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

    # -------- persistence --------

    def _save_state(self) -> None:
        if self.model is None:
            return
        try:
            torch.save(self.model.state_dict(), self.model_path)
            meta = {
                'past_window': self.past_window,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'static_mean': self.static_mean.tolist() if self.static_mean is not None else None,
                'static_std': self.static_std.tolist() if self.static_std is not None else None,
                'obs_mean': self.obs_mean.tolist() if self.obs_mean is not None else None,
                'obs_std': self.obs_std.tolist() if self.obs_std is not None else None,
                'duration_mean': self.duration_mean,
                'duration_std': self.duration_std,
                'last_fit': asdict(self.last_fit) if self.last_fit else None,
            }
            # T2.3: SHA-256 sidecar — load() refuses tampered TFT meta.
            from safe_loader import safe_save
            safe_save(meta, self.meta_path)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"TFT save failed: {exc}")

    def _try_load_state(self) -> None:
        if not self.model_path.exists() or not self.meta_path.exists():
            return
        try:
            from safe_loader import safe_load
            meta = safe_load(self.meta_path)
            self.past_window = meta.get('past_window', self.past_window)
            self.d_model = meta.get('d_model', self.d_model)
            self.n_heads = meta.get('n_heads', self.n_heads)
            self.static_mean = np.asarray(meta['static_mean']) if meta.get('static_mean') else None
            self.static_std = np.asarray(meta['static_std']) if meta.get('static_std') else None
            self.obs_mean = np.asarray(meta['obs_mean']) if meta.get('obs_mean') else None
            self.obs_std = np.asarray(meta['obs_std']) if meta.get('obs_std') else None
            self.duration_mean = float(meta.get('duration_mean', 90.0))
            self.duration_std = float(meta.get('duration_std', 30.0))
            if meta.get('last_fit'):
                self.last_fit = TFTFitResult(**meta['last_fit'])
            self.model = TFTLite(
                static_dim=DEFAULT_STATIC_DIM,
                obs_dim=DEFAULT_OBS_DIM,
                past_window=self.past_window,
                d_model=self.d_model,
                n_heads=self.n_heads,
            )
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
            self.model.eval()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"TFT load failed: {exc}")
            self.model = None
            self.last_fit = None

    def _append_history(self, fit: TFTFitResult) -> None:
        try:
            with self.history_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(asdict(fit)) + '\n')
        except Exception as exc:  # pragma: no cover
            logger.warning(f"TFT history append failed: {exc}")


def _auc_safe(y_true: np.ndarray, p: np.ndarray) -> Optional[float]:
    if y_true.min() == y_true.max():
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, p))
    except Exception:
        return None


__all__ = [
    'TORCH_AVAILABLE',
    'TFTTrainer',
    'TFTFitResult',
    'DEFAULT_PAST_WINDOW',
    'DEFAULT_QUANTILES',
]
if TORCH_AVAILABLE:
    __all__ += ['TFTLite', 'GatedResidualNetwork', 'VariableSelectionNetwork', 'pinball_loss']
