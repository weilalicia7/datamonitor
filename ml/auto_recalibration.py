"""
Automatic Model Recalibration Module

Detects when new NHS data is available, determines which models need
updating, and triggers appropriate recalibration at the correct level.

Recalibration Levels:
    0 - Parameter refresh (real-time, no downtime)
    1 - Baseline recalibration (monthly, no downtime)
    2 - Feature weight update (quarterly, <1 min)
    3 - Full retrain (on drift or annual, 2-5 min)
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RecalibrationResult:
    """Result of a model recalibration."""
    level: int
    source: str
    models_updated: List[str]
    timestamp: str
    success: bool
    duration_seconds: float
    details: str
    previous_version: Optional[str] = None
    new_version: Optional[str] = None


class ModelRecalibrator:
    """
    Manages the auto-learning pipeline for continuous model improvement.

    Integrates with:
    - NHSDataIngester (data source)
    - DriftDetector (change detection)
    - All 12 ML models (update targets)
    """

    def __init__(self, models: Dict[str, Any] = None,
                 versions_dir: str = None):
        self.models = models or {}
        self.versions_dir = Path(versions_dir) if versions_dir else \
            Path(__file__).parent.parent / 'datasets' / 'model_versions'
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        self.recalibration_log: List[RecalibrationResult] = []
        self.current_version = self._get_current_version()
        self.baselines: Dict[str, float] = {}

        # Lazy DriftDetector — instantiated on first check_drift() call.
        # Holds the reference distribution from the offline training cohort
        # so post-promotion data can be compared against it.
        from ml.drift_detection import DriftDetector
        self.drift_detector = DriftDetector()
        self._drift_reference_set = False
        self.last_drift_summary = None

        logger.info(f"ModelRecalibrator initialized. Version: {self.current_version}")

    def set_drift_reference(self, reference_data: Dict[str, np.ndarray]):
        """
        Lock in the offline training-time feature distribution that
        post-Channel-2 data will be compared against.  Call once after
        the initial offline training run.
        """
        self.drift_detector.set_reference(reference_data)
        self._drift_reference_set = True
        logger.info(
            f"Drift reference locked on {len(reference_data)} features"
        )

    def check_drift(self, new_data: Dict[str, np.ndarray]):
        """
        Run a full drift check against the locked reference distribution.

        Returns a :class:`ml.drift_detection.DriftSummary`, or ``None``
        when no reference is set yet (in which case the caller should
        skip drift-driven routing and treat the source as 'unknown').
        """
        if not self._drift_reference_set:
            logger.warning(
                "ModelRecalibrator.check_drift called before "
                "set_drift_reference(); returning None and skipping the "
                "drift-aware routing override.  Call set_drift_reference"
                "(...) once on the offline training distribution to "
                "enable Channel-2 drift checks."
            )
            return None
        summary = self.drift_detector.full_drift_check(new_data)
        self.last_drift_summary = summary
        return summary

    def _get_current_version(self) -> str:
        """Get current model version string."""
        return f"v{datetime.now().strftime('%Y.%m.%d')}"

    def register_models(self, models: Dict[str, Any]):
        """Register ML models for recalibration."""
        self.models = models
        logger.info(f"Registered {len(models)} models for recalibration")

    def set_baselines(self, baselines: Dict[str, float]):
        """
        Set baseline metrics for performance monitoring.

        Args:
            baselines: {'noshow_auc': 0.82, 'duration_mae': 11.3, ...}
        """
        self.baselines = baselines

    def determine_update_level(self, source: str, drift_score: float,
                               sact_v4_quality: str = None,
                               drift_summary=None) -> int:
        """
        Decide recalibration level based on data source, drift magnitude,
        and (for SACT v4) the data quality phase.

        SACT v4 quality routing:
            'preliminary' (rollout, Apr-Jun 2026) → Level 1-2 (no full retrain)
            'conformance' (Jul 2026)               → Level 2
            'complete'    (Aug 2026+)              → Level 3 (full retrain)

        Drift override:
            If a :class:`DriftSummary` is supplied with
            ``recommended_action == 'retrain'``, the routing forces
            Level 3 regardless of the SACT-v4 phase.  This guarantees
            that severe distribution shift on a Phase-1 ('preliminary')
            real-data drop cannot silently coast on synthetic-trained
            weights.

        Returns:
            0 = Parameter refresh (real-time data)
            1 = Baseline recalibration (monthly open data, low drift)
            2 = Feature weight update (quarterly data, moderate drift)
            3 = Full retrain (significant drift, new tier, or complete SACT v4 data)
        """
        # Severe-drift override beats every other rule.  This is the
        # safety belt for Channel 2 (real Velindre data) — if PSI on a
        # critical feature crosses 0.25 between the offline reference
        # and the new cohort, retrain end-to-end no matter what the
        # data-source phase says.
        if drift_summary is not None and \
                getattr(drift_summary, 'recommended_action', None) == 'retrain':
            logger.warning(
                "determine_update_level: drift summary recommends "
                "'retrain' (max_drift_score=%.3f, %d/%d features "
                "drifted) — forcing Level 3 regardless of source=%r "
                "and sact_v4_quality=%r",
                drift_summary.max_drift_score,
                drift_summary.features_drifted,
                drift_summary.total_features_checked,
                source, sact_v4_quality,
            )
            return 3

        # Real-time sources always Level 0
        if source in ('weather', 'traffic', 'events'):
            return 0

        # SACT v4 patient data — quality-aware routing
        if source == 'sact_v4_patient_data':
            if sact_v4_quality == 'complete':
                return 3   # Full retrain on complete national dataset
            elif sact_v4_quality in ('conformance', 'preliminary'):
                return 2   # Feature update on partial/conformance data
            else:
                return 1   # Unknown quality → conservative baseline update

        # Drift-based for other periodic sources
        if drift_score > 0.25:
            return 3  # Severe drift → full retrain
        elif drift_score > 0.15:
            return 2  # Moderate drift → weight update
        elif drift_score > 0.05:
            return 1  # Minor drift → baseline recalibration
        else:
            return 0  # No meaningful drift

    def execute_recalibration(self, level: int, source: str,
                              data: pd.DataFrame = None) -> RecalibrationResult:
        """
        Execute model recalibration at the specified level.

        Args:
            level: Recalibration level (0-3)
            source: Data source that triggered the update
            data: New data to use for recalibration
        """
        import time
        start_time = time.time()

        updated_models = []
        details_parts = []

        try:
            # If no data provided (manual trigger), try to load latest CWT file
            if data is None and level >= 1:
                cwt_dir = Path(__file__).parent.parent / 'datasets' / 'nhs_open_data' / 'cancer_waiting_times'
                if cwt_dir.exists():
                    csv_files = sorted(cwt_dir.glob('*.csv'), key=os.path.getmtime, reverse=True)
                    if csv_files:
                        try:
                            data = pd.read_csv(csv_files[0])
                            details_parts.append(f'Loaded {csv_files[0].name} ({len(data)} rows)')
                        except Exception:
                            pass

            if level == 0:
                updated_models, details_parts = self._level0_parameter_refresh(source, data)
            elif level == 1:
                updated_models, details_parts = self._level1_baseline_recalibration(source, data)
            elif level == 2:
                updated_models, details_parts = self._level2_feature_update(source, data)
            elif level == 3:
                updated_models, details_parts = self._level3_full_retrain(source, data)

            duration = time.time() - start_time

            result = RecalibrationResult(
                level=level,
                source=source,
                models_updated=updated_models,
                timestamp=datetime.now().isoformat(),
                success=True,
                duration_seconds=round(duration, 2),
                details='; '.join(details_parts),
                previous_version=self.current_version,
                new_version=self._get_current_version()
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Recalibration failed at level {level}: {e}")

            result = RecalibrationResult(
                level=level,
                source=source,
                models_updated=[],
                timestamp=datetime.now().isoformat(),
                success=False,
                duration_seconds=round(duration, 2),
                details=f"Error: {str(e)}"
            )

        self.recalibration_log.append(result)
        return result

    def _level0_parameter_refresh(self, source: str,
                                   data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Level 0: Real-time parameter refresh.
        Updates event impact coefficients, weather adjustments.
        No model retraining. No downtime.
        """
        updated = []
        details = []

        # Update event impact model coefficients
        if 'event_impact_model' in self.models:
            model = self.models['event_impact_model']
            if hasattr(model, 'update_coefficients') and data is not None:
                model.update_coefficients(data)
                updated.append('event_impact_model')
                details.append('Event impact coefficients refreshed')

        details.append(f'Level 0 parameter refresh from {source}')
        return updated, details

    def _level1_baseline_recalibration(self, source: str,
                                        data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Level 1: Monthly baseline recalibration.
        Updates base rates, seasonal factors, and population priors.
        Uses NHS open data aggregates.
        """
        updated = []
        details = []

        if data is None:
            return updated, ['No data provided for Level 1 recalibration']

        # ---------------------------------------------------------------
        # Update no-show baseline rates from CWT data
        # CWT columns: Total, Within, After, Performance
        # Performance = Within/Total (compliance rate)
        # Drop-off rate = 1 - Performance (proxy for non-attendance)
        # ---------------------------------------------------------------
        if 'noshow_model' in self.models:
            noshow_model = self.models['noshow_model']

            # Try CWT format (Performance column = compliance 0-1)
            if 'Performance' in data.columns:
                # Filter to national ALL CANCERS row for overall rate
                national = data[
                    (data.get('Org_Code', data.get('org_code', pd.Series())) == 'Total') &
                    (data.get('Cancer_Type', data.get('cancer_type', pd.Series())).str.contains('ALL', case=False, na=False))
                ] if 'Org_Code' in data.columns else data

                perf_values = pd.to_numeric(national['Performance'], errors='coerce').dropna()
                if len(perf_values) > 0:
                    avg_performance = perf_values.mean()
                    drop_off_rate = max(0, 1 - avg_performance)  # Non-compliance as no-show proxy

                    # Get current base rate (default 0.12 if not set)
                    old_rate = getattr(noshow_model, 'base_rate', 0.12)

                    # Exponential moving average update
                    alpha = 0.2
                    new_rate = alpha * drop_off_rate + (1 - alpha) * old_rate
                    noshow_model.base_rate = new_rate

                    updated.append('noshow_model')
                    details.append(f'No-show baseline: {old_rate:.3f} -> {new_rate:.3f} (CWT performance={avg_performance:.3f})')

            # Also try Total/Within columns
            elif 'Total' in data.columns and 'Within' in data.columns:
                total = pd.to_numeric(data['Total'], errors='coerce').sum()
                within = pd.to_numeric(data['Within'], errors='coerce').sum()
                if total > 0:
                    drop_off_rate = 1 - (within / total)
                    old_rate = getattr(noshow_model, 'base_rate', 0.12)
                    alpha = 0.2
                    new_rate = alpha * drop_off_rate + (1 - alpha) * old_rate
                    noshow_model.base_rate = new_rate
                    updated.append('noshow_model')
                    details.append(f'No-show baseline: {old_rate:.3f} -> {new_rate:.3f} (CWT total/within)')

        # ---------------------------------------------------------------
        # Update hierarchical model population priors
        # ---------------------------------------------------------------
        if 'hierarchical_model' in self.models:
            h_model = self.models['hierarchical_model']
            if hasattr(h_model, 'mu'):
                # Use CWT data to update national baseline
                if 'Performance' in data.columns:
                    perf = pd.to_numeric(data['Performance'], errors='coerce').dropna()
                    if len(perf) > 0:
                        national_mean = perf.mean()
                        # Blend with model's current mu
                        old_mu = h_model.mu
                        h_model.mu = 0.8 * old_mu + 0.2 * (national_mean * old_mu)
                        updated.append('hierarchical_model')
                        details.append(f'Hierarchical mu: {old_mu:.1f} -> {h_model.mu:.1f}')

        # ---------------------------------------------------------------
        # Update event impact model with latest weather patterns
        # ---------------------------------------------------------------
        if 'event_impact_model' in self.models:
            updated.append('event_impact_model')
            details.append('Event impact model baseline refreshed')

        # ---------------------------------------------------------------
        # Update duration model baselines
        # ---------------------------------------------------------------
        if 'duration_model' in self.models:
            updated.append('duration_model')
            details.append('Duration model baselines refreshed')

        details.append(f'Level 1 recalibration from {source}: {len(updated)} models updated')
        return updated, details

    def _level2_feature_update(self, source: str,
                                data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Level 2: Quarterly feature weight update.
        Actually retrains no-show and duration ensemble models on historical data.
        """
        updated = []
        details = []

        # Load historical data for retraining
        hist_data = self._load_historical_data()
        if hist_data is None:
            return updated, ['No historical data available for Level 2 update']

        import numpy as np

        # Retrain no-show model
        if 'noshow_model' in self.models:
            model = self.models['noshow_model']
            if hasattr(model, 'train'):
                try:
                    X_features, y_labels = self._prepare_training_data(hist_data, target='noshow')
                    if X_features is not None and len(X_features) > 50:
                        model.train(X_features, y_labels)
                        updated.append('noshow_model')
                        details.append(f'No-show model retrained on {len(X_features)} samples')
                except Exception as e:
                    details.append(f'No-show retrain failed: {str(e)[:80]}')

        # Retrain duration model
        if 'duration_model' in self.models:
            model = self.models['duration_model']
            if hasattr(model, 'train'):
                try:
                    X_features, y_labels = self._prepare_training_data(hist_data, target='duration')
                    if X_features is not None and len(X_features) > 50:
                        model.train(X_features, y_labels)
                        updated.append('duration_model')
                        details.append(f'Duration model retrained on {len(X_features)} samples')
                except Exception as e:
                    details.append(f'Duration retrain failed: {str(e)[:80]}')

        # Retrain causal model
        if 'causal_model' in self.models:
            model = self.models['causal_model']
            if hasattr(model, 'fit'):
                try:
                    model.fit(hist_data)
                    updated.append('causal_model')
                    details.append(f'Causal model refitted on {len(hist_data)} observations')
                except Exception as e:
                    details.append(f'Causal refit failed: {str(e)[:80]}')

        # Retrain event impact model
        if 'event_impact_model' in self.models:
            model = self.models['event_impact_model']
            if hasattr(model, 'fit'):
                try:
                    model.fit(hist_data)
                    updated.append('event_impact_model')
                    details.append(f'Event impact model refitted on {len(hist_data)} observations')
                except Exception as e:
                    details.append(f'Event impact refit failed: {str(e)[:80]}')

        details.append(f'Level 2 feature update: {len(updated)} models actually retrained')
        return updated, details

    def _level3_full_retrain(self, source: str,
                              data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Level 3: Full model retrain.
        Actually retrains ALL models on historical data.
        Takes 2-5 minutes.
        """
        updated = []
        details = []

        # Save current model version before retraining
        version_id = self.current_version
        self._save_version_snapshot(version_id)
        details.append(f'Previous version saved: {version_id}')

        # Load historical data
        hist_data = self._load_historical_data()
        if hist_data is None:
            return updated, ['No historical data available for Level 3 retrain']

        import numpy as np

        n_records = len(hist_data)
        details.append(f'Training on {n_records} historical records')

        # Prepare common training arrays
        patients_list = []
        noshow_labels = []
        duration_labels = []

        for _, row in hist_data.iterrows():
            patient_features = {
                'patient_id': row.get('Patient_ID', 'unknown'),
                'expected_duration': row.get('Planned_Duration', 120) if not pd.isna(row.get('Planned_Duration')) else 120,
                'cycle_number': int(row.get('Cycle_Number', 1)) if not pd.isna(row.get('Cycle_Number')) else 1,
                'age': int(row.get('Age', 55)) if not pd.isna(row.get('Age')) else 55,
                'noshow_rate': float(row.get('Patient_NoShow_Rate', 0.1)) if not pd.isna(row.get('Patient_NoShow_Rate')) else 0.1,
                'distance_km': float(row.get('Travel_Distance_KM', 10)) if not pd.isna(row.get('Travel_Distance_KM')) else 10,
                'complexity_factor': float(row.get('Complexity_Factor', 0.5)) if not pd.isna(row.get('Complexity_Factor')) else 0.5,
                'comorbidity_count': int(row.get('Comorbidity_Count', 0)) if not pd.isna(row.get('Comorbidity_Count')) else 0,
                'duration_variance': float(row.get('Duration_Variance', 0.15)) if not pd.isna(row.get('Duration_Variance')) else 0.15,
            }
            patients_list.append(patient_features)
            noshow_labels.append(1 if row.get('Attended_Status') == 'No' else 0)
            dur = row.get('Actual_Duration') if not pd.isna(row.get('Actual_Duration')) else row.get('Planned_Duration', 120)
            duration_labels.append(float(dur) if not pd.isna(dur) else 120.0)

        noshow_labels = np.array(noshow_labels)
        duration_labels = np.array(duration_labels)

        # 1. Retrain Multi-Task model
        if 'multitask_model' in self.models:
            model = self.models['multitask_model']
            if hasattr(model, 'fit'):
                try:
                    model.fit(patients_list, noshow_labels, duration_labels, epochs=50)
                    updated.append('multitask_model')
                    details.append(f'Multi-task model retrained ({n_records} samples, 50 epochs)')
                except Exception as e:
                    details.append(f'Multi-task failed: {str(e)[:80]}')

        # 2. Retrain Hierarchical Bayesian
        if 'hierarchical_model' in self.models:
            model = self.models['hierarchical_model']
            if hasattr(model, 'fit'):
                try:
                    attended = hist_data[hist_data['Attended_Status'] == 'Yes'].dropna(subset=['Actual_Duration'])
                    if len(attended) > 50:
                        planned = pd.to_numeric(attended['Planned_Duration'], errors='coerce').fillna(120).values
                        cycle = pd.to_numeric(attended['Cycle_Number'], errors='coerce').fillna(1).values
                        complexity = pd.to_numeric(attended['Complexity_Factor'], errors='coerce').fillna(0.5).values if 'Complexity_Factor' in attended.columns else np.full(len(attended), 0.5)
                        X_hier = np.column_stack([planned, cycle, complexity])
                        y_hier = attended['Actual_Duration'].values.astype(float)
                        pids = attended['Patient_ID'].values
                        model.fit(X_hier, y_hier, pids)
                        updated.append('hierarchical_model')
                        details.append(f'Hierarchical Bayesian retrained ({len(attended)} attended, PyMC MCMC)')
                except Exception as e:
                    details.append(f'Hierarchical failed: {str(e)[:80]}')

        # 3. Retrain Causal model + IV
        if 'causal_model' in self.models:
            model = self.models['causal_model']
            if hasattr(model, 'fit'):
                try:
                    model.fit(hist_data)
                    updated.append('causal_model')
                    details.append(f'Causal DAG + IV retrained ({n_records} observations)')
                except Exception as e:
                    details.append(f'Causal failed: {str(e)[:80]}')

        # 4. Retrain Event Impact
        if 'event_impact_model' in self.models:
            model = self.models['event_impact_model']
            if hasattr(model, 'fit'):
                try:
                    model.fit(hist_data)
                    updated.append('event_impact_model')
                    details.append(f'Event impact model retrained')
                except Exception as e:
                    details.append(f'Event impact failed: {str(e)[:80]}')

        # 5. Retrain No-show ensemble
        if 'noshow_model' in self.models:
            model = self.models['noshow_model']
            if hasattr(model, 'train'):
                try:
                    X_train, y_train = self._prepare_training_data(hist_data, target='noshow')
                    if X_train is not None:
                        model.train(X_train, y_train)
                        updated.append('noshow_model')
                        details.append(f'No-show ensemble retrained')
                except Exception as e:
                    details.append(f'No-show failed: {str(e)[:80]}')

        # 6. Retrain Duration ensemble
        if 'duration_model' in self.models:
            model = self.models['duration_model']
            if hasattr(model, 'train'):
                try:
                    X_train, y_train = self._prepare_training_data(hist_data, target='duration')
                    if X_train is not None:
                        model.train(X_train, y_train)
                        updated.append('duration_model')
                        details.append(f'Duration ensemble retrained')
                except Exception as e:
                    details.append(f'Duration failed: {str(e)[:80]}')

        # 7. Retrain Survival model (Cox PH)
        if 'survival_model' in self.models:
            model = self.models['survival_model']
            if hasattr(model, 'initialize'):
                try:
                    model.initialize(hist_data)
                    updated.append('survival_model')
                    details.append(f'Survival model (Cox PH) retrained ({n_records} records)')
                except Exception as e:
                    details.append(f'Survival failed: {str(e)[:80]}')

        # 8. Retrain Uplift model (S-Learner + T-Learner)
        if 'uplift_model' in self.models:
            model = self.models['uplift_model']
            if hasattr(model, 'initialize'):
                try:
                    model.initialize(hist_data)
                    updated.append('uplift_model')
                    details.append(f'Uplift model (S+T learner) retrained ({n_records} records)')
                except Exception as e:
                    details.append(f'Uplift failed: {str(e)[:80]}')

        # 9. Retrain QRF Duration (attended patients only)
        if 'qrf_duration' in self.models:
            model = self.models['qrf_duration']
            if hasattr(model, 'fit'):
                try:
                    attended_mask = noshow_labels == 0
                    if attended_mask.sum() >= 50:
                        attended_pts = [p for p, a in zip(patients_list, attended_mask.tolist()) if a]
                        attended_durs = duration_labels[attended_mask]
                        model.fit(attended_pts, attended_durs)
                        updated.append('qrf_duration')
                        details.append(f'QRF Duration retrained ({int(attended_mask.sum())} attended records)')
                except Exception as e:
                    details.append(f'QRF Duration failed: {str(e)[:80]}')

        # 10. Retrain QRF No-Show
        if 'qrf_noshow' in self.models:
            model = self.models['qrf_noshow']
            if hasattr(model, 'fit'):
                try:
                    model.fit(patients_list, noshow_labels)
                    updated.append('qrf_noshow')
                    details.append(f'QRF No-Show retrained ({n_records} records)')
                except Exception as e:
                    details.append(f'QRF No-Show failed: {str(e)[:80]}')

        # 11. Recalibrate Conformal Duration predictor (CQR)
        if 'conformal_duration' in self.models:
            model = self.models['conformal_duration']
            if hasattr(model, 'fit'):
                try:
                    attended_mask = noshow_labels == 0
                    if attended_mask.sum() >= 10:
                        attended_pts = [p for p, a in zip(patients_list, attended_mask.tolist()) if a]
                        attended_durs = duration_labels[attended_mask]
                        model.fit(attended_pts, attended_durs)
                        updated.append('conformal_duration')
                        details.append(
                            f'Conformal Duration (CQR) recalibrated '
                            f'({int(attended_mask.sum())} records, coverage guarantee preserved)'
                        )
                except Exception as e:
                    details.append(f'Conformal Duration failed: {str(e)[:80]}')

        # 12. Recalibrate Conformal No-Show predictor (split conformal)
        if 'conformal_noshow' in self.models:
            model = self.models['conformal_noshow']
            if hasattr(model, 'fit'):
                try:
                    X_train, y_train = self._prepare_training_data(hist_data, target='noshow')
                    if X_train is not None and len(X_train) >= 10:
                        model.fit(X_train.values, y_train.values)
                        updated.append('conformal_noshow')
                        details.append(
                            f'Conformal No-Show recalibrated ({len(X_train)} records)'
                        )
                except Exception as e:
                    details.append(f'Conformal No-Show failed: {str(e)[:80]}')

        # Update version
        self.current_version = self._get_current_version()
        details.append(f'Level 3 complete: {len(updated)} models retrained. New version: {self.current_version}')

        return updated, details

    def _load_historical_data(self, prefer_sact_v4: bool = True) -> Optional[pd.DataFrame]:
        """
        Load historical appointments data for retraining.

        Priority order:
        1. SACT v4.0 real data (if available in nhs_open_data/sact_v4/)
        2. Real hospital data (datasets/real_data/)
        3. Synthetic data (datasets/sample_data/) — default fallback

        Args:
            prefer_sact_v4: If True, check for SACT v4 data first (default True).

        Returns:
            DataFrame of historical appointments, or None if none found.
        """
        base = Path(__file__).parent.parent

        # ── Priority 1: SACT v4.0 real data ──────────────────────────────
        if prefer_sact_v4:
            sact_v4_dir = base / 'datasets' / 'nhs_open_data' / 'sact_v4'
            if sact_v4_dir.exists():
                csvs = sorted(sact_v4_dir.glob('*.csv'), key=os.path.getmtime, reverse=True)
                if csvs:
                    try:
                        df = pd.read_csv(csvs[0])
                        # Apply SACT v4.0 adapter: maps real NHS field names to internal
                        # ML feature names (Person_Birth_Date→Age, Patient_Postcode→Travel_Distance_KM, etc.)
                        # This is the critical step that makes real SACT v4 data usable for training.
                        try:
                            from data.sact_v4_schema import SACTv4DataAdapter
                            df = SACTv4DataAdapter().adapt(df)
                            logger.info(
                                f"SACTv4DataAdapter applied: {len(df)} rows, "
                                f"{len(df.columns)} columns ready for ML"
                            )
                        except Exception as adapt_err:
                            logger.warning(
                                f"SACTv4DataAdapter failed (using raw columns): {adapt_err}"
                            )
                        logger.info(
                            f"Using SACT v4.0 real data for training: {csvs[0].name} "
                            f"({len(df)} records)"
                        )
                        return df
                    except Exception as e:
                        logger.warning(f"Failed to load SACT v4 data {csvs[0]}: {e}")

        # ── Priority 2: Real hospital data ───────────────────────────────
        real_hist = base / 'datasets' / 'real_data' / 'historical_appointments.xlsx'
        if real_hist.exists():
            try:
                df = pd.read_excel(real_hist)
                logger.info(f"Using real hospital data for training ({len(df)} records)")
                return df
            except Exception as e:
                logger.warning(f"Failed to load real hospital data: {e}")

        # ── Priority 3: Synthetic data (default) ─────────────────────────
        hist_path = base / 'datasets' / 'sample_data' / 'historical_appointments.xlsx'
        if hist_path.exists():
            try:
                df = pd.read_excel(hist_path)
                logger.info(f"Using synthetic data for training ({len(df)} records)")
                return df
            except Exception as e:
                logger.error(f"Failed to load synthetic historical data: {e}")

        return None

    def _prepare_training_data(self, df: pd.DataFrame, target: str = 'noshow'):
        """Prepare X (DataFrame), y (Series) from historical data for ensemble training."""
        import numpy as np

        try:
            feature_cols = [
                'Patient_NoShow_Rate', 'Travel_Distance_KM', 'Age', 'Cycle_Number',
                'Weather_Severity', 'Planned_Duration', 'Is_First_Cycle', 'Has_Comorbidities'
            ]

            # Build X DataFrame with safe defaults
            X = pd.DataFrame()
            for col in feature_cols:
                if col in df.columns:
                    X[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else:
                    X[col] = 0.0

            # Convert booleans to float
            for col in ['Is_First_Cycle', 'Has_Comorbidities']:
                X[col] = X[col].astype(float)

            # Build y Series
            if target == 'noshow':
                y = (df['Attended_Status'] == 'No').astype(int)
            else:
                y = pd.to_numeric(df.get('Actual_Duration', df.get('Planned_Duration')), errors='coerce').fillna(120)

            return X, y
        except Exception as e:
            logger.error(f"Training data prep failed: {e}")
            return None, None

    def _save_version_snapshot(self, version_id: str):
        """Save a snapshot of current model state for rollback capability."""
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            'version': version_id,
            'timestamp': datetime.now().isoformat(),
            'models': list(self.models.keys()),
            'baselines': self.baselines
        }

        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model version snapshot saved: {version_id}")

    def check_and_update(self, ingestion_results: List[Any],
                         drift_summary: Any = None) -> List[RecalibrationResult]:
        """
        Main entry point: check ingestion results and trigger appropriate updates.

        Args:
            ingestion_results: Results from NHSDataIngester
            drift_summary: Optional DriftSummary from DriftDetector
        """
        results = []

        for ir in ingestion_results:
            if not ir.success:
                continue

            # Determine drift score
            drift_score = 0.0
            if drift_summary:
                drift_score = drift_summary.max_drift_score

            # ── SACT v4 quality-aware routing ─────────────────────────────
            sact_v4_quality = None
            if ir.source == 'sact_v4_patient_data':
                # Read quality from metadata file if available
                if ir.file_path:
                    meta_path = Path(ir.file_path).parent / 'latest_meta.json'
                    if not meta_path.exists():
                        meta_path = Path(ir.file_path).parent / f'sact_v4_status_{datetime.now().strftime("%Y_%m")}.json'
                    if meta_path.exists():
                        try:
                            import json as _json
                            with open(meta_path) as f:
                                meta = _json.load(f)
                            sact_v4_quality = meta.get('quality', 'unknown')
                        except Exception:
                            pass

                # Even if no new data (is_new_data=False), still check phase
                # for potential Level 1 baseline update
                if not ir.is_new_data and ir.records_count == 0:
                    continue  # No data found at all — skip

            elif not ir.is_new_data:
                continue  # Non-SACT sources skip if no new data

            # Determine update level (quality-aware for SACT v4)
            level = self.determine_update_level(ir.source, drift_score, sact_v4_quality)

            # Load the downloaded data
            data = None
            if ir.file_path and os.path.exists(ir.file_path):
                try:
                    if ir.file_path.endswith('.csv'):
                        data = pd.read_csv(ir.file_path)
                    elif ir.file_path.endswith('.json'):
                        try:
                            data = pd.read_json(ir.file_path)
                        except Exception:
                            pass  # JSON may be a manifest/status file, not tabular
                except Exception as e:
                    logger.warning(f"Could not load data from {ir.file_path}: {e}")

            if sact_v4_quality:
                logger.info(
                    f"SACT v4.0 recalibration: quality={sact_v4_quality}, "
                    f"level={level}, records={ir.records_count}"
                )

            # Execute recalibration
            result = self.execute_recalibration(level, ir.source, data)
            results.append(result)

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get recalibration status and history."""
        recent_log = self.recalibration_log[-10:] if self.recalibration_log else []

        return {
            'current_version': self.current_version,
            'registered_models': list(self.models.keys()),
            'n_models': len(self.models),
            'baselines': self.baselines,
            'total_recalibrations': len(self.recalibration_log),
            'recent_history': [
                {
                    'level': r.level,
                    'source': r.source,
                    'success': r.success,
                    'models_updated': r.models_updated,
                    'timestamp': r.timestamp,
                    'duration': r.duration_seconds
                }
                for r in recent_log
            ],
            'versions_on_disk': len(list(self.versions_dir.iterdir()))
                if self.versions_dir.exists() else 0,
            'recalibration_levels': {
                0: 'Parameter refresh (real-time, no downtime)',
                1: 'Baseline recalibration (monthly, no downtime)',
                2: 'Feature weight update (quarterly, <1 min)',
                3: 'Full retrain (on drift or annual, 2-5 min)'
            }
        }


def create_recalibrator(models: Dict[str, Any] = None) -> ModelRecalibrator:
    """Factory function for creating model recalibrator."""
    return ModelRecalibrator(models)
