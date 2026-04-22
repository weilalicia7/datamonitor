# Auto-Learning Pipeline: Continuous Model Recalibration from NHS Open Data

## SACT Scheduling System - Automated Data Ingestion and Model Update Architecture

**Version:** 1.1
**Date:** April 2026
**Purpose:** Design a system that automatically reads publicly available NHS data, learns from it, and updates the ML models and optimization parameters in real-time.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [NHS Open Data Sources](#2-nhs-open-data-sources)
3. [Data Ingestion Architecture](#3-data-ingestion-architecture)
4. [Auto-Learning Pipeline](#4-auto-learning-pipeline)
5. [Model Recalibration Strategy](#5-model-recalibration-strategy)
6. [Three-Tier Data Strategy](#6-three-tier-data-strategy)
7. [Implementation Design](#7-implementation-design)
8. [Monitoring and Drift Detection](#8-monitoring-and-drift-detection)
9. [SACT v4.0 Integration Readiness](#9-sact-v40-integration-readiness)

---

## 1. Design Philosophy

### 1.1 The Problem

ML models trained on static data degrade over time as:
- Cancer treatment patterns evolve (new regimens, changing protocols)
- Patient demographics shift (aging population, new populations)
- Seasonal patterns change (COVID aftermath, winter pressures)
- NHS policy changes affect behaviour (new targets, pathway redesigns)

### 1.2 The Solution

A **three-tier data strategy** that automatically ingests open NHS data to recalibrate models, while being ready for real patient data when it becomes available:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     THREE-TIER AUTO-LEARNING ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TIER 1: NHS OPEN DATA (Available now, automated)                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Cancer Waiting Times CSV ──┐                                            ││
│  │  SACT Activity Dashboard ───┤──▶ Auto-Ingest ──▶ Recalibrate Baselines ││
│  │  Cancer Treatments Data ────┤                                            ││
│  │  NHSBSA Prescribing API ────┘                                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│                                     ▼                                        │
│  TIER 2: LOCAL OPERATIONAL DATA (When hospital provides access)             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ChemoCare exports ─────────┐                                            ││
│  │  ChemoSchedule bookings ────┤──▶ Feed Models ──▶ Patient-Level Learning ││
│  │  Local attendance records ──┤                                            ││
│  │  Staff rosters ─────────────┘                                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│                                     ▼                                        │
│  TIER 3: SACT v4.0 SUBMISSIONS (First complete dataset: August 2026)       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Monthly SACT v4.0 CSV ─────┐                                            ││
│  │  60 standardized fields ────┤──▶ Full Pipeline ──▶ Complete Retraining  ││
│  │  National registry data ────┘                                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. NHS Open Data Sources

### 2.1 Cancer Waiting Times (Primary Source)

| Property | Detail |
|----------|--------|
| **URL** | https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/ |
| **Format** | CSV and Excel (.xlsx) |
| **Update frequency** | Monthly |
| **Granularity** | Provider-level, by cancer type, by standard |
| **Coverage** | April 2022 onwards (historical from 2009 in time series) |
| **Access** | Free, no registration, direct download |
| **File size** | ~5-60 MB per monthly combined CSV |

**Fields available:**
- Provider (NHS Trust) identifier
- Cancer type classification
- 31-day standard compliance (decision to first treatment)
- 62-day standard compliance (referral to first treatment)
- Faster Diagnosis Standard compliance
- Patient counts by standard
- Monthly time series

**What this gives our ML system:**
- Baseline compliance rates per trust (calibrate expected performance)
- Seasonal patterns (monthly variation in waiting times)
- Cancer-type specific trends (which cancers have worse compliance)
- Trend detection (improving or deteriorating performance)

### 2.2 SACT Activity Dashboard (Secondary Source)

| Property | Detail |
|----------|--------|
| **URL** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/sact-activity |
| **Format** | Interactive dashboard with download capability |
| **Update frequency** | Annually (confirmed April 2026; previously quarterly) |
| **Granularity** | England / Cancer Alliance / ICB level |
| **Coverage** | January 2019 onwards |
| **Access** | Free, no registration |

**Fields available:**
- Patient counts by tumour group
- Regimen counts
- Drug administration counts
- Breakdowns by: age, gender, deprivation, ethnicity, treatment intent, admin route

**What this gives our ML system:**
- National regimen distribution (validate/update our regimen mix)
- Demographic patterns (age/gender effects on treatment volumes)
- Treatment intent distribution (curative vs non-curative ratios)
- Temporal trends in SACT activity

### 2.3 Cancer Treatments Dashboard (Tertiary Source)

| Property | Detail |
|----------|--------|
| **URL** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-treatments |
| **Dashboard** | https://nhsd-ndrs.shinyapps.io/cancer_treatments/ |
| **Format** | Interactive dashboard (R Shiny) with download |
| **Update frequency** | Annually |
| **Granularity** | Cancer type, stage, age, gender, deprivation, ethnicity, comorbidity, Cancer Alliance |
| **Coverage** | 2013-2022 |
| **Access** | Free, no registration |

**Fields available:**
- Proportion of tumours receiving SACT
- Breakdown by stage at diagnosis
- Breakdown by comorbidity count
- Cancer Alliance level analysis

**What this gives our ML system:**
- Comorbidity impact on treatment (calibrate duration model)
- Stage-specific treatment patterns
- Regional variation in treatment rates

### 2.4 NHSBSA Secondary Care Medicines Data with Indicative Price (Supplementary)

| Property | Detail |
|----------|--------|
| **URL** | https://opendata.nhsbsa.net/dataset/secondary-care-medicines-data-indicative-price |
| **Package ID** | `secondary-care-medicines-data-indicative-price` |
| **API** | `https://opendata.nhsbsa.net/api/3/action/package_show?id=secondary-care-medicines-data-indicative-price` |
| **Format** | JSON API (CKAN), CSV download |
| **Update frequency** | Monthly (~20th of each month, 2-month lag; latest: January 2026) |
| **Access** | Free, no registration, programmatic API |
| **Note** | ⚠ Old dataset `secondary-care-medicines-data` retired June 2022. Use new SCMD-IP package. |

**What this gives our ML system:**
- Drug prescribing trends (new regimens appearing in prescribing data)
- Volume changes (demand forecasting signals)

### 2.5 External Data Sources (Already Integrated)

| Source | API | Update | Already In System |
|--------|-----|--------|-------------------|
| **Open-Meteo** | REST | Hourly | Yes (weather monitor) |
| **TomTom Traffic** | REST | 5 min | Yes (traffic monitor) |
| **BBC Wales RSS** | RSS | 15 min | Yes (news monitor) |
| **Gov.uk Alerts** | RSS | Real-time | Yes (emergency alerts) |

---

## 3. Data Ingestion Architecture

### 3.1 Automated Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTOMATED DATA INGESTION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SCHEDULERS                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ││
│  │  │  MONTHLY     │  │  ANNUALLY    │  │   DAILY      │  │  REAL-TIME │  ││
│  │  │              │  │              │  │              │  │            │  ││
│  │  │ Cancer Wait  │  │ SACT Activity│  │ Weather      │  │ Traffic    │  ││
│  │  │ Times CSV    │  │ Dashboard    │  │ Forecast     │  │ Incidents  │  ││
│  │  │ NHSBSA API   │  │ Cancer Treat │  │ Local Events │  │ News/RSS   │  ││
│  │  │              │  │              │  │              │  │            │  ││
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  ││
│  │         │                 │                  │                │         ││
│  └─────────┼─────────────────┼──────────────────┼────────────────┼─────────┘│
│            ▼                 ▼                  ▼                ▼          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      DATA VALIDATION LAYER                               ││
│  │                                                                          ││
│  │  ├── Schema validation (expected columns present?)                      ││
│  │  ├── Range checks (values within expected bounds?)                      ││
│  │  ├── Completeness checks (missing data percentage?)                     ││
│  │  ├── Freshness checks (data actually updated since last pull?)          ││
│  │  └── Anomaly detection (sudden jumps in values?)                        ││
│  │                                                                          ││
│  └──────────────────────────────┬──────────────────────────────────────────┘│
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      DATA TRANSFORMATION LAYER                           ││
│  │                                                                          ││
│  │  ├── Normalize field names to internal schema                           ││
│  │  ├── Convert date formats (NHS DD/MM/YYYY → ISO)                       ││
│  │  ├── Aggregate to required granularity                                  ││
│  │  ├── Calculate derived metrics (rates, trends, moving averages)         ││
│  │  └── Store in versioned data lake                                       ││
│  │                                                                          ││
│  └──────────────────────────────┬──────────────────────────────────────────┘│
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      MODEL UPDATE TRIGGER                                ││
│  │                                                                          ││
│  │  IF new_data_available AND data_valid AND drift_detected:               ││
│  │      → Trigger model recalibration                                      ││
│  │  ELSE IF scheduled_retrain_due:                                         ││
│  │      → Trigger scheduled retraining                                     ││
│  │  ELSE:                                                                   ││
│  │      → Log "No update needed" and continue                              ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Ingestion Schedule

| Source | Frequency | Method | Trigger |
|--------|-----------|--------|---------|
| Cancer Waiting Times | 1st of each month | HTTP download CSV | Cron job |
| SACT Activity | Annually (check Jan each year) | Scrape dashboard download | Cron job |
| Cancer Treatments | Once per year | Download all data | Manual + cron |
| NHSBSA SCMD-IP (new) | 20th of each month | CKAN API (package_show) | Cron job |
| Open-Meteo Weather | Every hour | REST API | Existing monitor |
| TomTom Traffic | Every 5 minutes | REST API | Existing monitor |
| News/RSS Feeds | Every 15 minutes | RSS parse | Existing monitor |
| SACT v4.0 data | Monthly from Aug 2026 | NDRS portal / manual | Manual (then automate) |
| Local hospital data | When provided | File upload / DB query | Manual or webhook |

---

## 4. Auto-Learning Pipeline

### 4.1 What Each Data Source Teaches the Models

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  WHAT EACH DATA SOURCE TEACHES THE MODELS                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CANCER WAITING TIMES (Monthly CSV)                                         │
│  ├─▶ No-Show Model: Update baseline no-show rates by cancer type           │
│  ├─▶ Duration Model: Calibrate expected wait-to-treatment durations        │
│  ├─▶ Optimizer: Adjust capacity targets based on compliance gaps           │
│  └─▶ Trend Model: Detect seasonal patterns in cancellation rates           │
│                                                                              │
│  SACT ACTIVITY DASHBOARD (Annually)                                         │
│  ├─▶ Duration Model: Update regimen distribution weights                    │
│  ├─▶ No-Show Model: Recalibrate demographic risk factors (age/deprivation) │
│  ├─▶ Uplift Model: Validate intervention effectiveness assumptions         │
│  └─▶ Hierarchical: Update population-level priors                           │
│                                                                              │
│  CANCER TREATMENTS DATA (Annual)                                            │
│  ├─▶ Duration Model: Comorbidity impact coefficients                        │
│  ├─▶ Survival Model: Stage-specific hazard adjustments                      │
│  └─▶ Causal Model: Update confounding variable distributions                │
│                                                                              │
│  WEATHER/TRAFFIC/EVENTS (Real-Time)                                         │
│  ├─▶ Event Impact: Continuous coefficient learning                          │
│  ├─▶ No-Show Model: Live feature updates                                    │
│  └─▶ Optimizer: Dynamic constraint adjustment                               │
│                                                                              │
│  LOCAL HOSPITAL DATA (When Available)                                       │
│  ├─▶ ALL MODELS: Full retraining on patient-level data                      │
│  ├─▶ Hierarchical: Patient-specific random effects                          │
│  └─▶ Sequence Model: Individual patient trajectories                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Learning Hierarchy

Not all model updates are equal. The system uses a hierarchy of update strategies:

| Update Level | Trigger | What Changes | Downtime |
|-------------|---------|--------------|----------|
| **Level 0: Parameter Refresh** | Real-time data | Weather coefficients, event impacts | None |
| **Level 1: Baseline Recalibration** | Monthly open data | No-show base rates, duration baselines, compliance targets | None |
| **Level 2: Feature Weight Update** | Quarterly data | Feature importance weights, demographic adjustments | <1 min |
| **Level 3: Model Retrain** | Significant drift detected or annual | Full model retraining, hyperparameter tuning | 2-5 min |
| **Level 4: Architecture Change** | New data source or paradigm shift | New model components, feature engineering | Manual |

---

## 5. Model Recalibration Strategy

### 5.1 No-Show Prediction Model

**Monthly recalibration from Cancer Waiting Times data:**

```python
# Pseudocode: Monthly no-show baseline update
def update_noshow_baselines(new_cwt_data):
    """
    Recalibrate no-show base rates using latest Cancer Waiting Times data.

    Cancer Waiting Times don't directly report no-shows, but cancellation
    and non-attendance rates can be inferred from the gap between
    'seen' and 'referred' counts.
    """
    # Calculate implied non-attendance rate by provider
    for provider in new_cwt_data.providers:
        referred = provider.total_referred
        treated = provider.total_treated_in_target
        treated_late = provider.total_treated_beyond_target

        # Implied drop-off rate (includes no-shows, cancellations, pathway changes)
        drop_off_rate = 1 - (treated + treated_late) / referred

        # Update provider-specific baseline
        update_baseline(provider.id, drop_off_rate)

    # Detect seasonal patterns
    monthly_rates = calculate_monthly_trend(new_cwt_data)
    update_seasonal_adjustments(monthly_rates)
```

**What gets updated:**
- `P_base` (baseline no-show rate): Adjusted per cancer type
- Seasonal multipliers: Monthly adjustment factors (e.g., winter = +3%)
- Cancer-type specific offsets: Different cancers have different compliance

### 5.2 Duration Prediction Model

**Annual recalibration from SACT Activity data:**

```python
# Pseudocode: Annual duration model update
def update_duration_baselines(sact_activity_data):
    """
    Update regimen distribution weights and expected durations
    from SACT Activity Dashboard data.
    """
    # Update regimen frequency weights
    regimen_counts = sact_activity_data.regimen_breakdown
    total = sum(regimen_counts.values())
    for regimen, count in regimen_counts.items():
        new_weight = count / total
        update_regimen_weight(regimen, new_weight)

    # Update demographic adjustments
    age_distribution = sact_activity_data.age_breakdown
    update_age_coefficients(age_distribution)

    # Update intent distribution (curative vs non-curative)
    intent_split = sact_activity_data.intent_breakdown
    update_intent_priors(intent_split)
```

### 5.3 Hierarchical Bayesian Model

**Population prior updates from national data:**

$$\mu_{prior}^{(new)} = \frac{n_{local} \cdot \bar{y}_{local} + n_{national} \cdot \bar{y}_{national}}{n_{local} + n_{national}}$$

When national data updates:
- Population mean $\mu$ is recalibrated
- Between-group variance $\tau^2$ is updated from national variation
- Shrinkage factors are recalculated for patients with few observations

### 5.4 Event Impact Model

**Continuous learning from real-time events:**

```python
# Pseudocode: Continuous event coefficient update
def update_event_coefficients(actual_noshow_rate, predicted_rate, active_events):
    """
    After each day, compare predicted vs actual no-show rates
    and update event impact coefficients using exponential moving average.
    """
    error = actual_noshow_rate - predicted_rate
    learning_rate = 0.1

    for event in active_events:
        # Update coefficient for this event type
        old_coeff = get_coefficient(event.type)
        adjustment = learning_rate * error * event.severity
        new_coeff = old_coeff + adjustment
        set_coefficient(event.type, new_coeff)
```

### 5.5 Optimizer Parameters

**Adaptive optimization weights:**

| Parameter | Updated From | Frequency | Formula |
|-----------|-------------|-----------|---------|
| `w_priority` | Compliance gap data | Monthly | Increase if 62-day target being missed |
| `w_noshow` | Actual vs predicted rates | Weekly | Calibrate penalty to match observed impact |
| `w_overbooking` | No-show rate trend | Monthly | If no-shows rising, increase overbooking threshold |
| `capacity_buffer` | Duration prediction errors | Weekly | Widen buffer if MAE increasing |

---

## 6. Three-Tier Data Strategy

### 6.1 Tier Transition Plan

```
CURRENT STATE (April 2026 — Dissertation submission)
│
│  Tier 1 Only: Synthetic data + NHS open data
│  ├── Synthetic data generated to match SACT v4.0 schema
│  ├── NHS open data (CWT Jan 2026, SCMD-IP Jan 2026) for calibration
│  └── All 12 ML models operational on synthetic data
│
▼
NEAR-TERM (Post-dissertation, 3-6 months — by August 2026)
│
│  Tier 1 + 2: Open data + Local hospital exports
│  ├── Velindre provides ChemoCare exports
│  ├── Models retrained on real attendance data
│  ├── SACT v4.0 first complete dataset (August 2026) triggers Level 3 retrain
│  └── Hierarchical model learns real patient effects
│
▼
FULL DEPLOYMENT (6-12 months — by early 2027)
│
│  Tier 1 + 2 + 3: All sources connected
│  ├── SACT v4.0 data flowing monthly (from August 2026 onwards)
│  ├── Full auto-learning pipeline active
│  ├── Models continuously recalibrate on real treatment data
│  └── Optimization adapts to real performance data from Velindre
```

### 6.2 Graceful Degradation

The system works at every tier:

| Tier Available | Capability Level | Accuracy |
|---------------|-----------------|----------|
| **Tier 1 only** (now) | National baselines, synthetic patient models | Moderate |
| **Tier 1 + 2** | Local calibration, real patient patterns | Good |
| **Tier 1 + 2 + 3** | Full pipeline, standardized SACT data | Best |
| **Offline** (no external data) | Last-known parameters, rule-based fallback | Basic |

### 6.3 Runtime promotion from Tier 1 → Tier 2 (file-drop auto-switch)

Tier 2 activation requires **no restart and no configuration change**.  The
`ch2-watcher` daemon thread polls `datasets/real_data/` every 60 s.  When a
hospital drop is detected and the stability gate passes, `initialize_data()`
is called and all 12 ML models retrain on the real outcomes in a background
thread.

**What the hospital does:** place two Excel files in the folder:

```
sact_scheduler/
└── datasets/
    └── real_data/
        ├── patients.xlsx                  (required, SACT v4.0 patient schema)
        ├── historical_appointments.xlsx   (required, SACT v4.0 appointments)
        └── appointments.xlsx              (optional, future schedule)
```

**What the system does automatically:**

| Time | Event |
|---|---|
| 0 s | Files arrive on disk |
| ≤ 60 s | Watcher notices new/updated files |
| 60–120 s | Stability gate — waits one more polling cycle to confirm write has finished |
| ≈ 2 min | Active channel flips Ch1 → Ch2, data reloaded |
| 2 – 5 min | All 12 models retrain in a background thread (no downtime — requests keep serving) |
| 5 min + | Every prediction / optimisation / fairness audit uses the real-data-trained ensemble |

**Guards:**
- **Required-files gate** — a partial drop (only one of the two required files) is refused with a log line; system stays on Ch1.
- **Stability gate** — an export still being written (`mtime` changing) is not promoted until `mtime` is stable across 2 consecutive polls.
- **No auto-downgrade** — removing real files while Ch2 is active logs a warning but does NOT revert to synthetic; manual `POST /api/data/source {"channel":"synthetic"}` is required.

**Monitoring:**
- `GET /api/data/channel2-watcher` — JSON status (active_channel, real_data_ready, real_data_missing, last_rejection_reason, pending_since_mtime, last_check, last_switch).
- **Header channel badge** in `viewer.html` auto-polls every 30 s; purple `CH1 SYNTHETIC` turns green `CH2 REAL HOSPITAL` within ~30 s of promotion. Tooltip explains current state (missing files / stability-gate pending / confirmed Ch2). Visible on every tab.
- Regression-tested with 8 unit tests in `tests/test_channel2_watcher.py`.

---

## 7. Implementation Design

### 7.1 New Module: `data/nhs_data_ingestion.py`

```python
"""
NHS Open Data Ingestion Module

Automated downloading, validation, and processing of NHS open datasets
for continuous model recalibration.
"""

class NHSDataIngester:
    """
    Manages automated ingestion of NHS open data sources.

    Sources:
    - Cancer Waiting Times (monthly CSV)
    - SACT Activity Dashboard (annually)
    - Cancer Treatments Dashboard (annual)
    - NHSBSA Prescribing API (monthly)
    """

    def __init__(self, data_dir='datasets/nhs_open_data'):
        self.data_dir = data_dir
        self.sources = {
            'cancer_waiting_times': {
                'url_template': 'https://www.england.nhs.uk/statistics/...',
                'format': 'csv',
                'frequency': 'monthly',
                'last_downloaded': None
            },
            'sact_activity': {
                'url': 'https://nhsd-ndrs.shinyapps.io/sact_activity/',
                'format': 'dashboard_download',
                'frequency': 'annually',
                'last_downloaded': None
            },
            'nhsbsa_prescribing': {
                'api_url': 'https://opendata.nhsbsa.net/api/3/action/datastore_search_sql',
                'format': 'json_api',
                'frequency': 'monthly',
                'last_downloaded': None
            }
        }

    def check_for_updates(self):
        """Check all sources for new data."""
        ...

    def download_cancer_waiting_times(self, year_month):
        """Download latest CWT CSV from NHS England."""
        ...

    def query_nhsbsa_api(self, sql_query):
        """Query NHSBSA CKAN API for prescribing data."""
        ...

    def validate_data(self, data, schema):
        """Validate downloaded data against expected schema."""
        ...

    def transform_to_internal(self, data, source):
        """Transform NHS data format to internal schema."""
        ...
```

### 7.2 New Module: `ml/auto_recalibration.py`

```python
"""
Automatic Model Recalibration Module

Detects when new data is available, determines which models need
updating, and triggers appropriate recalibration.
"""

class ModelRecalibrator:
    """
    Manages the auto-learning pipeline.

    Recalibration levels:
    0 - Parameter refresh (real-time, no downtime)
    1 - Baseline recalibration (monthly, no downtime)
    2 - Feature weight update (quarterly, <1 min)
    3 - Full retrain (on drift or annual, 2-5 min)
    """

    def __init__(self, models, ingester):
        self.models = models  # Dict of all ML models
        self.ingester = ingester
        self.drift_detector = DriftDetector()
        self.update_log = []

    def check_and_update(self):
        """Main entry point: check for new data and update models."""
        new_data = self.ingester.check_for_updates()

        for source, data in new_data.items():
            if data is not None:
                drift = self.drift_detector.detect(source, data)
                level = self.determine_update_level(source, drift)
                self.execute_update(level, source, data)

    def determine_update_level(self, source, drift_score):
        """Decide recalibration level based on drift magnitude."""
        if drift_score > 0.3:
            return 3  # Full retrain
        elif drift_score > 0.15:
            return 2  # Feature weight update
        elif drift_score > 0.05:
            return 1  # Baseline recalibration
        else:
            return 0  # Parameter refresh only

    def execute_update(self, level, source, data):
        """Execute the appropriate recalibration."""
        ...
```

### 7.3 New Module: `ml/drift_detection.py`

```python
"""
Data and Model Drift Detection

Monitors for distribution shifts in incoming data that indicate
models need retraining.
"""

class DriftDetector:
    """
    Detects when model assumptions no longer hold.

    Methods:
    - Population Stability Index (PSI) for feature drift
    - Kolmogorov-Smirnov test for distribution changes
    - CUSUM for gradual drift detection
    - Performance monitoring (accuracy decay)
    """

    def detect_feature_drift(self, reference_data, new_data):
        """PSI-based feature drift detection."""
        ...

    def detect_concept_drift(self, predictions, actuals, window=30):
        """Monitor prediction accuracy over time."""
        ...

    def detect_prior_shift(self, old_distribution, new_distribution):
        """KS test for prior distribution changes."""
        ...
```

### 7.4 API Endpoints

```
NEW ENDPOINTS:

GET  /api/data/nhs/status          - Show all NHS data source status
POST /api/data/nhs/check-updates   - Trigger check for new data
POST /api/data/nhs/download/{src}  - Download specific source
GET  /api/data/nhs/history         - Show ingestion history

GET  /api/ml/recalibration/status  - Show model recalibration status
POST /api/ml/recalibration/run     - Trigger manual recalibration
GET  /api/ml/drift/report          - Show drift detection report
GET  /api/ml/drift/history         - Show drift history over time
```

### 7.5 File Structure

```
sact_scheduler/
  data/
    nhs_data_ingestion.py        # NEW: Download and validate NHS open data
    nhs_data_transformer.py      # NEW: Transform to internal format
  ml/
    auto_recalibration.py        # NEW: Model recalibration pipeline
    drift_detection.py           # NEW: Drift detection algorithms
    model_versioning.py          # NEW: Version and rollback models
  datasets/
    nhs_open_data/               # NEW: Downloaded NHS data storage
      cancer_waiting_times/
        cwt_2026_01.csv
        cwt_2026_02.csv
        ...
      sact_activity/
        sact_q1_2026.json
        ...
      prescribing/
        nhsbsa_2026_01.json
        ...
    model_versions/              # NEW: Versioned model snapshots
      v1.0_2026_03/
      v1.1_2026_04/
      ...
  config.py                      # UPDATE: Add data source URLs and schedules
  flask_app.py                   # UPDATE: Add new API endpoints
```

### 7.6 v4.0 Advanced ML Modules

The following modules are integrated into the auto-learning pipeline:

| Module | File | Trigger | Effect on Optimization |
|--------|------|---------|----------------------|
| Sensitivity Analysis | `ml/sensitivity_analysis.py` | On-demand via API | Feature importance context |
| Model Cards | `ml/model_cards.py` | On-demand via API | Transparency & fairness audit |
| Causal Validation | `ml/causal_validation.py` | After recalibration | 7 integrity tests (placebo + falsification) |
| Online Learning | `ml/online_learning.py` | Each outcome observed | Bayesian posterior blend (Beta-Bernoulli) |
| RL Scheduler | `ml/rl_scheduler.py` | Each optimization run | Pre-opt recommendation + post-opt learning |

**Ensemble Training at Startup:**
- NoShow ensemble (RF + GB + XGBoost) trains in background thread
- Training data: historical_appointments.xlsx (1,899 records, 102 columns with SACT v4.0 fields, Grade A 100/100; travel distribution calibrated against pseudonymised Velindre patient travel data, n=5,116 — distances real, identifiers synthetic)
- AUC: RF=0.635, GB=0.644, XGB=0.620, stacking=0.635
- Server responds immediately — predictions update when training completes

**Advanced Enhancement Pipeline:**
Called automatically in `run_ml_predictions()`:
1. Hierarchical Bayesian → patient/site random effects
2. Online Learning → Beta(273, 1655) posterior blend
3. Conformal → p90 overrun buffer (7min per patient)
4. RL Agent → pre-optimization action recommendation
5. Causal Model → weather ATE=0.046 adjustment

---

## 8. Monitoring and Drift Detection

### 8.1 Drift Detection Methods

| Method | What It Detects | Applied To |
|--------|----------------|-----------|
| **Population Stability Index (PSI)** | Feature distribution shift | Input features vs training distribution |
| **Kolmogorov-Smirnov test** | Distribution change | National rates vs model assumptions |
| **CUSUM** | Gradual drift | Prediction error trend over time |
| **Performance decay** | Accuracy drop | AUC-ROC / MAE on recent predictions |

### 8.2 PSI Formula

$$PSI = \sum_{i=1}^{B} (P_i^{new} - P_i^{ref}) \cdot \ln\left(\frac{P_i^{new}}{P_i^{ref}}\right)$$

Where $P_i$ is the proportion of observations in bin $i$.

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant change | No action |
| 0.1 - 0.25 | Moderate change | Level 1 recalibration |
| > 0.25 | Significant change | Level 3 full retrain |

### 8.3 Monitoring Dashboard Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MODEL HEALTH DASHBOARD                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DATA FRESHNESS                       MODEL PERFORMANCE                      │
│  ├── CWT Data:    15 days old         ├── No-Show AUC:    0.82 (stable)    │
│  ├── SACT Data:   45 days old         ├── Duration MAE:   11.3 (stable)    │
│  ├── Weather:     32 minutes old      ├── Conformal Cov:  91% (stable)     │
│  └── Traffic:     3 minutes old       └── Event Impact:   +6.8% (stable)   │
│                                                                              │
│  DRIFT INDICATORS                     LAST RECALIBRATION                     │
│  ├── Feature PSI:  0.08 (OK)         ├── Level 0: 3 min ago               │
│  ├── Target Drift: 0.03 (OK)         ├── Level 1: 12 days ago             │
│  ├── Prior Shift:  0.12 (WATCH)      ├── Level 2: 45 days ago             │
│  └── CUSUM:        Normal             └── Level 3: Never (initial train)   │
│                                                                              │
│  NEXT SCHEDULED UPDATES                                                      │
│  ├── CWT Download: 1 May 2026 (Feb 2026 data)                             │
│  ├── NHSBSA SCMD-IP: 20 May 2026 (Mar 2026 data)                          │
│  ├── SACT v4.0 first data: August 2026 (not April — rollout period)       │
│  └── Full Retrain: August 2026 (Tier 3 first complete SACT v4.0 data)     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. SACT v4.0 Integration Readiness

> **DISSERTATION NOTE:**
> SACT v4.0 data collection commenced 1 April 2026.
> Rollout data (April–June 2026) is **partially usable** for Level 1-2 recalibration,
> though nationally incomplete (reflects submitting trusts only, not full England).
> **First complete dataset expected August 2026** — triggers Level 3 full retrain.
> This dissertation uses synthetic v4.0-schema-compatible data; auto-detection code is
> ready to process real data as soon as CSV files are placed in `sact_v4/`.

### 9.1 When SACT v4.0 Data Becomes Available (August 2026)

| SACT v4.0 Timeline | Data Quality | System Response |
|---------------------|-------------|-----------------|
| April 2026 — Collection commenced | `preliminary` | ✓ Usable for Level 1-2 recalibration (partial trusts) |
| April–June 2026 — Rollout period | `preliminary` | Auto-learning triggers Level 1-2 on any local CSV found |
| July 2026 — Full conformance | `conformance` | All trusts submitting — Level 2 recalibration |
| August 2026 — First complete dataset | `complete` | ✓ Level 3 full retrain triggered automatically |
| Steady state (monthly) | `complete` | Monthly Level 1, quarterly Level 2, annual Level 3 |

**Auto-detection:** Place SACT v4.0 CSV in `datasets/nhs_open_data/sact_v4/` — scheduler detects within 24h, reads quality phase from date, routes to appropriate recalibration level.

**API trigger:** `POST /api/data/sact-v4/check` — manually trigger check + recalibration immediately.

### 9.2 SACT v4.0 Fields That Directly Improve Models

| SACT v4.0 Field | Model Improved | How |
|-----------------|---------------|-----|
| Performance_Status (WHO 0-4) | No-show + Duration | Strong predictor of fitness and compliance |
| Intent_Of_Treatment | Optimizer | Curative patients get higher priority |
| Cycle_Delay + Reason | No-show model | Direct no-show/delay outcome data |
| Regimen_Modification_Reason | Causal model | Understanding WHY changes happen |
| Toxicity_Grade | Duration model | Toxicity extends treatment times |
| Administration_Timestamp | Duration model | Actual vs planned timing |
| End_Of_Regimen_Summary | Survival model | Treatment outcome for survival analysis |

### 9.3 The Auto-Learning Advantage

```
STATIC SYSTEM (Traditional)           AUTO-LEARNING SYSTEM (This Project)
──────────────────────────            ─────────────────────────────────────

Train once on historical data         Train on synthetic → recalibrate monthly
       │                                     │
       ▼                                     ▼
Deploy model                          Deploy model + ingestion pipeline
       │                                     │
       ▼                                     ▼
Performance degrades over time        Monthly CWT data → baseline update
       │                                     │
       ▼                                     ▼
Manual retrain (if anyone notices)    Annual SACT data → weight update
       │                                     │
       ▼                                     ▼
Long periods of suboptimal            Drift detected → auto retrain
performance                                  │
                                             ▼
                                      Continuous improvement
                                      Performance maintained or improves
```

---

## Appendix: NHS Open Data URL Reference

| Source | URL | Notes |
|--------|-----|-------|
| Cancer Waiting Times Statistics | https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/ | Monthly CSV downloads |
| 2025-26 Monthly Data | https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/monthly-data-and-summaries/2025-26-monthly-cancer-waiting-times-statistics/ | Current year data |
| SACT Activity Dashboard | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/sact-activity | Annually, with download |
| Cancer Treatments Dashboard | https://nhsd-ndrs.shinyapps.io/cancer_treatments/ | Annual, downloadable |
| SACT Dataset Spec | https://digital.nhs.uk/ndrs/data/data-sets/sact | v4.0 documentation |
| NHSBSA Open Data API | https://opendata.nhsbsa.net/api/3/action/datastore_search_sql | CKAN SQL API |
| NHS Developer API Catalogue | https://digital.nhs.uk/developer/api-catalogue | All available APIs |
| CWT Data Contact | england.cancerwaitsdata@nhs.net | For data queries |
| NDRS Data Contact | ndrsenquiries@nhs.net | For SACT data queries |

---

*Document prepared for MSc Data Science Dissertation*
*Cardiff University - School of Mathematics*
*SACT Scheduling Optimization Project*

*Version 1.1 | April 2026 — Updated: NHSBSA SCMD package corrected to SCMD-IP, SACT v4.0 availability updated to August 2026 (first complete dataset)*
