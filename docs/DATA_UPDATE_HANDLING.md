# Data Update Handling

## How the SACT Scheduling System Automatically Ingests, Validates, and Learns from NHS Open Data

**Version:** 1.1
**Date:** April 2026
**Status:** Operational — CWT (Jan 2026) and SCMD-IP (Jan 2026) downloading; SACT v4.0 first complete data expected August 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Sources](#3-data-sources)
4. [Auto-Learning Scheduler](#4-auto-learning-scheduler)
5. [Data Ingestion Pipeline](#5-data-ingestion-pipeline)
6. [Drift Detection](#6-drift-detection)
7. [Model Recalibration](#7-model-recalibration)
8. [Data Validation](#8-data-validation)
9. [File Storage](#9-file-storage)
10. [API Endpoints](#10-api-endpoints)
11. [Monitoring and Logging](#11-monitoring-and-logging)
12. [SACT v4.0 Readiness](#12-sact-v40-readiness)

---

## 1. Overview

The system maintains a continuous learning loop:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   AUTO-LEARNING DATA UPDATE CYCLE                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐       │
│   │  INGEST  │────▶│ VALIDATE │────▶│  DETECT  │────▶│RECALIBRATE│      │
│   │          │     │          │     │  DRIFT   │     │          │       │
│   │ NHS Open │     │ Schema   │     │ PSI, KS  │     │ Level 0-3│       │
│   │ Data     │     │ NaN, Type│     │ CUSUM    │     │ Models   │       │
│   └─────────┘     └──────────┘     └──────────┘     └──────────┘       │
│        ▲                                                    │            │
│        │              Every 24 Hours                        │            │
│        └────────────────────────────────────────────────────┘            │
│                                                                          │
│   Background Thread (daemon) — starts 5 min after server boot           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
- Zero manual intervention required for routine updates
- Graceful degradation — system works without any external data
- Version snapshots before every recalibration for rollback
- All operations logged for audit trail

---

## 2. Architecture

### 2.1 Components

| Component | File | Role |
|-----------|------|------|
| **NHSDataIngester** | `data/nhs_data_ingestion.py` | Downloads and validates NHS open data |
| **DriftDetector** | `ml/drift_detection.py` | Detects distribution shifts in incoming data |
| **ModelRecalibrator** | `ml/auto_recalibration.py` | Triggers model updates at appropriate level |
| **Background Scheduler** | `flask_app.py` (thread) | Runs the cycle every 24 hours |
| **API Endpoints** | `flask_app.py` | Manual triggers and status monitoring |

### 2.2 Data Flow

```
NHS England Website ──┐
                      │   HTTP GET
NHSBSA Open Data ─────┤──────────▶ NHSDataIngester
                      │                   │
NHS Digital NDRS ─────┘                   │
                                          ▼
                                   ┌──────────────┐
                                   │   Validate    │
                                   │ • Schema check│
                                   │ • NaN removal │
                                   │ • Type cast   │
                                   │ • Hash compute│
                                   └──────┬───────┘
                                          │
                                   Is data new?
                                   (hash comparison)
                                          │
                               ┌──────────┴──────────┐
                               │ YES                  │ NO
                               ▼                      ▼
                        ┌──────────────┐       Log "up to date"
                        │ Save to disk │       and skip
                        │ Update hash  │
                        │ Log ingestion│
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ Drift        │
                        │ Detection    │
                        │ PSI + KS     │
                        └──────┬───────┘
                               │
                        Drift score > threshold?
                               │
                    ┌──────────┴──────────┐
                    │ YES                  │ NO
                    ▼                      ▼
             ┌──────────────┐       Level 0 refresh
             │ Recalibrate  │       (parameters only)
             │ Level 1-3    │
             │ based on     │
             │ drift score  │
             └──────────────┘
```

---

## 3. Data Sources

### 3.1 Cancer Waiting Times (CWT)

| Property | Detail |
|----------|--------|
| **Publisher** | NHS England |
| **URL** | Scraped from `england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/` |
| **Method** | Scrape page for latest CSV link, then HTTP GET |
| **Format** | CSV (~5 MB, ~27,000 rows) |
| **Update frequency** | Monthly |
| **Content** | Provider-level waiting times by cancer type, standard (31-day, 62-day, FDS) |
| **File naming** | `{Month}-{Year}-Monthly-Combined-CSV-Provisional.csv` |
| **Storage** | `datasets/nhs_open_data/cancer_waiting_times/` |
| **Status** | Operational — 27,003 records downloaded |
| **⚠ Decommissioning** | **NHS England CWT system decommissions June 2026.** Final monthly release will be June 2026. Successor source (NHS NDRS Cancer Data Hub / Faster Diagnosis Standard reporting) will be needed for deployments beyond June 2026. |

**How it's downloaded:**
```python
# 1. Scrape the statistics page for CSV links
page = requests.get(stats_url)
csv_links = re.findall(r'href="([^"]*Combined-CSV[^"]*\.csv)"', page.text)

# 2. Download the latest (first) CSV
response = requests.get(csv_links[0])

# 3. Compute hash to check if data is new
data_hash = sha256(response.content)[:16]
is_new = (data_hash != last_hash)

# 4. Save to disk
with open(output_file, 'wb') as f:
    f.write(response.content)
```

**What the system learns from CWT data:**
- Baseline no-show/cancellation rates by provider and cancer type
- Seasonal patterns in waiting times (monthly trends)
- 62-day and 31-day target compliance rates
- Performance benchmarks for the optimizer

### 3.2 NHSBSA Secondary Care Medicines Data with Indicative Price (SCMD-IP)

> **⚠ IMPORTANT:** The original `secondary-care-medicines-data` dataset was retired June 2022.
> The replacement is `secondary-care-medicines-data-indicative-price` (SCMD-IP).
> Code updated accordingly.

| Property | Detail |
|----------|--------|
| **Publisher** | NHS Business Services Authority |
| **API** | `opendata.nhsbsa.net/api/3/action/package_show` |
| **Package ID** | `secondary-care-medicines-data-indicative-price` |
| **Method** | CKAN API → package metadata → resource catalogue |
| **Format** | JSON (catalogue of CSV resources) |
| **Update frequency** | Monthly (~20th, 2-month lag; latest: January 2026) |
| **Content** | Hospital secondary care prescribing with indicative prices, including chemotherapy drugs |
| **Storage** | `datasets/nhs_open_data/prescribing/` |
| **Status** | Operational — resource catalogue downloaded |

**How it's downloaded:**
```python
# 1. Query CKAN API for package metadata (SCMD with Indicative Price — replaces retired SCMD)
response = requests.get(
    'https://opendata.nhsbsa.net/api/3/action/package_show',
    params={'id': 'secondary-care-medicines-data-indicative-price'}  # NOT 'secondary-care-medicines-data' (retired)
)

# 2. Extract resource list (each resource = one month of data)
resources = response.json()['result']['resources']

# 3. Save catalogue with download URLs for each monthly dataset
# (Individual CSVs are ~100-500 MB, so we catalogue rather than download all)
```

**What the system learns from SCMD data:**
- Drug prescribing trends (new regimens appearing in hospital data)
- Volume changes (demand forecasting signals)
- Which drugs are being prescribed at which trusts

### 3.3 SACT Activity Dashboard Metadata

| Property | Detail |
|----------|--------|
| **Publisher** | NHS Digital NDRS |
| **URL** | `digital.nhs.uk/ndrs/data/data-sets/sact` |
| **Method** | Fetch page metadata (dashboard is interactive Shiny app) |
| **Format** | JSON (metadata) |
| **Update frequency** | Annually (confirmed April 2026 — previously listed as quarterly) |
| **Content** | SACT v4.0 specification status, dataset documentation |
| **Storage** | `datasets/nhs_open_data/sact_activity/` |
| **Status** | Operational — metadata saved |
| **Note** | Interactive dashboard data requires manual download via UI |

**Limitation:** The SACT Activity Dashboard is an R Shiny application (`nhsd-ndrs.shinyapps.io/sact_activity/`) which cannot be scraped programmatically. The system fetches the SACT specification page metadata and monitors for updates. Users can manually download data from the dashboard and place CSV/Excel files in `datasets/nhs_open_data/sact_activity/` — the system will detect and ingest them.

### 3.4 SACT v4.0 Patient-Level Data (First complete dataset: August 2026)

> **⚠ DISSERTATION NOTE:** SACT v4.0 collection commenced 1 April 2026.
> Data is NOT available externally during the rollout period (April–June 2026).
> The first complete monthly dataset is expected **August 2026**.

| Property | Detail |
|----------|--------|
| **Collection commenced** | 1 April 2026 |
| **Rollout period** | April–June 2026 (partial trust submissions — usable for preliminary Level 1-2 recalibration; nationally incomplete) |
| **Full conformance** | 1 July 2026 |
| **First complete dataset** | **August 2026** (July data + NDRS processing lag) |
| **Format** | CSV, 60 data items across 7 sections |
| **Content** | Patient-level treatment records |
| **System readiness** | Field names already aligned in synthetic data generator |

When SACT v4.0 data becomes available (August 2026 onwards):
1. Trust places monthly CSV export in `datasets/nhs_open_data/sact_v4/`
2. System auto-detects new file
3. Validates against SACT v4.0 schema (60 fields, 7 sections)
4. Triggers Level 3 full model retrain
5. All 12 ML models recalibrate on real patient data

---

## 4. Auto-Learning Scheduler

### 4.1 Background Thread

The scheduler runs as a **daemon thread** in the Flask process:

```python
def start_auto_learning_scheduler():
    """Background thread that checks NHS data every 24 hours."""

    def auto_learning_loop():
        time.sleep(300)           # First check: 5 minutes after startup
        check_interval = 86400   # Then every 24 hours

        while True:
            # 1. Check all NHS data sources
            results = ingester.check_and_download_all()

            # 2. If new data found, run drift detection
            if any(r.is_new_data for r in results):
                drift_summary = detector.full_drift_check(new_data)

                # 3. Trigger recalibration at appropriate level
                recalibrator.check_and_update(results, drift_summary)

            time.sleep(check_interval)

    thread = threading.Thread(target=auto_learning_loop, daemon=True)
    thread.start()
```

### 4.2 Schedule

| Event | Timing | What Happens |
|-------|--------|--------------|
| Server starts | t=0 | All 12 ML models trained on synthetic data |
| First auto-check | t=5 min | Download NHS CWT, NHSBSA, SACT metadata |
| Subsequent checks | Every 24h | Re-check all sources for updates |
| New data found | Triggered | Drift detection → recalibration |
| Manual check | On demand | Via "Check Now" button or API |
| Manual recalibrate | On demand | Via "Recalibrate" button or API |

### 4.3 Thread Safety

- The scheduler thread is a **daemon** — dies automatically when the Flask process exits
- Data writes use file-level locking (one file per download, no concurrent writes)
- Model recalibration updates model objects in-place (atomic reference swap)
- Ingestion history saved to `datasets/nhs_open_data/ingestion_history.json`

### 4.4 Channel 2 watcher — real hospital data file-drop (runtime auto-switch)

A second daemon, `start_real_data_watcher()`, monitors `datasets/real_data/`
every 60 s for hospital export files (no restart required to activate).  Behaviour:

```python
while True:
    fp = _real_data_fingerprint()             # checks both required files
    if fp.ready and active_channel != 'real':
        if fp.mtime unchanged from previous poll:   # stability gate
            DATA_SOURCE_CONFIG['active_channel'] = 'real'
            initialize_data()                  # reload + retrain all 12 models
    sleep(60)
```

**Required files (both must be present):**

| File | Purpose |
|---|---|
| `datasets/real_data/patients.xlsx` | Patient master list in SACT v4.0 schema |
| `datasets/real_data/historical_appointments.xlsx` | Attendance outcomes for ML training |
| `datasets/real_data/appointments.xlsx` | Optional — future schedule |

**End-to-end timeline:**

| Time | Event |
|---|---|
| 0 s | Files written to `datasets/real_data/` |
| ≤ 60 s | Watcher detects new/updated files |
| 60–120 s | Stability gate confirms write finished (mtime unchanged across 2 polls) |
| ≈ 2 min | `active_channel` flips `synthetic` → `real`; data reloaded |
| 2–5 min | All 12 ML models retrain in background thread — no downtime |
| 5 min + | All predictions and optimisations use real-data-trained models |

**Correctness guards:**
- **Required-files gate** — partial drops (only `patients.xlsx`, missing `historical_appointments.xlsx`) rejected; system stays on Ch1 with clear log.
- **Stability gate** — prevents racing with an in-flight export write.
- **No auto-downgrade** — files removed while Ch2 is active logs a warning but does not revert; manual `POST /api/data/source {"channel":"synthetic"}` required.

**Monitoring:**
- **API:** `GET /api/data/channel2-watcher` returns live state (`active_channel`, `real_data_ready`, `real_data_missing`, `last_rejection_reason`, `pending_since_mtime`, `last_check`, `last_switch`).
- **Web UI:** the viewer header shows a persistent **channel badge** (purple `CH1 SYNTHETIC` / green `CH2 REAL HOSPITAL`) that polls the watcher endpoint every 30 s, so the auto-promotion is visible on every tab within ~30 s of the switch. Hover for a tooltip that explains the current state (missing files, stability gate pending, or confirmed Ch2).
- **Tests:** 8 unit tests in `tests/test_channel2_watcher.py`.

---

## 5. Data Ingestion Pipeline

### 5.1 Ingestion Steps

For each data source:

```
Step 1: CHECK IF DUE
  └── Compare last_downloaded timestamp against frequency
      Monthly source: due if > 28 days since last download
      Quarterly source: due if > 85 days

Step 2: DOWNLOAD
  └── Source-specific method (scrape, API, or file check)
      Timeout: 60 seconds per download
      Retries: none (will retry next cycle)

Step 3: VALIDATE
  └── Check HTTP status code
      Verify content is not empty
      Parse to confirm valid CSV/JSON

Step 4: HASH COMPARISON
  └── SHA-256 hash of downloaded content
      Compare to stored hash from last download
      If identical → data unchanged, skip processing

Step 5: SAVE TO DISK
  └── Write to datasets/nhs_open_data/{source}/{filename}
      Update source config (last_downloaded, last_hash)

Step 6: LOG RESULT
  └── Append to ingestion_history.json
      Log success/failure with timestamp and record count
```

### 5.2 Error Handling

| Error | Handling |
|-------|----------|
| Network timeout | Log warning, skip source, retry next cycle |
| HTTP 404 | Log error, source URL may have changed |
| HTTP 500 | Log error, server-side issue, retry next cycle |
| Parse error | Save raw content, log warning |
| Disk full | Log error, skip save |
| Hash unchanged | Log "up to date", skip processing |

All errors are non-fatal — the system continues operating with existing data.

---

## 6. Drift Detection

### 6.1 When Drift Detection Runs

Drift detection triggers when new data is successfully downloaded. It compares the current model's training data distribution against the new incoming data.

### 6.2 Methods

**Population Stability Index (PSI):**

$$PSI = \sum_{i=1}^{B} (P_i^{new} - P_i^{ref}) \cdot \ln\left(\frac{P_i^{new}}{P_i^{ref}}\right)$$

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | No significant shift | No action |
| 0.10 - 0.25 | Moderate shift | Level 1-2 recalibration |
| > 0.25 | Significant shift | Level 3 full retrain |

**Kolmogorov-Smirnov Test:**

Two-sample test comparing reference and current distributions. Reject H₀ (same distribution) if p-value < 0.05.

**CUSUM (Cumulative Sum):**

Detects gradual drift in model performance over time. Alarm when cumulative standardized residuals exceed threshold h=5.

### 6.3 What Gets Compared

| Feature | Reference (Training Data) | Current (New Data) |
|---------|--------------------------|-------------------|
| No-show rates | Historical appointment outcomes | CWT compliance rates |
| Duration distributions | Regimen-specific durations | Updated regimen volumes |
| Demographic shifts | Patient age/gender distribution | SACT Activity demographics |
| Drug volumes | Regimen frequency weights | NHSBSA prescribing trends |

---

## 7. Model Recalibration

### 7.1 Recalibration Levels

| Level | Trigger | What Changes | Models Affected | Downtime |
|-------|---------|--------------|-----------------|----------|
| **0** | Real-time event data | Weather/traffic coefficients | Event Impact | None |
| **1** | Monthly open data, PSI < 0.10 | No-show base rates, seasonal factors | No-Show, Duration, Hierarchical | None |
| **2** | Quarterly data, PSI 0.10-0.25 | Feature weights, demographic adjustments | All ensemble models, Uplift, Causal | < 1 min |
| **3** | PSI > 0.25 or new data tier | Full model retrain with new data | All 12 models | 2-5 min |

> **Floor rule:** `recommended_recalibration_level` always returns at least **1** (never `None`).
> During the SACT v4.0 rollout period (April–June 2026) when no local CSV files are present,
> the ingester defaults to Level 1 — synthetic data is always sufficient for baseline recalibration.
>
> **Synthetic baseline (v4.6):** 1,899 historical records (102 cols, Grade A 100/100), 250 patients (82 cols, Grade A 91.4/100), 819 appointments (70 cols, Grade A 100/100). Travel distribution calibrated against pseudonymised Velindre patient travel data (n=5,116; distances real, identifiers synthetic): Near (<20 min) ~38%, Medium (20–45 min) ~60%, Remote (>45 min) ~2.4%. All SACT v4.0 sections 1–7 fields present across all files. AUC5 drug dosing (Carboplatin) uses Calvert formula. Use `generate_sample_data.py` to regenerate with `POSTCODE_WEIGHTS` applied.

### 7.2 Level 1: Baseline Recalibration

Updates population-level parameters using exponential moving average:

$$P_{base}^{new} = 0.2 \cdot P_{observed} + 0.8 \cdot P_{base}^{old}$$

Applied to:
- No-show baseline rates (from CWT compliance data)
- Duration baselines (from SACT activity regimen volumes)
- Hierarchical model population priors (μ, τ²)

### 7.3 Level 3: Full Retrain

```
1. Save current model version snapshot
2. Load new training data
3. Retrain all 12 models:
   - Ensemble (RF+GB+XGB)
   - Sequence (GRU)
   - Survival (Cox PH)
   - Uplift (S+T Learner)
   - Multi-Task (PyTorch)
   - Quantile Forest
   - Hierarchical (PyMC MCMC)
   - Causal DAG + IV + DML
   - Event Impact
   - Conformal Prediction
   - MC Dropout (inherits from retrained NN)
4. Run validation on holdout set
5. If performance acceptable, deploy new models
6. Log recalibration result
```

### 7.4 Version Snapshots

Before every Level 2+ recalibration:

```
datasets/model_versions/
  v2026.03.21/
    metadata.json      # Version, timestamp, models, baselines
  v2026.04.01/
    metadata.json      # After first real CWT data update
```

Enables rollback if a recalibration degrades performance.

### Level 4: Advanced ML Enhancement Pipeline

When data updates trigger recalibration, the advanced ML enhancement pipeline also runs:

1. **Ensemble retrain**: RF + GB + XGBoost on new historical data
2. **Online Learning calibration**: Beta posterior updated from Attended_Status counts
3. **Conformal recalibration**: p90 overrun recalculated from Actual_Duration
4. **Causal validation**: 7 tests re-run to verify model integrity
5. **Model Cards refresh**: Subgroup metrics recalculated for transparency

Training runs in background thread — server remains responsive during recalibration.

---

## 8. Data Validation

### 8.1 Schema Validation

Each downloaded file is validated against expected schema:

**CWT CSV expected columns:**
```
Basis, Org_Code, Parent_Org, Org_Name, Standard_or_Item,
Cancer_Type, Referral_Route_or_Stage, Measure, Value
```

**NHSBSA SCMD-IP expected structure:**
```json
{
  "package": "secondary-care-medicines-data-indicative-price",
  "description": "NHS hospital secondary care prescribing with indicative prices (replaces retired SCMD dataset, June 2022 onwards)",
  "total_resources": 58,
  "latest_resource": "SCMD_PROVISIONAL_202601",
  "resources": [...]
}
```

### 8.2 Data Quality Checks

| Check | What It Catches |
|-------|----------------|
| Empty content | Failed downloads, empty responses |
| NaN proportion | Corrupted data (reject if >50% NaN) |
| Row count | Suspiciously small files (< 100 rows for CWT) |
| Column count | Schema changes in published data |
| Date range | Data freshness (reject if older than 6 months) |

---

## 9. File Storage

### 9.1 Directory Structure

```
datasets/
  nhs_open_data/
    ingestion_history.json                      # Log of all downloads
    cancer_waiting_times/
      January-2026-Monthly-Combined-CSV.csv     # 5 MB, 27,003 rows
      cwt_latest.csv                            # Symlink to latest
    prescribing/
      nhsbsa_scmd_catalogue_2026_03.json        # Resource catalogue
    sact_activity/
      sact_metadata_2026_03.json                # Page metadata
  model_versions/
    v2026.03.21/
      metadata.json                             # Version snapshot
  sample_data/
    patients.xlsx                               # 250 patients, 82 cols
    appointments.xlsx                           # 819 appointments, 70 cols
    historical_appointments.xlsx                # 1,899 records, 102 cols (Grade A 100/100)
    regimens.xlsx                               # 20 SACT protocols
    sites.xlsx                                  # 5 Velindre sites
    staff.xlsx                                  # 68 staff, NHS bands
```

### 9.2 Storage Management

- Each source keeps its latest file plus the previous version
- `ingestion_history.json` keeps the last 100 entries
- Model version snapshots kept indefinitely (small metadata files)
- Total disk usage: ~15 MB (dominated by CWT CSV at 5 MB)

---

## 10. API Endpoints

### 10.1 Status and Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data/nhs/status` | GET | Current status of all data sources |
| `/api/ml/recalibration/status` | GET | Model recalibration status and history |
| `/api/ml/drift/report` | GET | Drift detector configuration |

**Example response — `/api/data/nhs/status`:**
```json
{
  "sources": {
    "cancer_waiting_times": {
      "name": "Cancer Waiting Times",
      "frequency": "monthly",
      "last_downloaded": "2026-03-21T20:30:00",
      "files_on_disk": 2,
      "is_due": false,
      "description": "Monthly provider-level cancer waiting times"
    },
    "nhsbsa_prescribing": {
      "name": "NHSBSA Prescribing Data",
      "frequency": "monthly",
      "last_downloaded": "2026-03-21T20:30:05",
      "files_on_disk": 1,
      "is_due": false
    },
    "sact_activity": {
      "name": "SACT Activity Dashboard",
      "frequency": "annually",
      "last_downloaded": "2026-03-21T20:30:08",
      "files_on_disk": 1,
      "is_due": false
    }
  },
  "total_ingestions": 3,
  "last_check": "2026-03-21T20:30:08"
}
```

### 10.2 Manual Triggers

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data/nhs/check-updates` | POST | Immediately check all sources and download |
| `/api/ml/recalibration/run` | POST | Manually trigger recalibration |
| `/api/ml/drift/check` | POST | Run drift detection on current data |

**Example — Manual check:**
```bash
curl -X POST http://localhost:1421/api/data/nhs/check-updates
```

**Response:**
```json
{
  "results": [
    {"source": "cancer_waiting_times", "success": true, "is_new_data": false, "records": 27003},
    {"source": "nhsbsa_prescribing", "success": true, "is_new_data": false, "records": 58},
    {"source": "sact_activity", "success": true, "is_new_data": false, "records": 1}
  ],
  "new_data_found": false
}
```

### 10.3 Viewer Integration

The ML Models tab on the Schedule Viewer (`/viewer`) shows:
- Live status of all data sources (green check / amber warning)
- Files on disk count
- "Check Now" button for immediate download
- "Recalibrate" button for manual model update
- Recalibration levels explained inline

---

## 11. Monitoring and Logging

### 11.1 Log Messages

All data operations are logged via Python `logging` module:

```
INFO  - Auto-learning scheduler started
INFO  - Auto-learning: checking NHS data sources...
INFO  - Scraping CWT page for download links...
INFO  - Downloading CWT: January-2026-Monthly-Combined-CSV-Provisional.csv
INFO  - CWT downloaded: January-2026-..., 27003 records, new=True
INFO  - Fetching NHSBSA SCMD with Indicative Price package info...
INFO  - NHSBSA SCMD-IP catalogue saved: 41 resources, latest=SCMD_IP_202601
INFO  - Fetching SACT dataset page metadata...
INFO  - SACT metadata saved
INFO  - Auto-learning: 3 new data sources found
INFO  - Drift check: 0/10 features drifted, severity=none, action=none
INFO  - Auto-learning: Level 0 recalibration — no significant drift
```

### 11.2 Ingestion History

Stored in `datasets/nhs_open_data/ingestion_history.json`:

```json
[
  {
    "source": "cancer_waiting_times",
    "timestamp": "2026-03-21T20:30:00",
    "success": true,
    "records": 27003,
    "is_new": true,
    "hash": "a3f2b7c1e9d4f6a8"
  },
  {
    "source": "nhsbsa_prescribing",
    "timestamp": "2026-03-21T20:30:05",
    "success": true,
    "records": 41,
    "is_new": true,
    "hash": "b7d1e4f2a8c3b6d9"
  }
]
```

### 11.3 Alerts

The system does NOT send external alerts (email/Slack) as this is a local deployment. Monitoring is via:
- Flask server logs (console output)
- `/api/data/nhs/status` API endpoint
- Viewer ML Models tab (visual status)
- `ingestion_history.json` file

---

## 12. SACT v4.0 Readiness

### 12.1 Current State (April 2026)

> **SACT v4.0 Data Status:** Collection commenced 1 April 2026.
> **Data NOT yet available to external researchers** (rollout period April–June 2026).
> **First complete dataset expected: August 2026.**
> This dissertation uses synthetic v4.0-compatible data.

The system is fully prepared for SACT v4.0 data when it becomes available:

| Readiness Item | Status |
|---------------|--------|
| Field names match SACT v4.0 | Yes — 60 fields aligned in synthetic data |
| CSV ingestion pipeline | Yes — `check_and_download_all()` handles CSV |
| Validation against schema | Yes — column count and type checks |
| Drift detection | Yes — PSI + KS on numeric features |
| Level 3 retrain capability | Yes — full model retrain in 2-5 min |
| Version snapshots | Yes — saved before every retrain |

### 12.2 What Happens When SACT v4.0 Data Arrives (August 2026)

```
Timeline:
  August 2026    First COMPLETE monthly SACT v4.0 CSV available from NDRS
                 (July 2026 data after 4-week processing lag)
       │
       ▼
  Trust/NDRS     Place monthly CSV in sact_activity/ directory
       │
       ▼
  Auto-detect    System detects new file in next 24h check
       │
       ▼
  Validate       Check 60-field schema, 7 sections
       │
       ▼
  Drift detect   Compare distributions to synthetic training data
       │         (Expected: significant drift → PSI > 0.25)
       │
       ▼
  Level 3        Full retrain of all 12 models on real data
       │
       ▼
  Version snap   Save v2026.04.xx snapshot (rollback available)
       │
       ▼
  Deploy         New models active, serving real predictions
       │
       ▼
  Monitor        Continuous drift detection on subsequent months
```

### 12.3 Expected Performance Changes

| Metric | On Synthetic | Expected on Real | Reason |
|--------|-------------|------------------|--------|
| AUC-ROC (no-show) | 0.82 | 0.78-0.85 | Real data may have more noise |
| MAE (duration) | 11.3 min | 10-15 min | Depends on data documentation quality |
| Conformal coverage | 91% | 88-92% | Calibration adjusts automatically |
| IV F-statistic | 2,237 | Varies | Depends on real weather-traffic correlation |

---

## Appendix: Quick Reference

### Manual Operations

```bash
# Check for new NHS data now
curl -X POST http://localhost:1421/api/data/nhs/check-updates

# Trigger Level 1 recalibration
curl -X POST http://localhost:1421/api/ml/recalibration/run \
  -H "Content-Type: application/json" \
  -d '{"level": 1, "source": "manual"}'

# Run drift detection
curl -X POST http://localhost:1421/api/ml/drift/check

# View current status
curl http://localhost:1421/api/data/nhs/status
curl http://localhost:1421/api/ml/recalibration/status
```

### File Locations

```
Data sources:     datasets/nhs_open_data/
Model versions:   datasets/model_versions/
Ingestion log:    datasets/nhs_open_data/ingestion_history.json
Flask log:        sact_scheduler.log (console)
Config:           config.py (OPTIMIZATION_WEIGHTS, PARETO_WEIGHT_SETS)
```

### Key Classes

```python
from data.nhs_data_ingestion import NHSDataIngester
from ml.drift_detection import DriftDetector
from ml.auto_recalibration import ModelRecalibrator

# Manual usage
ingester = NHSDataIngester()
results = ingester.check_and_download_all()

detector = DriftDetector()
detector.set_reference(training_data)
summary = detector.full_drift_check(new_data)

recalibrator = ModelRecalibrator(models)
recalibrator.check_and_update(results, summary)
```

---

*Document prepared for MSc Data Science Dissertation*
*Cardiff University — School of Mathematics*
*SACT Scheduling Optimization Project*

*Version 1.0 | March 2026*
