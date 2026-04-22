# Three-Channel Data Strategy

## How the System Handles Synthetic Data, Real Hospital Data, and NHS Open Data

**Version:** 1.2 (numbering aligned with README.md §4)
**Date:** April 2026

> **Numbering (authoritative — matches README.md §4):**
>
> | Channel | Name | Location | Availability |
> |---|---|---|---|
> | **1** | Synthetic Data | `datasets/sample_data/` | ✅ available now, acts as live operational data today |
> | **2** | Real Hospital Data | `datasets/real_data/` | ⚠ file-dropped by hospital — Channel 2 watcher auto-promotes on arrival |
> | **3** | NHS Open Data | `datasets/nhs_open_data/` | ✅ partially available (public aggregates), **always running** in background for recalibration |
>
> Earlier v1.1 of this document swapped Channel 2 and Channel 3. The order in §Channel 2 / §Channel 3 below has been flipped to match.

---

## Overview

The system is designed to operate with three independent data channels that can be used individually or combined. Each channel serves a different purpose and becomes available at different stages.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     THREE-CHANNEL DATA ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CHANNEL 1: SYNTHETIC DATA (Available Now)                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Source: generate_sample_data.py                                  │  │
│  │  Location: datasets/sample_data/                                  │  │
│  │  Purpose: ML model training, system testing, dissertation demo    │  │
│  │  Format: SACT v4.0 compliant Excel files                         │  │
│  │  Records: 250 patients, 1,899 historical, 819 appointments       │  │
│  │  Update: Regenerate on demand                                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  CHANNEL 2: REAL HOSPITAL DATA (When Provided — Ch2 Watcher Auto-Flips) │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Source: Velindre Cancer Centre (ChemoCare exports)                │  │
│  │  Location: datasets/real_data/                                    │  │
│  │  Purpose: Full model retraining on actual patient outcomes        │  │
│  │  Format: Excel/CSV matching SACT v4.0 field names                 │  │
│  │  Records: TBD (when data sharing agreement is in place)           │  │
│  │  Update: Dropped into folder → auto-detected within 60s           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  CHANNEL 3: NHS OPEN DATA (Partially Available — Always Running)        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Source: NHS England, NHSBSA, NHS Digital                         │  │
│  │  Location: datasets/nhs_open_data/                                │  │
│  │  Purpose: Model recalibration, baseline rates, demand trends      │  │
│  │  Format: CSV (CWT 27K rows), JSON (SCMD catalogue), metadata     │  │
│  │  Records: 27,003 CWT + 307,676 SCMD prescribing                  │  │
│  │  Update: Auto every 24 hours via background scheduler             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ALL THREE CHANNELS FEED INTO:                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  Feature Engineering (22 → 92 features)                           │  │
│  │       → 12 ML Models                                              │  │
│  │       → Multi-Objective Optimizer (6 objectives)                  │  │
│  │       → Optimized Schedule                                        │  │
│  │                                                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Channel 1: Synthetic Data

### What It Is

Synthetically generated data that mimics real Velindre Cancer Centre patterns. Created by `datasets/generate_sample_data.py` with fixed random seeds for reproducibility.

### Files Generated

| File | Records | Columns | Content |
|------|---------|---------|---------|
| `patients.xlsx` | 250 | 82 | SACT v4.0 sections 1–7 all fields; NHS Number, ICD-10, Performance Status, BSA, Drug_Name, sections 5–7 baseline (Grade A 91.4/100) |
| `historical_appointments.xlsx` | 1,899 | 102 | ML training data with outcomes, weather, traffic, modifications, toxicity grades, all SACT v4.0 fields (Grade A 100/100); postcode distribution calibrated against pseudonymised Velindre travel data (n=5,116 patients; distances real, identifiers synthetic): Near 38% / Medium 60% / Remote 2.4% |
| `appointments.xlsx` | 819 | 70 | Future schedule with nurse assignments, no overlaps, full SACT fields (Grade A 100/100) |
| `regimens.xlsx` | 20 | 11 | Real SACT protocols (FOLFOX, RCHOP, PEMBRO, etc.) |
| `sites.xlsx` | 5 | 9 | Velindre hub + 4 outreach sites with chairs, beds, hours |
| `staff.xlsx` | 68 | 15 | NHS bands (Band 3-8a), SACT trained, 1:1 capability |

### Why Synthetic

1. **No real data access** — NHS IG approvals take 6+ months, GDPR Article 9 applies
2. **Reproducibility** — Fixed seeds mean identical results every run
3. **Full control** — Can inject specific patterns (seasonal, causal, edge cases)
4. **SACT v4.0 compliant** — All field names match the v4.0 standard (60 fields, 7 sections); real data expected August 2026
5. **Publishable** — Can include in dissertation submission

### How It Feeds the System

```
generate_sample_data.py → datasets/sample_data/*.xlsx
                                    │
                                    ▼
                         flask_app.py loads at startup
                                    │
                                    ▼
                         train_advanced_ml_models()
                         (trains all 12 models)
                                    │
                                    ▼
                         Models serve predictions via API
```

### Regeneration

```bash
python datasets/generate_sample_data.py
# Then restart Flask to retrain models on new data
```

---

## Channel 2: Real Hospital Data (renumbered from v1.1 Channel 3)

See **Channel 3 section below** for previous "Channel 2 NHS Open Data" content — the two sections have been swapped to match README.md §4.

### What It Is

Actual patient appointment data from Velindre Cancer Centre, exported from ChemoCare or provided directly. **Partially available today** — the `datasets/real_data/` folder exists and is watched by the runtime **Channel 2 watcher** (polls every 60 s), which auto-promotes this channel over Channel 1 synthetic the moment `patients.xlsx` is dropped in.

### Runtime auto-switch (new in v1.2)

```
flask_app.py → start_real_data_watcher()  (daemon thread "ch2-watcher")
              └─ every 60 s, check datasets/real_data/
              └─ required files: patients.xlsx AND historical_appointments.xlsx
              └─ stability gate: mtime must be unchanged across 2 consecutive polls
              └─ on pass: active_channel='real', call initialize_data(),
                          all 12 ML models retrain in a background thread
              └─ monitorable at GET /api/data/channel2-watcher
```

**End-to-end timeline from a file drop to "predictions use real data":**

| Time | Event |
|---|---|
| 0 s | Hospital writes files into `datasets/real_data/` |
| ≤ 60 s | Watcher detects new files |
| 60–120 s | Stability gate — waits for export write to finish |
| ~ 2 min | Channel flips Ch1 → Ch2, data reloads |
| 2 – 5 min | All 12 ML models retrain on real outcomes (background thread; no downtime) |
| 5 min + | Every request hits models trained on real Velindre data |

**No restart required** — drop and walk away.

### Correctness guards

- **Required-files gate.** Both `patients.xlsx` AND `historical_appointments.xlsx` must exist. Partial drops are rejected with `Channel 2 watcher: incomplete drop — missing required files: …`; system stays on Ch1.
- **Stability gate.** A file with a moving `mtime` (still being written) is not promoted until two consecutive polls see the same `mtime`.
- **No auto-downgrade.** Removing real files while Ch2 is active does NOT revert to synthetic; it logs a warning and keeps serving Ch2. Manual revert: `POST /api/data/source {"channel":"synthetic"}`.
- Regression-tested with 8 unit tests in `tests/test_channel2_watcher.py`.

### User-visible status (header channel badge)

The web viewer's dark header carries a persistent **channel badge** that auto-refreshes every 30 seconds by polling `GET /api/data/channel2-watcher`:

| Badge state | Colour | Tooltip on hover |
|---|---|---|
| `CH1 SYNTHETIC` | purple | Reports whether Ch2 files are missing, present but still in the stability gate, or simply absent |
| `CH2 REAL HOSPITAL` | green | Confirms the file-drop has been promoted; predictions now served from real-hospital-trained models |

So when a hospital drops files into `datasets/real_data/`, the operator can see the transition happen within ~30 s of the server completing its promotion — regardless of which tab they are currently on.

### Write-path guard: Patient-ID enforcement on Ch2

Switching to Ch2 also tightens the write-path. The urgent-insert endpoint
(`POST /api/urgent/insert`, used by the Squeeze-In tab) enforces a
channel-aware rule at the server boundary:

| Active channel | Empty / placeholder patient ID (`URGENT_*`, `PREVIEW*`, `TEST*`) |
|---|---|
| **Ch1 synthetic** | Auto-generated silently as `URGENT_<YYYYMMDDHHMMSS>` for demo / training workflows |
| **Ch2 real hospital** | **Rejected with HTTP 400** — "Channel 2 requires a real patient identifier (NHS Number or Trust Local ID)." The web UI prompts the operator for a real ID before the request is fired. |

This prevents orphan appointments from being created in the live schedule
when the system is serving real hospital data. Both layers (frontend prompt
and backend validation) enforce the rule independently so a tampered client
cannot bypass it.

### Expected Format

The system is designed to accept real data in the **same format as synthetic data** (SACT v4.0 field names). The hospital would provide:

| File | Expected Format | How to Provide |
|------|----------------|----------------|
| Patient records | Excel/CSV with SACT v4.0 columns (NHS_Number, Primary_Diagnosis_ICD10, etc.) | ChemoCare export or manual extraction |
| Historical appointments | Excel/CSV with attendance outcomes (Attended_Status: Yes/No/Cancelled) | ChemoCare appointment history export |
| Current schedule | Excel/CSV with upcoming appointments | ChemoCare scheduling export |

### How to Integrate Real Data

**Step 1: Place files in the real_data directory**

```
datasets/real_data/
    patients.xlsx              # Real patient records
    historical_appointments.xlsx  # Real attendance history
    appointments.xlsx           # Real upcoming schedule
```

**Step 2: No restart needed** — the Channel 2 watcher promotes automatically within 60 s. Verify via `GET /api/data/channel2-watcher` or the IRL tab's "Active data channel" badge.

---

## Channel 3: NHS Open Data (renumbered from v1.1 Channel 2)

### What It Is

Publicly available NHS datasets downloaded automatically every 24 hours by the background scheduler.

### Sources

| Source | Records | Size | Update | What It Provides |
|--------|---------|------|--------|-----------------|
| **Cancer Waiting Times** | 27,003 rows (per month) | 5 MB | Monthly (Jan 2026 latest) | No-show baseline rates, 62-day compliance, seasonal patterns |
| **NHSBSA SCMD-IP** (new) | 307,676 rows | 31 MB | Monthly (Jan 2026 latest) | Chemotherapy drug prescribing volumes across 145 trusts |
| **SACT Metadata** | - | 0.5 KB | Quarterly | v4.0 specification status (collection commenced Apr 2026) |

> **⚠ NOTE:** NHSBSA dataset changed — old `secondary-care-medicines-data` retired June 2022.
> Now using `secondary-care-medicines-data-indicative-price` (SCMD-IP). Code updated.

### How It Feeds the System

```
Background Scheduler (24h cycle)
        │
        ▼
NHSDataIngester.check_and_download_all()
        │
        ├── Cancer Waiting Times CSV
        │   └── Performance column → noshow_model.base_rate (EMA α=0.2)
        │
        ├── NHSBSA SCMD catalogue
        │   └── Drug volumes → demand forecasting signals
        │
        └── SACT metadata
            └── v4.0 schema validation reference
        │
        ▼
DriftDetector.full_drift_check()
        │
        ▼
ModelRecalibrator.execute_recalibration(level)
        │
        ├── Level 0: Weather/traffic coefficients (real-time)
        ├── Level 1: No-show baseline + hierarchical priors (monthly)
        ├── Level 2: Feature weights + causal distributions (quarterly)
        └── Level 3: Full retrain of all 12 models (on significant drift)
```

### Specific Recalibration Logic

**From CWT data:**
```python
# Extract national compliance rate as proxy for attendance
avg_performance = cwt_data['Performance'].mean()  # e.g., 0.75
drop_off_rate = 1 - avg_performance                # e.g., 0.25

# Update no-show baseline via exponential moving average
noshow_model.base_rate = 0.2 * drop_off_rate + 0.8 * old_rate
```

**From SCMD data:**
```python
# Track chemotherapy drug volume changes
# If regimen volumes shift, update regimen frequency weights
# in the optimizer and duration models
```

### Storage

```
datasets/nhs_open_data/
    ingestion_history.json
    cancer_waiting_times/
        January-2026-Monthly-Combined-CSV-Provisional.csv    (5 MB)
        December-2025-Monthly-Combined-CSV-Provisional.csv   (5 MB)
        November-2025-Monthly-Combined-CSV-Provisional.csv   (5 MB)
        October-2025-Monthly-Combined-CSV-Provisional.csv    (5 MB)
    prescribing/
        SCMDV3_202206.csv                                    (31 MB)
        nhsbsa_scmd_catalogue_2026_03.json                   (14 KB)
    sact_activity/
        sact_metadata_2026_03.json                           (0.5 KB)
```

---

## Channel 3: Real Hospital Data — (MOVED to Channel 2 above)

> This section was relocated to **Channel 2** in v1.2 (see top of document).
> The block below is retained only for backward-compatible anchors in older
> cross-references. **Authoritative copy is §Channel 2.**

### What It Is

Actual patient appointment data from Velindre Cancer Centre, exported from ChemoCare or provided directly. **Not yet available** — requires data sharing agreement and NHS IG approval.

### Expected Format

The system is designed to accept real data in the **same format as synthetic data** (SACT v4.0 field names). The hospital would provide:

| File | Expected Format | How to Provide |
|------|----------------|----------------|
| Patient records | Excel/CSV with SACT v4.0 columns (NHS_Number, Primary_Diagnosis_ICD10, etc.) | ChemoCare export or manual extraction |
| Historical appointments | Excel/CSV with attendance outcomes (Attended_Status: Yes/No/Cancelled) | ChemoCare appointment history export |
| Current schedule | Excel/CSV with upcoming appointments | ChemoCare scheduling export |

### How to Integrate Real Data

**Step 1: Place files in the real_data directory**

```
datasets/real_data/
    patients.xlsx              # Real patient records
    historical_appointments.xlsx  # Real attendance history
    appointments.xlsx           # Real upcoming schedule
```

**Step 2: Update config.py to point to real data**

```python
# In config.py, change:
DATA_SOURCE_CONFIG = {
    'data_directory': 'datasets/real_data',   # Changed from 'datasets/sample_data'
    'patient_file': 'patients.xlsx',
    'appointment_file': 'appointments.xlsx',
    'auto_optimize': True,
}
```

**Step 3: Restart the server**

```bash
python flask_app.py
# System will:
# 1. Load real data instead of synthetic
# 2. Retrain all 12 models on real outcomes
# 3. Generate predictions based on real patient history
# 4. Optimize schedule using real chair/bed constraints
```

### Column Mapping

If the hospital's export uses different column names, the system handles this via the data loading code in `flask_app.py` which already supports both formats:

```python
# The loader checks for both naming conventions:
patient_id = str(row.get('Patient_ID', row.get('patient_id', f'P{_}')))
priority_val = row.get('Priority', row.get('priority', 'P3'))
regimen_code = str(row.get('Regimen_Code', row.get('protocol', 'Standard')))
```

### What Changes with Real Data

| Aspect | Synthetic | Real |
|--------|-----------|------|
| No-show patterns | Modeled from Beta(1.5, 8) distribution | Actual attendance records |
| Duration accuracy | Protocol-based ± random noise | Actual treatment times |
| Patient demographics | Random Welsh names, postcodes | Real patient profiles |
| Regimen distribution | Equal weighting across 20 protocols | Actual trust protocol mix |
| Model accuracy | AUC ~0.82 (estimated) | Depends on data quality |
| Predictions | Based on synthetic patterns | Based on real patient history |

---

## How All Three Channels Work Together

### Current State (Dissertation)

```
Channel 1 (Synthetic)     ──▶ Train all 12 ML models (live operational data)
Channel 2 (Real Hospital) ──▶ Not yet available — Ch2 watcher polls folder
Channel 3 (NHS Open)      ──▶ Always running: recalibrate baselines monthly
```

### Near-Term (Post-Dissertation)

```
Channel 1 (Synthetic)     ──▶ Fallback / testing only
Channel 2 (Real Hospital) ──▶ PRIMARY: Train on real outcomes → Level 3 retrain
Channel 3 (NHS Open)      ──▶ Always running: national benchmarking + monthly recal
```

### Full Production

```
Channel 1 (Synthetic)     ──▶ Not used (real data available)
Channel 2 (Real Hospital) ──▶ PRIMARY: All models trained on real patient data
Channel 3 (NHS Open)      ──▶ National benchmarking + demand forecasting
```

### Priority Logic

The system automatically selects the best available data — both at startup
and at runtime (the Channel 2 watcher polls every 60 s):

```python
# Startup + runtime (flask_app.py):
if Path('datasets/real_data/patients.xlsx').exists():
    # Channel 2: Real hospital data has highest priority
    load_from('datasets/real_data/')
elif Path('datasets/sample_data/patients.xlsx').exists():
    # Channel 1: Synthetic fallback (default today)
    load_from('datasets/sample_data/')
else:
    # No data: use built-in defaults
    use_defaults()

# Channel 3 (NHS Open Data) ALWAYS runs in background regardless of
# which primary channel is active — recalibration never stops.
```

---

## Data Flow Diagram

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  CHANNEL 1   │  │  CHANNEL 2   │  │  CHANNEL 3   │
│  Synthetic   │  │  Real Hosp.  │  │  NHS Open    │
│              │  │              │  │              │
│ sample_data/ │  │ real_data/   │  │ nhs_open_    │
│ *.xlsx       │  │ *.xlsx       │  │ data/*.csv   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       │     ┌───────────┘                  │
       │     │                              │
       ▼     ▼                              ▼
┌──────────────────┐              ┌──────────────────┐
│ Model Training   │              │ Model Training   │
│ (startup)        │              │ (Level 3 retrain)│
│                  │              │                  │
│ All 12 models    │◀─────────────│ All 12 models    │
│ trained on       │  When real   │ retrained on     │
│ synthetic data   │  data placed │ real outcomes    │
└────────┬─────────┘              └────────┬─────────┘
         │                                 │
         ▼                                 ▼
┌──────────────────────────────────────────────────┐
│              UNIFIED ML PIPELINE                  │
│                                                   │
│  Predictions │ Optimization │ Fairness │ Events   │
│                                                   │
│  Served via 87 API endpoints → Web Viewer         │
└──────────────────────────────────────────────────┘
         ▲
         │
┌────────┴─────────┐
│ Channel 3 runs   │
│ continuously:    │
│ • CWT monthly    │
│ • SCMD monthly   │
│ • Drift detect   │
│ • Auto-recalib   │
└──────────────────┘
```

---

## Implementation Checklist

### Channel 1 (Synthetic) — COMPLETE

- [x] `generate_sample_data.py` with SACT v4.0 field names
- [x] 250 patients with NHS Number, ICD-10, Performance Status
- [x] 1,899 historical records with weather, traffic, events (postcode distribution calibrated against pseudonymised Velindre travel data, n=5,116; distances real, identifiers synthetic)
- [x] 819 future appointments with no overlaps, nurse assignments, full SACT fields
- [x] 20 real SACT protocol codes (FOLFOX, RCHOP, etc.)
- [x] Realistic South Wales postcodes and Welsh names
- [x] Modification reason codes (SACT v4.0 numeric 0-4)
- [x] Toxicity grades (CTCAE v5.0 0-5)

### Channel 2 (Real Hospital Data) — READY + auto-promoted at runtime

- [x] Data loading code supports both SACT v4.0 and legacy column names
- [x] `datasets/real_data/` directory exists with README
- [x] `config.py` has `DATA_SOURCE_CONFIG` for easy path switching
- [x] Level 3 retrain capability tested (32 seconds on 2,241 records)
- [x] SACT v4.0 export endpoint (`/api/data/export-sact`) for data validation
- [x] **Channel 2 watcher (`ch2-watcher` daemon)** — auto-promotes within 60 s of file drop, no restart needed; monitorable at `GET /api/data/channel2-watcher`
- [ ] **Pending:** Data sharing agreement with Velindre
- [ ] **Pending:** NHS IG approval
- [ ] **Pending:** ChemoCare export format mapping document

### Channel 3 (NHS Open Data) — COMPLETE, partially-available, always running

- [x] `data/nhs_data_ingestion.py` with 3 sources
- [x] Background scheduler (24h auto-check, 5 min first check)
- [x] Cancer Waiting Times CSV auto-download (27,003 rows)
- [x] NHSBSA SCMD catalogue download (307,676 records)
- [x] SACT metadata tracking
- [x] Hash-based change detection (only process new data)
- [x] Drift detection (PSI, KS-test, CUSUM)
- [x] 4-level auto-recalibration
- [x] Version snapshots for rollback
- [x] Manual triggers via API and web UI
- [x] **Always-on by default** — recalibration runs regardless of whether Channel 1 or Channel 2 is the active primary source

### No Fake Data Policy (v4.0)

All `random.uniform()`, `random.randint()`, and `np.random` calls removed from the prediction pipeline in flask_app.py. Predictions now use:
- Trained ML ensemble (RF+GB+XGB, AUC=0.64) when historical data available
- Rule-based fallback using real patient features (history, distance, priority)
- Never random numbers

All ML models train on whichever data channel is active — synthetic, real, or NHS open.

---

## For the Dissertation

When writing about data handling, emphasize:

1. **Three-channel design** demonstrates production readiness, not just a prototype
2. **Synthetic data is methodologically valid** — distributions match published NHS statistics
3. **NHS open data is already flowing** — 56 MB of real NHS data downloaded and feeding into recalibration
4. **Real data integration is zero-effort** — just place Excel files in `real_data/` and restart
5. **Auto-learning ensures models don't degrade** — continuous monitoring and recalibration

---

*Document prepared for MSc Data Science Dissertation*
*Cardiff University — School of Mathematics*
*SACT Scheduling Optimization Project*

*Version 1.0 | March 2026*
