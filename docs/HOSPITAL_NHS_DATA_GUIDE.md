# NHS Open Data — Reference Guide for Velindre Cancer Centre

**Document type:** Information for clinical/IT sponsor
**System:** SACT Intelligent Scheduling Optimisation System
**Prepared by:** MSc Data Science Dissertation, Cardiff University
**Date:** April 2026
**Intended audience:** Clinical informatics, IT department, data governance lead

---

## Purpose of This Document

This document explains which NHS open data sources are used by the SACT scheduling system, what each source contains, where to access it, and what actions — if any — your team needs to take. All sources listed in Section 1 are publicly available, free, and require no NHS login or data-sharing agreement.

---

## 1. NHS Open Data Sources Used by the System

The system automatically downloads and processes data from five NHS open data sources. Each runs on a background schedule — no manual action is required during normal operation.

---

### 1.1 Cancer Waiting Times (CWT) Statistics

| | |
|--|--|
| **Publisher** | NHS England |
| **Status** | Active — monthly releases; **final release June 2026** |
| **Website** | https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/ |
| **Monthly data page** | https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/monthly-data-and-summaries/2025-26-monthly-cancer-waiting-times-statistics/ |
| **Format** | Monthly XLSX + combined CSV |
| **Latest available** | January 2026 (released March 2026) |
| **Licence** | Open Government Licence v3.0 — free to use |
| **No registration needed** | Yes — direct download |

**What this data contains:**

- Provider (NHS Trust) identifiers and names
- Number of patients referred and treated within each standard
- 62-day standard: urgent referral to first treatment (national target: 85%)
- 31-day standard: decision to treat to first treatment (national target: 96%)
- Faster Diagnosis Standard: referral to diagnosis within 28 days
- Monthly time series from October 2009 onwards

**How the system uses it:**

The system downloads the monthly combined CSV on the 1st of each month and uses it to:
- Update baseline no-show rate estimates by cancer type and provider
- Calibrate waiting-time targets used in the scheduling optimiser
- Detect seasonal patterns in cancellation and delay rates
- Feed Level 1 model recalibration (parameter updates, < 30 seconds)

**⚠ Important — Decommissioning June 2026:**

NHS England is decommissioning the Cancer Waiting Times collection system in June 2026. The final monthly data release will cover June 2026. From 1 July 2026 the system automatically switches to the NHS NDRS Cancer Data Hub (see Section 1.5 below). No manual action is needed — this transition is handled automatically.

---

### 1.2 SACT Activity Dashboard

| | |
|--|--|
| **Publisher** | NHS England NDRS (National Disease Registration Service) |
| **Status** | Active — updated annually (confirmed April 2026) |
| **Main page** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/sact-activity |
| **Interactive dashboard** | https://nhsd-ndrs.shinyapps.io/sact_activity/ |
| **Format** | Interactive R Shiny dashboard with CSV download |
| **Coverage** | January 2019 onwards |
| **Granularity** | England / Cancer Alliance / Integrated Care Board |
| **Licence** | Open Government Licence |
| **No registration needed** | Yes |

**What this data contains:**

- Patient counts by tumour group (colorectal, breast, lymphoma, etc.)
- Regimen counts by treatment type
- Drug administration counts by route (IV, oral, subcutaneous)
- Breakdowns by: age group, gender, deprivation decile, ethnicity
- Treatment intent: curative vs non-curative proportions
- Cancer Alliance and ICB level summaries

**How the system uses it:**

Downloaded annually and used to:
- Validate that the synthetic regimen mix reflects national practice
- Update demographic risk factors in the no-show prediction model (age/deprivation effects)
- Calibrate the hierarchical Bayesian model's population-level priors
- Feed Level 2 model recalibration (feature weight updates, < 1 minute)

**How to download manually (if needed):**

1. Open the dashboard at https://nhsd-ndrs.shinyapps.io/sact_activity/
2. Apply any filters required (e.g., Cancer Alliance = South East Wales)
3. Click **"Download data for selected inputs"**
4. Place the downloaded CSV in `datasets/nhs_open_data/sact_activity/`

---

### 1.3 NHSBSA Secondary Care Medicines Data with Indicative Price (SCMD-IP)

| | |
|--|--|
| **Publisher** | NHS Business Services Authority (NHSBSA) |
| **Status** | Active — monthly releases |
| **Dataset page** | https://opendata.nhsbsa.net/dataset/secondary-care-medicines-data-indicative-price |
| **API endpoint** | https://opendata.nhsbsa.net/api/3/action/package_show?id=secondary-care-medicines-data-indicative-price |
| **Format** | JSON API (CKAN) + CSV download |
| **Latest available** | January 2026 (released ~20th March 2026; 2-month lag) |
| **Update schedule** | ~20th of each month |
| **Licence** | Open Government Licence |
| **No registration needed** | Yes — fully open API |

> **Note:** The original NHSBSA "Secondary Care Medicines Data" dataset was retired in June 2022. The replacement is the "Secondary Care Medicines Data **with Indicative Price**" (SCMD-IP). The system uses the new package — please ensure any manual references also use the new dataset name.

**What this data contains:**

- Drug name by BNF (British National Formulary) code
- Quantity dispensed per hospital/organisation
- Item count and cost (Net Ingredient Cost)
- Monthly breakdown per organisation
- Relevant BNF sections for oncology:
  - **0801** — Cytotoxic drugs (chemotherapy)
  - **0802** — Drugs affecting immune response (immunotherapy)
  - **0803** — Supportive drugs in oncology

**How the system uses it:**

Downloaded monthly and used to:
- Detect new chemotherapy regimens appearing in national prescribing data
- Monitor volume trends as demand signals (rising prescribing → increased scheduling pressure)
- Provide context for the event impact model
- Feed Level 1 baseline recalibration

---

### 1.4 SACT v4.0 Patient-Level Data

| | |
|--|--|
| **Publisher** | NHS England NDRS |
| **Status** | ⚠ Collection commenced 1 April 2026 — rollout period, data not yet publicly available |
| **Specification page** | https://digital.nhs.uk/ndrs/data/data-sets/sact |
| **Technical guidance** | https://digital.nhs.uk/ndrs/data/data-sets/sact (v4.0 documentation) |
| **Format** | CSV (submitted by each NHS Trust to NDRS monthly) |
| **Fields** | 60 data items across 7 sections |
| **Rollout period** | April–June 2026 (partial trust submissions) |
| **Full conformance** | 1 July 2026 (all trusts required to submit) |
| **First complete dataset** | **August 2026** (July 2026 data + ~4-week NDRS processing lag) |

**What this data contains (60 fields, 7 sections):**

| Section | Key fields |
|---------|-----------|
| 1 — Linkage | NHS Number, Local Patient Identifier |
| 2 — Demographics | Date of birth, gender, postcode, ODS code |
| 3 — Clinical Status | ICD-10 diagnosis, morphology, performance status, specialty |
| 4 — Regimen | Regimen code, treatment intent, context, height/weight/BSA |
| 5 — Modifications | Regimen/dose modifications with reasons, toxicity grade (CTCAE v5.0) |
| 6 — Drug Details | Drug name, dose, route, cycle length |
| 7 — Outcome | End of regimen summary code |

**How the system uses it:**

This is the primary data source for full model retraining:

| Phase | Dates | What the system does |
|-------|-------|----------------------|
| Rollout (partial) | April–June 2026 | Level 1–2 recalibration from partial submissions |
| Full conformance | July 2026 | Level 2 recalibration |
| **First complete dataset** | **August 2026** | **Level 3 — full retrain of all 12 ML models** |

**Action required from Velindre:**

If Velindre Cancer Centre wishes to use its own SACT v4.0 submission data within the local system (not the national aggregate), your informatics team should:

1. Export the trust's monthly SACT v4.0 CSV from your NDRS submission interface
2. Place the CSV file in the folder: `datasets/nhs_open_data/sact_v4/`
3. The system detects the file within 24 hours and triggers the appropriate recalibration level automatically
4. No other configuration is needed — the system reads the standard v4.0 field names directly

The system will validate the uploaded file against the full SACT v4.0 schema (60 fields, 7 sections) and report a quality grade (A–F) and section-by-section coverage before using it for training.

---

### 1.5 NHS NDRS Cancer Data Hub — CWT Successor Source

| | |
|--|--|
| **Publisher** | NHS England NDRS |
| **Status** | Activates automatically 1 July 2026 (CWT decommissions June 2026) |
| **Cancer Data Hub main page** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub |
| **Faster Diagnosis Standard statistics** | https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/faster-diagnosis-standard/ |
| **Format** | Mixed (dashboards, CSV/XLSX downloads, Shiny apps) |
| **Licence** | Open Government Licence |
| **No registration needed** | Yes |

**Background:**

The NHS England Cancer Waiting Times (CWT) system publishes its final monthly dataset for June 2026 and is then decommissioned. From July 2026, waiting-time and performance data is available through:

1. **NHS NDRS Cancer Data Hub** — aggregate cancer outputs across multiple dashboards
2. **Faster Diagnosis Standard (FDS) statistics** — the replacement metric for the 62-day standard; measures whether patients receive a diagnosis within 28 days of referral (target: 75% by 2024, increasing to 80%)

**What the system does from 1 July 2026:**

- Automatically disables the CWT source (no further downloads attempted)
- Activates the `ndrs_cancer_data_hub` source on a monthly schedule
- Polls both the Cancer Data Hub page and the FDS statistics page for downloadable CSV/XLSX files
- If direct download links are found, downloads the most recent file automatically
- If no direct download links are available, saves a manifest with the discovered links for manual review (stored in `datasets/nhs_open_data/ndrs_cancer_data_hub/`)
- The downloaded data feeds Level 1 recalibration (same role as CWT)

**FDS data available on the statistics page:**

- Monthly time series of FDS compliance by provider (NHS Trust)
- 28-day diagnosis target attainment rates
- Breakdown by cancer type (colorectal, lung, breast, etc.)
- Provider extract files (XLSX)
- Direct download links for combined monthly data

**Additional Cancer Data Hub outputs (available now, all free):**

| Dashboard | URL | Update |
|-----------|-----|--------|
| Cancer Registration Statistics | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-registration-statistics | Annual |
| Rapid Cancer Registration Data | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/rapid-cancer-registration-data-dashboards | Monthly |
| Cancer Prevalence | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-prevalence | Annual |
| Cancer Treatments | https://nhsd-ndrs.shinyapps.io/cancer_treatments/ | Annual |
| Get Data Out Programme | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/get-data-out | Periodic |

---

## 2. Real-Time External Data (No NHS Data)

In addition to NHS open data, the system uses three external real-time feeds for event monitoring. These are non-NHS sources and require no NHS data-sharing agreement.

| Source | URL | Update | Purpose |
|--------|-----|--------|---------|
| Open-Meteo Weather | https://api.open-meteo.com | Hourly | Predict weather-related no-shows |
| TomTom Traffic API | https://developer.tomtom.com | Every 5 min | Predict travel delays |
| BBC Wales RSS | https://feeds.bbci.co.uk/news/wales/rss.xml | Every 15 min | Detect local events |
| Wales Online RSS | https://www.walesonline.co.uk/news/rss.xml | Every 15 min | Detect local events |
| Gov.uk Emergency Alerts | https://www.gov.uk/search/news-and-communications.atom | Real-time | Emergency alerts |

All are free. TomTom provides 2,500 free API calls per day (sufficient for production use at a single hospital).

---

## 3. Data Your Team Needs to Provide

The system can operate in three modes. Mode selection is automatic based on what data is available.

| Mode | Data required | Who provides it | ML quality |
|------|--------------|----------------|------------|
| **Synthetic** (default) | None — generated automatically | — | Calibrated to national patterns |
| **Real hospital data** | patients.xlsx + historical_appointments.xlsx | **Your informatics team** | Best — trained on your actual patients |
| **NHS open data** | Auto-downloaded (Sections 1.1–1.5 above) | — | National-level calibration |

### What to provide for real hospital data mode

To switch the system from synthetic data to your hospital's real data, your informatics team needs to export and format two files:

**File 1: `patients.xlsx`**
A list of current patients in the SACT programme. Required fields:

| Field | Source in your system | Format |
|-------|-----------------------|--------|
| NHS_Number | Patient demographics | 10-digit string |
| Local_Patient_Identifier | Trust patient ID / ChemoCare ID | String |
| Person_Birth_Date | Demographics | YYYY-MM-DD |
| Person_Stated_Gender_Code | Demographics | 1=Male, 2=Female, 9=Not stated |
| Patient_Postcode | Demographics | e.g. CF14 2TL |
| Primary_Diagnosis_ICD10 | Diagnosis record | e.g. C18.9 |
| Performance_Status | Clinical record | 0–4 (WHO) |
| Regimen_Code | ChemoCare / treatment plan | e.g. FOLFOX |
| Intent_Of_Treatment | Treatment plan | 06=Curative, 07=Non-curative |
| Cycle_Number | Current cycle | Integer |
| Priority | Clinical assessment | P1–P4 |
| Height_At_Start | Clinical record | Metres (e.g. 1.72) |
| Weight_At_Start | Clinical record | Kilograms (e.g. 78.5) |

**File 2: `historical_appointments.xlsx`**
The appointment history used to train the machine learning models. This is the most important file — at least 500 records are needed; 1,000+ records are recommended for reliable predictions.

| Field | Source in your system | Format | Notes |
|-------|-----------------------|--------|-------|
| Patient_ID | ChemoCare / trust ID | String | Matches Local_Patient_Identifier |
| Date | Appointment record | YYYY-MM-DD | |
| Time | Appointment record | HH:MM | Scheduled start time |
| **Attended_Status** | Appointment outcome | Yes / No / Cancelled | **Most important field — required for ML** |
| Actual_Duration | Appointment record | Integer (minutes) | For attended appointments |
| Planned_Duration | Appointment record | Integer (minutes) | |
| Regimen_Code | Treatment record | String | e.g. FOLFOX |
| Site_Code | Location | WC / PCH / RGH / POW / CWM | |
| Cycle_Number | Treatment record | Integer | |

**How to export from ChemoCare:**
Ask your ChemoCare system administrator for a "treatment history extract" covering the past 2–3 years. The `Attended_Status` field is the most critical — this tells the system which patients attended, did not attend, or cancelled, which is what the no-show prediction model learns from.

**Where to place the files:**
Copy both files to the folder `datasets/real_data/` on the server running the scheduling system. The system detects the files automatically on the next startup and retrains all 12 machine learning models on your data. No other configuration is needed.

**Data security:**
- Both files are excluded from version control (`.gitignore`) — they will not be committed to any repository
- The system does not transmit patient data to any external service
- All processing is local to the server
- Apply your trust's standard data access controls to the `datasets/real_data/` folder

---

## 4. Data Access for Restricted NHS Sources (Optional, For Future Reference)

Two additional NHS datasets contain richer patient-level data but require a formal application. These are not required to run the system — they would enhance ML model quality if obtained.

### 4.1 Historical Patient-Level SACT Data (pre-v4.0)

| | |
|--|--|
| **Source** | NHS England NDRS |
| **Access route** | Data Access Request Service (DARS) |
| **DARS portal** | https://digital.nhs.uk/services/data-access-request-service-dars |
| **Enquiries** | ndrsenquiries@nhs.net |
| **Coverage** | April 2012 onwards (SACT v3.0 data) |
| **Typical timeline** | 3–6 months from application to data receipt |

This dataset contains patient-level SACT treatment records from all England NHS Trusts. A successful DARS application would provide the highest-quality training data for the ML models and would supplement or replace the synthetic data entirely.

**Requirements for DARS application:**
- Legitimate research/quality improvement purpose (clearly applicable here)
- Data Sharing Agreement
- Information Governance compliance (DSPT Level 2+)
- Named Data Access Agreement holder at your trust

### 4.2 NHS Cancer Data Hub — NHS Login Required

Some Cancer Data Hub outputs require an NHS OpenID Connect login:

| Dashboard | URL | What it contains |
|-----------|-----|-----------------|
| Cancer Services Profiles | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-services-profiles | Provider-level service performance indicators |
| Emergency Presentations | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-emergency-presentations | Emergency presentation rates by trust |

These are not currently used by the scheduling system but may be relevant for service planning.

---

## 5. Complete URL Reference

### Free, Open, No Login Required

| Source | Primary URL |
|--------|-------------|
| Cancer Waiting Times (monthly data) | https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/monthly-data-and-summaries/2025-26-monthly-cancer-waiting-times-statistics/ |
| Cancer Waiting Times (main page) | https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/ |
| SACT Activity Dashboard | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/sact-activity |
| SACT Activity (interactive) | https://nhsd-ndrs.shinyapps.io/sact_activity/ |
| SACT v4.0 Dataset Specification | https://digital.nhs.uk/ndrs/data/data-sets/sact |
| NHSBSA SCMD-IP (dataset page) | https://opendata.nhsbsa.net/dataset/secondary-care-medicines-data-indicative-price |
| NHSBSA SCMD-IP (API — JSON) | https://opendata.nhsbsa.net/api/3/action/package_show?id=secondary-care-medicines-data-indicative-price |
| NHS NDRS Cancer Data Hub | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub |
| Faster Diagnosis Standard statistics | https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/faster-diagnosis-standard/ |
| Cancer Registration Statistics | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-registration-statistics |
| Rapid Cancer Registration Data | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/rapid-cancer-registration-data-dashboards |
| Cancer Treatments Dashboard | https://nhsd-ndrs.shinyapps.io/cancer_treatments/ |
| Cancer Prevalence | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-prevalence |
| Get Data Out Programme | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/get-data-out |

### Restricted Access (Application Required)

| Source | URL |
|--------|-----|
| DARS Application Portal | https://digital.nhs.uk/services/data-access-request-service-dars |
| NDRS Enquiries | ndrsenquiries@nhs.net |

---

## 6. Timeline Summary

| Date | Event | System action |
|------|-------|--------------|
| 1 April 2026 | SACT v4.0 collection commenced | System enters rollout-phase monitoring |
| April–June 2026 | SACT v4.0 rollout period | Level 1–2 recalibration if local CSV provided |
| 1 July 2026 | Full SACT v4.0 conformance required | Level 2 recalibration available |
| **1 July 2026** | **CWT system decommissions** | **Auto-switch to NDRS Cancer Data Hub source** |
| **August 2026** | **First complete SACT v4.0 dataset** | **Level 3 — full retrain of all 12 ML models** |
| Ongoing (monthly) | NHSBSA SCMD-IP updated ~20th | Level 1 baseline recalibration |
| Ongoing (annually) | SACT Activity Dashboard updated | Level 2 feature weight update |

---

## 7. System Data Status Check

You can check the current status of all data sources at any time via the web interface or API:

**Web interface:** Open the system at http://localhost:1421 → ML Models tab → NHS Data Pipeline panel

**API call:**
```
GET http://localhost:1421/api/data/nhs/status
```

This returns the last download date, number of files on disk, and whether each source is due for an update. The NDRS Cancer Data Hub entry also shows a countdown to the CWT decommission date and whether the successor source is active.

**To manually trigger a data check:**
```
POST http://localhost:1421/api/data/nhs/check-updates
```

Or from the web interface: ML Models tab → NHS Data Pipeline → "Check for Updates" button.

---

## 8. Summary of Actions Required

| Action | Who | When | Priority |
|--------|-----|------|----------|
| No action needed for CWT → NDRS CDH transition | — | Automatic July 2026 | — |
| Export historical appointment data from ChemoCare | IT / Informatics | When ready | High — improves ML quality |
| Place SACT v4.0 monthly CSV in `sact_v4/` folder | Informatics | August 2026 | High — triggers real-data ML retrain |
| Consider DARS application for historical SACT data | Data Governance Lead | Optional | Medium — further ML improvement |
| Provide TomTom API key for traffic monitoring | IT | Optional | Low — enables traffic-aware scheduling |

---

*Prepared for Velindre Cancer Centre NHS Trust*
*MSc Data Science Dissertation — Cardiff University, School of Mathematics*
*Supervisor: [Supervisor name]*
*April 2026*

*For technical queries about this document or the scheduling system, contact the dissertation author.*
