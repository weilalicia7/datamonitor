# NHS Open Data Inventory for SACT Scheduling System

## What's Available Now vs What Requires Waiting

**Date:** April 2026
**Purpose:** Definitive list of all NHS open data sources relevant to SACT scheduling, categorized by availability.

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA AVAILABILITY TIMELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  AVAILABLE NOW (April 2026)                8 datasets, free, downloadable   │
│  ════════════════════════════                                                │
│  ├── Cancer Waiting Times CSV             Monthly (latest: Jan 2026)       │
│  ├── SACT Activity Dashboard              Quarterly, aggregate             │
│  ├── Cancer Treatments Dashboard          Annual, downloadable             │
│  ├── Cancer Registration Statistics       Annual, incidence/mortality      │
│  ├── Get Data Out Programme               Per cancer type, downloadable    │
│  ├── NHSBSA SCMD w/ Indicative Price      Monthly (latest: Jan 2026)       │
│  ├── Rapid Cancer Registration Data       Monthly, early estimates         │
│  └── Cancer Prevalence Dashboard          Annual, downloadable             │
│                                                                              │
│  COLLECTION COMMENCED — DATA NOT YET AVAILABLE EXTERNALLY                   │
│  ════════════════════════════════════════════════════════                    │
│  └── SACT v4.0 Patient-Level Data         Collection started 1 April 2026  │
│      ⚠ Rollout period: April–June 2026 (partial trust submissions only)    │
│      ⚠ Full conformance: 1 July 2026                                       │
│      ⚠ First COMPLETE dataset expected: August 2026                        │
│      ⚠ NOT available to external researchers until post-June 2026          │
│                                                                              │
│  RESTRICTED (Requires DARS Application)    2 datasets, formal process      │
│  ═══════════════════════════════════════                                     │
│  ├── Patient-Level SACT (historical)      Via Data Access Request Service  │
│  └── Cancer Consolidated Dataset          Research applications only       │
│                                                                              │
│  AVAILABLE BUT NHS-INTERNAL ONLY           2 datasets                      │
│  ════════════════════════════════                                            │
│  ├── Cancer Services Profiles             NHS login required               │
│  └── Emergency Presentations Data         NHS login required               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PART 1: AVAILABLE NOW (Free, Open, No Registration)

### 1.1 Cancer Waiting Times Statistics

| Property | Detail |
|----------|--------|
| **Status** | AVAILABLE NOW |
| **URL** | https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/ |
| **Format** | CSV + Excel (.xlsx) |
| **Update** | Monthly (latest: January 2026) |
| **Granularity** | Provider-level, by cancer type, by standard |
| **Coverage** | April 2022 onwards (time series from Oct 2009) |
| **Size** | 4-60 MB per monthly combined CSV |
| **Access** | Direct download, no registration |
| **Licence** | Open Government Licence |

**Data fields include:**
- Provider (NHS Trust) identifier and name
- Cancer type classification
- 31-day standard: Decision to first treatment (target: 96%)
- 62-day standard: Urgent referral to first treatment (target: 85%)
- Faster Diagnosis Standard compliance
- Patient counts (referred, treated within/beyond target)
- Monthly time series

**Value for our system:**
- Baseline no-show/cancellation rates by provider
- Seasonal patterns in waiting times
- Performance benchmarks (68.4% vs 85% target)
- Trend detection for demand forecasting

**⚠ CRITICAL NOTE (April 2026):**
- **CWT system is being decommissioned in June 2026.** Final monthly release will be June 2026.
- From November 2025, the CRS Provider and CRS Commissioner files were consolidated into a single **Monthly Time Series** file.
- Latest combined CSV covers Apr–Sep 2025 only (25.4 MB). Monthly XLSX extends to Jan 2026.
- Provider extract (Oct 2025–Jan 2026): `CWT-CRS-Oct-2025-to-Jan-2026-Data-Extract-Provider.xlsx`

**How to download:**
```
Monthly data page → Select month → Download "Monthly Time Series with Revisions" (XLSX)
Or combined CSV: 2025-26-Apr-Sep-Monthly-Combined-CSV-Provisional.csv (25.4 MB, up to Sep 2025)
URL pattern: england.nhs.uk/statistics/wp-content/uploads/sites/2/{year}/{month}/...
```

---

### 1.2 SACT Activity Dashboard

| Property | Detail |
|----------|--------|
| **Status** | AVAILABLE NOW |
| **URL** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/sact-activity |
| **Dashboard** | Interactive (with download buttons) |
| **Update** | Annually (previously quarterly; confirmed April 2026) |
| **Granularity** | England / Cancer Alliance / ICB level |
| **Coverage** | January 2019 onwards |
| **Access** | Free, no registration, downloadable via dashboard |

**Data fields include:**
- Patient counts by tumour group
- Regimen counts by type
- Drug administration counts
- Breakdowns by: age group, gender, deprivation, ethnicity
- Treatment intent (curative vs non-curative)
- Administration route (IV, oral, subcutaneous)

**Value for our system:**
- National regimen distribution (validate our synthetic mix)
- Demographic patterns (age/gender effects)
- Treatment intent ratios (curative vs palliative)
- Quarterly trend tracking

**How to download:**
```
Open dashboard → Apply filters → Click "Download data for selected inputs"
OR: Go to Downloads tab → "Download all data"
```

---

### 1.3 Cancer Treatments Dashboard

| Property | Detail |
|----------|--------|
| **Status** | AVAILABLE NOW |
| **URL** | https://nhsd-ndrs.shinyapps.io/cancer_treatments/ |
| **Format** | R Shiny dashboard with download capability |
| **Update** | Annually (last: May 2025, next: Spring 2026) |
| **Granularity** | Cancer type, stage, age, gender, deprivation, ethnicity, comorbidity, Cancer Alliance |
| **Coverage** | 2013-2022 |
| **Access** | Free, no registration |

**Data fields include:**
- Proportion of tumours receiving SACT vs radiotherapy vs surgery
- Stage at diagnosis breakdown
- Comorbidity count impact on treatment
- Cancer Alliance level analysis
- Deprivation quintile analysis

**Value for our system:**
- Comorbidity impact coefficients (duration model calibration)
- Stage-specific treatment patterns
- Regional variation in treatment rates
- Baseline rates for hierarchical model priors

---

### 1.4 Cancer Registration Statistics

| Property | Detail |
|----------|--------|
| **Status** | AVAILABLE NOW |
| **URL** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-registration-statistics |
| **Format** | Dashboard with download |
| **Update** | Annually (next: early 2026 with 2023 data) |
| **Granularity** | Cancer site, age, gender, geography |
| **Coverage** | 1995-2022 |
| **Access** | Free, downloadable |

**Data fields include:**
- Cancer incidence (new cases) by site
- Cancer mortality (deaths) by site
- Age-standardised rates
- Geographic breakdown
- Trends over time

**Value for our system:**
- Demand forecasting (incidence trends predict future treatment volumes)
- Cancer-type specific scheduling patterns

---

### 1.5 Get Data Out Programme

| Property | Detail |
|----------|--------|
| **Status** | AVAILABLE NOW |
| **URL** | https://www.cancerdata.nhs.uk/getdataout/data |
| **Moving to** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/get-data-out |
| **Format** | CSV download per cancer type |
| **Update** | Periodic |
| **Granularity** | Per tumour group, ~100 patient cohorts |
| **Access** | Free, Open Government Licence |

**Data fields include:**
- Incidence statistics per cancer type
- Routes to diagnosis
- Treatment patterns (SACT, surgery, radiotherapy)
- Survival statistics
- Demographic breakdowns

**Cancer types available:**
Bladder, Bone, Brain, Breast, Cervical, Colorectal, Eye, Head & Neck, Kidney, Leukaemia, Liver, Lung, Lymphoma, Melanoma, Mesothelioma, Myeloma, Oesophageal, Ovarian, Pancreatic, Prostate, Sarcoma, Stomach, Testicular, Thyroid, Uterine

**Value for our system:**
- Cancer-type specific treatment rates (validate regimen distribution)
- Routes to diagnosis (affects booking lead times)
- Treatment pattern validation

---

### 1.6 NHSBSA Secondary Care Medicines Data with Indicative Price (SCMD-IP)

| Property | Detail |
|----------|--------|
| **Status** | AVAILABLE NOW |
| **URL** | https://opendata.nhsbsa.net/dataset/secondary-care-medicines-data-indicative-price |
| **API endpoint** | `https://opendata.nhsbsa.net/api/3/action/package_show?id=secondary-care-medicines-data-indicative-price` |
| **Format** | JSON API (CKAN) + CSV download |
| **Update** | Monthly (~20th, 2-month lag; latest: January 2026) |
| **Granularity** | Hospital/organisation level, per drug item |
| **Access** | Free, no registration, programmatic API |
| **Note** | ⚠ Old dataset `secondary-care-medicines-data` retired June 2022. Always use the new `secondary-care-medicines-data-indicative-price` package. |

**Data fields include:**
- Drug name (BNF code)
- Quantity dispensed
- Item count
- Cost (NIC)
- Practice/organisation identifier
- Monthly breakdown

**Relevant BNF sections:**
- 0801: Cytotoxic drugs
- 0802: Drugs affecting immune response
- 0803: Supportive drugs in oncology

**Value for our system:**
- Drug prescribing trends (new regimens appearing)
- Volume changes (demand signals)
- Cost data for economic analysis

**Example API query:**
```sql
SELECT "YEAR_MONTH", "BNF_CHEMICAL_SUBSTANCE", "CHEMICAL_SUBSTANCE_BNF_DESCR",
       "TOTAL_QUANTITY", "ITEM_COUNT"
FROM "EPD"
WHERE "BNF_CHEMICAL_SUBSTANCE" LIKE '0801%'
ORDER BY "YEAR_MONTH" DESC
LIMIT 500
```

---

### 1.7 Rapid Cancer Registration Data (RCRD)

| Property | Detail |
|----------|--------|
| **Status** | AVAILABLE NOW |
| **URL** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/rapid-cancer-registration-data-dashboards |
| **Format** | Dashboard with download |
| **Update** | Monthly (latest: 29 January 2026) |
| **Granularity** | Geography, demographics |
| **Access** | Public |

**Data fields include:**
- Indicative cancer incidence
- Early-stage proportion
- Geographic breakdowns
- Demographic breakdowns

**Value for our system:**
- Most up-to-date incidence data
- Early-stage detection trends (affects treatment complexity)

---

### 1.8 Cancer Prevalence Dashboard

| Property | Detail |
|----------|--------|
| **Status** | AVAILABLE NOW |
| **URL** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-prevalence |
| **Format** | Dashboard with download |
| **Update** | Annual |
| **Access** | Public |

**Data fields include:**
- Numbers of people living with/beyond cancer
- By cancer type, age, region

**Value for our system:**
- Long-term demand planning

---

## PART 2: COLLECTION COMMENCED — NOT YET AVAILABLE EXTERNALLY

> **⚠ IMPORTANT DISSERTATION NOTE:**
> SACT v4.0 data collection commenced 1 April 2026.
> The dataset is NOT available to external researchers until after the rollout period ends.
> **First complete monthly dataset expected: August 2026.**
> This dissertation uses synthetic data that mirrors the v4.0 schema in preparation for
> future real-data integration.

### 2.1 SACT v4.0 Patient-Level Submissions

| Property | Detail |
|----------|--------|
| **Status** | ⚠ COLLECTION IN PROGRESS — ROLLOUT PERIOD (partial data usable) |
| **Collection commenced** | 1 April 2026 |
| **Rollout period** | April–June 2026 (partial submissions — ✓ usable for Level 1-2 recalibration) |
| **Full conformance** | 1 July 2026 (all trusts required to submit — ✓ usable for Level 2) |
| **First complete dataset** | **August 2026** (July 2026 data + processing lag — ✓ triggers Level 3 full retrain) |
| **Auto-detection** | Place CSV in `datasets/nhs_open_data/sact_v4/` — scheduler detects within 24h |
| **Format** | CSV (submitted by each NHS Trust monthly to NDRS) |
| **Fields** | 60 data items across 7 sections |
| **Synthetic alignment** | patients.xlsx 91.4/100 Grade A · historical_appointments.xlsx 100/100 Grade A · appointments.xlsx 100/100 Grade A; 7/7 sections covered |
| **Synthetic dataset** | 1,899 historical records (102 cols), 250 patients (82 cols), 819 appointments (70 cols); travel distribution calibrated against pseudonymised Velindre patient travel data (n=5,116; distances real, identifiers synthetic): Near ~38% / Medium ~60% / Remote ~2.4% |
| **Validate endpoint** | `GET /api/data/sact-v4/validate` — returns score, grade, section coverage |
| **Status endpoint** | `GET /api/data/sact-v4/status` — returns quality_phase, recalibration level (floor: 1) |
| **Granularity** | Patient-level (individual treatment records) |
| **Access** | Trusts submit to NDRS; aggregated data published via NDRS portal |

**What's new in v4.0 (vs v3.0):**
- Expanded to 60 data items
- New modification tracking (regimen, cycle, dose with reasons)
- New toxicity grading (CTCAE v5.0)
- New treatment context field (neoadjuvant/adjuvant/SACT-only)
- New line of treatment tracking (curative/non-curative lines)
- Removal of redundant fields
- Improved clinical accuracy

**7 sections:**
1. Linkage (NHS_Number, Local_ID)
2. Demographics (DOB, Gender, Postcode)
3. Clinical Status (ICD-10, Morphology, SNOMED, Specialty)
4. Regimen (Intent, Context, Regimen, Performance Status, Height/Weight)
5. Modifications (Regimen/Cycle/Dose changes with reasons and toxicity)
6. Drug Details (Drug name, dose, route, timestamp, cycle length)
7. Outcome (End of regimen summary)

**Value for our system:**
- **THIS IS THE GOLD STANDARD** — patient-level treatment data with all clinical details
- Direct retraining of all 12 ML models on real data
- Drug-level duration data for precise scheduling
- Modification/toxicity data for no-show/cancellation prediction
- Outcome data for survival model calibration

**What to prepare NOW:**
- Synthetic data uses SACT v4.0 field names — schema-compatible from day one
- System can ingest SACT v4.0 CSV format directly when data becomes available
- Auto-learning pipeline ready to trigger Level 3 retrain on first complete dataset

**Timeline:**
```
April 2026     SACT v4.0 data collection COMMENCED (1 April 2026)
               Partial trust submissions during rollout
               ✓ USABLE for preliminary Level 1-2 recalibration
               ⚠ Nationally incomplete — results reflect submitting trusts only
               Auto-learning scheduler checks quality phase and routes to Level 1-2

May–June 2026  Rollout continues — increasing trust submissions
               ✓ Progressively more representative — still Level 1-2 recalibration
               ⚠ Not yet suitable for Level 3 full retrain (incomplete national picture)

July 2026      Full conformance required for all trusts
               First full month of compliant submissions
               ✓ Level 2 recalibration appropriate

August 2026    ✓ FIRST COMPLETE DATASET EXPECTED
               (July 2026 data + ~4 week NDRS processing lag)
               ✓ Level 3 full retrain triggered automatically
               Place CSV in datasets/nhs_open_data/sact_v4/ to trigger

September 2026 Validate synthetic model assumptions against real data
               All 12 ML models retrained on real SACT v4.0 data
```

**Dissertation Note:**
> "SACT v4.0 data collection commenced 1 April 2026. The first complete monthly dataset
> is expected in August 2026 following the 3-month rollout period and NDRS processing lag.
> This dissertation uses synthetically generated data that mirrors the v4.0 schema,
> enabling seamless future integration once real data becomes available."

---

## PART 3: RESTRICTED ACCESS (Requires Application)

### 3.1 Historical Patient-Level SACT Data (v3.0)

| Property | Detail |
|----------|--------|
| **Status** | RESTRICTED — Requires DARS Application |
| **URL** | https://digital.nhs.uk/services/data-access-request-service-dars |
| **Format** | Extract (CSV/custom) |
| **Coverage** | April 2012 onwards (pre-July 2014 data has completeness issues) |
| **Granularity** | Patient-level |
| **Access** | Formal application through Data Access Request Service |

**Application requirements:**
- Organisation must demonstrate legitimate purpose
- Data sharing agreement
- Information governance compliance
- Typically 3-6 months process
- Contact: ndrsenquiries@nhs.net

**Value for our system:**
- Historical patient-level data for ML training
- Would replace synthetic data entirely
- Real no-show patterns, real duration data

### 3.2 NDRS Cancer Consolidated Dataset

| Property | Detail |
|----------|--------|
| **Status** | RESTRICTED — Research applications only |
| **Access** | Via DARS, research governance required |
| **Content** | Linked cancer registration + treatment + outcomes data |

---

## PART 4: NHS-INTERNAL ONLY

### 4.1 Cancer Services Profiles

| Property | Detail |
|----------|--------|
| **Status** | NHS LOGIN REQUIRED |
| **URL** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-services-profiles |
| **Content** | Service-level performance indicators |

### 4.2 Emergency Presentations of Cancer

| Property | Detail |
|----------|--------|
| **Status** | NHS LOGIN REQUIRED |
| **URL** | https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-emergency-presentations |
| **Content** | Emergency presentation routes and rates |

---

## PART 5: EXTERNAL OPEN DATA (Already Integrated)

These are non-NHS open data sources already integrated into our real-time monitoring:

| Source | API | Update | Status |
|--------|-----|--------|--------|
| Open-Meteo Weather | REST (free, unlimited) | Hourly | Integrated |
| TomTom Traffic | REST (2,500/day free) | 5 min | Integrated |
| BBC Wales RSS | RSS feed | 15 min | Integrated |
| Wales Online RSS | RSS feed | 15 min | Integrated |
| Gov.uk Emergency Alerts | RSS feed | Real-time | Integrated |
| NHS Wales Health Alerts | RSS/Web | Hourly | Integrated |

---

## PART 6: AUTO-LEARNING INGESTION SCHEDULE

### What Our System Should Automatically Pull

| Dataset | When Available | Ingestion Frequency | Recalibration Level |
|---------|---------------|--------------------|--------------------|
| Cancer Waiting Times CSV | **NOW** (Jan 2026 latest) | Monthly (1st of month) | Level 1: Baseline recalibration |
| SACT Activity Dashboard | **NOW** | Quarterly | Level 2: Feature weight update |
| Cancer Treatments Dashboard | **NOW** | Annually | Level 2: Feature weight update |
| NHSBSA SCMD-IP (new dataset) | **NOW** (Jan 2026 latest) | Monthly (20th) | Level 1: Baseline recalibration |
| Get Data Out CSV | **NOW** | When updated | Level 1: Baseline recalibration |
| RCRD Dashboard | **NOW** | Monthly | Level 0: Parameter refresh |
| Weather/Traffic/Events | **NOW** | Real-time | Level 0: Parameter refresh |
| SACT v4.0 Patient Data (rollout) | **April–June 2026** (partial) | Monthly | Level 1-2: Preliminary recalibration |
| SACT v4.0 Patient Data (complete) | **August 2026** (first complete) | Monthly | Level 3: Full retrain |
| Historical SACT (DARS) | **ON APPLICATION** | One-time + updates | Level 3: Full retrain |

### Implementation Priority

```
PRIORITY 1 — Implement Now (April 2026)
├── Cancer Waiting Times CSV auto-download     → Monthly baseline calibration
├── NHSBSA SCMD-IP API (NEW package ID)        → Demand signals (use 'secondary-care-medicines-data-indicative-price')
├── Weather/Traffic/Events (already working)    → Real-time parameter refresh
└── Drift detection on all incoming data        → Know when to retrain

PRIORITY 2 — Implement by August 2026
├── SACT Activity Dashboard scraper            → Quarterly feature updates
├── Get Data Out CSV parser                    → Cancer-type validation
└── SACT v4.0 CSV ingestion pipeline           → Ready for August 2026 first complete dataset

PRIORITY 3 — Implement When Access Granted
├── DARS historical SACT data loader           → Full model retraining
└── Local ChemoCare export parser              → Hospital-specific calibration
```

---

## Quick Reference: URL List

```
OPEN DATA (Available Now):
  Cancer Waiting Times    → england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/
  SACT Activity          → digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/sact-activity
  Cancer Treatments      → nhsd-ndrs.shinyapps.io/cancer_treatments/
  Cancer Registration    → digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-registration-statistics
  Get Data Out           → cancerdata.nhs.uk/getdataout/data
  NHSBSA SCMD-IP (NEW)   → opendata.nhsbsa.net/dataset/secondary-care-medicines-data-indicative-price
  NHSBSA API endpoint    → opendata.nhsbsa.net/api/3/action/package_show?id=secondary-care-medicines-data-indicative-price
  RCRD                   → digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/rapid-cancer-registration-data-dashboards
  Cancer Prevalence      → digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub/cancer-prevalence

SACT v4.0 — COLLECTION IN PROGRESS, NOT YET AVAILABLE:
  Collection started     1 April 2026 (rollout period, partial data only)
  First complete dataset August 2026
  SACT v4.0 Spec         → digital.nhs.uk/ndrs/data/data-sets/sact

RESTRICTED:
  DARS Application       → digital.nhs.uk/services/data-access-request-service-dars
  Contact                → ndrsenquiries@nhs.net

EXTERNAL (Already Integrated):
  Weather                → open-meteo.com
  Traffic                → developer.tomtom.com
  News                   → feeds.bbci.co.uk/news/wales/rss.xml
```

---

*Document prepared for MSc Data Science Dissertation*
*Cardiff University - School of Mathematics*
*SACT Scheduling Optimization Project*

*Sources:*
- *NHS England — Cancer Waiting Times Statistics*
- *NHS Digital NDRS — Cancer Data Hub*
- *NHS Digital — SACT Dataset v4.0*
- *NHSBSA — Open Data Portal*
- *CancerData.nhs.uk — Get Data Out Programme*

*Version 1.1 | April 2026 — Updated: SCMD package ID corrected (retired → SCMD-IP), SACT v4.0 availability clarified (August 2026 first complete dataset)*
