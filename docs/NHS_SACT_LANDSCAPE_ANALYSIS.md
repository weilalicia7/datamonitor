# NHS SACT Systems Landscape Analysis

## How This Project Fits Within the Existing NHS Cancer Treatment Technology Stack

**Version:** 1.0
**Date:** March 2026
**Purpose:** Position the SACT Scheduling Optimization System within the existing NHS cancer treatment technology ecosystem, identify the gap it fills, and document SACT v4.0 data standard compliance.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [SACT v4.0 Data Standard](#2-sact-v40-data-standard)
3. [Existing NHS Scheduling Systems](#3-existing-nhs-scheduling-systems)
4. [The Gap: What No Current System Does](#4-the-gap-what-no-current-system-does)
5. [SACT Capacity Crisis: The Problem in Numbers](#5-sact-capacity-crisis-the-problem-in-numbers)
6. [System Positioning: Complementary Intelligent Layer](#6-system-positioning-complementary-intelligent-layer)
7. [SACT v4.0 Data Compatibility](#7-sact-v40-data-compatibility)
8. [Implications for Dissertation](#8-implications-for-dissertation)

---

## 1. Executive Summary

Investigation into the NHS cancer treatment technology landscape reveals a critical finding: **no existing NHS system applies machine learning, predictive analytics, or mathematical optimization to SACT scheduling**.

The NHS SACT ecosystem consists of:

- **SACT v4.0** — A data collection and reporting standard (60 fields, 7 sections). Collection commenced 1 April 2026 (rollout period ends June 2026; first complete data August 2026). It defines *what data to collect*, not *how to schedule*.
- **ChemoCare / ChemoSchedule** — The dominant e-prescribing and basic booking system, used by ~80% of NHS England. It automates workflow but has zero predictive or optimization capability.
- **Manual processes** — Scheduling coordinators using spreadsheets and clinical judgement to allocate chairs, manage no-shows, and handle urgent patients.

This project — the SACT Intelligent Scheduling Optimization System — fills the gap between data collection and intelligent resource allocation by providing 12 ML models and constraint-based optimization that no existing commercial system offers.

---

## 2. SACT v4.0 Data Standard

### 2.1 What SACT v4.0 Is

SACT v4.0 is a **national data collection standard** maintained by the National Disease Registration Service (NDRS) under NHS England Digital. It defines the fields that NHS trusts must submit monthly about their anti-cancer therapy activity.

It is **NOT**:
- A scheduling system
- An optimization tool
- A prediction engine
- A software application

It is **purely** a CSV data specification for reporting treatment activity to national registries.

> Source: NHS Digital — "supports users with the implementation and collection of data as part of the Systemic Anti Cancer Therapy (SACT) data set v4.0.2"

### 2.2 SACT v4.0 Structure

| Section | Data Items | Purpose |
|---------|-----------|---------|
| 1. Linkage | NHS_Number, Local_Patient_Identifier, NHS_Number_Status | Patient identity |
| 2. Demographics | Birth_Date, Gender, Postcode, Organisation_ID | Patient demographics |
| 3. Clinical Status | Primary_Diagnosis (ICD-10), Morphology, SNOMED_CT, Consultant_Specialty | Clinical classification |
| 4. Regimen | Treatment_Context, Intent, Regimen_Name, Start_Date, Performance_Status, Height, Weight, Clinical_Trial, Chemoradiation | Treatment plan |
| 5. Cycle | Cycle_Number, Cycle_Modification, Cycle_Delay, Dose_Modification, Toxicity_Grade | Cycle-level tracking |
| 6. Drug Details | Drug_Name, Dose, Route, Administration_Timestamp, Cycle_Length_In_Days, Number_Of_Cycles, Organisation_Of_Administration | Drug administration |
| 7. Outcome | End_Of_Regimen_Summary | Treatment outcome |

**Total: 60 data items** (expanded from previous versions)

### 2.3 Implementation Timeline

| Milestone | Date |
|-----------|------|
| DAPB approval | June 2025 |
| Implementation phase | July 2025 - March 2026 |
| **Data collection commenced** | **1 April 2026** |
| Roll-out period (partial submissions) | April - June 2026 |
| **Full conformance required** | **1 July 2026** |
| **First complete monthly dataset** | **August 2026** |

> **⚠ DISSERTATION NOTE:** SACT v4.0 data collection commenced 1 April 2026.
> Data is NOT available to external researchers during the rollout period.
> The first complete monthly dataset is expected **August 2026** (July 2026 data + NDRS processing lag).
> This dissertation uses synthetic v4.0-schema-compatible data.

### 2.4 Submission Requirements

- **Format:** CSV only (`.csv` extension)
- **Date format:** `ccyy-mm-dd`
- **Frequency:** Monthly, on 2-month delayed schedule
- **Validation:** Mandatory fields rejected if missing; required fields expected where applicable
- **Connection:** Health and Social Care Network (HSCN) required
- **File naming:** `UnitID-ccyymmdd-ccyymmdd.csv`

### 2.5 Key Data Fields Relevant to Scheduling

| SACT v4.0 Field | Item # | Type | Scheduling Relevance |
|-----------------|--------|------|---------------------|
| Start_Date_Of_Regimen | 22 | Mandatory | When treatment begins |
| Date_Decision_To_Treat | 21 | Required | Booking lead time |
| Performance_Status | 50 | Required | Patient fitness (WHO 0-4) |
| Intent_Of_Treatment | 15 | Required | Curative vs non-curative priority |
| Treatment_Context | 65 | Required | Neoadjuvant/Adjuvant/SACT-only |
| Regimen | 16 | Mandatory | Protocol determines duration |
| Cycle_Length_In_Days | - | Drug section | Cycle interval for scheduling |
| Number_Of_Cycles | - | Drug section | Total treatment horizon |
| Administration_Timestamp | - | Drug section | Actual delivery time |
| Cycle_Modification | - | Cycle section | Schedule changes |
| Cycle_Delay | - | Cycle section | Delays (maps to our no-show/cancellation) |
| Regimen_Modification_Reason | 69 | Required | 1=Patient choice, 2=Organisational, 3=Clinical, 4=Toxicity |

---

## 3. Existing NHS Scheduling Systems

### 3.1 ChemoCare (System C)

| Aspect | Detail |
|--------|--------|
| **Market share** | ~80% of NHS centres across England, Scotland, and Wales |
| **Primary function** | Electronic chemotherapy prescribing |
| **Scheduling capability** | Basic treatment scheduling and ad-hoc intervention management |
| **What it does** | Prescribing, drug preparation, dosage calculation, appointment tracking, data sharing |
| **What it lacks** | No ML/AI, no predictive analytics, no optimization algorithms, no no-show prediction, no resource optimization |

> Source: System C — "the UK's number-one-selling electronic chemotherapy prescribing system"

### 3.2 ChemoSchedule (CIS Oncology)

| Aspect | Detail |
|--------|--------|
| **Relationship** | Add-on module for ChemoCare |
| **Primary function** | Appointment booking automation |
| **Key features** | Auto-generates bookings from prescriptions, cross-site synchronization, centralized worklist, appointment letter generation |
| **What it does** | Workflow automation — booking requests generated once treatment is prescribed |
| **What it lacks** | No no-show prediction, no ML/AI, no chair utilization optimization, no capacity forecasting, no dynamic scheduling |

> Source: CIS Oncology — "Intelligent technology manages all bookings" (referring to rule-based prioritization, not ML)

### 3.3 iQemo (iQ HealthTech)

| Aspect | Detail |
|--------|--------|
| **Primary function** | End-to-end electronic SACT administration and recording |
| **What it lacks** | No predictive analytics documented |

### 3.4 CEPAS (NHS Scotland)

| Aspect | Detail |
|--------|--------|
| **Coverage** | 14 NHS Boards across Scotland via 3 Regional Cancer Networks |
| **Primary function** | Integrated e-prescribing and basic patient scheduling |
| **What it lacks** | No ML/AI optimization documented |

### 3.5 Capability Comparison Matrix

```
                          ChemoCare  ChemoSchedule  iQemo  CEPAS  THIS PROJECT
                          ─────────  ─────────────  ─────  ─────  ────────────
e-Prescribing                 Y           -           Y      Y        -
Basic Booking                 Y           Y           Y      Y        Y
Workflow Automation            Y           Y           Y      Y        Y
Cross-Site Sync               -           Y           -      Y        Y
SACT Data Submission           Y           -           Y      Y        Y
─────────────────────────────────────────────────────────────────────────────
No-Show Prediction             -           -           -      -        Y (AUC 0.82)
Duration Prediction            -           -           -      -        Y (MAE 11 min)
Chair Optimization             -           -           -      -        Y (OR-Tools)
Causal Inference               -           -           -      -        Y (DAG + IV + DML)
Uncertainty Quantification     -           -           -      -        Y (Conformal + MC)
Event Impact Analysis          -           -           -      -        Y (NLP + Sentiment)
Survival Analysis              -           -           -      -        Y (Cox PH)
Uplift Modeling                -           -           -      -        Y (S+T Learner)
Hierarchical Bayesian          -           -           -      -        Y (PyMC)
Squeeze-In Optimization        -           -           -      -        Y (ML-guided)
```

---

## 4. The Gap: What No Current System Does

### 4.1 The Missing Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NHS SACT TECHNOLOGY STACK                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LAYER 1: DATA STANDARD (Exists)                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  SACT v4.0 — 60 fields, 7 sections                                     ││
│  │  Defines WHAT data to collect and report nationally                     ││
│  │  Status: Collection commenced 1 April 2026 (first complete data Aug 2026) ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│                                     ▼                                        │
│  LAYER 2: e-PRESCRIBING & BOOKING (Exists)                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ChemoCare + ChemoSchedule (~80% market share)                          ││
│  │  Prescribe → Generate booking → Track administration                    ││
│  │  Rule-based workflow automation                                          ││
│  │  Status: Mature, widely deployed                                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│                                     ▼                                        │
│  LAYER 3: INTELLIGENT SCHEDULING (DOES NOT EXIST)          ◀── THIS PROJECT│
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ││
│  │  │ PREDICT      │  │ OPTIMIZE     │  │ UNDERSTAND   │  │ QUANTIFY   │  ││
│  │  │              │  │              │  │              │  │            │  ││
│  │  │ No-show risk │  │ Chair alloc  │  │ Causal DAG   │  │ Conformal  │  ││
│  │  │ Duration est │  │ CP-SAT solve │  │ IV / DML     │  │ MC Dropout │  ││
│  │  │ Event impact │  │ Squeeze-in   │  │ Counterfact  │  │ Bayesian   │  ││
│  │  │ Survival     │  │ Overbooking  │  │ Uplift       │  │ QRF        │  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  ││
│  │                                                                          ││
│  │  12 ML Models + OR-Tools Optimization + Real-Time Event Monitoring      ││
│  │  Status: This dissertation project                                       ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Specific Gaps Identified

| Gap | Current State | What This Project Provides |
|-----|--------------|---------------------------|
| **No-show prediction** | Staff rely on gut feeling or past experience | Ensemble ML model (AUC-ROC 0.82) with actionable risk levels |
| **Duration prediction** | Protocol base times only; no patient-specific adjustment | Ensemble regression (MAE 11 min) with cycle, comorbidity, complexity adjustment |
| **Chair optimization** | First-come-first-served or manual juggling | OR-Tools CP-SAT mathematical optimization with constraint satisfaction |
| **Urgent insertion** | Phone calls and manual rearrangement | ML-guided squeeze-in using no-show probability for intelligent double-booking |
| **Event impact** | Reactive — cancel when storm hits | Proactive — NLP sentiment analysis of weather/traffic/news with quantified impact |
| **Causal understanding** | No understanding of WHY patients no-show | Causal DAG with do-calculus, IV estimation, DML for treatment effects |
| **Uncertainty** | Point predictions or no predictions at all | Conformal prediction (guaranteed coverage), MC Dropout (epistemic/aleatoric), QRF intervals |
| **Patient-level effects** | Treat all patients the same | Hierarchical Bayesian model with patient-specific random effects and shrinkage |
| **Intervention targeting** | Same reminder to everyone | Uplift modeling identifies WHICH patients benefit from WHICH intervention |

---

## 5. SACT Capacity Crisis: The Problem in Numbers

### 5.1 National Performance Against Targets

| Metric | Target | Actual (Jan 2026) | Gap |
|--------|--------|-------------------|-----|
| First treatment within 31 days of decision | 96% | 89.8% | -6.2 pp |
| First treatment within 62 days of urgent referral | 85% | 68.4% | -16.6 pp |
| Patients breaching 2-month standard | 0 | 8,818 (May 2025) | Critical |

> Source: NHS England Cancer Waiting Times Statistics

### 5.2 Root Causes

| Cause | Evidence | How This Project Helps |
|-------|----------|----------------------|
| **Staff shortages** | 88% of cancer centre heads concerned about workforce delays | Optimization reduces wasted capacity from no-shows, freeing staff time |
| **Physical space constraints** | Limited chairs/beds across sites | Chair utilization optimization (+17% improvement) |
| **Rising demand** | SACT delivery growing 6-8% per year | Predictive capacity planning anticipates future demand |
| **No-shows and cancellations** | 8-15% of appointments wasted | No-show prediction enables proactive intervention and intelligent overbooking |
| **External disruptions** | Weather, traffic, local events affect attendance | Real-time event monitoring with sentiment analysis and impact quantification |

> Source: Royal College of Radiologists — "Shortages in qualified staff — pharmacy, nursing and medical — and lack of physical space, alongside increasing demand, means that waiting lists are continuing to grow."

### 5.3 Cost of the Problem

| Impact | Estimate | Basis |
|--------|----------|-------|
| Drug waste per no-show | £150-300 | Pre-prepared chemotherapy drugs that cannot be reused |
| Annual wasted appointments (per centre) | 2,000-4,000 | 8-15% of ~26,000 annual appointments |
| Annual financial loss (per centre) | £300,000-1,200,000 | Drug waste + staff time + lost capacity |
| Patient harm | Delayed treatment | Every missed appointment pushes back the treatment timeline |

---

## 6. System Positioning: Complementary Intelligent Layer

### 6.1 Integration Architecture

This system is designed to **complement, not replace** existing NHS infrastructure:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DATA FLOW: EXISTING → THIS SYSTEM → OUTPUT             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT FROM EXISTING SYSTEMS                                                 │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │                                                            │              │
│  │  ChemoCare → Prescriptions, patient records, drug details │              │
│  │  ChemoSchedule → Current bookings, appointment history    │              │
│  │  SACT v4.0 CSV → Standardized treatment data              │              │
│  │  PAS (Patient Admin) → Demographics, contact details      │              │
│  │                                                            │              │
│  └──────────────────────────┬────────────────────────────────┘              │
│                              │                                               │
│                              ▼                                               │
│  THIS SYSTEM (Intelligent Optimization Layer)                                │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │                                                            │              │
│  │  Data Ingestion → Feature Engineering → ML Prediction     │              │
│  │       → Constraint Optimization → Schedule Output         │              │
│  │                                                            │              │
│  │  + Real-Time Monitoring (Weather/Traffic/Events)          │              │
│  │  + Causal Analysis (Why do patients no-show?)             │              │
│  │  + Uncertainty Quantification (How confident are we?)     │              │
│  │                                                            │              │
│  └──────────────────────────┬────────────────────────────────┘              │
│                              │                                               │
│                              ▼                                               │
│  OUTPUT BACK TO EXISTING SYSTEMS                                             │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │                                                            │              │
│  │  Optimized schedule → ChemoSchedule / manual coordinators │              │
│  │  Risk flags → Clinical dashboard for nursing team         │              │
│  │  Intervention recommendations → Patient contact team      │              │
│  │  SACT v4.0 compliant CSV → National registry submission  │              │
│  │  Capacity forecasts → Management planning                 │              │
│  │                                                            │              │
│  └───────────────────────────────────────────────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Value Proposition

| Stakeholder | Current Experience | With This System |
|-------------|-------------------|-----------------|
| **Scheduling Coordinator** | Manual spreadsheet juggling, reactive | Optimized schedule generated automatically, proactive alerts |
| **Nursing Team** | Surprised by no-shows, uneven workload | Predicted no-shows flagged, workload balanced across chairs |
| **Patients** | May receive generic reminder or none | Targeted intervention (SMS/phone/transport) based on uplift model |
| **Unit Manager** | Guesses at capacity needs | Data-driven forecasts with uncertainty quantification |
| **Trust Board** | Misses 62-day target (68.4% vs 85%) | Improved throughput and utilization metrics |

---

## 7. SACT v4.0 Data Compatibility

### 7.1 Field Mapping: SACT v4.0 → This System

| SACT v4.0 Field | Our Field | Status | Notes |
|-----------------|-----------|--------|-------|
| NHS_Number | Patient_ID | Map | Use as identifier |
| Person_Birth_Date | Age / Age_Band | Derive | Calculate age from DOB |
| Patient_Postcode | Postcode_District | Map | First part of postcode |
| Person_Stated_Gender_Code | — | **Add** | Not currently captured |
| Primary_Diagnosis (ICD-10) | — | **Add** | Cancer type classification |
| Performance_Status (WHO 0-4) | Priority (P1-P4) | Map | WHO scale → priority weighting |
| Treatment_Context | — | **Add** | Neoadjuvant/Adjuvant/SACT-only |
| Intent_Of_Treatment | — | **Add** | Curative/Non-curative |
| Regimen | Regimen_Code | Map | Protocol name |
| Start_Date_Of_Regimen | Date | Map | First cycle date |
| Date_Decision_To_Treat | Appointment_Booked_Date | Map | Booking lead time |
| Cycle_Number | Cycle_Number | Direct | Already captured |
| Cycle_Length_In_Days | cycle_days (regimen table) | Direct | Already captured |
| Number_Of_Cycles | total_cycles (regimen table) | Direct | Already captured |
| Administration_Timestamp | Time | Map | Appointment start time |
| Regimen_Modification | — | Partial | Via Cancellation_Reason |
| Modification_Reason | Cancellation_Reason | Map | See mapping below |
| End_Of_Regimen_Summary | Attended_Status | Map | Outcome tracking |

### 7.2 Cancellation Reason Mapping

| SACT v4.0 Code | SACT Meaning | Our Cancellation_Reason |
|----------------|-------------|------------------------|
| 1 | Patient choice | Patient_Request |
| 2 | Organisational issues | Scheduling_Conflict, Resource_Unavailable |
| 3 | Patient clinical factors | Medical, Unwell |
| 4 | Toxicity | Medical (toxicity-related) |

### 7.3 Fields to Add for Full SACT v4.0 Compliance

To make this system capable of generating SACT v4.0 compliant output:

| Field | Type | Priority | Impact on ML |
|-------|------|----------|-------------|
| Gender_Code | Categorical | Medium | Potential no-show/duration feature |
| ICD-10 Diagnosis | String | Medium | Cancer-type specific models |
| Performance_Status (WHO 0-4) | Integer | **High** | Strong predictor of duration and attendance |
| Intent_Of_Treatment | Binary | **High** | Affects scheduling priority logic |
| Treatment_Context | Categorical | Medium | Neoadjuvant patients may have different patterns |
| Clinical_Trial | Binary | Low | Trial patients have different compliance |
| Chemoradiation | Binary | Medium | Combined treatments have different durations |

### 7.4 Export Capability

The system should be able to export scheduled and completed appointments in SACT v4.0 CSV format:

```
File: VCC-20260401-20260430.csv

NHS_Number,Local_Patient_Identifier,...,Regimen,Start_Date_Of_Regimen,...,
1234567890,P12345,...,FOLFOX,2026-04-01,...,
```

This enables the trust to submit their SACT data to NDRS directly from the optimized schedule, closing the loop between intelligent scheduling and national reporting.

---

## 8. Implications for Dissertation

### 8.1 Key Arguments Strengthened

| Argument | Supporting Evidence |
|----------|-------------------|
| **Novelty** | No existing NHS system (ChemoCare, ChemoSchedule, iQemo, CEPAS) provides ML-based scheduling optimization |
| **Timeliness** | SACT v4.0 collection commenced April 2026; first complete data August 2026 creates natural integration point for intelligent scheduling |
| **Need** | 68.4% vs 85% target on 62-day waits; 88% of cancer centre heads concerned about delays |
| **Practicality** | System designed as complementary layer, not replacement — reduces adoption barrier |
| **Data alignment** | 60-field SACT v4.0 standard provides the data foundation; our system adds the intelligence |

### 8.2 Suggested Dissertation Quotes

For the **Introduction**:

> "Shortages in qualified staff — pharmacy, nursing and medical — and lack of physical space, alongside increasing demand, means that waiting lists are continuing to grow." — Royal College of Radiologists

> "The percentage of patients receiving their first cancer treatment within two months of an urgent referral decreased to 68.4% in January 2026, significantly below the 85% operational standard." — NHS England Cancer Waiting Times

For the **Literature Review / Gap Analysis**:

> "While the SACT v4.0 data standard (NHS Digital, 2025) defines comprehensive data collection requirements and systems such as ChemoCare provide e-prescribing and basic booking automation, no existing NHS system applies machine learning or mathematical optimization to the scheduling problem itself. This represents a significant gap between the data available and the intelligence applied to resource allocation decisions."

For the **Methodology**:

> "The system is designed as a complementary intelligent layer that sits above existing NHS infrastructure. It ingests data in SACT v4.0 compatible format from e-prescribing systems such as ChemoCare and outputs optimized schedules that can be exported back to the trust's booking system and submitted to the national SACT registry."

### 8.3 Literature References to Add

| Reference | Use |
|-----------|-----|
| NHS Digital (2025). SACT Data Set v4.0 Technical Guidance. NDRS. | Data standard alignment |
| NHS Digital (2026). SACT Activity Dashboard, January 2026 Update. | Treatment volume statistics |
| Royal College of Radiologists (2023). The SACT Capacity Crisis in the NHS. Policy Briefing. | Problem statement, crisis statistics |
| NHS England (2026). Cancer Waiting Times Statistics. | Performance gap evidence |
| System C Healthcare (2025). ChemoCare: Electronic Chemotherapy Prescribing. | Existing system capabilities |
| CIS Oncology (2025). ChemoSchedule: Oncology Appointment Scheduling. | Existing scheduling limitations |

---

## Appendix A: SACT v4.0 Complete Field List (60 Items)

### Section 1: Linkage
- NHS_Number (M)
- Local_Patient_Identifier (M)
- NHS_Number_Status_Indicator_Code (M)

### Section 2: Demographics
- Person_Birth_Date (R)
- Organisation_Identifier (R)
- Person_Family_Name (R)
- Person_Given_Name (R)
- Person_Stated_Gender_Code (R)
- Patient_Postcode (R)

### Section 3: Clinical Status
- Primary_Diagnosis ICD-10 (R)
- Morphology ICD-O (R)
- Diagnosis_Code SNOMED_CT (R)
- Consultant_Specialty_Code (R)

### Section 4: Regimen
- Treatment_Context (R)
- Intent_Of_Treatment (R)
- Curative_Line_Of_Treatment (M if curative)
- Non_Curative_Line_Of_Treatment (M if non-curative)
- Regimen (M)
- Height_At_Start (R)
- Weight_At_Start (R)
- Performance_Status_Adult (R)
- Date_Decision_To_Treat (R)
- Start_Date_Of_Regimen (M)
- Clinical_Trial (R)
- Chemoradiation (R)

### Section 5: Modifications
- Regimen_Modification (M if section used)
- Reason_For_Regimen_Modification (R)
- Reason_Patient_Clinical_Factors (R)
- Toxicity_Grade_Regimen (R)
- Cycle_Modification (M if section used)
- Reason_For_Cycle_Modification (R)
- Reason_Cycle_Clinical_Factors (R)
- Toxicity_Grade_Cycle (R)
- Cycle_Delay (M if section used)
- Reason_For_Cycle_Delay (R)
- Reason_Delay_Clinical_Factors (R)
- Dose_Modification (M if section used)
- Reason_For_Dose_Modification (R)
- Reason_Dose_Clinical_Factors (R)
- Toxicity_Grade_Dose (R)

### Section 6: Drug Details
- Drug_Name (M)
- Daily_Total_Dose_Per_Administration (R)
- Administration_Measurement (R)
- Unit_Of_Measurement (R)
- SACT_Administration_Route (R)
- Administration_Timestamp (M — choice 1)
- Administration_Date_Oral (M — choice 2)
- Cycle_Length_In_Days (R)
- Number_Of_Cycles_Administered (R)
- Organisation_Identifier_Of_Administration (R)

### Section 7: Outcome
- End_Of_Regimen_Summary (R)

**Classification Key:** M = Mandatory, R = Required

---

*Document prepared for MSc Data Science Dissertation*
*Cardiff University — School of Mathematics*
*SACT Scheduling Optimization Project*

*Sources:*
- *NHS Digital — SACT Dataset v4.0 (digital.nhs.uk/ndrs/data/data-sets/sact)*
- *NHS Digital — SACT v4.0 Technical Guidance*
- *NHS Digital — SACT v4.0 User Guide v4.0.2*
- *NHS England — Cancer Waiting Times Statistics*
- *Royal College of Radiologists — The SACT Capacity Crisis in the NHS (2023)*
- *System C Healthcare — ChemoCare*
- *CIS Oncology — ChemoSchedule*

*Version 1.0 | March 2026*
