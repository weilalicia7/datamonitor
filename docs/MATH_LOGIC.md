# Mathematical Logic and Algorithms

## SACT Scheduling System - Mathematical Framework

This document describes all mathematical formulas, algorithms, and optimization logic used in the SACT (Systemic Anti-Cancer Therapy) Scheduling System for Velindre Cancer Centre.

**Version:** 5.0.9 (Updated April 2026)
**New in v5.0.9:** §5.6.2 fairness-mitigation head-to-head — new `ml/benchmark_fairness_mitigation.py` runs the CP-SAT solver twice on the same cohort with `ScheduleOptimizer._fairness_constraints_enabled` False vs True and writes Four-Fifths-ratio + utilisation comparison rows to `data_cache/fairness_benchmark/results.jsonl`.  `dissertation_analysis.R §21b` reads the JSONL and emits 13 new macros (`\fmSource`, `\fmNpatients`, `\fm{Age,Gender,Site}{Off,On,Delta}`, `\fmUtil{Off,On,Delta}`).  Dissertation §5.6.2 new subsection reports the table with live numbers.  4 new `TestFairnessMitigationToggle` regression tests lock the default-True, feasible-when-off, internal-penalty-list-empty, and JSONL-schema invariants.  Regression for the external-review finding that the fairness audit showed disparities without a mitigation evaluation (world-class system standard).
**New in v5.0.8:** Two dissertation-honesty clarifications (external review).  (i) §15.8 / §A.1.7: the seven causal-validation tests run against a synthetic DAG-generated frame, so passing them verifies implementation correctness, not real-world causality — new `\causalValidationScope` macro + `TestCausalValidationScope` (2 regression tests).  (ii) §A.1.7 / dissertation §4.5.8: the online feature store's `as_of()` API is NOT consumed by the current training path; a new "Scope in the current pipeline" subsection documents this explicitly and `TestTrainingIntegrationScope` (2 tests) locks the contract — one test fails if training imports the store, the other fails if the online hook is unwired.
**New in v5.0.7:** §2.13.8 IRL held-out cross-validated agreement — new `_heldout_agreement()` helper runs LOO CV for $N \leq 30$ and 5-fold CV for $N > 30$; results flow through `IRLFitResult.mean_agreement_heldout` into the JSONL history and into three new R macros (`\irlAgreementHeldout`, `\irlHeldoutMethod`, `\irlHeldoutFolds`).  Dissertation §4.5.3 prose rewritten to report the training-set agreement as optimistic, with the cross-validated number as the honest generalisation claim.  4 new regression tests (`TestHeldoutCrossValidation`) lock the LOO-vs-k-fold switch, the too-few-samples guard, and the generalisation-not-exceeding-training invariant.  Regression for the external-review finding "100.0% pairwise agreement over 22 training pairs" (no test set reported → almost certainly overfitting).
**New in v5.0.6:** §A.13.8 Stochastic MPC budget-latency consistency — new `TestBudgetLatencyInvariants` (3 regression tests), new macros `\mpcBudgetMs`, `\mpcPninetyfiveBudgetStatus`, `\mpcPninetyfiveOvershootMs` emitted by `dissertation_analysis.R §34` with a defensive warning if `p95 > budget` AND `fallback_rate = 0`.  Dissertation §4.5.20 prose rewritten to frame the 500 ms `total_timeout_s` as a fallback *trigger* (not an SLA), report p95 honestly as "slightly above the 500 ms trigger by X ms", and surface the fallback rate as a separate metric rather than lumping it into the same "under 500 ms" clause.  Regression for the external-review finding "p95 = 505.7 ms ... well under the 500 ms budget" contradiction (505.7 > 500, and a percentage can't be "under" a millisecond).
**New in v5.0.5:** §3.8 TFT head-to-head held-out benchmark — new `ml/benchmark_tft_vs_ensemble.py` fits a fresh `TFTTrainer` and a fresh `GradientBoostingClassifier` on the SAME 80/20 train/test split of the eligibility-filtered cohort and writes one row per run to `data_cache/ensemble_benchmark/results.jsonl`.  `dissertation_analysis.R §20` reads it as the single source of truth for dissertation §4.5.6 and emits five new macros (`\tftHeldoutAUC`, `\ensembleHeldoutAUC`, `\heldoutDeltaAUC`, `\heldoutNtrain`, `\heldoutNtest`).  The §4.5.6 prose is rewritten to present TFT as an experimental alternative and to cite the honest head-to-head result alongside the training-set diagnostics from `tft_history.jsonl`.  Two new structural invariants in `tests/test_temporal_fusion_transformer.py::TestHeadToHeadBenchmarkSchema` lock the JSONL schema and the AUC/delta consistency.  Regression for the external-review finding that the original prose compared TFT training-set AUC (0.706) against the tree-ensemble's 0.635 literal from an unrelated validation split.
**New in v5.0.4:** §2.12.11 Reproducible CG scalability benchmark — new `ml/benchmark_column_generation.py` runs both solvers against `patients.xlsx` and writes timed rows to `data_cache/cg_benchmark/results.jsonl`.  Dissertation §4.5.1 Table 4.4 reads this JSONL via `dissertation_analysis.R §16` (with the calibrated empirical model as fallback) and emits `\cgTimeoutSeconds` + `\cgSource` macros so the table caption cannot drift from the actual benchmark setup.  Two new structural invariants in `tests/test_column_generation.py` lock the speedup-vs-cells consistency and the timeout-flag-vs-measured-time consistency.  Regression for the original Table 4.4 inconsistency where the speedup column (4.6×) disagreed with the displayed CP-SAT cell ("timeout") because the two were computed from different hidden values.
**New in v5.0.3:** §A.8.7 Counterfactual fairness audit — decision-threshold safety bounds.  `ScheduleabilityPredictor.fit()` previously set `_decision_threshold = y.mean()` on degenerate inputs, which collapsed to 0.0 when the audit was called against an all-rejected cohort and produced a vacuous PASS with `delta_prob = 0` for every patient (the §4.5.15 dissertation showed `Decision threshold 0.000`).  New `_clamp_threshold()` helper + `DECISION_THRESHOLD_{FLOOR,CEILING,NEUTRAL}` constants + a dedicated `degenerate_fallback` predictor method now guarantee the threshold is strictly inside `[0.05, 0.95]` on every code path, with an honest "vacuously PASS" narrative when the input is degenerate.  Locked by `tests/test_counterfactual_fairness.py::TestDecisionThresholdInvariants` (9 tests).
**New in v5.0.2:** §A.7.6 Safety-guardrails verdict-narrative consistency — new `TestVerdictInvariants` regression class asserts the runtime invariant ($\textsc{reject} \iff n_{\textsc{C}} > 0$) and that the `narrative` string's leading verdict word matches `report.verdict`.  `dissertation_analysis.R §28` adds a defensive `stop()` before emitting `\safExample*` macros if any reports.jsonl row breaks the invariant, and a new `\safExample*` macro family sources the §4.5.14 example block from the most recent `critical_slack_floor` REJECT row so the example's verdict, violation counts, and rules tripped are guaranteed-consistent (regression for the dissertation rendering that mixed `\safLatestVerdict=ACCEPT` with a hardcoded `REJECT --- 1 total violations` narrative).
**New in v5.0.1:** §2.12.6 Column-generation wall-clock guard — `ColumnGenerator(time_limit_s=...)` now honours the outer `time_limit_seconds` passed by `_optimize_column_generation`, breaking the master loop at iteration boundaries (and mid-pricing-pass) when the budget is exhausted and returning a feasible solution tagged `CG_TIME_LIMIT`. Without this guard the §A.9 auto-scaler's per-worker budgets were unenforceable on instances that routed to CG: a 2 s budget on a 202-patient cohort previously took $\approx 350$ s (175× overshoot). Regression-tested at `tests/test_column_generation.py::test_cg_respects_wall_clock_time_limit`. Pre-fix evidence preserved at `data_cache/auto_scaling/runs.pre_cg_time_limit_fix.jsonl.bak`.
**New in v5.0:** Production-readiness sweep across the prediction pipeline.  (i) MPC reward §A.13.2 corrected from $(5 - \text{prio})$ to $(6 - \text{prio})$ + matching $\max(5 - \text{prio}, 1)$ floor in the terminal penalty so priority-5 patients still reward completion / penalise abandonment; `ChairState.priority_at_assignment` field carries priority through the `OCCUPIED→IDLE` transition.  (ii) Auto-scaling §A.9 parallel race exposes a new `solve_with_weights(patients, weights, time_limit)` injection point so each worker observes its own weights — no shared mutable optimiser state, no cross-contamination.  (iii) SHA-256-verified pickle loader (`safe_loader.py`) gates every model `save`/`load` in 7 modules; sidecar `<file>.sha256` lets boot-time integrity probes refuse tampered weights before deserialisation.  (iv) Production hardening: role-based auth (admin ⊇ operator ⊇ viewer), input caps + DML whitelist + `MAX_CONTENT_LENGTH=16MB`, CSRF + session-cookie hardening, structured JSON logging with patient-ID redaction, audit trail (after_request hook auto-emits one row per mutating request, route-template keyed for bounded cardinality), Prometheus `/metrics` (HTTP + `sact_optimizer_solve_seconds{cpsat,mpc}` + `sact_ml_prediction_seconds{noshow,duration}` histograms), `/health/{live,ready}`, optional OpenTelemetry, CVE-pinned deps, multi-stage Dockerfile + nginx TLS + GitHub Actions CI, `secrets_manager.py` (env > .env > AWS/Vault) with rotation runbook, NHS data-protection playbook + DPIA, `retention_enforcer.py` for TTL pruning + GDPR right-to-erasure.  Test suite grew from 441 to 961 (every untested module covered).  No new UI panels — every integration is invisible to the prediction pipeline; observability lives at `/api/*/status`, `/health/*`, `/metrics`.
**New in v4.9:** Column generation for large instances (Section 2.12). Dantzig-Wolfe decomposition splits the monolithic CP-SAT into a master set-partitioning LP (GLOP) + per-chair pricing subproblems (CP-SAT). Handles 100+ patients/day where the monolithic formulation becomes impractical. Integrates with warm-start cache and GNN pruning. `/api/optimizer/colgen` endpoint for CG diagnostics.
**New in v4.8:** GNN feasibility pre-filter (Section 2.11). Bipartite message-passing GNN (numpy + sklearn, no deep-learning framework required) prunes infeasible (patient, chair) pairs before CP-SAT runs. 2–5× solve-time reduction after warm-up. `/api/optimizer/gnn` endpoint for live training stats and prune rates.
**New in v4.7:** Warm-start with CP-SAT solution hints (Section 2.10). `model.AddHint()` seeds solver with prior feasible assignment keyed by instance fingerprint (day-of-week, patient count, priority distribution, no-show bucket, duration band, chair count). 50-80% solve time reduction for recurring schedule patterns. `/api/optimizer/cache` endpoint exposes hit rates and eviction stats for operational monitoring.
**New in v4.6:** All SACT v4.0 sections 1–7 fields now present across all three data files (patients, historical_appointments, appointments). AUC5 drug dosing (Carboplatin) corrected to use Calvert formula: dose = AUC × (CrCl + 25), where CrCl is estimated via simplified Cockcroft-Gault. `Patient_NoShow_Rate` added to patients.xlsx and appointments.xlsx as section-0 scheduling field (was missing, causing false Grade B). SACT compliance validator scoring formula corrected: optional-field denominator (`n_optional`) now consistent with missing-field count (both section-1+ only), eliminating spurious penalty for appointment-level fields absent from patient registry. Verified Grade A across all three files: patients.xlsx 91.4/100, historical_appointments.xlsx 100/100, appointments.xlsx 100/100. `Modification_Reason_Code` valid range corrected to 0–4 (0=not applicable). Historical dataset: 1,899 records (was 1,929 in v4.5 — further RNG state shift from `postcode_weights` sampling).
**New in v4.5:** Postcode sampling recalibrated against pseudonymised Velindre patient travel data (n=5,116 patients; the miles/minutes to each Velindre site are real NHS postcode routing, but patient identifiers are synthetic pseudonyms — see `prepare doc/Patient Data ANONYMISED.csv`). Previous uniform sampling over 25 postcodes produced 20% Near / 72% Medium / 8% Remote — inflating remote patients vs. real Welsh geography. New weighted sampling targets 40% Near / 57% Medium / 3% Remote (empirical CDF from the pseudonymised travel records). Historical dataset regenerated: 1,929 records (was 1,978 — natural variation from RNG state shift). Fairness Audit updated: 'Unknown' gender (Person_Stated_Gender_Code 0/9) excluded from Equal_Opportunity pairwise comparison to suppress spurious violations. Pareto Frontier endpoint upgraded to reload real-time from source + re-run ML predictions on each call, replacing stale post-optimization snapshot.
**v4.4:** Squeeze-in double-booking thresholds recalibrated (HIGH: 0.30→0.25, MEDIUM: 0.20→0.15, LOW: 0.15→0.10, REJECT: <0.15→<0.10) to increase utilisation of no-show predictions and reduce idle chair time; driven by ML ensemble retraining on regenerated SACT v4.0 dataset (102 cols, corrected BSA) which produces 13–14% no-show predictions — below the previous 0.15 floor. Risk classification in §3.4 aligned to match. Validation on hold-out set of 396 appointments (20% of 1,978) showed +12% increase in successful squeeze-in insertions with no significant rise in chair overutilisation (pending real-data confirmation).
**v4.3:** BSA formula corrected (H must be in cm: `H_cm = H_m × 100`); synthetic dataset regenerated to 102 columns / 1,929 rows with Grade A SACT v4.0 alignment (98.2/100); `prepare_dataframe_for_ml()` added to FeatureEngineer; `/api/data/sact-v4/validate` endpoint added; `/analytics` now serves live NHS Open Data & Auto-Learning panel; recalibration level floor set to 1 (was None when no real data available)
**v4.2:** SACT v4.0 data availability clarified (collection commenced April 2026; first complete dataset August 2026); NHSBSA SCMD package corrected to SCMD-IP
**v4.1:** DRO (Wasserstein ambiguity set), CVaR objective in CP-SAT (Rockafellar & Uryasev 2000)
**v4.0:** Sensitivity Analysis, Model Cards, full ML-optimizer integration, ensemble training on historical data
**v3.0:** Auto-learning pipeline, drift detection, SACT v4.0 data compliance, NHS open data integration
**v2.0:** Advanced ML models (Sections 10-20)
**v1.0:** Core scheduling models (Sections 1-9)

---

## Table of Contents

### Core Models
1. [Data Requirements (SACT v4.0)](#1-data-requirements)
   - 1.3 [Data Flow Pipeline (end-to-end function composition)](#13-data-flow-pipeline-end-to-end-function-composition)
2. [Constraint Programming Optimization](#2-constraint-programming-optimization) — CP-SAT with 6 objectives + fairness constraints
   - 2.10 [Warm-Start with Solution Hints](#210-warm-start-with-solution-hints)
   - 2.11 [GNN Feasibility Pre-Filter](#211-gnn-feasibility-pre-filter)
   - 2.12 [Column Generation for Large Instances](#212-column-generation-for-large-instances)
   - 2.13 [Multi-Objective with Learned Preferences (Inverse RL)](#213-multi-objective-with-learned-preferences-inverse-rl)
3. [No-Show Prediction Model](#3-no-show-prediction-model) — Stacked Ensemble (RF + GB + XGBoost)
   - 3.6 [RNN Sequence Model for Patient History](#36-rnn-sequence-model-for-patient-history)
   - 3.7 [Decision-Focused Learning (SPO+)](#37-decision-focused-learning-smart-predict-then-optimise)
   - 3.8 [Temporal Fusion Transformer](#38-temporal-fusion-transformer-joint-multi-output)
4. [Duration Prediction Model](#4-duration-prediction-model) — Protocol-specific with variance
5. [Feature Engineering](#5-feature-engineering) — 80+ features in 5 categories
6. [Travel Time Estimation](#6-travel-time-estimation)
7. [Risk Scoring](#7-risk-scoring)
8. [Performance Metrics](#8-performance-metrics)
9. [Urgent Patient Insertion](#9-urgent-patient-insertion) — Squeeze-in with robustness-aware scoring (9.8)

### Production-Grade Subsystems (v5.0 — Appendix A)
- A.1 [Online Feature Store](#a1-online-feature-store-31--streaming-ml)
- A.2 [Micro-Batch Optimizer](#a2-micro-batch-optimizer-32--three-tier-orchestration)
- A.3 [Digital Twin (What-If Simulation)](#a3-digital-twin-33--what-if-simulation)
- A.4 [Drift Root-Cause Attribution](#a4-drift-root-cause-attribution-34)
- A.5 [Distributionally Robust Fairness (DRO)](#a5-distributionally-robust-fairness-41)
- A.6 [Individual (Lipschitz) Fairness](#a6-individual-lipschitz-fairness-42)
- A.7 [Safety Guardrails with Runtime Monitoring](#a7-safety-guardrails-with-runtime-monitoring-43)
- A.8 [Counterfactual Fairness Audit](#a8-counterfactual-fairness-audit-44)
- A.9 [Auto-scaling Optimizer with Timeout Guarantees](#a9-auto-scaling-optimizer-with-timeout-guarantees-51)
- A.10 [Human-in-the-Loop Override Learning](#a10-human-in-the-loop-override-learning-52)
- A.11 [Explainable Rejection Reports](#a11-explainable-rejection-reports-53)
- A.12 [SACT Version Adapter](#a12-sact-version-adapter-54)
- A.13 [Stochastic MPC Scheduler](#a13-stochastic-mpc-scheduler-55)

### Advanced ML Models (v2.0)
10. [Survival Analysis (Cox Proportional Hazards)](#10-survival-analysis)
11. [Uplift Modeling (S-Learner & T-Learner)](#11-uplift-modeling)
12. [Multi-Task Learning (Neural Network)](#12-multi-task-learning)
13. [Quantile Regression Forest](#13-quantile-regression-forest)
14. [Hierarchical Bayesian Model](#14-hierarchical-bayesian-model)
15. [Causal Inference (DAG & do-Calculus)](#15-causal-inference) — includes Counterfactual Explanations (15.7) and Causal Validation (15.8)
15b. [Sensitivity Analysis](#15b-sensitivity-analysis) — Local S_i, Global I_j, Elasticity E_i
15c. [Model Cards for Transparency](#15c-model-cards-for-transparency) — NHS AI Ethics compliance
16. [Instrumental Variables (2SLS)](#16-instrumental-variables)
17. [Double Machine Learning](#17-double-machine-learning)
18. [Event Impact Model](#18-event-impact-model)
19. [Conformal Prediction](#19-conformal-prediction) — incl. 19.10 Risk-Adaptive α
20. [Monte Carlo Dropout](#20-monte-carlo-dropout)

### Auto-Learning & Data Pipeline (v3.0)
22. [Auto-Learning Pipeline](#22-auto-learning-pipeline) — Online Learning (22.6), RL (22.7), MARL (22.8)
23. [Drift Detection](#23-drift-detection) — PSI, KS-test, CUSUM
24. [SACT v4.0 Data Standard](#24-sact-v40-data-standard)

### Uncertainty-Aware Optimization (v4.0)
24b. [DRO & CVaR Optimization](#24b-uncertainty-aware-optimization-dro) — Wasserstein DRO (24b.1-24b.3) + CVaR in CP-SAT (24b.7, Rockafellar & Uryasev 2000)
25. [Complete Model Summary](#25-model-summary-table)

### Performance & Complexity (v5.0)
27. [Algorithmic Complexity Guarantees](#27-algorithmic-complexity-guarantees) — Time / space / convergence bounds for CP-SAT (27.1), GNN (27.2), Column Generation (27.3), Conformal (27.4), MPC rollout (27.5), DRO (27.6)
28. [Numerical Stability Notes](#28-numerical-stability-notes) — log/exp/sqrt/division safeguards across modules
29. [Hyperparameter Selection Methodology](#29-hyperparameter-selection-methodology) — How every ε, α, λ, τ, etc. in the codebase is chosen
30. [Validation Invariants (Mathematical Tests)](#30-validation-invariants-mathematical-tests) — what the test suite proves about each formula
31. [Pseudocode Style Guide](#31-pseudocode-style-guide) — single convention for every algorithm box
32. [Known Limitations per Model](#32-known-limitations-per-model) — explicit failure modes for CP-SAT / no-show / conformal / DRO / IRL / MPC

---

## Notation Conventions

This document follows a single typographic convention across every formula
so symbols can be read at a glance.  Where you see a deviation in older
sections, treat it as legacy markup pending a stylistic copy-edit, not a
semantic difference.

| Object               | Style                          | Example                                       |
|----------------------|--------------------------------|-----------------------------------------------|
| Vector               | Bold lowercase                 | $\mathbf{x},\ \mathbf{w},\ \boldsymbol{\theta}$ |
| Matrix               | Bold uppercase                 | $\mathbf{X},\ \mathbf{W},\ \boldsymbol{\Sigma}$ |
| Scalar               | Italic                         | $x,\ y,\ \alpha,\ \tau$                        |
| Set / family         | Calligraphic                   | $\mathcal{A},\ \mathcal{D},\ \mathcal{S}$      |
| Random variable      | Italic uppercase               | $X,\ Y,\ T$                                   |
| Estimator / fitted   | Hat                            | $\hat{\theta},\ \hat{\mathbf{w}},\ \hat{V}$    |
| Indicator            | $\mathbb{1}_{\{\cdot\}}$       | $\mathbb{1}_{\text{complete}}$                |
| Expectation, prob.   | $\mathbb{E}[\cdot],\ \Pr(\cdot)$ | $\mathbb{E}[R_t],\ \Pr(\text{override})$    |

Subscripts denote indices ($x_i$ for sample $i$, $w_p$ for patient $p$);
superscripts in math mode are powers unless explicitly tagged a label
(e.g., $h_0(t)$ for "baseline hazard at $t$").  Bracketed superscripts —
$\mathbf{x}^{(t)}$ — denote a value at decision epoch $t$.

---

## Global Notation Table (Symbol Glossary)

A single reference for every symbol that recurs across multiple sections.
Local symbols introduced and discharged within one subsection are not
listed here — see the section text for those.

| Symbol                  | Description                                                       | First used        | Domain / Constraints                                              |
|-------------------------|-------------------------------------------------------------------|-------------------|-------------------------------------------------------------------|
| $P$                     | Set of patients to schedule                                       | §2.1              | $\lvert P \rvert = n$                                            |
| $C$                     | Set of chairs                                                     | §2.1              | $\lvert C \rvert = m$, typically 45                              |
| $H$                     | Operating horizon (minutes)                                       | §2.1              | 480–600 (8–10 h working day)                                     |
| $d_p$                   | Expected treatment duration for patient $p$ (min)                 | §2.1              | $\mathbb{R}^+$                                                   |
| $\pi_p$                 | ML-predicted no-show probability for $p$                          | §2.2              | $[0, 1]$                                                         |
| $\lambda_i$             | Weight of objective $i$ in CP-SAT scalarisation                   | §2.2              | $\sum_i \lambda_i = 1,\ \lambda_i \geq 0$                        |
| $Z_{\text{pri}}$        | Priority-weighted assignment score                                | §2.2              | $100 \cdot (5 - \text{priority}_p)$                              |
| $Z_{\text{util}}$       | Early-start preference (negative of start time)                   | §2.2              | $-\sum_p s_p$                                                    |
| $Z_{\text{noshow}}$     | No-show risk penalty                                              | §2.2              | $-\lfloor 100 \pi_p \rfloor$                                     |
| $Z_{\text{wait}}$       | Waiting-time bonus                                                | §2.2              | $\min(\text{days\_waiting}, 62) \cdot 5$                         |
| $Z_{\text{robust}}$     | Schedule robustness                                               | §2.2              | $-\max(0, \lfloor (D_p - 120)/30 \rfloor)$                       |
| $Z_{\text{travel}}$     | Travel-distance penalty                                           | §2.2              | $-\lfloor t_p^{\text{travel}}/10 \rfloor$                        |
| $R(S)$                  | Normalised schedule robustness                                    | §2.2              | $\frac{1}{\lvert P \rvert}\sum_p \min(1,\ \text{Slack}_p / d_p)$ |
| $\varepsilon$           | Wasserstein radius for DRO                                        | §24b.1            | $\varepsilon \geq 0$, default $0.05$                             |
| $\alpha$                | CVaR quantile / conformal miscoverage level                       | §19, §24b.4       | $\alpha \in (0, 1)$, default $0.10$                              |
| $\beta_1, \beta_2$      | Risk-adaptive conformal coefficients                              | §19.10            | $\beta_1 = 0.15,\ \beta_2 = 0.08$                                |
| $\boldsymbol{\theta}$   | Inverse-RL preference vector                                      | §2.13             | $\boldsymbol{\theta} \in \mathbb{R}^6_{\geq 0},\ \sum_i \theta_i = 1$ |
| $\tau$                  | Double-booking threshold (lowest tier)                            | §9.4              | $0.10$                                                           |
| $\tau_{\text{suggest}}$ | Override-suggestion threshold                                     | §A.10.4           | $0.80$                                                           |
| $L$                     | Lipschitz constant for individual fairness                        | §A.6              | $L \geq 0$, default $1.0$                                        |
| $\delta$                | Fairness parity budget                                            | §2.4, §A.5        | $0.15$ (demographic), $0.10$ (equal opportunity)                  |
| $K$                     | Number of DRO scenarios / MPC rollouts                            | §24b.7, §A.13     | $K = 10$ (default)                                               |
| $T$                     | Number of MC-Dropout forward passes                               | §20               | $T = 100$ (default)                                              |
| $\text{PSI}$            | Population Stability Index                                        | §23.1             | $\text{PSI} \in [0, \infty)$                                     |

When a symbol is overloaded across sections (e.g., $\alpha$ for both the
CVaR quantile and the Gamma-distribution shape parameter in the
Bayesian arrival model of §A.13.5), the local meaning is fixed by the
nearest defining sentence; the glossary above lists the most common
usage.  Where a section needs a private alias for a glossary symbol, it
is introduced immediately and discharged before the next subsection.

---

## 1. Data Requirements

The system uses a data model aligned with **SACT v4.0** (NHS England, April 2026). Fields are organized into the 7 SACT sections plus scheduling-specific extensions.

### 1.1 SACT v4.0 Core Fields (60 items across 7 sections)

| Section | Field | Type | Values | Usage |
|---------|-------|------|--------|-------|
| **1. Linkage** | NHS_Number | String(10) | 10-digit numeric | Mandatory patient identifier |
| | Local_Patient_Identifier | String | P10000-P99999 | Internal system ID |
| | NHS_Number_Status_Indicator_Code | String(2) | 01=Verified | Identifier validation |
| **2. Demographics** | Person_Given_Name | String | First name | Patient identification |
| | Person_Family_Name | String | Surname | Patient identification |
| | Person_Birth_Date | Date | ccyy-mm-dd | Age derivation |
| | Person_Stated_Gender_Code | Integer | 1=M, 2=F, 9=NS | Demographics |
| | Patient_Postcode | String | CF10 4YW | Geographic features |
| | Organisation_Identifier | String | RQF (Velindre) | Site tracking |
| **3. Clinical** | Primary_Diagnosis_ICD10 | String | C18.9, C50.9... | Cancer type classification |
| | Morphology_ICD_O | String | 8140/3, 8500/3... | Tumour morphology |
| | Performance_Status | Integer | WHO 0-4 | No-show & duration prediction |
| | Consultant_Specialty_Code | String | 370, 800, 303 | Clinical pathway |
| **4. Regimen** | Regimen_Code | String | FOLFOX, RCHOP, PEMBRO... | Treatment protocol (SACT standard) |
| | Intent_Of_Treatment | String(2) | 06=Curative, 07=Non-curative | Scheduling priority |
| | Treatment_Context | String(2) | 01=Neo, 02=Adj, 03=SACT | Treatment pathway |
| | Start_Date_Of_Regimen | Date | ccyy-mm-dd | Treatment timeline |
| | Date_Decision_To_Treat | Date | ccyy-mm-dd | Booking lead time |
| | Height_At_Start | Float | metres | BSA dose calc — convert to cm before DuBois formula |
| | Weight_At_Start | Float | kg | BSA dose calc |
| | Clinical_Trial | String(2) | 01=In trial, 02=Not | Trial status |
| | Chemoradiation | String(1) | Y/N | Combined treatment |
| **5. Modifications** | Regimen_Modification | String(1) | Y/N | Schedule change flag |
| | Modification_Reason_Code | Integer | 0-4 (SACT v4.0) | 0=N/A, 1=Patient, 2=Org, 3=Clinical, 4=Toxicity |
| | Toxicity_Grade | Integer | 0-5 (CTCAE v5.0) | Adverse event grading |
| **6. Drug Details** | Drug_Name | String | Oxaliplatin, Fluorouracil... | Drug identification |
| | Daily_Total_Dose | Float | Per administration | Dose tracking |
| | Unit_Of_Measurement | String | mg/m2, mg, AUC | Dose unit |
| | SACT_Administration_Route | String | IV, PO, SC | Administration route |
| | Cycle_Length_In_Days | Integer | 7-60 | Scheduling interval |
| **7. Outcome** | End_Of_Regimen_Summary | String | Outcome code | Treatment outcome |

### 1.1a Body Surface Area (BSA) Formula

Drug doses in oncology are weight-normalised using the DuBois Body Surface Area formula:

```
BSA (m²) = 0.007184 × H_cm^0.725 × W_kg^0.425
```

Where:
- **H_cm** = patient height in **centimetres** (`H_cm = Height_At_Start_metres × 100`)
- **W_kg** = patient weight in kilograms

> ⚠️ **Important:** The constant 0.007184 assumes height in **cm**, not metres. In `data/sact_v4_schema.py` and `datasets/generate_sample_data.py`, `Height_At_Start` is stored in metres and converted with `h_cm = h * 100` before applying the formula. A prior bug (v4.2 and earlier) omitted this conversion, producing BSA values of ~0.07 instead of the correct ~1.7–1.9 m². Fixed in v4.3.

**Population values (typical adult cancer patient):**

| Statistic | Value | Source |
|-----------|-------|--------|
| Median BSA | 1.73 m² (default fallback) | Standard oncology reference |
| Synthetic dataset range | 1.60–2.16 m² | `generate_sample_data.py` (v4.3) |
| Fixed-dose drugs (mg) | Flat dose; BSA not applied | e.g. Pembrolizumab 200 mg fixed |
| Weight-based drugs (mg/kg) | Daily dose = dose_per_kg × weight_kg | e.g. Trastuzumab 6 mg/kg |
| m²-based drugs | Daily dose = dose_per_m² × BSA | e.g. Oxaliplatin 85 mg/m² |
| AUC-based drugs | Calvert formula: dose = AUC × (CrCl + 25) | e.g. Carboplatin AUC5 ≈ 425–700 mg |

> **Calvert formula (AUC dosing):** `dose (mg) = AUC × (GFR + 25)`, where GFR is estimated via simplified Cockcroft-Gault: `CrCl = (140 − age) × weight_kg / 72` (× 0.85 for female). Minimum CrCl clamped at 30 mL/min. Used for Carboplatin in CARBPAC and PEME regimens.

### 1.2 Scheduling-Specific Fields (22 operational fields)

| # | Field | Type | Values | Usage |
|---|-------|------|--------|-------|
| 1 | Appointment_ID | String | APT500001... | Unique appointment ID |
| 2 | Appointment_Date | Date | ccyy-mm-dd | Scheduling |
| 3 | Site_Code | Categorical | WC, NP, BGD, CWM, SA | Resource allocation |
| 4 | Regimen_Code | String | FOLFOX, RCHOP, PEMBRO... | Treatment protocol |
| 5 | Cycle_Number | Integer | 1-20 | Duration prediction |
| 6 | Treatment_Day | String | Day 1, Day 8, Day 15 | Multi-day scheduling |
| 7 | Planned_Duration | Integer | 30-360 minutes | Optimization |
| 8 | Actual_Duration | Integer | Actual minutes | ML training target |
| 9 | Chair_Number | Integer | 1-12 | Resource constraint |
| 10 | Travel_Time_Min | Integer | 5-80 min | No-show prediction |
| 11 | Attended_Status | Categorical | Yes/No/Cancelled | ML target variable |
| 12 | Day_Of_Week | Categorical | Mon-Fri | Temporal features |
| 13 | Priority | Categorical | P1, P2, P3, P4 | Objective weighting |
| 14 | Age_Band | Categorical | <40, 40-60, 60-75, >75 | Risk stratification |
| 15 | Has_Comorbidities | Boolean | Yes/No | Duration adjustment |
| 16 | IV_Access_Difficulty | Boolean | Yes/No | Duration/resource |
| 17 | Requires_1to1_Nursing | Boolean | Yes/No | Staffing constraint |
| 18 | Date_Decision_To_Treat | Date | ccyy-mm-dd | Lead time features |
| 19 | Modification_Reason_Code | Integer | 0-4 (SACT) | Pattern analysis |
| 20 | Previous_NoShows | Integer | 0-10 | No-show prediction |
| 21 | Previous_Cancellations | Integer | 0-10 | Risk scoring |
| 22 | Contact_Preference | Categorical | SMS/Phone/Email/Post | Engagement analysis |

---

## 1.3 Data Flow Pipeline (end-to-end function composition)

The transformation from raw SACT v4.0 inputs to a published schedule is
a composition of well-typed functions; every stage has an explicit
mathematical signature so a reader can trace a single value end-to-end.

ASCII overview (reads top-to-bottom; each arrow is one function call,
boxed labels point at the section that defines the function):

```
                ┌──────────────────────────────────────┐
                │  Raw SACT v4.0 inputs                │
                │  R = {patients.xlsx,                 │
                │       historical_appointments.xlsx,  │
                │       appointments.xlsx}             │
                └──────────────────────────┬───────────┘
                                           │  A  (version adapter, §A.12)
                                           ▼
                ┌──────────────────────────────────────┐
                │  D_canonical  ∈ R^{n × 24}           │
                │  Canonical schema (24 cols)          │
                └──────────────────────────┬───────────┘
                                           │  F  (feature engineering, §5)
                                           ▼
                ┌──────────────────────────────────────┐
                │  X  ∈ R^{n × f},  f = 60+            │
                │  Feature matrix (5 categories)       │
                └─────────────┬────────────────────────┘
                              │
              ┌───────────────┴────────────────┐
              │                                │
              │ M_ns (§3 + §3.6 RNN + §3.8 TFT)│  M_dur (§4)
              ▼                                ▼
   ┌──────────────────────┐         ┌──────────────────────┐
   │  π ∈ [0, 1]^n        │         │  d ∈ R^n_+           │
   │  No-show prob/patient│         │  Duration (min)      │
   └──────────┬───────────┘         └──────────┬───────────┘
              │  M_DFL  (calibration head, §3.7, optional)
              ▼
   ┌──────────────────────┐
   │  π'  ∈ [0, 1]^n      │
   │  π'_p = σ(a·logit(π_p) + b),  a ≥ 0
   └──────────┬───────────┘
              │
              │ + constraints (chairs C, horizon H, fairness §2.4,
              │   safety §A.7, lipschitz §A.6, DRO §A.5)
              │ + weights λ ∈ Δ⁵ (§2.2 Quick Reference §2.2.1)
              ▼
   ┌──────────────────────────────────────────────────────┐
   │  O — CP-SAT solve (§2)                               │
   │  with warm-start (§2.10) + GNN pre-filter (§2.11) +  │
   │  column generation (§2.12) + auto-scaling (§A.9)     │
   └──────────┬───────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────────┐
   │  S  = published schedule  (appointment list)         │
   └──────────┬───────────────────────────────────────────┘
              │
              │  steady-state real-time refinement
              │
       ┌──────┴───────┬──────────────────────┐
       │              │                      │
       │  B (§A.2)    │  R_MPC (§A.13)       │  Audit + Observability
       │  micro-batch │  stochastic MPC      │  (T4.4 + T4.5)
       ▼              ▼                      ▼
   ┌────────┐    ┌─────────────────┐    ┌────────────────────┐
   │ S_t+δ  │    │ S_t+1 = R_MPC   │    │ Prometheus metrics │
   │ live    │   │ (S_t, Q_t,      │    │ + audit JSONL +    │
   │ update │    │       λ(t))     │    │ /health/* probes   │
   └────────┘    └─────────────────┘    └────────────────────┘
```

Formal function signatures (one line per stage):

| Stage | Function | Type | §                  |
|-------|----------|------|--------------------|
| Adapter        | $\mathcal{A}: R \to D_{\text{canonical}}$                                                       | raw → 24-col canonical    | §A.12 |
| Features       | $\mathcal{F}: D_{\text{canonical}} \to \mathbf{X} \in \mathbb{R}^{n \times f},\ f \approx 60$ | canonical → feature matrix | §5    |
| No-show        | $\mathcal{M}_{\text{ns}}: \mathbf{X} \to \boldsymbol{\pi} \in [0, 1]^{n}$                       | features → P(no-show)      | §3    |
| Duration       | $\mathcal{M}_{\text{dur}}: \mathbf{X} \to \mathbf{d} \in \mathbb{R}_{+}^{n}$                    | features → minutes         | §4    |
| Calibration    | $\mathcal{M}_{\text{DFL}}: \pi \mapsto \sigma(a\cdot\operatorname{logit}(\pi) + b),\ a \geq 0$  | scalar calibration head    | §3.7  |
| Optimisation   | $\mathcal{O}: (\boldsymbol{\pi}', \mathbf{d}, P, C, H, \boldsymbol{\lambda}) \to S$              | CP-SAT solve               | §2    |
| Micro-batch    | $\mathcal{B}: (S_t,\ \Delta\text{events}) \to S_{t+\delta}$                                     | streaming patch            | §A.2  |
| MPC            | $\mathcal{R}_{\text{MPC}}: (S_t, Q_t, \lambda(t)) \to S_{t+1}$                                  | receding-horizon refit     | §A.13 |

The composition is

$$
S = \mathcal{O}\bigl(\,\mathcal{M}_{\text{DFL}}(\mathcal{M}_{\text{ns}}(\mathbf{X}))
,\ \mathcal{M}_{\text{dur}}(\mathbf{X})
,\ P,\ C,\ H,\ \boldsymbol{\lambda}\bigr)
\quad\text{where}\quad
\mathbf{X} = \mathcal{F}(\mathcal{A}(R))
$$

with $\mathcal{B}$ + $\mathcal{R}_{\text{MPC}}$ applied in steady state to
keep $S_t$ aligned with live arrivals, no-shows, and cancellations.

**Reproducibility invariants** (every intermediate written to JSONL):

| Stage | Log file                                              | Used by                               |
|-------|-------------------------------------------------------|---------------------------------------|
| Adapter        | `data_cache/sact_adapter/events.jsonl`               | dissertation §32                      |
| Feature store  | `data_cache/feature_store/serving_latency.jsonl`     | dissertation §28                      |
| Predictions    | `data_cache/predictions/latest.json`                  | downstream callers                    |
| Optimiser      | `data_cache/auto_scaling/runs.jsonl`                  | dissertation §30                      |
| Micro-batch    | `data_cache/micro_batch/latency.jsonl`               | dissertation §29                      |
| MPC            | `data_cache/mpc_scheduler/{decisions,simulations}.jsonl` | dissertation §34                  |
| Audit          | `data_cache/audit/<YYYY-MM-DD>.jsonl`                 | T4.4 audit trail + GDPR Art 30       |

This composition + log inventory is what makes every dissertation
number reproducible from a fresh checkout: replay the JSONL trail
through `dissertation_analysis.R` and the macros regenerate exactly.

---

## 2. Constraint Programming Optimization

### 2.1 Problem Formulation

The scheduling problem is formulated as a **Constraint Programming (CP)** problem using Google OR-Tools CP-SAT solver.

#### Decision Variables

For each patient $p \in P$ and chair $c \in C$:

$$x_{p,c} \in \{0, 1\}$$ - Binary: patient $p$ assigned to chair $c$

$$s_p \in [0, H - d_p]$$ - Integer: start time of patient $p$ (minutes from day start)

$$a_p \in \{0, 1\}$$ - Binary: patient $p$ is assigned

Where:
- $H$ = horizon (operating hours in minutes, typically 600 for 8:00-18:00)
- $d_p$ = expected duration for patient $p$

#### Constraints

**1. Single Assignment Constraint:**
Each patient can be assigned to at most one chair:

$$\sum_{c \in C} x_{p,c} = a_p, \quad \forall p \in P$$

**2. No-Overlap Constraint:**
For each chair, no two appointments can overlap:

$$\text{NoOverlap}(\{[s_p, s_p + d_p] : x_{p,c} = 1\}), \quad \forall c \in C$$

**3. Bed Requirement Constraint:**
Patients requiring beds must be assigned to bed-capable chairs:

$$x_{p,c} = 0, \quad \forall p \in P_{bed}, c \in C_{chair}$$

**4. Operating Hours Constraint:**

$$0 \leq s_p \leq H - d_p, \quad \forall p \in P$$

### 2.2 Multi-Objective Function

The scheduling objective uses **weighted scalarization** of 6 objectives, with configurable weights for Pareto frontier exploration:

$$\max Z = \lambda_1 Z_{priority} + \lambda_2 Z_{util} + \lambda_3 Z_{noshow} + \lambda_4 Z_{wait} + \lambda_5 Z_{robust} + \lambda_6 Z_{travel}$$

Subject to: $\sum_{i=1}^{6} \lambda_i = 1$

#### Individual Objectives

**1. Priority-Weighted Assignment ($Z_{priority}$):**

$$Z_{priority} = \sum_{p \in P} (5 - \text{priority}_p) \times 100 \times a_p$$

- P1: 400, P2: 300, P3: 200, P4: 100

**2. Utilization — Early Start Preference ($Z_{util}$):**

$$Z_{util} = -\sum_{p \in P} s_p$$

Earlier start times pack the schedule tighter, increasing chair utilization.

**3. No-Show Risk Minimization ($Z_{noshow}$):**

$$Z_{noshow} = -\sum_{p \in P} \lfloor \pi_p \times 100 \rfloor \times a_p$$

Where $\pi_p$ = ML-predicted no-show probability for patient $p$.

**4. Waiting Time Minimization ($Z_{wait}$):**

$$Z_{wait} = \sum_{p \in P} \min(d_p^{wait}, 62) \times 5 \times a_p$$

Where $d_p^{wait}$ = days since booking (capped at 62-day NHS target). Patients who have waited longer receive higher scheduling priority.

**5. Schedule Robustness ($Z_{robust}$):**

$$Z_{robust} = -\sum_{p \in P} \max(0, \lfloor(D_p - 120) / 30\rfloor)$$

Penalizes long treatments (>120 min) that are more susceptible to overruns.

**Robustness Score $R(S)$ (normalized 0-1):**

$$R(S) = \frac{1}{|P|} \sum_{p \in P} \min\left(1, \frac{\text{Slack}_p}{\text{Duration}_p}\right)$$

Where $R(S) \in [0,1]$. Higher = more robust. A score of 1.0 means every appointment has slack at least equal to its own duration.

**Slack Calculation (sequential on same chair):**

$$\text{Slack}_p = s_p - (s_{p-1} + d_{p-1})$$

Where $s_p$ = start time, $d_{p-1}$ = duration of previous appointment. For first appointment: $\text{Slack}_p = s_p - \text{day\_start}$.

**Slack Categories:**

| Slack Duration | Classification | Robustness Impact |
|---------------|---------------|-------------------|
| < 10 min | **Critical** | High risk of cascade delays |
| 10-20 min | **Tight** | Moderate risk |
| 20-60 min | **Adequate** | Low risk |
| > 60 min | **Ample** | Very robust |

**Robustness-Aware Squeeze-In Scoring:**

$$S_{\text{total}} = S_{\text{base}} + S_{\text{robustness}} + S_{\text{priority}}$$

$$S_{\text{robustness}} = \begin{cases} -15 & \text{if remaining\_slack} < 10 \\ -5 & \text{if } 10 \leq \text{remaining\_slack} < 20 \\ 0 & \text{if } 20 \leq \text{remaining\_slack} < 60 \\ +10 & \text{if remaining\_slack} \geq 60 \end{cases}$$

**Remaining Slack after insertion:**

$$\text{Remaining\_Slack} = \min(s_{\text{new}} - t_{\text{prev}}, s_{\text{next}} - (s_{\text{new}} + d_{\text{new}}))$$

**Robustness Impact:**

$$\text{Robustness\_Impact} = \text{Slack}_{\text{before}} + \text{Slack}_{\text{after}} - 2 \cdot \text{Remaining\_Slack}$$

This measures how much total buffer is lost due to the insertion.

**6. Travel Distance Minimization ($Z_{travel}$):**

$$Z_{travel} = -\sum_{p \in P} \lfloor t_p^{travel} / 10 \rfloor$$

Where $t_p^{travel}$ = estimated travel time in minutes.

#### Default Weight Configuration

| Objective | Weight ($\lambda$) | Rationale |
|-----------|-------------------|-----------|
| Priority | 0.30 | Clinical urgency is primary driver |
| Utilization | 0.25 | Resource efficiency critical for NHS targets |
| No-show risk | 0.15 | Reduces wasted capacity |
| Waiting time | 0.15 | Supports 62-day and 31-day targets |
| Robustness | 0.10 | Schedule stability for operational reliability |
| Travel | 0.05 | Patient convenience (secondary) |

#### 2.2.1 Quick Reference: CP-SAT Six Objectives

Compact one-row-per-objective summary so you don't have to chase the
formulas across §2.2 + §2.3.  $a_p \in \{0, 1\}$ is the assignment
indicator (1 iff patient $p$ is scheduled); $s_p$ is the integer start
time in minutes; the other symbols follow §2.1 + the Global Notation
Table.

| Symbol                  | Objective              | Direction | Default $\lambda_i$ | Mathematical form                                                       | Live in `optimization/optimizer.py`     |
|-------------------------|------------------------|-----------|---------------------|-------------------------------------------------------------------------|------------------------------------------|
| $Z_{\text{priority}}$   | Clinical priority      | Maximise  | $0.30$              | $\sum_p (5 - \text{prio}_p) \cdot 100 \cdot a_p$                        | `optimizer.py:683` (`* w_priority`)      |
| $Z_{\text{util}}$       | Chair utilisation      | Maximise  | $0.25$              | $-\sum_p s_p$                                                            | `optimizer.py:689` (`-pvars['start'] * w_util`) |
| $Z_{\text{noshow}}$     | No-show risk           | Minimise  | $0.15$              | $-\sum_p \lfloor 100 \pi_p \rfloor \cdot a_p$                            | `optimizer.py:702–703` (`-noshow_penalty * w_noshow`) |
| $Z_{\text{wait}}$       | Waiting time           | Minimise  | $0.15$              | $\sum_p \min(\text{days\_waiting}_p, 62) \cdot 5 \cdot a_p$              | `optimizer.py:712` (`waiting_bonus * w_waiting`) |
| $Z_{\text{robust}}$     | Schedule robustness    | Maximise  | $0.10$              | $-\sum_p \max\!\bigl(0,\ \lfloor (d_p - 120)/30 \rfloor\bigr)$           | `optimizer.py:721` (`-duration_risk * w_robust`) |
| $Z_{\text{travel}}$     | Travel distance        | Minimise  | $0.05$              | $-\sum_p \lfloor t_p^{\text{travel}} / 10 \rfloor$                       | `optimizer.py:728` (`-travel_penalty * w_travel`) |

All six $Z_k$ are scaled so that their typical magnitudes sit within one
order of magnitude before weighting (the leading constants $100$, $5$,
$10$, etc. exist only to keep CP-SAT's integer coefficients in a
balanced range).  The composite objective fed to CP-SAT is the
weighted sum

$$
Z(S) = \sum_{k=1}^{6} \lambda_k \cdot Z_k(S),
\qquad
\sum_k \lambda_k = 1,\ \lambda_k \geq 0,
$$

with the weights drawn from `OPTIMIZATION_WEIGHTS` in `config.py:38–45`
or any of the `PARETO_WEIGHT_SETS` profiles (`config.py:48–54`).

The "Direction" column is the maximisation/minimisation intent — every
term is *expressed* with a sign such that the CP-SAT solver always
maximises a single composite ($Z_{\text{util}}$, $Z_{\text{noshow}}$,
$Z_{\text{robust}}$, $Z_{\text{travel}}$ already negate inside the
formula so a smaller $s_p$ / $\pi_p$ / $d_p$ / $t_p^{\text{travel}}$
yields a larger contribution).

### 2.3 Pareto Frontier Generation

Since objectives conflict (e.g., maximizing utilization may reduce robustness), the system generates a **Pareto frontier** by solving with multiple weight vectors:

$$\mathcal{P} = \{Z^*(\lambda) : \lambda \in \Lambda\}$$

Where $\Lambda$ = set of pre-defined weight vectors:

| Profile | $\lambda_{pri}$ | $\lambda_{util}$ | $\lambda_{risk}$ | $\lambda_{wait}$ | $\lambda_{rob}$ | $\lambda_{trav}$ |
|---------|-------|------|------|------|------|------|
| **Balanced** | 0.30 | 0.25 | 0.15 | 0.15 | 0.10 | 0.05 |
| **Max Throughput** | 0.15 | 0.45 | 0.10 | 0.10 | 0.10 | 0.10 |
| **Patient First** | 0.40 | 0.10 | 0.10 | 0.25 | 0.05 | 0.10 |
| **Risk Averse** | 0.20 | 0.15 | 0.30 | 0.10 | 0.20 | 0.05 |
| **Robust** | 0.20 | 0.15 | 0.15 | 0.10 | 0.30 | 0.10 |

#### Pareto Dominance

Solution $A$ dominates solution $B$ ($A \succ B$) if:

$$\forall i: Z_i^A \geq Z_i^B \quad \text{and} \quad \exists j: Z_j^A > Z_j^B$$

The Pareto frontier is the set of all non-dominated solutions:

$$\mathcal{F} = \{A \in \mathcal{P} : \nexists B \in \mathcal{P}, B \succ A\}$$

Decision-makers select from the frontier based on current operational priorities.

> **v4.5 implementation note:** `POST /api/optimize/pareto` now reloads patients directly from `patients.xlsx` and re-runs ML predictions on each call. This guarantees the frontier reflects the current data state even after a regular `POST /api/optimize` clears the pending patient queue. Previous versions returned an error ("No patients to schedule") when called after optimization.

### 2.4 Fairness Constraints

Scheduling must not systematically disadvantage patients based on protected characteristics (Equality Act 2010, NHS Constitution). The optimizer enforces three fairness criteria:

#### Demographic Parity

The scheduling rate should be approximately equal across demographic groups:

$$|P(\text{scheduled} | G = g_1) - P(\text{scheduled} | G = g_2)| \leq \epsilon_{DP}$$

Where $G$ is a protected attribute (age band, gender, distance group) and $\epsilon_{DP} = 0.15$.

**Implementation:** For groups $g_1, g_2$ with sizes $n_1, n_2$ and assigned counts $s_1, s_2$:

$$|s_1 \cdot n_2 - s_2 \cdot n_1| \leq \epsilon_{DP} \cdot n_1 \cdot n_2$$

This cross-multiplication avoids division in the integer CP-SAT solver.

#### Disparate Impact Ratio (Four-Fifths Rule)

The scheduling rate for any disadvantaged group must be at least 80% of the advantaged group:

$$\frac{P(\text{scheduled} | G = g_{minority})}{P(\text{scheduled} | G = g_{majority})} \geq 0.80$$

**Distance-based equity constraint:**

$$\frac{s_{remote}}{n_{remote}} \geq 0.8 \cdot \frac{s_{local}}{n_{local}}$$

Ensures remote patients (>45 min travel) are not disadvantaged relative to local patients.

#### Equal Opportunity

Among patients who would attend (positive outcome), scheduling rates should be equal:

$$|P(\text{scheduled} | Y = 1, G = g_1) - P(\text{scheduled} | Y = 1, G = g_2)| \leq \epsilon_{EO}$$

Where $Y = 1$ indicates the patient attended, $\epsilon_{EO} = 0.10$.

This ensures the system doesn't deny scheduling to reliable patients from disadvantaged groups.

#### Protected Attributes Monitored

| Attribute | Groups | Legal Basis |
|-----------|--------|-------------|
| Age Band | <40, 40-60, 60-75, >75 | Equality Act 2010 (Age) |
| Gender | Male, Female | Equality Act 2010 (Sex) |
| Distance Group | Local (<20min), Medium (20-45min), Remote (>45min) | NHS Constitution (equal access) |
| Deprivation | Via postcode proxy | Health Equity Framework |

> **v4.5 note:** `Person_Stated_Gender_Code` values 0 (Not Known) and 9 (Not Specified) are mapped to 'Unknown' and **excluded** from Equal Opportunity pairwise comparisons. Comparing Unknown vs Male/Female produces spurious violations because 'Unknown' is an administrative category, not a protected characteristic. Demographic Parity is unaffected — it uses only Male and Female groups.

#### Fairness Penalty in Objective

Fairness violations are penalized in the objective function as a soft constraint:

$$Z_{fairness} = -w_{fair} \cdot \sum_{(g_1, g_2)} |s_{g_1} \cdot n_{g_2} - s_{g_2} \cdot n_{g_1}|$$

Where $w_{fair} = 10$ (penalty weight per unit of disparity).

### 2.5 Greedy Fallback Algorithm

When OR-Tools is unavailable, a greedy heuristic is used:

```
Algorithm: Greedy Schedule
Input: Patients P, Chairs C
Output: Schedule S

1. Sort P by (priority ASC, is_urgent DESC, duration ASC)
2. Initialize chair_availability[c] = day_start for all c
3. For each patient p in P:
   a. best_chair = null, best_start = infinity
   b. For each chair c in C:
      - If p.requires_bed AND NOT c.is_bed: continue
      - available = chair_availability[c]
      - If available + p.duration <= day_end:
        - If available < best_start:
          - best_chair = c, best_start = available
   c. If best_chair != null:
      - Add (p, best_chair, best_start) to S
      - chair_availability[best_chair] = best_start + p.duration
   d. Else:
      - Add p to unscheduled
4. Return S
```

---

### 2.10 Warm-Start with Solution Hints

#### 2.10.1 Motivation

CP-SAT solves each scheduling instance from scratch, which takes 5–10 s for a typical 45-chair, 30-patient SACT session.  Because SACT sessions follow strong day-of-week patterns (Monday oncology, Wednesday haematology, etc.), consecutive days of the same weekday share 80–90% of their structural features.  Seeding the solver with a prior feasible assignment exploits this recurrence, reducing solve time by 50–80%.

#### 2.10.2 Instance Fingerprint

A lightweight 5-tuple identifies the structural class of a scheduling instance.  Only **DRO-stable** fields are used — `noshow_probability` and `expected_duration` are intentionally excluded because `_apply_event_adjustment()` and `_apply_dro_robustness()` mutate them in-place before each solve, which would otherwise produce a different fingerprint on the second call for the same patient set, defeating the warm-start cache.

```
FP = (dow, n, (P1, P2, P3, P4), L, n_chairs)
```

| Component | Symbol | Description | DRO-stable? |
|-----------|--------|-------------|-------------|
| Day of week | dow | 0=Mon … 6=Sun; captures protocol mix | ✓ |
| Patient count | n | Direct slot-pressure indicator | ✓ |
| Priority distribution | (P1…P4) | Count per priority tier | ✓ |
| Long-infusion count | L | Recliner demand | ✓ |
| Chair count | n_chairs | Resource configuration | ✓ |

Two instances with the same fingerprint are structurally equivalent and benefit maximally from hint reuse.

#### 2.10.3 Hint Injection

For each patient `p` present in the cached solution:

```python
model.AddHint(assigned_p,  1)
model.AddHint(start_p,     prior['start'])
for c_idx, chair_var in enumerate(chair_vars_p):
    model.AddHint(chair_var, 1 if c_idx == prior['chair_idx'] else 0)
```

CP-SAT treats hints as a **soft initial assignment** — the solver departs freely if hints conflict with constraints.  The warm start only provides a head start on the branch-and-bound tree.

#### 2.10.4 Cache Management

- **Storage:** `ScheduleOptimizer._solution_cache` dict, lives for the Flask process lifetime (module-level singleton).
- **Key:** fingerprint tuple.
- **Value:** `{patient_assignments, timestamp, prior_solve_time, hits}`.
- **Eviction:** LRU-style; when `cache_size ≥ 50`, the entry with the oldest `timestamp` is removed.
- **Monitoring:** `GET /api/optimizer/cache` returns per-fingerprint hit counts, cached patient counts, and prior solve times.

#### 2.10.5 Expected Performance Impact

| Scenario | Expected Speedup |
|----------|-----------------|
| Same weekday, identical patient list | 70–80% solve-time reduction |
| Same weekday, similar patient mix (±3 patients) | 50–65% reduction |
| Different weekday, same fingerprint class | 30–50% reduction |
| Novel fingerprint (cache miss) | 0% — full solve |

For a 10 s baseline solve time, recurring Monday sessions converge in ~2–3 s after the first run.

---

### 2.11 GNN Feasibility Pre-Filter

#### 2.11.1 Motivation

The CP-SAT search space for a 40-patient, 45-chair instance contains 1,800 potential (patient, chair) assignment variables.  Most are structurally infeasible or sub-optimal (long-infusion patient assigned to a non-recliner; P1 patient scheduled on a distant site; very short appointment occupying the only all-day recliner).  Eliminating them before the solver runs shrinks the search tree, yielding 2–5× solve-time reduction.

#### 2.11.2 Graph Structure

Bipartite graph **G = (V_P ∪ V_C, E)**:

| Component | Description | Dim |
|-----------|-------------|-----|
| V_P — patient nodes | priority, duration, noshow_prob, travel_time, long_infusion, is_urgent | 6 |
| V_C — chair nodes | is_recliner, position_norm, recliner_density, non_recliner_density | 4 |
| E — candidate edges | all n_P × n_C pairs (1,800 for 40 × 45) | — |

#### 2.11.3 Message Passing (2 Rounds, Mean Pooling)

```
Round r:
  patient_emb ← concat(patient_emb_prev,  mean_{c ∈ V_C}(chair_emb_prev))
  chair_emb   ← concat(chair_emb_prev,    mean_{p ∈ V_P}(patient_emb_prev))
```

No learned weight matrices — the downstream classifier handles non-linear transformation.  Dimension growth per round:

| Round | patient_dim | chair_dim |
|-------|------------|-----------|
| 0 (raw) | 6 | 4 |
| 1 | 10 | 10 |
| 2 | 20 | 20 |

#### 2.11.4 Edge Feature and Classifier

For each (p, c) pair:

```
x_{pc} = concat(patient_emb_final,   # 20 dims
                chair_emb_final,     # 20 dims
                edge_compat)         # 4 dims: meets_bed_req, site_match,
                                     #         urgency×recliner, dur_fraction
```

**Classifier**: `RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')`

Output: P(patient p assigned to chair c in the optimal CP-SAT solution)

#### 2.11.5 Training (Online, from CP-SAT Solutions)

After each successful solve:
1. Extract solution_assignments = {patient_id → chair_idx}
2. Label all n_P × n_C edges: y_{pc} = 1 if p was assigned to c, else 0
3. Append (X, y) to training buffer
4. Every `train_every` (default 5) solves, refit the RandomForest

Positive rate ≈ 1/n_chairs ≈ 2.2%.  `balanced_subsample` corrects this without discarding negatives.

#### 2.11.6 Pruning and Safety Invariants

Prune pair (p, c) if P(assigned) < `prune_threshold` (default 0.15).

**Three-layer safety**:

1. **Hard-rule layer** (always active, no training required):
   long_infusion patient → non-recliner chair is always pruned
2. **Model layer**: prune if P < threshold
3. **Safety restore**: if any patient drops below `min_viable_chairs` (default 5) valid options, restore the most compatible chairs until the minimum is met

#### 2.11.7 Expected Performance

| Phase | Description | Prune Rate | Speedup |
|-------|-------------|-----------|---------|
| Solves 1–4 | Untrained (hard rules only) | ~30% | ~1.4× |
| Solves 5–19 | Partially trained | 40–55% | ~2× |
| Solves 20+ | Converged model | 55–70% | 2–5× |

#### 2.11.8 API Monitoring

`GET /api/optimizer/gnn` returns:

```json
{
  "is_trained": true,
  "n_solves_seen": 25,
  "lifetime_prune_rate": 0.62,
  "prune_threshold": 0.15,
  "min_viable_chairs": 5,
  "feature_dim": 44
}
```

### 2.12 Column Generation for Large Instances

#### 2.12.1 Motivation

The monolithic CP-SAT formulation (§2.1) creates O(|P| × |C|) binary decision variables and O(|C| × |P|²) no-overlap interval constraints. For Velindre's 45 chairs and typical 30–50 patients, this is tractable (≤2,250 binary variables). At 100+ patients/day the variable count exceeds 4,500 and the constraint interaction space grows quadratically, pushing solve times beyond operational limits.

Column generation (Dantzig-Wolfe decomposition) exploits the problem's **block-diagonal structure**: each chair operates independently once patients are assigned to it. We decompose into a compact master problem that selects bundles and per-chair subproblems that generate new bundles.

#### 2.12.2 Definitions

| Symbol | Meaning |
|--------|---------|
| P | Set of patients, \|P\| = n |
| C | Set of chairs, \|C\| = m (typically 45) |
| K | Set of generated columns (grows iteratively) |
| K_c ⊆ K | Columns for chair c |
| a_{p,k} ∈ {0,1} | Whether patient p appears in column k |
| cost_k | Weighted objective value of column k |
| λ_k ∈ [0,1] | Master LP variable: fraction of column k used |
| π_p | Dual price for patient coverage constraint |
| μ_c | Dual price for chair usage constraint |

A **column** k is a feasible schedule for a single chair: a set of patients with start times that satisfy no-overlap, bed-requirement, and operating-hours constraints.

#### 2.12.3 Column Cost

Each column's cost aggregates per-patient objective terms (matching §2.2):

```
cost_k = Σ_{p: a_{p,k}=1} [ w_pri · (5 - priority_p) · 100
                            - w_ns · π_p^noshow · 100
                            + w_wait · min(days_waiting_p, 62) · 5 ]
```

where w_pri, w_ns, w_wait are the OPTIMIZATION_WEIGHTS scaled by 1000.

#### 2.12.4 Restricted Master Problem (RMP)

```
max   Σ_k  cost_k · λ_k

s.t.  Σ_k  a_{p,k} · λ_k  ≤  1     ∀p ∈ P     [dual: π_p]     ... (coverage)
      Σ_{k ∈ K_c}  λ_k     ≤  1     ∀c ∈ C     [dual: μ_c]     ... (one-per-chair)
      0 ≤ λ_k ≤ 1                                                ... (LP relaxation)
```

Solved with Google GLOP (OR-Tools LP solver). The LP relaxation provides an **upper bound** on the integer optimum.

#### 2.12.5 Pricing Subproblem

For each chair c, find the column with maximum **reduced cost**:

```
rc_c = max_{x}  Σ_p (cost_p − π_p) · x_p  −  μ_c

s.t.  No-overlap on chair c
      If p.long_infusion: chair c must be recliner
      Start + duration ≤ horizon
      x_p ∈ {0,1}
```

This is a single-machine weighted job scheduling problem, solved via a small CP-SAT model with ~n variables (vs. n×m in the monolithic formulation).

If rc_c > ε (tolerance = 10⁻⁴), the new column is added to the master. Otherwise, chair c cannot produce an improving bundle.

#### 2.12.6 Convergence

The CG loop terminates when:
1. **No improving column**: all subproblems return rc ≤ ε, OR
2. **Iteration limit**: 100 iterations reached, OR
3. **Wall-clock budget exhausted**: `time.time() − t0 ≥ time_limit_s` (status `CG_TIME_LIMIT`).

At termination the LP relaxation is optimal (case 1) or near-optimal (cases 2/3). The wall-clock check fires at iteration boundaries *and* mid-iteration during the per-chair pricing pass, so a single slow subproblem cannot blow past the budget by more than one subproblem's runtime. The integer-rounding step always runs after the loop exits, so all three termination paths return a feasible (possibly suboptimal) schedule.

**Why a hard wall-clock guard.** Without the `time_limit_s` parameter the CG loop ran to `max_iterations` regardless of wall time. On a 202-patient cohort with `max_iterations=100` and 14 chairs, total wall time reached $\approx 350$ s for a 2 s budget — a $175\times$ overshoot that defeated the auto-scaling wrapper's per-worker timing guarantees (§A.9). The fix passes the outer `time_limit_seconds` from `_optimize_column_generation` straight into `ColumnGenerator(time_limit_s=...)`; the master loop checks elapsed time every iteration and breaks gracefully. Regression-tested at `tests/test_column_generation.py::test_cg_respects_wall_clock_time_limit` (a 0.1 s budget must terminate within 1 s).

#### 2.12.7 Integer Rounding (Branch-and-Price Lite)

After LP convergence, the solution may be fractional. We round via:

1. **Fix near-integer columns**: λ_k > 0.99 → fix to 1, lock chair and patients.
2. **Restricted CP-SAT**: for remaining (fractional) patients, solve a compact CP-SAT using only chairs and time slots suggested by the LP. This is much smaller than the original monolithic formulation.

#### 2.12.8 Integration with Warm-Start and GNN

| Feature | Integration |
|---------|-------------|
| Warm-start cache (§2.10) | Cached solutions seed initial columns (one column per chair from prior assignment). Eliminates cold-start penalty. |
| GNN pruning (§2.11) | Valid pairs from GNN restrict candidate patients in each subproblem, reducing subproblem size by ~60%. |
| Fingerprint | Same 5-field fingerprint as §2.10; CG solutions cached identically. |

#### 2.12.9 Complexity Analysis

| Metric | Monolithic CP-SAT | Column Generation |
|--------|------------------|-------------------|
| Binary variables | n × m | ~n per subproblem |
| Master LP variables | — | \|K\| (grows iteratively, typically 2-5×m) |
| Subproblem size | — | n variables, single-chair |
| Scalability | ~60 patients | 100+ patients |
| Overhead | None | CG iteration loop |

#### 2.12.10 Activation

Column generation activates automatically when `len(patients) > COLUMN_GEN_THRESHOLD` (default 50). The threshold is configurable in `config.py`.

#### 2.12.11 Reproducible scalability benchmark

`ml/benchmark_column_generation.py` runs both solvers against
`patients.xlsx` for a fixed sequence of cohort sizes and writes one
timed row per (n, solver) pair to `data_cache/cg_benchmark/results.jsonl`.
The dissertation §4.5.1 Table 4.4 reads this JSONL via `dissertation_analysis.R §16`; when the file is absent R falls back to a calibrated empirical model that scales the published constants.  Two structural invariants are locked by `tests/test_column_generation.py`:

* **`test_benchmark_speedup_equals_ratio_of_cells`** — every row's recorded `speedup` equals `cpsat_time_s / cg_time_s` to within rounding.  Regression for the original Table 4.4 inconsistency where the speedup column (4.6×) disagreed with the displayed CP-SAT cell ("timeout") because the speedup was computed from a hidden 23 s CP-SAT time; the Python benchmark now computes the speedup at the measurement site so the dissertation table cells cannot drift.
* **`test_benchmark_timeout_flag_consistent_with_measured_time`** — the `cpsat_timed_out` flag agrees with `cpsat_time_s ≥ time_limit_s − 0.5` (slop for OR-Tools' soft cap).  The dissertation Table 4.4 only renders "timeout (≥ X s)" when this flag is True.

Run manually:

```bash
python -m ml.benchmark_column_generation \
    --patient-counts 30,50,75,100,150 \
    --time-limit-seconds 30 \
    --chairs 45
```

#### 2.12.12 API Monitoring

`GET /api/optimizer/colgen` returns:

```json
{
  "enabled": true,
  "threshold": 50,
  "iterations": 23,
  "columns_generated": 87,
  "lp_bound": 45230.5,
  "status": "CG_OPTIMAL",
  "solve_time": 4.2
}
```

### 2.13 Multi-Objective with Learned Preferences (Inverse RL)

#### 2.13.1 Motivation

The fixed `OPTIMIZATION_WEIGHTS` vector in `config.py` is a designer-chosen point on the Pareto front. In deployment, clinicians frequently override optimiser proposals to reach a different point. Each override is a **preference datum**: the edited schedule `M` dominated the proposal `P` under the clinician's internal utility. We learn this utility directly from revealed preferences instead of asking clinicians to hand-tune weights.

#### 2.13.2 Feature Vector Z(λ)

For any schedule λ we compute the same six un-weighted objective components that enter §2.2:

```
Z(λ) = [ Z_priority,   Z_utilization,   Z_noshow,
         Z_waiting,    Z_robustness,    Z_travel ]  ∈ R^6
```

Every component is signed so that "higher = better" matches the CP-SAT maximisation direction. Implementation: `ml.inverse_rl_preferences.compute_objective_features(patients, assignments)` — the single source of truth shared with `optimizer.compute_schedule_features()`.

#### 2.13.3 Pairwise-Softmax Choice Model

For each override pair i we observe ΔZ_i = Z(M_i) − Z(P_i) and model

```
P(M_i ≻ P_i | θ) = σ( θ · ΔZ_i )
```

where σ is the logistic function (Bradley–Terry / pairwise softmax).

#### 2.13.4 Training Objective

```
θ* = argmax_{θ ≥ 0}   Σ_i  log σ( θ · ΔZ_i )  −  λ ‖θ‖²
```

- **θ ≥ 0** keeps every objective monotone (higher Z_k → more utility).
- **L2 prior (λ = 0.05)** pulls θ toward the fixed `OPTIMIZATION_WEIGHTS` when data are scarce.
- Solver: `scipy.optimize.minimize(L-BFGS-B, bounds=[(0,None)]*6)`.

The programme is strictly convex after the ridge, so it has a unique global optimum; computing 500 L-BFGS-B iterations takes < 50 ms for N ≤ 1,000 pairs.

#### 2.13.5 Scaling Back to Objective Space

Features are standardised by per-component sample SD σ_Z before fitting so the optimiser sees comparable gradients. The learned θ is then inverted to objective scale and renormalised:

```
θ̃_k  =  θ*_k / σ_Z,k
w_k  =  θ̃_k / Σ_j θ̃_j        ← drop-in replacement for OPTIMIZATION_WEIGHTS
```

#### 2.13.6 Bootstrap for Cold Start

When the override log has < 20 real entries, the learner auto-seeds N = 200 synthetic pairs drawn from a latent clinician truth (`BOOTSTRAP_PRIOR` in `ml/inverse_rl_preferences.py`):

| index | objective       | weight |
|-------|-----------------|--------|
| 0     | priority        | 0.35   |
| 1     | utilization     | 0.10   |
| 2     | noshow_risk     | 0.25   |
| 3     | waiting_time    | 0.20   |
| 4     | robustness      | 0.05   |
| 5     | travel          | 0.05   |
| **Σ** |                 | **1.00** |

The vector has exactly six entries (one per objective in `OBJECTIVE_KEYS`) and sums to 1.0; both invariants are asserted at module import time and locked by `tests/test_irl_preferences.py::TestBootstrapPrior` so a silent shape regression cannot desynchronise the dissertation, this document, and the code. Synthetic overrides are flagged (`source="synthetic"`) and are eventually outweighed by real data as the system accumulates history.

#### 2.13.7 Persistence & API

| File | Purpose |
|------|---------|
| `data_cache/irl_overrides.jsonl` | Append-only log of overrides (one event/line) |
| `data_cache/irl_weights_history.jsonl` | One row per fit — feeds `dissertation_analysis.R` |
| `models/inverse_rl_preferences.pkl` | Serialised learner state (θ, σ_Z, last fit metadata) |

Endpoints:

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/irl/status` | GET | Current learner state + active optimiser weights |
| `/api/irl/log_override` | POST | Record a clinician override (features or full schedules) |
| `/api/irl/train` | POST | Fit θ (bootstrap_if_empty + optional apply_to_optimizer) |
| `/api/irl/apply` | POST | Push learned θ into `optimizer.weights` |
| `/api/irl/reset` | POST | Revert to fixed `OPTIMIZATION_WEIGHTS` |
| `/api/irl/overrides` | GET | Recent overrides for UI/review |

Web UI: `viewer.html` gains an **IRL Preferences** tab with refit, apply/reset, a manual override form, and a diagnostics panel (log-likelihood + agreement rate).

#### 2.13.8 Held-out cross-validated agreement (§4.5.3 regression)

`IRLFitResult.mean_agreement = P(θ·ΔZ > 0)` is measured on the training set the fit optimises against — it is trivially close to 1.00 when six parameters of the softmax head are fitted against a small N (the original dissertation run reported 100 % over 22 pairs).  The honest generalisation number is the cross-validated agreement:

* **Leave-one-out CV** for $N \leq 30$ (22 pairs → 22 folds)
* **5-fold CV** for $N > 30$

For each fold the helper `_heldout_agreement(deltas, λ)` re-fits $\theta$ on the train rows (same L-BFGS-B, same ridge, same standardisation built from the train rows only) and records `P(θ·Δz_heldout > 0)`.  The aggregate mean is written to `IRLFitResult.mean_agreement_heldout` with `heldout_method ∈ {"loo", "kfold-5", "none"}` and `heldout_n_folds`; `dissertation_analysis.R §17` surfaces all three via `\irlAgreementHeldout`, `\irlHeldoutMethod`, `\irlHeldoutFolds` so the dissertation §4.5.3 prose cites the cross-validated number alongside the (optimistic) training number.

Invariants locked by `tests/test_irl_preferences.py::TestHeldoutCrossValidation` (4 new tests): LOO is used iff $N \leq 30$, 5-fold iff $N > 30$, `n_folds` matches the method, $N < 5$ returns `None` (no CV attempted), and — for any real fit — $\text{mean\_agreement\_heldout} \leq \text{mean\_agreement}$ (generalisation cannot structurally exceed training).  The R pipeline also loud-warns if the JSONL row violates the generalisation invariant, catching any future regression where the CV driver is accidentally trained on the wrong split.

---

## 3. No-Show Prediction Model

### 3.1 Ensemble Architecture

The model uses **Stacked Generalization** (meta-ensemble) with three base classifiers:

**Level-1 (Base Models):**
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

**Level-2 (Meta-Learner):**

$$\hat{P}_{noshow} = \sigma\left(\beta_0 + \sum_{i=1}^{3} \beta_i \cdot \hat{P}_i + \sum_{i=1}^{3} \sum_{j>i} \beta_{ij} \cdot \hat{P}_i \cdot \hat{P}_j\right)$$

Where:
- $\sigma$ is the sigmoid function
- $\hat{P}_i$ are base model predictions
- $\beta_{ij}$ capture interaction effects between models

**Implementation:**
```python
meta_features = [
    rf_prob, gb_prob, xgb_prob,           # Base predictions
    rf_prob * gb_prob,                     # Interaction terms
    rf_prob * xgb_prob,
    gb_prob * xgb_prob
]
meta_learner = LogisticRegression(C=1.0)
```

**Alternative 1: Bayesian Model Averaging (BMA)**

When stacking is disabled, BMA can be used to automatically adapt weights based on model performance:

$$P(y|x) = \sum_{i=1}^{K} P(y|x, M_i) \cdot P(M_i|D)$$

**Posterior Weight Computation:**

The posterior probability for each model is computed from log-likelihood:

$$P(M_i|D) \propto P(D|M_i) \cdot P(M_i)$$

With uniform prior $P(M_i) = 1/K$, using log-likelihood:

$$LL_i = \sum_{n=1}^{N} \left[ y_n \log(p_{i,n}) + (1-y_n) \log(1-p_{i,n}) \right]$$

$$w_i = \frac{\exp(LL_i / T)}{\sum_{j=1}^{K} \exp(LL_j / T)}$$

Where $T$ is the temperature parameter:
- $T = 1$: Standard BMA (may over-concentrate on single model)
- $T = 10$: Recommended default (balanced weights)
- Higher $T$: More uniform weights

**Implementation (temperature-scaled softmax):**
```python
def compute_bma_weights(y_true, model_probs, temperature=10.0):
    log_likelihoods = {}
    for name, probs in model_probs.items():
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        ll = np.sum(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
        log_likelihoods[name] = ll

    max_ll = max(log_likelihoods.values())
    exp_lls = {name: np.exp((ll - max_ll) / temperature) for name, ll in log_likelihoods.items()}
    total = sum(exp_lls.values())
    return {name: exp_ll / total for name, exp_ll in exp_lls.items()}
```

**Example Output (T=10):**
| Model | Log-Likelihood | BMA Weight |
|-------|----------------|------------|
| Random Forest | -54.68 | 0.60 |
| XGBoost | -65.16 | 0.21 |
| Gradient Boosting | -66.23 | 0.19 |

**Advantage:** Automatically adapts weights based on which model performs best on validation data while maintaining ensemble diversity.

**Alternative 2: Fixed Weights (Fallback):**

$$\hat{P}_{noshow} = \frac{w_{RF} \cdot P_{RF} + w_{GB} \cdot P_{GB} + w_{XGB} \cdot P_{XGB}}{w_{RF} + w_{GB} + w_{XGB}}$$

Where:
- $w_{RF} = 0.4$ (Random Forest weight)
- $w_{GB} = 0.35$ (Gradient Boosting weight)
- $w_{XGB} = 0.25$ (XGBoost weight)

### 3.2 Model Hyperparameters

**Random Forest Classifier:**
- n_estimators: 100
- max_depth: 10
- min_samples_split: 5
- min_samples_leaf: 2

**Gradient Boosting Classifier:**
- n_estimators: 100
- max_depth: 5
- learning_rate: 0.1

**XGBoost Classifier:**
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1

### 3.3 Rule-Based Fallback

When model is not trained, a rule-based prediction is used:

$$P_{noshow} = P_{base} + f_{power}(r) + \sum_{i} w_i \cdot f_i$$

Where $P_{base} = 0.08$ (8% baseline rate).

#### Non-Linear Historical Rate Transformation

The historical no-show rate uses a **power transformation** to capture the non-linear relationship observed in data (high-risk patients have disproportionately higher no-show rates):

$$f_{power}(r) = 0.4r + 0.4r^{1.5}$$

| Historical Rate (r) | Linear (0.4×r) | Power f(r) | Actual Data | Error |
|---------------------|----------------|------------|-------------|-------|
| 0.062 (0 prev) | 0.025 | 0.031 | 11.4% | 4.7% |
| 0.142 (1 prev) | 0.057 | 0.078 | 21.1% | 0.3% |
| 0.198 (2 prev) | 0.079 | 0.114 | 24.8% | 0.4% |
| 0.231 (3 prev) | 0.092 | 0.137 | 26.8% | 0.1% |
| 0.332 (4+ prev) | 0.133 | 0.209 | 34.7% | 0.8% |

This transformation reduces total absolute error from 0.234 to 0.063 (73% improvement).

#### Other Factors

| Factor | Weight | Feature | Condition |
|--------|--------|---------|-----------|
| Long distance | +0.08 | `is_long_distance` | Binary: distance ≥ 30km |
| Medium distance | +0.03 | `is_medium_distance` | Binary: 10km ≤ distance < 30km |
| Weather severity | +0.15 | `weather_severity` | Multiplied by severity (0.0-1.0) |
| Traffic severity | +0.10 | `traffic_severity` | Multiplied by severity (0.0-1.0) |
| New patient | +0.05 | `is_new_patient` | Binary: total_appointments = 0 |
| Long advance booking | +0.05 | `booked_long_advance` | Binary: days_until > 30 |
| Priority adjustment | +0.02 | `priority_level` | (priority - 1) × 0.02 |
| Monday/Friday effect | +0.02 | `is_mon` OR `is_fri` | Binary: either day |
| Age <40 | -0.02 | `age_band` | Younger patients: lower risk |
| Age 40-60 | -0.02 | `age_band` | Middle age: lower risk |
| Age 60-75 | +0.03 | `age_band` | Older patients: higher risk |
| Age >75 | +0.03 | `age_band` | Elderly: highest risk |

Age band adjustments are derived from actual data showing older patients (60+) have ~4-5% higher no-show rates than younger patients.

Final probability is clamped: $P_{noshow} = \min(0.9, \max(0.01, P_{noshow}))$

#### Important Distinction

- **Patient_NoShow_Rate** (stored in data): Historical rate = Previous_NoShows / Total_Appointments
- **Predicted P_noshow** (formula output): Calculated probability for THIS specific appointment

#### Example Calculation (APT101458)

Data from `historical_appointments.xlsx` row 11:
- Patient_ID: P84264
- Patient_NoShow_Rate: 0.433 (6 previous no-shows)
- Travel_Distance_KM: 28km
- Day_Of_Week: Mon
- Priority: P3
- Weather_Severity: 0.0

| Component | Value | Calculation |
|-----------|-------|-------------|
| Base rate | 0.080 | Fixed baseline |
| Historical (power) | 0.287 | 0.4×0.433 + 0.4×0.433^1.5 = 0.173 + 0.114 |
| Medium distance | 0.030 | 10km ≤ 28km < 30km |
| Priority P3 | 0.040 | (3 - 1) × 0.02 |
| Monday effect | 0.020 | is_mon = True |
| Weather | 0.000 | 0.15 × 0.0 |
| Traffic | 0.000 | 0.10 × 0.0 |
| **Total Predicted** | **0.457** | Sum, clamped to [0.01, 0.9] |

**Comparison**:
| Method | Prediction | vs Actual (34.7%) |
|--------|------------|-------------------|
| Linear (0.4×r) | 0.343 | -4.0% error |
| Power (r + r^1.5) | 0.457 | Accounts for high-risk |

The power transformation correctly identifies this patient as very high risk due to their 6 previous no-shows.

### 3.4 Risk Level Classification

$$\text{Risk Level} = \begin{cases}
\text{very\_high} & \text{if } P \geq 0.50 \\
\text{high} & \text{if } P \geq 0.25 \\
\text{medium} & \text{if } P \geq 0.10 \\
\text{low} & \text{otherwise}
\end{cases}$$

> **v4.4:** Boundaries updated from (0.50/0.30/0.15) to (0.50/0.25/0.10) to align with the
> operational double-booking thresholds in §9.4. `medium` now maps directly to the squeeze-in
> MEDIUM tier (double-book with caution); `low` maps to REJECT (do not double-book).

### 3.5 Prediction Confidence

$$\text{Confidence} = C_{base} + C_{history} + C_{location} + C_{external}$$

Where:
- $C_{base} = 0.60$
- $C_{history} = \begin{cases} 0.20 & \text{if } appointments > 10 \\ 0.10 & \text{if } appointments > 5 \\ 0 & \text{otherwise} \end{cases}$
- $C_{location} = 0.10$ if distance known
- $C_{external} = 0.05$ if weather data available

Capped at: $\text{Confidence} = \min(0.95, \text{Confidence})$

### 3.6 RNN Sequence Model for Patient History

For patients with >5 appointments, a recurrent neural network (GRU or LSTM) captures temporal patterns in attendance behavior that tree-based models miss.

**Mathematical Formulation:**

The hidden state evolves with each appointment:

$$h_t = \text{GRU}(x_t, h_{t-1})$$

$$\hat{y}_t = \sigma(W \cdot h_t + b)$$

Where:
- $x_t$ = feature vector for appointment $t$
- $h_t$ = hidden state capturing patient trajectory
- $\hat{y}_t$ = no-show probability prediction

**GRU Update Equations:**

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \quad \text{(update gate)}$$

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \quad \text{(reset gate)}$$

$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b) \quad \text{(candidate)}$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(new state)}$$

**Sequence Features (per appointment):**

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `gap_days` | Days since last appointment | $\min(gap/30, 6)$ |
| `cumulative_attend_rate` | Historical attendance up to $t$ | [0, 1] |
| `attendance_streak` | Consecutive attended | $streak/10$ |
| `noshow_streak` | Consecutive no-shows | $streak/5$ |
| `prev_attended_1,2,3` | Lag features | Binary/0.5 |
| `cycle_number` | Treatment cycle | $\min(cycle/10, 1)$ |
| `day_of_week_sin/cos` | Cyclical encoding | $\sin/\cos(2\pi \cdot dow/7)$ |
| `month_sin/cos` | Seasonal encoding | $\sin/\cos(2\pi \cdot month/12)$ |

**Model Architecture:**

```
Input (19 features) → BatchNorm → GRU (64 hidden, 2 layers) → Dropout(0.3) → Linear → Sigmoid
```

| Parameter | Value |
|-----------|-------|
| Hidden size | 64 |
| Num layers | 2 |
| Dropout | 0.3 |
| Learning rate | 0.001 |
| Batch size | 32 |
| Early stopping patience | 10 |

**Variable-Length Sequence Handling:**

For batches with different sequence lengths, use `pack_padded_sequence`:

```python
x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
gru_out, _ = self.gru(x_packed)
gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
```

**Masked Loss for Padded Positions:**

$$L = \frac{\sum_{t=1}^{T} m_t \cdot BCE(y_t, \hat{y}_t)}{\sum_{t=1}^{T} m_t}$$

Where $m_t = 1$ for valid positions and $m_t = 0$ for padding.

**Ensemble Integration:**

For patients with sufficient history (>5 appointments), combine sequence and tree predictions:

$$\hat{P}_{combined} = (1 - w_{seq}) \cdot \hat{P}_{tree} + w_{seq} \cdot \hat{P}_{seq}$$

Where $w_{seq} = 0.3$ by default (adjustable via `set_sequence_model_weight()`).

**Implementation:**

```python
# Enable sequence model
model = NoShowModel(use_sequence_model=True, sequence_model_type='gru')

# Train with appointment history
model.train(X, y, appointments_df=appointments_df)

# Predict (automatically uses sequence model for patients with >5 appts)
result = model.predict(patient_data, appointment_data)

# Check sequence model stats
stats = model.get_sequence_model_stats()
print(f"Patients with history: {stats['patients_with_history']}")
print(f"Best validation AUC: {stats['best_val_auc']:.3f}")
```

**Expected Improvement:**

| Metric | Tree-Only | With Sequence | Improvement |
|--------|-----------|---------------|-------------|
| AUC-ROC (all patients) | 0.68 | 0.69 | +1% |
| AUC-ROC (>5 appts) | 0.70 | 0.77 | +5-7% |

The sequence model is most beneficial for:
- Patients with rich appointment history
- Detecting behavior changes (e.g., patient becoming unreliable)
- Capturing seasonal patterns in individual patient behavior

### 3.7 Decision-Focused Learning (Smart Predict-then-Optimise)

#### 3.7.1 Motivation

Standard cross-entropy training makes the ensemble **statistically accurate**, but scheduling cost depends on the **decision** taken on the back of the prediction (assign a dedicated slot vs. double-book):

- a probability slightly too low that still sits below the double-booking threshold τ produces the right decision at zero cost;
- a probability slightly too high that flips the decision produces a crowding cost even though the prediction is only marginally miscalibrated.

Decision-focused learning (DFL) closes this gap by training on the scheduling cost itself — the gap flagged by Elmachtoub & Grigas (2022) and by the blackbox-solver-gradient route of Wilder et al. (2019).

#### 3.7.2 Calibration head

A two-parameter head sits on top of the XGBoost / GB / RF ensemble:

```
g(p) = σ( a · logit(p) + b ),    a ≥ 0
```

The non-negativity constraint on `a` keeps the calibration monotone so the ordering of patients by no-show risk (used by the squeeze-in double-booking score in §9.3) is preserved.

#### 3.7.3 Smooth decision-cost surrogate

```
c̃(p, y; τ, β) = (1 − σ_τ(p)) · y · C_waste  +  σ_τ(p) · (1 − y) · C_crowd
σ_τ(p) = σ( β · (p − τ) )          (β = sharpness ≈ 20)
```

- `y ∈ {0, 1}` — observed no-show outcome
- `τ` — double-booking threshold (default 0.40, from `NOSHOW_THRESHOLDS['high']`)
- `C_waste = 100 · w_utilization` — cost of an empty chair when patient no-shows
- `C_crowd = 100 · w_robustness` — cost of crowding when a predicted-risky patient attends

Both cost constants are derived from the current `OPTIMIZATION_WEIGHTS` (fixed prior or learned θ from §2.13) so the DFL loss stays consistent with whatever Pareto point the optimiser is using.

#### 3.7.4 SPO+ gradient

The SPO+ loss of Elmachtoub & Grigas (2022) for a threshold decision reduces exactly to `c̃` above, so the gradient w.r.t. `(a, b)` is analytic:

```
∂c̃/∂a = Σ [(1-y)·C_crowd − y·C_waste] · β · σ_τ(1-σ_τ) · p_cal(1-p_cal) · logit(p_raw)
∂c̃/∂b = Σ [(1-y)·C_crowd − y·C_waste] · β · σ_τ(1-σ_τ) · p_cal(1-p_cal)
```

Fitted by L-BFGS-B with an L2 prior pulling `(a, b)` toward identity `(1, 0)` and explicit box bounds $a \in [10^{-4}, 5.0]$ and $b \in [-3, 3]$ — chosen so the calibrated probability cannot collapse to the saturating ends of the sigmoid (an earlier $b \in [-20, 20]$ bound let the optimiser drive the bias to $\pm 20$ on the synthetic dataset, inflating cross-entropy 5× before the bound was tightened). No CP-SAT call inside the training loop — the Wilder et al. (2019) blackbox-perturbation route is reserved for a future upgrade if the head is lifted to a deeper MLP.

**Predict-vs-decide trade-off.**  The SPO+ surrogate optimises *decision* cost, not log-likelihood, so it is expected — and observed in this implementation — that cross-entropy *increases* slightly while decision regret falls.  Live numbers from the synthetic dataset (1,900 records, latest fit on the per-patient `Patient_NoShow_Rate` column the optimiser actually consumes): regret $10334.34 \to 9298.50$ (+10.0 % improvement), cross-entropy $0.522 \to 0.656$ (1.26× increase from a 13 % base-rate distribution where predictions are centred near the prior); the optimiser sits on the upper bias bound ($b = 3.000$, `bound_active = true`), the early-warning signal that the data wants more upward shift than the bound permits.  This is the predict-vs-decide trade-off discussed in §3.7.1 and is the central justification for SPO+ over plain cross-entropy minimisation.

#### 3.7.5 Integration

| File | Purpose |
|------|---------|
| `ml/decision_focused_learning.py` | `DFLCalibrator`, fit/calibrate/reset, pickle persistence |
| `data_cache/dfl_history.jsonl` | One row per fit — feeds dissertation_analysis.R §18 |
| `models/dfl_calibrator.pkl` | Serialised `(a, b)` + last-fit diagnostics |
| `flask_app.py` `_apply_dfl(p)` | Single injection point — routes every `noshow_model.predict()` through `g(·)` when fitted |

The calibration is invisible to every downstream module — CP-SAT objective, squeeze-in engine, IRL learner, fairness audit — because it writes to the same `patient.noshow_probability` attribute they already read.

Endpoints:

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/ml/dfl/status` | GET | Report `(a, b)` + last-fit diagnostics |
| `/api/ml/dfl/train` | POST | Fit head on historical Attended_Status outcomes |
| `/api/ml/dfl/reset` | POST | Revert to identity calibration |

### 3.8 Temporal Fusion Transformer (joint multi-output)

A single attention-based model that replaces the 3-pillar stack
(tree ensemble + sequence GRU + multi-task MLP) when fitted.

**Inputs:**
- `static ∈ R⁵` — age, gender, priority, postcode band, baseline no-show rate
- `past ∈ R^{T×6}` — last T=10 appointments (attended flag, actual duration, cycle, weather severity, DoW, travel km)

**Architecture:**
```
s → GRN embeddings → VSN(5) ──► static context
p → per-feat GRN → VSN(6) ──► static-enriched LSTM ──► multi-head attention (H=4)
                                                          │
                                                          ▼
                                    ┌─────────────┬──────────────┬─────────────┐
                                    ▼             ▼              ▼             ▼
                              sigmoid(Wₙh)  sigmoid(Wᴄh)   Wq h ∈ R³     attn weights (T)
                              p_noshow      p_cancel       d_q10/50/90
```

**Gated Residual Network** (Lim et al. 2021): `LN( x_skip + GLU( W₂ drop( elu(W₁x + W_c c) ) ) )` — context-gated by static features.

**Variable Selection**: softmax over K feature embeddings; weights retained for interpretability.

**Joint loss:**
```
L = BCE(p̂_ns, y_ns) + λ_c·BCE(p̂_cx, y_cx) + λ_q·Σ_τ pinball(q̂_τ, y_dur, τ)
```
with λ_c = 0.5 and λ_q = 0.01. Pinball = `max(τ(y-q), (τ-1)(y-q))`.

**Integration:**

| File | Purpose |
|---|---|
| `ml/temporal_fusion_transformer.py` | `TFTLite` nn.Module + `TFTTrainer` façade |
| `models/tft_lite.pt` | PyTorch weights (torch.save) |
| `models/tft_lite_meta.pkl` | Architecture + normalisation stats + last-fit diagnostics |
| `data_cache/tft_history.jsonl` | One row per fit (consumed by `dissertation_analysis.R` §20) |
| `flask_app.py` `_apply_tft()` | Single hook after `_apply_dfl()` — overrides `patient.noshow_probability` and `patient.expected_duration` when fitted |

Endpoints:

| Route | Method | Purpose |
|---|---|---|
| `/api/ml/tft/status` | GET | Fit state + diagnostics |
| `/api/ml/tft/train` | POST | Fit on historical_appointments.xlsx |
| `/api/ml/tft/reset` | POST | Drop weights; revert to ensemble + DFL |
| `/api/ml/tft/attention` | POST | Per-patient attention vector + VSN feature importances |

**Native quantiles** — the duration head outputs τ ∈ {0.1, 0.5, 0.9} directly, so the TFT duration predictions carry their own 10/50/90 interval without an external CQR wrapper. The adaptive-α machinery in §3.6 still applies to the tree ensemble conformal pipeline; the TFT simply provides an alternative uncertainty channel.

No UI panel — feature runs invisibly; when TFT is unfitted, the pipeline is bit-identical to the pre-TFT code.

**Head-to-head held-out benchmark.** The `noshow_auc` field in `tft_history.jsonl` is TRAINING-set AUC — the model is evaluated on the same data it trained on, which is optimistic.  The dissertation §4.5.6 honest-comparison claim is instead backed by `ml/benchmark_tft_vs_ensemble.py`, which:

1. filters to patients with ≥ `past_window` prior appointments (the TFT's native eligibility gate),
2. splits 80/20 train/test deterministically by seed,
3. fits a fresh `TFTTrainer` on the train split AND a fresh `GradientBoostingClassifier` on the same train split,
4. evaluates both on the SAME held-out rows,
5. writes one row to `data_cache/ensemble_benchmark/results.jsonl` with `tft_noshow_auc`, `ensemble_noshow_auc`, `n_test`, `seed`, and a `comparison_note`.

`dissertation_analysis.R §20` reads the most recent row with `ensemble_available=True` and emits `\tftHeldoutAUC`, `\ensembleHeldoutAUC`, `\heldoutDeltaAUC`, `\heldoutNtrain`, `\heldoutNtest`, `\tftHeldoutSource` so the dissertation table cells cannot drift from the measurement.  Two `tests/test_temporal_fusion_transformer.py::TestHeadToHeadBenchmarkSchema` tests lock the JSONL schema + AUC invariants against silent regression.  Regression for the §4.5.6 external-review finding that the original prose compared TFT training-set AUC (0.706) against a stale historical literal of 0.635 for the tree ensemble from an unrelated validation split.

Run manually:

```bash
python -m ml.benchmark_tft_vs_ensemble --test-frac 0.20 --seed 42 --epochs 40
```

---

## 4. Duration Prediction Model

### 4.1 Ensemble Architecture

Similar to no-show model, uses weighted regression ensemble:

$$\hat{D} = \frac{w_{RF} \cdot D_{RF} + w_{GB} \cdot D_{GB} + w_{XGB} \cdot D_{XGB}}{w_{RF} + w_{GB} + w_{XGB}}$$

### 4.2 Protocol Base Durations

Durations are C1 (first cycle) values from the REGIMENS table. Regimen codes use **SACT standard protocol abbreviations**:

| SACT Code | Protocol | Cancer Type | C1 (min) | C2 (min) | C3+ (min) | Nursing |
|-----------|----------|-------------|----------|----------|-----------|---------|
| FOLFOX | FOLFOX | Colorectal | 180 | 165 | 150 | 1:3 |
| FOLFIRI | FOLFIRI | Colorectal | 180 | 165 | 150 | 1:3 |
| RCHOP | R-CHOP | Lymphoma | 360 | 300 | 240 | 1:1 |
| DOCE | Docetaxel | Breast/Prostate | 150 | 120 | 90 | 1:4 |
| PACW | Paclitaxel Weekly | Breast/Ovarian | 180 | 150 | 120 | 1:3 |
| CARBPAC | Carboplatin/Paclitaxel | Ovarian/Lung | 240 | 210 | 180 | 1:2 |
| PEMBRO | Pembrolizumab | Various | 60 | 45 | 30 | 1:5 |
| NIVO | Nivolumab | Various | 60 | 45 | 30 | 1:5 |
| TRAS | Trastuzumab | Breast | 120 | 90 | 60 | 1:4 |
| FECT | FEC-D | Breast | 120 | 90 | 90 | 1:3 |
| GEM | Gemcitabine | Pancreatic/Lung | 60 | 45 | 30 | 1:5 |
| CAPOX | CAPOX | Colorectal | 180 | 150 | 120 | 1:3 |
| RITUX | Rituximab Maintenance | Lymphoma | 300 | 240 | 180 | 1:1 |
| CISE | Cisplatin/Etoposide | Lung/Testicular | 300 | 270 | 240 | 1:1 |
| IPNIVO | Ipilimumab/Nivolumab | Melanoma | 120 | 90 | 60 | 1:2 |
| ZOLE | Zoledronic Acid | Bone Metastases | 30 | 20 | 15 | 1:6 |
| AC | Doxorubicin/Cyclophosphamide | Breast | 90 | 75 | 60 | 1:4 |
| BEVA | Bevacizumab Maintenance | Colorectal/Ovarian | 90 | 60 | 30 | 1:5 |
| PEME | Pemetrexed/Carboplatin | Lung | 120 | 90 | 75 | 1:3 |
| VINO | Vinorelbine | Breast/Lung | 30 | 20 | 15 | 1:6 |

### 4.3 Cycle-Based Duration Adjustment

$$D_{protocol}(c) = \begin{cases}
D_{C1} & \text{if } c = 1 \quad \text{(First cycle)} \\
D_{C2} & \text{if } c = 2 \quad \text{(Second cycle)} \\
D_{C3+} & \text{if } c \geq 3 \quad \text{(Subsequent cycles)}
\end{cases}$$

Example (R-CHOP):
- $D_{C1} = 360$ minutes
- $D_{C2} = 300$ minutes
- $D_{C3+} = 240$ minutes

### 4.4 Rule-Based Duration Prediction

$$\hat{D} = D_{base} + A_{first} + A_{complexity} + A_{new} + A_{weather} + A_{traffic} + A_{time}$$

**Note:** If a cycle-adjusted `Planned_Duration` is provided in appointment data, it is used directly as $D_{base}$ and $A_{first}$ is not applied (to avoid double-counting).

Where adjustments are:

| Adjustment | Formula | Notes |
|------------|---------|-------|
| First cycle | $A_{first} = 37 \cdot \mathbb{1}_{first\_cycle}$ | Only if $D_{base}$ not cycle-adjusted |
| Complexity | $A_{complexity} = 20 \cdot \mathbb{1}_{complexity > 2}$ | High complexity treatments |
| New patient | $A_{new} = 15 \cdot \mathbb{1}_{appointments < 3}$ | Setup time for new patients |
| Weather | $A_{weather} = 10 \cdot \mathbb{1}_{severity > 0.3}$ | Weather delays |
| Traffic | $A_{traffic} = 5 \cdot \mathbb{1}_{delay > 15}$ | Traffic delays |
| Time of day | $A_{time} = 5 \cdot \mathbb{1}_{hour < 9 \lor hour > 16}$ | Off-peak hours |

The +37 minute first cycle adjustment is derived from actual data analysis showing C1 averages 152 min vs C2+ at 115 min (difference: 37 min).

### 4.5 Confidence Interval

The 95% confidence interval is calculated as:

$$CI = [\hat{D} - 1.96\sigma, \hat{D} + 1.96\sigma]$$

Where $\sigma$ combines multiple independent variance sources:

$$\sigma = \sqrt{\sigma_{model}^2 + \sigma_{protocol}^2}$$

**Variance Components:**

| Component | Source | Description |
|-----------|--------|-------------|
| $\sigma_{model}$ | Ensemble disagreement | Weighted variance of model predictions |
| $\sigma_{protocol}$ | Historical data | Protocol-specific variance from training data |

**Protocol-Specific Historical Variance:**

| Protocol | Name | $\sigma_{protocol}$ |
|----------|------|---------------------|
| RCHOP | R-CHOP | 63 min (long, high variance) |
| CISE | Cisplatin/Etoposide | 45 min |
| FOLFOX | FOLFOX | 23 min |
| PEMBRO | Pembrolizumab | 13 min (short, low variance) |
| ZOLE | Zoledronic Acid | 6 min (very short) |

**Example CI Calculation (R-CHOP, 360 min planned):**
- $\sigma_{protocol}$ = 63 min (from historical data)
- $\sigma_{residual}$ = 21 min (average residual after protocol adjustment)
- $\sigma$ = $\sqrt{63^2 + 21^2}$ = 66.4 min
- 95% CI = [360 - 1.96×66.4, 360 + 1.96×66.4] = [229, 490] min

This approach provides protocol-appropriate confidence intervals:
- High-variance protocols (R-CHOP): wider CIs reflect actual uncertainty
- Low-variance protocols (Pembrolizumab): narrower CIs are more precise

### 4.6 Variance Classification

Thresholds calibrated to actual data variance:

$$\text{Variance Level} = \begin{cases}
\text{low} & \text{if } \sigma < 20 \text{ min} \\
\text{medium} & \text{if } 20 \leq \sigma < 45 \text{ min} \\
\text{high} & \text{if } \sigma \geq 45 \text{ min}
\end{cases}$$

---

## 5. Feature Engineering

### 5.0 SACT v4.0 Data Normalisation Pipeline (v4.3)

Before feature extraction, all DataFrames — whether from real SACT v4.0 CSVs or the synthetic dataset — pass through a unified adapter (`SACTv4DataAdapter.adapt()`) that produces identical ML features from both sources:

```python
# ml/feature_engineering.py — FeatureEngineer.prepare_dataframe_for_ml()
if len(sact_overlap) >= 3:          # SACT v4 fields detected
    df = _sact_adapter.adapt(df)    # normalise real → internal names + derive features
else:
    pass                            # synthetic already in internal format
```

**11-step adapter pipeline:**

| Step | Operation | Output |
|------|-----------|--------|
| 1 | Rename real SACT column aliases | Canonical field names |
| 2 | Derive BSA from H×W (DuBois, H in cm) | `BSA` |
| 3 | Derive age from `Person_Birth_Date` | `Age`, `Age_Band` |
| 4 | Derive lead time from DDT → appointment | `Days_To_Appointment` |
| 5 | Derive scheduling priority from PS + intent | `Priority` (P1–P4) |
| 6 | Derive travel band from postcode district | `Travel_Time_Min` |
| 7 | Fill missing ICD-10 from regimen code | `Primary_Diagnosis_ICD10` |
| 8 | Fill all missing fields from `DEFAULTS` dict | Sensible evidence-based defaults |
| 9 | Encode intent/context as numeric flags | `Is_Curative`, `Is_Neoadjuvant` |
| 10 | Derive no-show risk features | `Patient_NoShow_Rate`, `Complexity_Factor` |
| 11 | Tag data source | `Data_Source` = 'real_sact'/'synthetic' |

**Synthetic dataset SACT v4.0 alignment (v4.6, April 2026):**

| File | Score | Grade | Missing Mandatory | Missing Optional | Rows | Columns |
|------|-------|-------|-------------------|------------------|------|---------|
| patients.xlsx | **91.4 / 100** | **A** | 0 | 8 (appointment-level fields; N/A for patient registry) | 250 | 82 |
| historical_appointments.xlsx | **100.0 / 100** | **A** | 0 | 0 | 1,899 | 102 |
| appointments.xlsx | **100.0 / 100** | **A** | 0 | 0 | 819 | 70 |

The 8 missing optional fields in patients.xlsx are section-0 scheduling fields (`Appointment_ID`, `Date`, `Planned_Duration`, `Actual_Duration`, `Attended_Status`, `Site_Code`, `Chair_Number`, `Weather_Severity`) that are appointment-level — they do not belong in a patient registry and are correctly present in appointments.xlsx and historical_appointments.xlsx.

Validated live via `GET /api/data/sact-v4/validate`.

### 5.1 Feature Categories

The system creates features across six categories:

#### 5.1.1 History Features (Patient PDF Fields 20, 21)

$$f_{noshow\_rate} = \frac{\text{Previous\_NoShows}}{\max(1, \text{Total\_Appointments})}$$

$$f_{cancel\_rate} = \frac{\text{Previous\_Cancellations}}{\max(1, \text{Total\_Appointments})}$$

#### 5.1.2 Temporal Features (PDF Fields 2, 12, 18)

| Feature | Formula |
|---------|---------|
| Hour slot encoding | One-hot for 6 time slots |
| Day of week encoding | One-hot for Mon-Fri |
| Season encoding | One-hot for Winter/Spring/Summer/Autumn |
| Booking lead time | $\text{Appointment\_Date} - \text{Booked\_Date}$ |

#### 5.1.3 Geographic Features (PDF Field 10)

$$f_{travel} = \text{Travel\_Time\_Min}$$

Travel time categories (calibrated against pseudonymised Velindre patient travel data, n=5,116 patients; distances real, identifiers synthetic — source: `prepare doc/Patient Data ANONYMISED.csv`):

| Band | Threshold | Synthetic distribution (v4.5) | NHS empirical target |
|------|-----------|-------------------------------|---------------------|
| Near | < 20 min | ~38% | ~40% |
| Medium | 20–45 min | ~60% | ~57% |
| Remote | > 45 min | ~2.4% | ~3% |

> **v4.5:** Postcode sampling weights (`POSTCODE_WEIGHTS` in `generate_sample_data.py`) recalibrated to match the NHS empirical distribution. Previous uniform sampling over 25 postcodes produced 8% Remote — 2.7× the real rate. The recalibration reduces spurious Distance_Group fairness violations driven by data artefact rather than scheduling behaviour.

#### 5.1.4 Age Band Features (PDF Field 14)

One-hot encoding for age bands: <40, 40-60, 60-75, >75

#### 5.1.5 Clinical Features (PDF Fields 15, 16, 17)

Binary features:
- has_comorbidities
- iv_access_difficulty
- requires_1to1_nursing

#### 5.1.6 Treatment Features (PDF Fields 4, 5, 6, 7, 9)

| Feature | Description |
|---------|-------------|
| Protocol type | Immunotherapy vs Chemo combo |
| Cycle number | First, early (1-3), later |
| Treatment day | Day 1, multi-day |
| Duration category | Short (<45), Long (>120), Very Long (>180) |
| Resource type | Chair vs Bed |

### 5.2 Feature Count

Total engineered features: **60+** including:
- 12 history features
- 18 temporal features
- 12 geographic features
- 5 age features
- 3 clinical features
- 10+ treatment features

---

## 6. Travel Time Estimation

### 6.1 Haversine Distance Formula

For two points $(lat_1, lon_1)$ and $(lat_2, lon_2)$:

$$a = \sin^2\left(\frac{\Delta lat}{2}\right) + \cos(lat_1) \cdot \cos(lat_2) \cdot \sin^2\left(\frac{\Delta lon}{2}\right)$$

$$c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right)$$

$$d = R \cdot c$$

Where $R = 6371$ km (Earth's radius).

### 6.2 Travel Time Calculation

$$t_{travel} = \frac{d}{v_{avg}} \times 60$$

Where:
- $d$ = distance in km
- $v_{avg} = 40$ km/h (average speed assumption)
- Result in minutes

Bounds: $t_{travel} = \max(10, \min(120, t_{travel}))$

---

## 7. Risk Scoring

### 7.1 Combined External Severity

$$S_{external} = 0.3 \cdot S_{weather} + 0.4 \cdot S_{traffic} + 0.3 \cdot S_{event}$$

Where each severity $S \in [0, 1]$.

### 7.2 Weather Severity Mapping

| Condition | Severity |
|-----------|----------|
| Clear | 0.0 |
| Partly Cloudy | 0.0 |
| Cloudy | 0.05 |
| Light Rain | 0.1 |
| Rain | 0.2 |
| Heavy Rain | 0.35 |
| Fog | 0.15 |
| Snow | 0.5 |
| Ice | 0.6 |
| Storm | 0.7 |

### 7.3 No-Show Probability in Historical Data

For generating realistic training data:

$$P_{noshow} = P_{base} + 0.3 \cdot S_{weather} + 0.05 \cdot \mathbb{1}_{monday} + 0.03 \cdot \mathbb{1}_{friday} + 0.1 \cdot \frac{d}{100} - 0.1 \cdot \mathbb{1}_{first\_cycle}$$

Clamped to $[0, 0.6]$.

---

## 8. Performance Metrics

### 8.1 Chair Utilization

$$U_{chair} = \frac{\sum_{a \in A} d_a}{H \times |C|}$$

Where:
- $A$ = scheduled appointments
- $d_a$ = duration of appointment $a$
- $H$ = operating hours in minutes
- $|C|$ = number of chairs

### 8.2 Model Evaluation Metrics

For classification (no-show prediction):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{AUC-ROC} = \int_0^1 TPR(FPR^{-1}(x)) \, dx$$

For regression (duration prediction):

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

### 8.3 Cross-Validation

Two cross-validation strategies are available:

**Standard Stratified CV** (default):
5-fold stratified cross-validation with random splits:

$$\text{CV Score} = \frac{1}{5} \sum_{k=1}^5 \text{Score}_k$$

**Temporal Cross-Validation** (recommended for production):
Medical scheduling patterns evolve over time (seasonal flu, new protocols, changing demographics). Temporal CV respects time ordering:

```
Standard CV:    [Random Mix of All Data] → Splits randomly
Temporal CV:    [Past] → Train | [Future] → Validate
```

**Walk-Forward Validation:**

For each fold $k$:
- Train on data from $t_0$ to $t_k$
- Validate on data from $t_k$ to $t_{k+1}$
- Expand training window and repeat

```python
# Implementation
from sklearn.model_selection import TimeSeriesSplit

model = NoShowModel(use_temporal_cv=True)
metrics = model.train(X, y, date_column='Appointment_Date')

# Or run walk-forward validation
wf_results = model.walk_forward_validation(X, y, date_column='Appointment_Date', n_splits=5)
```

**Example Walk-Forward Results:**

| Fold | Train Period | Test Period | AUC-ROC |
|------|--------------|-------------|---------|
| 1 | Jan 2023 - Jun 2024 | Jul-Dec 2024 | 0.621 |
| 2 | Jan 2023 - Dec 2024 | Jan-Jun 2025 | 0.634 |
| 3 | Jan 2023 - Jun 2025 | Jul-Dec 2025 | 0.648 |

**Rationale:** Temporal CV typically shows lower but more realistic performance estimates because it captures how well the model generalizes to future data patterns.

$$\text{CV Std} = \sqrt{\frac{1}{K} \sum_{k=1}^K (\text{Score}_k - \text{CV Score})^2}$$

---

## 9. Urgent Patient Insertion

### 9.1 Problem Statement

When an urgent patient needs to be scheduled into an already-full day, the system must find the optimal slot that:
1. Minimizes disruption to existing patients
2. Maximizes the probability of successful treatment
3. Considers no-show predictions for intelligent double-booking

### 9.2 Strategy Priority Order

The system uses a three-tier strategy for urgent patient insertion:

| Priority | Strategy | Description | Risk Level |
|----------|----------|-------------|------------|
| 1 | **Gap-Based** | Find natural gaps in schedule | Lowest |
| 2 | **Double-Booking** | Book alongside slot with P_noshow ≥ 0.10 | Medium-High† |
| 3 | **Rescheduling** | Bump lower-priority patient | Highest |

† *Risk increases at lower probability thresholds. Mitigated by CVaR overrun scoring (§9.3) and robustness alerts (§9.7).*

### 9.3 Double-Booking Score Calculation

For each existing appointment, calculate a double-booking suitability score:

$$S_{double} = 50 \cdot P_{noshow} + 5 \cdot (priority - 1) + 5 + S_{duration} + S_{time}$$

Where:
- $P_{noshow}$ = No-show probability from ML model (0-1)
- $priority$ = Clinical priority (1-4, lower = higher priority)
- $S_{duration}$ = Duration bonus (15 if ≥120min, 10 if ≥90min, 5 if ≥60min)
- $S_{time}$ = Time bonus (10 if ≤9am, 7 if ≤11am, 4 if ≤2pm, 0 otherwise)

Only appointments with $P_{noshow} \geq 0.10$ (THRESHOLD_LOW) are evaluated; appointments below this floor receive a score of zero and are excluded from double-booking consideration. At the LOW tier (0.10–0.15), the 50·P component contributes 5.0–7.5 to the score, ensuring gap-based and HIGH-tier candidates always rank higher.

### 9.4 No-Show Probability Thresholds

| Confidence Level | Threshold | Action |
|------------------|-----------|--------|
| HIGH | P ≥ 0.25 | Safe to double-book |
| MEDIUM | 0.15 ≤ P < 0.25 | Double-book with caution |
| LOW | 0.10 ≤ P < 0.15 | Only if no other options |
| REJECT | P < 0.10 | Do not double-book |

> **v4.4 note:** Thresholds recalibrated from (0.30/0.20/0.15) to (0.25/0.15/0.10) after ML ensemble
> retrained on the regenerated SACT v4.0 dataset (102 columns, corrected BSA). Model now predicts
> 13–14% no-show probability for synthetic patients — the previous 0.15 floor rejected every candidate.

### 9.5 Algorithm Flow

```
Algorithm: Urgent Patient Insertion (squeeze_in_with_noshow)
Input: urgent_patient, existing_schedule, patient_data_map
Output: SqueezeInResult

1. STRATEGY 1 - Gap Search:
   gaps = find_squeeze_in_options(urgent_patient, existing_schedule)
   if gaps is not empty:
       return best gap option (strategy="gap")

2. STRATEGY 2 - No-Show Based Double-Booking:
   if noshow_model available:
       for each appointment in existing_schedule:
           p_noshow = noshow_model.predict(appointment)
           if p_noshow >= THRESHOLD_LOW:   # 0.10 (v4.4, was 0.15)
               calculate double_book_score
               add to candidates

       sort candidates by double_book_score (descending)
       if best candidate exists:
           return double-booking option (strategy="double_booking")

3. STRATEGY 3 - Rescheduling:
   if allow_rescheduling:
       lower_priority = find appointments with priority > urgent.priority
       if lower_priority exists:
           return reschedule option (strategy="rescheduling")

4. Return failure (strategy="none")
```

### 9.6 Implementation

```python
from optimization.squeeze_in import SqueezeInHandler

# Initialize with no-show model
handler = SqueezeInHandler(chairs=chairs, noshow_model=model)

# Patient data for predictions
patient_data_map = {
    'P001': {'patient_id': 'P001', 'total_appointments': 10, 'no_shows': 2, ...},
    'P002': {'patient_id': 'P002', 'total_appointments': 5, 'no_shows': 0, ...},
}

# Find best slot for urgent patient
result = handler.squeeze_in_with_noshow(
    patient=urgent_patient,
    existing_schedule=existing_schedule,
    patient_data_map=patient_data_map,
    allow_double_booking=True,
    allow_rescheduling=False,
    external_data={'weather': {'severity': 0.2}}
)

# Check result
if result.success:
    print(f"Strategy: {result.strategy_used}")
    print(f"Time: {result.appointment.start_time}")
    if result.strategy_used == "double_booking":
        print(f"No-show probability: {result.noshow_probability:.1%}")
        print(f"Confidence: {result.confidence_level}")
```

### 9.7 Overbooking Recommendations

For proactive capacity management, get daily double-booking recommendations:

```python
recommendations = handler.recommend_double_booking(
    existing_schedule=schedule,
    patient_data_map=patient_data,
    target_slots=3  # Number of slots to recommend
)

# Output:
# {
#   'recommended_slots': [
#     {'patient_id': 'P042', 'start_time': '10:30', 'noshow_probability': '35%', ...},
#     {'patient_id': 'P018', 'start_time': '14:00', 'noshow_probability': '28%', ...},
#   ],
#   'summary': {
#     'high_confidence': 1,
#     'medium_confidence': 2,
#     'expected_noshows': 0.92
#   }
# }
```

### 9.8 Robustness-Aware Insertion Scoring

When inserting urgent patients, the system considers schedule robustness to prevent cascading delays.

#### 9.8.1 Remaining Slack Calculation

For insertion between appointments $p_{\text{prev}}$ and $p_{\text{next}}$:

$$\text{Slack}_{\text{before}} = s_{\text{new}} - (s_{\text{prev}} + d_{\text{prev}})$$

$$\text{Slack}_{\text{after}} = s_{\text{next}} - (s_{\text{new}} + d_{\text{new}})$$

$$\text{Remaining\_Slack} = \min(\text{Slack}_{\text{before}}, \text{Slack}_{\text{after}})$$

#### 9.8.2 Robustness Score Function

$$S_{\text{robustness}} = \begin{cases} -15 & \text{if remaining\_slack} < 10 \\ -5 & \text{if } 10 \leq \text{remaining\_slack} < 20 \\ 0 & \text{if } 20 \leq \text{remaining\_slack} < 60 \\ +10 & \text{if remaining\_slack} \geq 60 \end{cases}$$

#### 9.8.3 Total Insertion Score

$$S_{\text{total}} = S_{\text{base}} + S_{\text{robustness}} + S_{\text{priority}}$$

Where:
- $S_{\text{base}}$ = no-show + duration + time-of-day score
- $S_{\text{priority}}$ = clinical priority score (P1 highest)

#### 9.8.4 Robustness Impact Metric

$$\text{Robustness\_Impact} = \text{Slack}_{\text{before}} + \text{Slack}_{\text{after}} - 2 \cdot \text{Remaining\_Slack}$$

This measures total buffer consumed by the insertion. Higher values indicate more disruption to schedule flexibility.

#### 9.8.5 Schedule Robustness Score

For the entire schedule:

$$R(S) = \frac{1}{|G|} \sum_{g \in G} f(\text{Slack}_g)$$

Where $f(\text{slack})$ is:

$$f(s) = \begin{cases} 0 & s < 10 \\ 0.3 & 10 \leq s < 20 \\ 0.7 & 20 \leq s < 60 \\ 1.0 & s \geq 60 \end{cases}$$

#### 9.8.6 Multi-Objective Optimization with Robustness

The initial schedule creation includes robustness as an objective:

$$\max Z = \sum_{p \in P} \left[ w_{\text{priority}}(p) - w_{\text{noshow}} \pi_p - w_{\text{early}} s_p \right] + w_{\text{robustness}} \cdot R(S)$$

Where $w_{\text{robustness}} = 0.10$ (10% weight on robustness).

#### 9.8.7 Robustness Alert System

After squeeze-in insertion, the system generates alerts:

| Remaining Slack | Alert Level | Action |
|-----------------|-------------|--------|
| < 5 min | **CRITICAL** | High cascade risk - consider alternatives |
| < 10 min | **CRITICAL** | Tight buffer - monitor closely |
| 10-20 min | **WARNING** | Moderate risk - watch for delays |
| > 20 min | None | Acceptable buffer |

### 9.9 Multi-Candidate Gap Sampling and UI Ordering

#### 9.9.1 Multi-candidate gap sampling

Earlier versions of the squeeze-in engine generated one candidate per
qualifying gap, placed at the earliest-fit start time. Combined with a
sharp `S_priority` bonus for hours ≤ 9, this caused the top-N display
to collapse onto 08:00 across every chair (see
`prepare doc/screenshot/Screenshot 2026-04-18 012631.png`).

The current engine generates a candidate at every 30-minute offset
within the **start-of-day** and **end-of-day** gaps. Mid-day gaps are
still sampled once at the earliest-fit position (they are typically
tight, with only one sensible start). `S_priority` was replaced with
a shallow triangular bell peaked at 11:00 (magnitude ≤ 3) so gap size,
CVaR and robustness remain decisive.

Result: on the default synthetic dataset the number of returned
candidates rises from ~36 to ~95 and the top-10 now spans 5+ unique
start times instead of clustering on a single minute.

#### 9.9.2 Two-stage display pipeline

Both squeeze-in panels in the web viewer share the same pattern:

```
(score or risk) desc  ─►  top-N filter  ─►  chronological re-sort  ─►  UI
```

- **Left panel (Found Best Slots)** — score picks the top-10, then
  the survivors are re-sorted by start-time.
- **Right panel (High No-Show Slots)** — risk-descending, then a
  30-minute bucket cap of 2 diversifies ties, then chronological sort.

Score and risk values remain on every row, so the ranking signal is
preserved while the panels read as a time-ordered walk through the day.

#### 9.9.3 Operator-forced manual insert

`POST /api/urgent/insert` accepts an optional `forced_slot` payload:

```json
{"forced_slot": {"chair_id": "WC-C03", "start_time": "11:30"}}
```

When present, the engine's own selection is bypassed and the urgent
patient is pinned into exactly that chair and minute. The UI exposes
this through a per-row **"Insert here"** button inside the scrollable
Found-slots list. The response carries `strategy: manual_operator_selected`
for audit-log traceability.

#### 9.9.4 Channel-aware Patient-ID enforcement

When the active data channel is **Channel 2 (real hospital data)**,
`/api/urgent/insert` rejects placeholder patient identifiers at the
server boundary:

```
supplied_id == ''  OR  supplied_id startswith ('URGENT_', 'PREVIEW', 'TEST')
    AND  app_state.active_channel == 'real'
    ⇒  HTTP 400 + explicit 'real identifier required' error
```

The frontend mirrors this check: on Ch2 a `prompt()` asks for a real
NHS Number / Trust Local ID before the request is fired; on Ch1
(synthetic) an empty field is filled silently with
`URGENT_<YYYYMMDDHHMMSS>`. The dual-layer enforcement (JS UX + server
authoritative) means a tampered client cannot bypass the rule.

---

## Implementation Files

### Core Models

| Component | File | Key Functions |
|-----------|------|---------------|
| Optimization | `optimization/optimizer.py` | `optimize()`, `_optimize_cpsat()` |
| Squeeze-In | `optimization/squeeze_in.py` | `squeeze_in_with_noshow()`, `find_high_noshow_slots()` |
| No-Show Model | `ml/noshow_model.py` | `predict()`, `train()`, `_ensemble_predict()` |
| Sequence Model | `ml/sequence_model.py` | `SequenceNoShowModel`, `PatientGRU`, `PatientLSTM` |
| Duration Model | `ml/duration_model.py` | `predict()`, `train()` |
| Feature Engineering | `ml/feature_engineering.py` | `create_patient_features()` |

### Advanced ML Models

| Component | File | Key Functions |
|-----------|------|---------------|
| Survival Analysis | `ml/survival_model.py` | `CoxProportionalHazards`, `predict_survival()` |
| Uplift Modeling | `ml/uplift_model.py` | `UpliftModel`, `recommend_intervention()` |
| Multi-Task Learning | `ml/multitask_model.py` | `MultiTaskNetwork`, `predict_joint()` |
| Quantile Forest | `ml/quantile_forest.py` | `QuantileRegressionForest`, `predict_quantiles()` |
| Hierarchical Bayesian | `ml/hierarchical_model.py` | `HierarchicalBayesianModel`, `fit()`, `predict()` |
| Causal Inference | `ml/causal_model.py` | `SchedulingCausalModel`, `CausalDAG`, `DoubleMachineLearning` |
| Event Impact | `ml/event_impact_model.py` | `EventImpactModel`, `SentimentAnalyzer` |
| Conformal Prediction | `ml/conformal_prediction.py` | `ConformalPredictor`, `ConformizedQuantileRegression` |
| MC Dropout | `ml/mc_dropout.py` | `MonteCarloDropout`, `predict_with_mc_dropout()` |

### Explainability & Transparency (v4.0)

| Component | File | Key Functions |
|-----------|------|---------------|
| Sensitivity Analysis | `ml/sensitivity_analysis.py` | `SensitivityAnalyzer`, `local_sensitivity()`, `global_importance()` |
| Model Cards | `ml/model_cards.py` | `ModelCardGenerator`, `generate_card()`, `generate_all_cards()` |
| Causal Validation | `ml/causal_validation.py` | `CausalValidator`, `run_all_tests()` (7 tests) |
| Fairness Audit | `ml/fairness_audit.py` | `FairnessAuditor`, `audit()`, Four-Fifths Rule |

### Uncertainty-Aware Optimization (v4.1)

| Component | File | Key Functions |
|-----------|------|---------------|
| DRO + CVaR | `optimization/uncertainty_optimization.py` | `UncertaintyAwareOptimizer`, `compute_robust_parameters()` |
| DRO in Optimizer | `optimization/optimizer.py` | `_apply_dro_robustness()`, CVaR aux vars in `_optimize_cpsat()` |
| DRO in Squeeze-In | `optimization/squeeze_in.py` | `S_uncertainty` component, CVaR overrun scoring |

### Auto-Learning Pipeline (v3.0)

| Component | File | Key Functions |
|-----------|------|---------------|
| Online Learning | `ml/online_learning.py` | `OnlineLearner`, SGD + Bayesian + EMA updates |
| RL Scheduler | `ml/rl_scheduler.py` | `SchedulingRLAgent`, Q-Learning, 8400 states |
| NHS Data Ingestion | `data/nhs_data_ingestion.py` | `NHSDataIngester`, `check_and_download_all()` |
| Auto-Recalibration | `ml/auto_recalibration.py` | `ModelRecalibrator`, `execute_recalibration()` |
| Drift Detection | `ml/drift_detection.py` | `DriftDetector`, `compute_psi()`, `ks_test()` |
| Data Generation | `datasets/generate_sample_data.py` | SACT v4.0 compliant, 102 columns (historical_appointments), Grade A 100/100 |
| Flask API | `flask_app.py` | 100+ endpoints, viewer + auto-learning APIs |
| Feature Store (§3.1) | `ml/feature_store.py` | `FeatureStore`, `FeatureView`, `get_online_features`, `as_of` (PIT correct) |

---

## Appendix A. Advanced Operational Modules

This appendix expands on §9 with the production-grade subsystems shipped
with v5.0.  Each module is a self-contained add-on to the prediction
pipeline (no UI panel; status visible only via `/api/*/status`,
`/health/*`, and Prometheus `/metrics`).  Sections are kept here rather
than interleaved with the core §9 material so a reader following the
main numerical hierarchy is not interrupted by operational concerns.

---

### A.1 Online Feature Store (§3.1 — Streaming ML)

#### A.1.1 Motivation

The legacy pipeline reloads `historical_appointments.xlsx` and recomputes 80+ features on every restart. §3.1 demands streaming ML — real-time feature updates, low-latency serving, versioned features, and point-in-time correctness.

#### A.1.2 Architecture

```
┌──────────────────────────────┐   append
│  historical_appointments.xlsx│─────────┐
└──────────────────────────────┘         ▼
                               ┌────────────────────┐
                               │   events.jsonl     │  (offline, append-only)
                               └────────────────────┘
                                     │
                                     ▼
                       ┌──────────────────────────┐
                       │  FeatureView compute fn  │
                       │  (30d stats, 90d stats,  │
                       │   cycle_ctx, trend)      │
                       └──────────────────────────┘
                                     │
                                     ▼
                       ┌──────────────────────────┐
                       │   online dict (pickle)   │  ← low-latency serving
                       └──────────────────────────┘
                                     │
                                     ▼
          ┌────────────────────────────────────────┐
          │  GET /api/features/online/<patient>    │  (<100 ms)
          │  POST /api/features/online/as_of       │  (PIT correct)
          │  POST /api/features/store/push_event   │  (streaming write)
          └────────────────────────────────────────┘
```

#### A.1.3 Point-in-time (PIT) correctness

For training labels generated at time `T`:

```
as_of(patient, T) = FeatureView.compute(
    [e for e in events(patient) if occurrence_time(e) <= T]
)
```

`occurrence_time(e)` is the actual appointment `Date` field, **not** the ingestion `event_ts` — this guarantees that a model fitted on labels from day `T` cannot see any feature derived from a post-`T` event.

#### A.1.4 Feature views shipped

| View | Version | Features |
|------|---------|----------|
| `patient_30d_stats` | v1.0.0 | no-show rate, appt count, mean duration, cancellation rate (30 d window) |
| `patient_90d_stats` | v1.0.0 | same four on 90 d window |
| `patient_cycle_ctx` | v1.0.0 | current cycle, total cycles, cycles since regimen modification, days since last visit |
| `patient_trend`     | v1.0.0 | attended streak, cancelled streak, duration trend (stable/growing/declining via rolling linear slope) |

#### A.1.5 Storage

| File | Purpose |
|------|---------|
| `data_cache/feature_store/online.pkl`           | Pickled dict for O(1) online lookup |
| `data_cache/feature_store/events.jsonl`         | Append-only event stream |
| `data_cache/feature_store/materialisations.jsonl` | One row per batch materialisation |
| `data_cache/feature_store/serving_latency.jsonl`  | Per-call latency log (fed into dissertation_analysis.R §22) |
| `data_cache/feature_store/schema.json`            | Schema version + feature view definitions |

#### A.1.6 Integration

| Surface | Role |
|---------|------|
| `_enrich_with_feature_store(patient_data, pid)` in `run_ml_predictions()` | Single injection point — every ensemble / TFT / IRL predictor auto-receives rolling-window features |
| `_auto_materialise_feature_store()` on startup                             | Non-blocking batch materialisation after historical data loads |
| `/api/features/store/{status,materialize,push_event}`                       | Diagnostics + manual / streaming ingest |
| `/api/features/online/<patient_id>`                                         | Low-latency serving endpoint with per-call latency tracking |
| `/api/features/online/as_of`                                                | Training-time PIT endpoint (exposed but NOT consumed by the current `NoShowModel.train` path — see scope note below) |

No UI panel — feature runs invisibly; when the store is empty (fresh install) the enrichment hook is a no-op and the legacy feature-engineering code runs unchanged.

#### A.1.7 Scope in the current pipeline (§4.5.8 regression)

The store is wired into the **online serving** path only: `_enrich_with_feature_store(patient_data, pid)` in `run_ml_predictions()` folds the online dict values into `patient_data` before the ensemble / TFT run.  The main training path (`NoShowModel.train()` in `ml/noshow_model.py`) materialises its features directly from `historical_appointments.xlsx` via `FeatureEngineer` and does **not** call `feature_store.as_of()`.  Temporal leakage in training is controlled separately by `NoShowModel.train(use_temporal_cv=True, date_column="Appointment_Date")`.  Wiring training through `as_of()` end-to-end would require backfilling the event log from the historical workbook and running `FeatureEngineer` through the store — flagged as future work, not current behaviour.  Regression for the external-review finding that the earlier prose implied training-time PIT consumption when the training pipeline never imported the store.  Locked by `tests/test_feature_store.py::TestTrainingIntegrationScope` (2 tests): (a) no training-path file imports `feature_store`; (b) the online `_enrich_with_feature_store` hook is still wired in `flask_app.py`.  If either invariant flips, the dissertation prose must be updated in the same commit.

### A.2 Micro-Batch Optimizer (§3.2 — Three-Tier Orchestration)

#### A.2.1 Motivation

Re-running CP-SAT on every scheduling change is a waste when that change is "insert one urgent patient". The §3.2 brief wants three tiers: fast heuristic insertion (<50 ms), slow full re-optimisation (every 15 min or 3+ changes), background RL (continuous incremental improvement).

#### A.2.2 Routing rule

```
path(c) = FAST    if c.type == 'insert' AND c.urgent
       |= SLOW    if queue >= change_threshold OR elapsed_since_last_reopt >= slow_path_interval_s
       |= QUEUED  otherwise
```

Fast-path exception → demote to QUEUED. Slow-path exception → re-queue every drained change (no data loss).

#### A.2.3 Wiring

| Primitive | Tier | File |
|-----------|------|------|
| `SqueezeInHandler.squeeze_in_with_noshow()` | FAST | `optimization/squeeze_in.py` |
| `ScheduleOptimizer.optimize()`              | SLOW | `optimization/optimizer.py` |
| `SchedulingRLAgent` (tick) | BG  | `ml/rl_scheduler.py` |

The coordinator `ml/micro_batch_optimizer.py` is constructed with 3 callables (no hard imports), so the Flask app injects its already-instantiated singletons via adapter functions (`_mb_fast_path`, `_mb_slow_path`, `_mb_rl_tick`).

#### A.2.4 Eventual consistency

Queued non-urgent changes are applied to `app_state['appointments']` only when the slow path fires. Between firings the schedule reflects fast-path insertions only. Lag is surfaced on `/api/microbatch/status` as `eventual_consistency_lag_s` so operators can audit freshness.

#### A.2.5 Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/api/microbatch/status` | GET  | Queue size + counters + next-slow-path ETA |
| `/api/microbatch/submit` | POST | Route a change (insert/cancel/reschedule) |
| `/api/microbatch/flush`  | POST | Force slow path now |
| `/api/microbatch/config` | POST | Retune thresholds live |

Every decision logs one row to `data_cache/micro_batch/latency.jsonl` with `{ts, path, change_type, latency_ms, success}` — consumed by `dissertation_analysis.R` §23. No UI panel.

---

### A.3 Digital Twin (§3.3 — What-If Simulation)

#### A.3.1 Motivation

An operator proposing to lower the double-book threshold from 0.30 to 0.05 has no safe way to preview the effect before it hits live patients. The §3.3 brief asks for a parallel simulated environment mirroring live state, advancing virtual time by historical arrival patterns, and letting policies be evaluated offline over a multi-day horizon.

#### A.3.2 Arrival model

Given a historical appointment log $\{(t_i)\}_{i=1}^{N}$ spanning $D$ distinct days, the twin fits per-(dow, hod) Poisson rates

$$
\lambda(d, h) = \frac{|\{i : \mathrm{dow}(t_i) = d \wedge \mathrm{hod}(t_i) = h\}|}{D / 7}
$$

and an urgent fraction $p_U = \Pr[\text{is\_urgent} \mid \text{arrival}]$. Arrivals in $[t_0, t_1)$ are sampled as $N_{\text{arr}} \sim \text{Poisson}(\lambda(\mathrm{dow}(t_0), \mathrm{hod}(t_0)) \cdot (t_1 - t_0))$ with urgency drawn independently per arrival.

#### A.3.3 Rollout loop

```
state := deepcopy(snapshot)
for i in 1..(horizon_days * 24 / step_hours):
    arrivals := sample_poisson(lambda, step_hours)
    for a in arrivals:
        outcome := squeeze_fn(a, state, policy)
        if outcome.success: state.appointments.append(outcome.appointment)
    for appt in state.appointments where start_time in [t0, t1):
        if rand() < noshow_fn(appt): noshows += 1
        else:                         completed += 1
    state.virtual_time += step_hours
```

`squeeze_fn` and `noshow_fn` default to the production `SqueezeInHandler.squeeze_in_with_noshow()` and `NoShowModel.predict()`, so twin metrics match what live ops would realise given identical arrivals.

#### A.3.4 Policy score and guardrails

$$
S(\pi) = 0.45 \cdot r_{\text{accept}} + 0.30 \cdot \frac{u}{100} - 0.15 \cdot r_{\text{double-book}} - 0.10 \cdot r_{\text{no-show}}
$$

Guardrails: `double_book_rate > 0.25`, `accept_rate < 0.50`, `noshow_rate > 0.35` are appended as string violations so the operator cannot commit a policy that trips any.

#### A.3.5 Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/api/twin/status`         | GET  | arrival-model summary + last snapshot + last eval |
| `/api/twin/snapshot`       | POST | deep-copy `app_state` into a frozen TwinState |
| `/api/twin/evaluate`       | POST | run one policy over {horizon_days, step_hours, rng_seed} |
| `/api/twin/compare`        | POST | rank multiple policies under the same snapshot + seed |
| `/api/twin/evaluations`    | GET  | list prior eval JSONs, most-recent first |
| `/api/twin/config`         | GET  | defaults + guardrail thresholds |

Every evaluation writes one JSON to `data_cache/digital_twin/evaluations/<ts>_<policy>.json` and one event row to `data_cache/digital_twin/twin_events.jsonl` — consumed by `dissertation_analysis.R` §24. No UI panel.

#### A.3.6 Determinism contract

$(state_0, \pi, \text{seed})$ ⇒ byte-identical `(total_arrivals, total_accepted, total_double_bookings, total_noshows_realised, policy_score)`. The test harness `tests/test_digital_twin.py::TestDeterminism::test_same_seed_reproduces` enforces this.

---

### A.4 Drift Root-Cause Attribution (§3.4)

#### A.4.1 Motivation

`ml/drift_detection.py` fires on PSI/KS/CUSUM but only tells operators *that* a distribution moved. The §3.4 brief asks for *why*: which feature contributes the most to the PSI increase and, within that feature, which histogram bin.

#### A.4.2 Per-bin contribution

For feature $j$ with reference $\mathbf{r}_j$ and current $\mathbf{c}_j$, histogram both on the same percentile breakpoints. With Laplace-smoothed bin proportions $P_{j,i}^{\text{ref}}, P_{j,i}^{\text{cur}}$:

$$
\delta_{j,i} = \left(P_{j,i}^{\text{cur}} - P_{j,i}^{\text{ref}}\right) \cdot \ln\!\left(\frac{P_{j,i}^{\text{cur}}}{P_{j,i}^{\text{ref}}}\right)
$$

Summing over bins recovers the per-feature PSI, $\text{PSI}_j = \sum_i \delta_{j,i}$, matching `DriftDetector.compute_psi()` exactly (enforced by `tests/test_drift_attribution.py::TestConsistencyWithDriftDetector`).

#### A.4.3 Shares

$$
\text{share}_j = \frac{\text{PSI}_j}{\sum_k \text{PSI}_k}, \qquad \sum_j \text{share}_j = 1
$$

The attributor emits a top-K feature breakdown (default K=5). Overall severity uses the same thresholds as `DriftDetector`: `PSI_j ≤ 0.1 → none`, `0.1-0.25 → moderate`, `> 0.25 → significant`.

#### A.4.4 Narrative

The top feature's top bin (by $|\delta_{j,i}|$) is rendered:

```
{share:.0f}% of PSI increase due to '{top_feature}' shift
  ({hint}: bin [lower, upper) {direction} from {p_ref}% -> {p_cur}%);
overall severity = {severity}.
```

`{hint}` is a small operator-maintained dictionary (`Travel_Time_Min → "more remote patients"`, `Age → "age mix shifted"`, …). Features without a hint fall back to the raw column name.

#### A.4.5 Integration

| Event                             | What happens                                                            |
|-----------------------------------|-------------------------------------------------------------------------|
| `POST /api/ml/drift/check`        | Returns `attribution` block alongside the report                       |
| `POST /api/drift/attribution`     | Ad-hoc attribution for custom (ref, cur) + features + hints            |
| `GET  /api/drift/attribution/last`| Full per-bin detail of the most recent attribution                     |
| `GET  /api/drift/attribution/status` | Counters + last narrative                                            |

Every run writes one JSONL row to `data_cache/drift_attribution/attributions.jsonl`. §25 of `dissertation_analysis.R` reads this log. No UI panel.

#### A.4.6 Consistency invariants

- `sum(fa.share_of_total for fa in breakdown) == 1.0` (within 1e-6)
- `sum(b.psi_contribution for b in fa.bins) == fa.psi` (within 1e-6)
- `fa.psi == DriftDetector(n_bins=N).compute_psi(ref, cur)` (within 1e-6)

---

### A.5 Distributionally Robust Fairness (§4.1)

#### A.5.1 Motivation

The legacy `FairnessAuditor` (Section 13) reports demographic parity as a SOFT penalty on the CP-SAT objective — any fairness weight can be traded off against throughput. §4.1 makes fairness a HARD constraint by producing a *certificate* of the worst-case parity gap over a 1-Wasserstein ball of radius ε centred on the empirical joint distribution. The certificate survives distributional shift within the ball, which is the key guarantee sought.

#### A.5.2 Upper bound

For binary outcome $Y = \mathbf{1}[\text{scheduled}]$ and the 1-Wasserstein metric with 0-1 label cost, the supremum-parity-gap over $B_\varepsilon(\hat P)$ admits the plug-in dual bound (Taskesen et al. FAccT 2021):

$$
\sup_{Q \in B_\varepsilon(\hat P)} |P_Q(Y = 1 \mid G = g_1) - P_Q(Y = 1 \mid G = g_2)| \leq |\hat \Delta| + \varepsilon \left( \frac{1}{\pi_1} + \frac{1}{\pi_2} \right)
$$

where $\hat \Delta = \hat P(Y=1|G=g_1) - \hat P(Y=1|G=g_2)$ and $\pi_g$ is the empirical mass of group $g$. The $\varepsilon/\pi_g$ term encodes the adversary's leverage: the smaller the group, the more one unit of mass shifts its conditional rate.

Certified iff worst-case ≤ δ budget.

#### A.5.3 Finite-sample adjustment

Two-sided Wald $z_{\alpha/2}$ inflation on the plug-in:

$$
\text{SE}(\hat\Delta) = \sqrt{\frac{\hat p_1(1-\hat p_1)}{n_1} + \frac{\hat p_2(1-\hat p_2)}{n_2}}, \qquad \hat\Delta^{\text{upper}} = \text{worst-case} + z_{\alpha/2} \cdot \text{SE}(\hat\Delta)
$$

Reported as `se_adjusted_upper`. Operators enforcing hard constraints should require both `certified` AND `certified_conservative`.

#### A.5.4 Integration

| Event                                | What happens                                                 |
|--------------------------------------|--------------------------------------------------------------|
| `POST /api/fairness/audit`           | Legacy response + `dro_certificate` block (invisible)        |
| `POST /api/fairness/dro/certify`     | On-demand Wasserstein-DRO certificate                        |
| `GET  /api/fairness/dro/status`      | Certifier config + last verdict + narrative                  |
| `GET  /api/fairness/dro/last`        | Full per-pair detail of the most recent certificate          |
| `POST /api/fairness/dro/config`      | Retune ε, δ, confidence, hard-constraint mode                |
| `run_optimization()`                 | Attaches certificate to `fairness_dro_certificate` result    |

When `enforce_as_hard_constraint=True` is set via `/api/fairness/dro/config`, a failing certificate sets `fairness_blocked=True` in the optimisation result, and downstream consumers can refuse to commit.

JSONL log at `data_cache/dro_fairness/certificates.jsonl`; §26 of `dissertation_analysis.R` reads it. No UI panel.

#### A.5.5 Invariants (enforced by `tests/test_dro_fairness.py`)

- `worst_case_gap ≥ empirical_gap` for all pairs
- `ε = 0 ⇒ worst_case_gap = empirical_gap` (DRO collapses to plug-in)
- `∂(worst_case_gap)/∂(1/π_g) > 0` (minority inflation)
- `certified ⇔ worst_case_gap ≤ δ`
- `se_adjusted_upper ≥ worst_case_gap`

---

### A.6 Individual (Lipschitz) Fairness (§4.2)

#### A.6.1 Motivation

§4.1 guarantees group-level parity under distributional perturbation. §4.2 adds the orthogonal Dwork et al. (2012) individual-level guarantee: patients whose features are close should receive close outcomes.

#### A.6.2 Lipschitz condition

For a distance metric $d(\cdot,\cdot)$ on patient features and outcome score $f(x) \in [0,1]$:

$$
\forall (i, j) : d(x_i, x_j) \leq \tau \implies |f(x_i) - f(x_j)| \leq L \cdot d(x_i, x_j)
$$

Excess is the amount by which the bound is breached:

$$
e_{ij} = \max\left(0,\; |f(x_i) - f(x_j)| - L \cdot d(x_i, x_j)\right)
$$

#### A.6.3 Distance metric

Features are min-max normalised to $[0, 1]$ over the training cohort. Euclidean distance is then divided by $\sqrt{d}$ (the number of features) so $\tau$ stays on an interpretable [0, 1] scale regardless of how many features the certifier tracks. Default features: `age, priority, expected_duration, distance_km, no_show_rate`.

#### A.6.4 Similar-pair search

`sklearn.neighbors.NearestNeighbors(radius=tau * sqrt(d))` runs in near-linear time for typical cohorts. Fallback path is an $O(n^2)$ pairwise scan so the module has no hard sklearn dependency.

#### A.6.5 Certificate verdict

| Quantity         | Rule                                         |
|------------------|----------------------------------------------|
| `violation_rate` | $\lvert \{(i,j): e_{ij} > 0\} \rvert / \lvert \{(i,j): d(x_i, x_j) \leq \tau\} \rvert$ |
| `certified`      | `violation_rate ≤ violation_budget` (default 0.05) |
| `strictly_lipschitz` | `n_violations == 0`                       |

#### A.6.6 Lazy-constraint emitter for CP-SAT

`LipschitzFairnessCertifier.iter_violating_pairs(...)` returns `[(patient_a, patient_b, excess), …]`. Callers translate each tuple into a CP-SAT clause forbidding the current $(\text{outcome}_a, \text{outcome}_b)$ assignment, enabling constraint-generation enforcement. The optimiser does not need refactoring — the emitter is dependency-injected.

#### A.6.7 Integration

| Event                                 | What happens                                      |
|---------------------------------------|---------------------------------------------------|
| `POST /api/fairness/audit`            | Legacy + DRO + **Lipschitz certificate** (invisible) |
| `POST /api/fairness/lipschitz/certify`| On-demand Lipschitz certificate                  |
| `GET  /api/fairness/lipschitz/status` | Config + last verdict + narrative                |
| `GET  /api/fairness/lipschitz/last`   | Full top-K violating pairs                       |
| `POST /api/fairness/lipschitz/config` | Retune L, τ, violation_budget, features, hard mode |
| `run_optimization()`                  | Attaches `lipschitz_certificate` + `lipschitz_blocked` |

JSONL log at `data_cache/individual_fairness/certificates.jsonl`. §27 of `dissertation_analysis.R` reads it. No UI panel.

#### A.6.8 Invariants (enforced by `tests/test_individual_fairness.py`)

- Constant outcomes on similar pairs ⇒ `strictly_lipschitz == True`
- Larger L ⇒ `n_violations` monotonically non-increasing
- Larger τ ⇒ `n_similar_pairs` monotonically non-decreasing
- `L = ∞ ⇒ n_violations == 0`
- Missing features are filled with their column mean, never NaN

---

### A.7 Safety Guardrails with Runtime Monitoring (§4.3)

#### A.7.1 Motivation

The §4.1 / §4.2 certificates block schedules that violate *statistical* fairness guarantees. §4.3 adds a third pillar: a runtime monitor that blocks outputs the CP-SAT optimiser might return which are *technically optimal but clinically dangerous*. The brief's motivating example:

```
if any(remaining_slack < 5 for critical_patient):
    reject_schedule("Would create cascade risk for high-priority patient")
```

#### A.7.2 Rule protocol

Each rule is a pure function:

```
rule(schedule: List[appt], patients_by_id: dict, context: dict) -> List[SafetyViolation]
```

Every rule carries a severity tag (`CRITICAL | HIGH | MODERATE | LOW`) and a human-readable `suggested_fix`. Rules are registered in `SafetyGuardrailsMonitor._rules`; new ones added at runtime via `register_rule`.

#### A.7.3 Verdict rule

$$
\text{verdict}(R) = \begin{cases}
  \text{reject} & \text{if } \exists v \in R : \text{sev}(v) = \text{CRITICAL}\\
  \text{warn}   & \text{else if } R \neq \emptyset\\
  \text{accept} & \text{otherwise}
\end{cases}
$$

When `enforce_as_hard_gate` is set (default `True`), a `reject` verdict raises `safety_blocked = True` on the optimisation result. Downstream consumers see this flag and refuse to commit.

#### A.7.4 Built-in rule set

| Rule                         | Severity | Trigger                                                    |
|------------------------------|----------|------------------------------------------------------------|
| `critical_slack_floor`       | CRITICAL | priority-{1,2} patient with < 5 min slack either side      |
| `chair_capacity`             | CRITICAL | concurrent appointments exceed site capacity               |
| `concurrent_chair_overlap`   | CRITICAL | two appointments overlap without `double_booked` flag      |
| `long_infusion_cutoff`       | HIGH     | infusion ≥ 180 min starts after 13:00                      |
| `travel_time_ceiling`        | HIGH     | assigned site travel > 120 min                             |
| `wait_time_ceiling`          | HIGH     | wait past `earliest_time` > 180 min                        |
| `consecutive_high_noshow`    | MODERATE | 3 consecutive high-NS-risk patients on the same chair      |
| `double_book_density`        | MODERATE | more than one double-booking in a chair-hour               |

Every parameter is tunable live via `POST /api/safety/config`.

#### A.7.5 Integration

| Event                           | What happens                                                          |
|---------------------------------|-----------------------------------------------------------------------|
| `run_optimization()`            | Attaches `safety_report` + `safety_blocked` to optimisation results   |
| `POST /api/safety/evaluate`     | Runs monitor on current schedule                                      |
| `GET  /api/safety/status`       | Config + last verdict + narrative                                     |
| `GET  /api/safety/last`         | Full violation list                                                   |
| `POST /api/safety/config`       | Toggle rules, retune thresholds, set `enforce_as_hard_gate`           |

JSONL log at `data_cache/safety_guardrails/reports.jsonl`. §28 of `dissertation_analysis.R` reads it. No UI panel.

#### A.7.6 Invariants (enforced by `tests/test_safety_guardrails.py`)

- Clean schedule ⇒ `verdict == "accept"` and `n_violations == 0`
- Any CRITICAL ⇒ `verdict == "reject"`
- Only HIGH or MODERATE ⇒ `verdict == "warn"`
- Disabled rule ⇒ no violations regardless of data
- `concurrent_chair_overlap` respects the `double_booked` flag
- Every violation carries `rule_name`, `severity`, `detail`, `suggested_fix`
- **Verdict-narrative consistency** (`TestVerdictInvariants`): the verdict and the narrative string MUST be derived from the same `evaluate()` call so the leading verdict word in `narrative` equals `verdict.upper()`.  Regression for the §4.5.14 dissertation contradiction: an earlier rendering mixed a macro-driven `\safLatestVerdict = ACCEPT` with a hardcoded `Safety guardrails: REJECT --- 1 total violations` narrative, an internal contradiction the runtime cannot produce but stale prose can.  The `dissertation_analysis.R §28` ingest now also stop()s before emitting `\safExample*` macros if any reports.jsonl row violates `verdict == "reject" iff n_critical > 0`.

---

### A.8 Counterfactual Fairness Audit (§4.4)

#### A.8.1 Motivation

§4.1/§4.2 catch explicit group / individual parity failures. §4.4 addresses the subtler **proxy-discrimination** failure mode: a model that never sees the protected attribute may still discriminate through correlates (e.g. Welsh postcode as a proxy for socioeconomic class).

#### A.8.2 Scheduleability predictor

Thin logistic regression fitted on-the-fly from the historical `(features, was_scheduled)` pair. Features (numeric only):

- `age`, `priority`, `expected_duration` — intrinsic
- `deprivation_score` (0–10, from Welsh postcode → WIMD lookup) — primary proxy
- `distance_km`, `travel_time_min` — downstream effect of postcode
- `no_show_rate` — indirectly reflects deprivation

Decision threshold = `median(predicted_prob | scheduled == 1)`. Falls back to a mean-ratio heuristic if sklearn is unavailable.

#### A.8.3 Flip rule

For a rejected patient at postcode $p_0$ (deprivation $d_0 > 4$), flip to counterfactual $p_c$ (default CF10, $d_c \approx 2$) and recompute probability. Declare a flip:

$$
\Pr_{\text{cf}}(\text{scheduled} \mid x_{\text{cf}}) \geq \tau_{\text{decision}}
\quad \text{AND} \quad
\Pr_{\text{cf}} - \Pr_{\text{orig}} \geq \Delta_{\min}
$$

with $\Delta_{\min} = 0.05$ by default. Certificate passes iff `flip_rate = n_flipped / n_rejected ≤ flip_budget` (default 0.10).

#### A.8.4 Label-only vs circumstances-rich modes

| Mode | `flip_downstream` | What changes | Question answered |
|------|-------------------|--------------|-------------------|
| Label-only (default) | False | only `deprivation_score` | Does the model use postcode *as a label*? |
| Circumstances-rich   | True  | deprivation + distance + travel + no_show | What if they *actually lived there*? |

#### A.8.5 Deprivation lookup

Hand-curated Welsh-postcode → WIMD-decile map in `DEFAULT_POSTCODE_DEPRIVATION`: CF10/CF23 (Cardiff affluent, 2–3), CF14/CF5 (mixed, 3–4), CF44/CF40/CF42 (Rhondda deprived, 8–9). Overridable in config.

#### A.8.6 Integration

| Event                                         | What happens                                   |
|-----------------------------------------------|------------------------------------------------|
| `POST /api/fairness/audit`                    | Legacy + DRO + Lipschitz + **counterfactual**  |
| `POST /api/fairness/counterfactual/audit`     | On-demand audit                                |
| `GET  /api/fairness/counterfactual/status`    | Config + last verdict + narrative              |
| `GET  /api/fairness/counterfactual/last`      | Full top-K flip cases                          |
| `POST /api/fairness/counterfactual/config`    | Retune cf postcode, min effect, budget, lookup |
| `run_optimization()`                          | Attaches `counterfactual_certificate`           |

JSONL log at `data_cache/counterfactual_fairness/audits.jsonl`. §29 of `dissertation_analysis.R` reads it. No UI panel.

#### A.8.7 Invariants (enforced by `tests/test_counterfactual_fairness.py`)

- Patients with affluent-bucket postcodes are skipped (nothing to flip)
- Unbiased cohort (single postcode) ⇒ `n_rejected == 0`
- **Decision threshold strictly inside `[DECISION_THRESHOLD_FLOOR=0.05, DECISION_THRESHOLD_CEILING=0.95]` on every code path** (logreg / mean_ratio / degenerate_fallback) — `_clamp_threshold()` enforces this at the source so the audit's `would_flip = (cf_prob ≥ threshold)` rule cannot collapse to "every patient flips" (threshold = 0) or "no patient ever flips" (threshold = 1).  Regression for the §4.5.15 dissertation finding which showed `Decision threshold 0.000`: the all-rejected input cohort took the degenerate-fallback branch which previously set `_decision_threshold = y.mean() = 0.0`, producing a vacuous PASS where every `delta_prob = 0` because `predict_proba` short-circuited to the threshold value.  Locked by `TestDecisionThresholdInvariants` (9 tests covering clamp helper, all three fit branches, end-to-end certificate, and the vacuously-PASS narrative).
- Predictor fits degenerate-class inputs (all scheduled / none scheduled) → `degenerate_fallback` method, neutral threshold (0.5), and a narrative that admits the audit was vacuous so operators are not misled.
- Affluent patients get higher Pr(scheduled) than deprived patients (end-to-end calibration sanity)
- JSON round-trip preserves the full certificate

---

### A.9 Auto-scaling Optimizer with Timeout Guarantees (§5.1)

#### A.9.1 Motivation

CP-SAT solve time is structurally unpredictable — the same instance can take 0.1s or 30s depending on branching decisions. Hospital deployment requires hard timing guarantees, not a single optimistic budget.

#### A.9.2 Three mechanisms (§5.1 brief)

1. **Cascade**: try [5s, 2s, 1s, 0.5s] budgets sequentially; if all fail → greedy fallback.
2. **Early stopping**: abort when `|objective - best_bound| / |objective| ≤ 0.01` (1% optimality gap).
3. **Parallel search**: race 4 weight configurations in a ThreadPoolExecutor; take best by `(-n_scheduled, solve_time, config_name)` (deterministic tie-break).  Each worker calls the injected `solve_with_weights(patients, weights, time_limit)` callable so weights flow through as a per-call argument — workers never share mutable optimiser state, and no two workers can trample each other's weights mid-solve.[^race-fix]  When the host caller injects only the legacy `set_weights` callable (no `solve_with_weights`), the race serialises behind a single lock — correct but no parallelism, with a one-shot `auto-scaling parallel race degraded to serial` warning logged.

[^race-fix]: Earlier revisions used a `set_weights` callback that mutated the shared optimiser's internal state.  Worker B could overwrite the weights mid-solve for worker A, returning a result computed against the wrong configuration.  The current API exposes `solve_with_weights(patients, weights, time_limit)` so each worker carries its own configuration as an argument, eliminating cross-worker contamination by construction.

#### A.9.3 Weight configurations raced

| Config | priority | utilization | noshow_risk | waiting_time | robustness | travel |
|--------|----------|-------------|-------------|--------------|------------|--------|
| balanced         | 0.30 | 0.25 | 0.15 | 0.15 | 0.10 | 0.05 |
| priority_heavy   | 0.55 | 0.15 | 0.10 | 0.10 | 0.05 | 0.05 |
| throughput       | 0.20 | 0.45 | 0.10 | 0.10 | 0.10 | 0.05 |
| robustness_heavy | 0.25 | 0.20 | 0.15 | 0.10 | 0.25 | 0.05 |

All weights sum to 1.0.

#### A.9.4 Greedy fallback

Priority-first placement: sort patients by `(priority ASC, earliest_time ASC)`; round-robin chairs from `DEFAULT_SITES`; place on first open slot with 5-minute slack between appointments. No overlap, respects chair count per site, obeys `day_start_hour=8` / `day_end_hour=18`. Always runs in <100ms for cohorts up to ~1000 patients.

#### A.9.5 Integration

| Event                                    | What happens                                    |
|------------------------------------------|-------------------------------------------------|
| `POST /api/optimize/autoscale/optimize`  | Runs full cascade + parallel race + greedy      |
| `GET  /api/optimize/autoscale/status`    | Config + last winner + narrative                |
| `GET  /api/optimize/autoscale/last`      | Full per-attempt detail                         |
| `POST /api/optimize/autoscale/config`    | Retune budgets, parallel configs, gap, flags    |
| `run_optimization()`                     | Attaches `auto_scaling_last` to results         |

JSONL log at `data_cache/auto_scaling/runs.jsonl`. §30 of `dissertation_analysis.R` reads it. No UI panel. Invisible integration: legacy `/api/optimize` keeps the single-solve path; auto-scaling is opt-in.

#### A.9.6 Invariants (enforced by `tests/test_auto_scaling_optimizer.py`)

- Successful parallel race returns `winner_stage == 'parallel'`
- All CP-SAT attempts failing triggers greedy fallback: `winner_stage == 'greedy'` AND `greedy_fallback == True`
- Parallel race runs exactly `parallel_configs` workers
- Cascade budgets tried in decreasing order (not including parallel's top budget)
- Greedy schedule: no chair overlaps, respects site capacity, priority-1 patient gets earliest slot
- `early_stopped == True` iff solve_time < 0.9 × budget AND success
- No base optimizer ⇒ falls straight through to greedy
- **Race-isolation invariants:** with `solve_with_weights` injected, four configs racing in parallel each observe their OWN weights — no cross-contamination (`TestParallelRaceWeightIsolation.test_per_call_weights_no_cross_contamination`), wall-time is dominated by ONE solve not all four (`...test_solve_with_weights_truly_parallel`); with the legacy `set_weights` path, max concurrent active solves is 1 (`...test_legacy_set_weights_serialises_under_lock`)
- **Wall-clock budget invariant (§2.12.6):** every parallel worker honours its `parallel_time_limit_s` — for instances large enough to route to column generation, `ColumnGenerator.solve()` checks elapsed time at every iteration boundary and returns a feasible solution tagged `CG_TIME_LIMIT` when the budget is hit (regression: `tests/test_column_generation.py::test_cg_respects_wall_clock_time_limit`). Before this guard a 2 s budget could blow past 300 s on a 202-patient cohort; the historical pre-fix evidence is preserved at `data_cache/auto_scaling/runs.pre_cg_time_limit_fix.jsonl.bak`.

---

### A.10 Human-in-the-Loop Override Learning (§5.2)

#### A.10.1 Motivation

The scheduler's throughput + fairness + safety guarantees mean nothing if clinicians don't trust it. §5.2 closes the feedback loop: every manual override is a fragment of preference data; the system should harvest, model, and preemptively suggest the clinician's preferred alternative.

#### A.10.2 Event schema

`OverrideEvent` dataclass:
- `ts` (ISO UTC)
- `patient_id`
- `original_{chair_id, start_time, duration}` — what the scheduler proposed
- `new_{chair_id, start_time, duration}` — what the clinician preferred
- `priority`, `site_code`, `noshow_prob` — contextual features
- `clinician_id`, `reason` — audit trail

Append-only JSONL log at `data_cache/override_learning/events.jsonl`.

#### A.10.3 Prediction model

Features: `(hour_of_day, day_of_week, priority, duration_min, noshow_prob, site_idx)` — numeric only, 6-dim.

Primary: **sklearn logistic regression** on `(X, y)` where
- Positive class (`y=1`): events where override happened
- Negative class (`y=0`): accepted appointments (in `app_state['appointments']` without a corresponding override event), capped at 4× positives for balance

$$
\Pr(\text{override} \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \tilde{\mathbf{x}} + b)
$$

where $\tilde{\mathbf{x}}$ is the standardised feature vector.

Fallback: **count-rate table** with Laplace smoothing, indexed by `(3-hour-bin, priority, site-hash-bucket)`:

$$
\Pr_{\text{count}}(\text{override} \mid \text{bucket}) = \frac{n_{\text{override}} + 1}{n_{\text{seen}} + 2}
$$

Cold start (before `min_events_for_fit`): returns `cold_start_prior` (default 0.10).

#### A.10.4 Preemptive suggestion rule

For candidate alternatives $\mathcal{A}$ (default: ±1, ±2 hours on the same chair):

$$
\text{suggest}(\text{appt}) = \begin{cases}
\arg\min_{a \in \mathcal{A}} \Pr(\text{override} \mid a) & \text{if } \Pr(\text{override} \mid \text{appt}) \geq \tau_{\text{suggest}} \\
& \quad \text{AND } \min_{a} \Pr(\text{override} \mid a) < \Pr(\text{override} \mid \text{appt}) \\
\text{None} & \text{otherwise}
\end{cases}
$$

where $\tau_{\text{suggest}} \in [0, 1]$ is the **suggestion threshold** — the minimum override probability the system must assign to the proposed slot before it offers any alternative.  Default $\tau_{\text{suggest}} = 0.80$ (per the §5.2 brief): the model must be at least 80\% confident the clinician would override the optimiser's pick before the system speaks up.  Tunable at runtime via `POST /api/override/config` and persisted on the `OverrideLearner` instance as the `suggest_threshold` attribute (see `ml/override_learning.py`).

The two-clause rule combines:

1. \emph{Confidence gate} — $\Pr(\text{override} \mid \text{appt}) \geq \tau_{\text{suggest}}$ ensures we only intervene on high-confidence override predictions.
2. \emph{Improvement gate} — $\min_a \Pr(\text{override} \mid a) < \Pr(\text{override} \mid \text{appt})$ ensures the proposed alternative scores strictly lower than the current pick.

The "no improvement" guard means the system stays silent when it cannot propose anything better — bad advice is worse than no advice.

#### A.10.5 Integration

| Event                                | What happens                                      |
|--------------------------------------|---------------------------------------------------|
| `POST /api/override/log`             | Record event; eagerly refit at `min_events_for_fit` |
| `POST /api/override/suggest`         | Bulk-suggest across `app_state['appointments']`   |
| `GET  /api/override/status`          | Config + event count + method + last narrative    |
| `GET  /api/override/last`            | Last 20 events + full status                      |
| `POST /api/override/config`          | Retune $\tau_{\text{suggest}}$, min-events, prior, neighbourhood |
| `run_optimization()`                 | Attaches `override_suggestions` to results        |

JSONL logs at `data_cache/override_learning/{events,suggestions}.jsonl`. §31 of `dissertation_analysis.R` reads them. No UI panel.

#### A.10.6 Invariants (enforced by `tests/test_override_learning.py`)

- Cold start (events < `min_events_for_fit`) ⇒ predicted prob = `cold_start_prior`
- Post-fit probabilities ∈ [0, 1]
- Late-afternoon overrides ⇒ late-afternoon slots score higher than morning slots (pattern recovery)
- Suggestions respect `suggest_threshold` (below-threshold appointments return None)
- Suggestion's `suggested_override_prob < original_override_prob` (strict improvement)
- Events persist through learner restart
- Custom `alternatives` list honoured when provided
- JSON round-trip safe

---

### A.11 Explainable Rejection Reports (§5.3)

#### A.11.1 Motivation

When CP-SAT cannot place a patient, today the only signal is a patient-id in `OptimizationResult.unscheduled`. §5.3 requires (i) a human-readable reason list per-chair and (ii) a concrete counterfactual alternative slot, matching the brief's template verbatim.

#### A.11.2 Blocker taxonomy

Seven stable categories:

| Category | Trigger |
|----------|---------|
| `insufficient_slack` | best gap on chair < patient's expected duration |
| `would_bump_higher_priority` | colliding appt has priority strictly higher (lower number) |
| `would_bump_same_or_lower_priority` | colliding appt has equal/lower priority AND `allow_reschedule=False` |
| `travel_exceeded` | chair site's travel > `max_travel_minutes` |
| `outside_operating_hours` | no open window in `day_start_hour`–`day_end_hour` |
| `earliest_time_exceeded` | patient's `earliest_time` past the day-end |
| `duration_too_long_for_day` | patient's duration > full-day minutes |

The bump classifier picks the **highest-priority** colliding appointment (lowest priority number) to match the brief's "would require bumping P1 patient" phrasing — the toughest swap to justify.

#### A.11.3 Gap analysis per chair

For each chair, walk its appointments sorted by `start_time`:

```
cursor = max(earliest_time, day_start_hour:00)
for each appointment a (in order):
    if a.end_time <= cursor: skip
    gap = a.start_time − cursor
    if gap >= duration: candidate
    cursor = max(cursor, a.end_time)
trailing_gap = day_end − cursor
```

Return `(best_slack, max_slack_any, colliding_appt)` where `best_slack` is the first gap >= duration, `max_slack_any` is the biggest gap seen at all (used for insufficient-slack detail even when none fit).

#### A.11.4 Alternative-slot finder

Bounded forward scan:

```
for day_offset in range(look_ahead_days):
    day = current_date + day_offset
    day_start_for_scan = day_offset == 0 ? max(earliest, day_start) 
                                          : day.replace(hour=max(earliest.hour, day_start))
    for chair in chairs:
        if best_slack(chair) >= duration + slack_buffer_min:
            return first_fitting_start(chair, day_start_for_scan, duration)
return None
```

Returns an `AlternativeSlot` with `chair_id`, `site_code`, `date`, `start_time`, `duration_minutes`, `wait_increase_minutes`, and a human-readable `narrative` like "Thursday 9am (24 h wait increase)".

#### A.11.5 Narrative template

Renders the §5.3 brief pattern verbatim:

```
Patient {pid} (P{priority}, previous no-shows: {prev_ns}) not scheduled because:
- {blocker_1.detail}
- {blocker_2.detail}
- {blocker_3.detail}
- Alternative: offer {alt.narrative} [Accept] [Decline]
```

Top-3 blockers deduplicated by `(blocker_type, chair_id)`; when no alternative is found, the last line is `"No alternative within {N} days lookahead."`

#### A.11.6 Integration

| Event                                | What happens                                        |
|--------------------------------------|-----------------------------------------------------|
| `POST /api/rejection/explain`        | Explain one patient by id                           |
| `POST /api/rejection/explain/all`    | Bulk-explain every patient not in `scheduled_patients` |
| `GET  /api/rejection/status`         | Config + last run summary                           |
| `GET  /api/rejection/last`           | Full detail of the most recent batch                |
| `POST /api/rejection/config`         | Retune day-hours, look-ahead, slack buffer, `allow_reschedule` |
| `run_optimization()`                 | Attaches `rejection_explanations` to results        |

JSONL log at `data_cache/rejection_explainer/explanations.jsonl`. §32 of `dissertation_analysis.R` reads it. No UI panel.

#### A.11.7 Invariants (enforced by `tests/test_rejection_explainer.py`)

- Narrative starts with exact brief header `"Patient {pid} (P{priority}, previous no-shows: {n}) not scheduled because:"`
- Narrative ends with `"Alternative: offer ..."` or `"No alternative within {N} days lookahead."`
- Insufficient slack reported with exact `"X min slack (requires Y min for this patient)"` phrasing
- Bump classifier selects the highest-priority colliding patient (lowest priority number)
- Alternative respects `earliest_time` on all look-ahead days (not just day 0)
- Travel-time ceiling fires before slack analysis (travel check is the outer gate)
- Full saturation across all `look_ahead_days` ⇒ `alternative is None` AND narrative says so

---

### A.12 SACT Version Adapter (§5.4)

#### A.12.1 Motivation

NHS SACT v4.0 (May 2025) will be superseded by v4.1 (~2028) with new molecular-marker fields and a gender-code rename. The production pipeline must not break on that day. §5.4 demands an ABC-based adapter layer so downstream ML binds to a canonical schema, not to version-specific raw column names.

#### A.12.2 ABC contract

```python
class SACTVersionAdapter(ABC):
    @abstractmethod
    def to_internal(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns from CANONICAL_COLUMNS."""
```

#### A.12.3 Canonical schema (24 columns, fixed point)

| Category | Columns |
|----------|---------|
| Identity | Patient_ID, NHS_Number_Hashed |
| Event | Date, Appointment_Hour, Attended_Status, Regimen_Code, Cycle_Number |
| Demographics | Age, Age_Band, Gender_Code, Ethnic_Category, Patient_Postcode |
| Clinical | Primary_Diagnosis_Code, Performance_Status |
| Resourcing | Planned_Duration, Actual_Duration, Site_Code, Chair_Id |
| External | Weather_Severity |
| v4.1 forward-compat (empty on v4.0) | Molecular_Marker_Status, Biomarker_Panel_Code, Subcutaneous_Administration_Flag, Comorbidity_Count, Commissioning_Organisation_Code |

#### A.12.4 Concrete adapters

- **`SACTv4Adapter`** (live): maps v4.0's 82 raw fields → 24 canonical columns via `SPEC_V4_0.canonical_map`. Handles both `Person_Stated_Gender_Code`/`Gender_Code`, `Ethnic_Category_Code`/`Ethnic_Category`, etc.

- **`SACTv4_1Adapter`** (placeholder per §5.4 brief): inherits v4.0 mapping, adds v4.1's new fields and renamed columns (`Gender_Identity_Code` → `Gender_Code`, `Provider_Code` → `Commissioning_Organisation_Code`). Coerces `Subcutaneous_Administration_Flag` from Y/N/1/0 → Python bool.

#### A.12.5 Auto-detection rule

`auto_detect_version(raw_df)` iterates registered versions from highest to lowest. Picks the first version whose `expected_fields` ⊆ `raw_df.columns`. Falls back to v4.0 with a WARNING log if no version matches strictly.

v4.0 signature: `{Patient_ID, Regimen_Code, Cycle_Number, Person_Stated_Gender_Code, Ethnic_Category_Code}`  
v4.1 signature: `{Molecular_Marker_Status, Biomarker_Panel_Code, Gender_Identity_Code, Commissioning_Organisation_Code}`

#### A.12.6 Registry

`ADAPTERS: Dict[str, SACTVersionAdapter]` + `SPECS: Dict[str, VersionSpec]` are the canonical lookups. Runtime registration via `register_adapter(version, adapter, spec)` lets third parties add v5.x adapters without touching core code.

#### A.12.7 Integration

| Event                              | What happens                                                            |
|------------------------------------|-------------------------------------------------------------------------|
| `load_data_from_source()`          | Auto-detects version; stores in `app_state['sact_version_detected']`    |
| `GET  /api/sact/status`            | Registered versions, canonical column list, last run summary            |
| `GET  /api/sact/versions`          | Full version diff: `expected_fields`, `new_since_prior`, renames         |
| `POST /api/sact/detect`            | Detect version of current historical_appointments_df or patients_df     |
| `POST /api/sact/adapt`             | Convert current frame to canonical schema (explicit version or auto)    |

JSONL log at `data_cache/sact_adapter/adapter_events.jsonl`. §33 of `dissertation_analysis.R` reads it. No UI panel.

#### A.12.8 Invariants (enforced by `tests/test_sact_version_adapter.py`)

- `SACTVersionAdapter()` raises TypeError (ABC)
- `CANONICAL_COLUMNS` is the set of output columns for **every** adapter, regardless of input version
- v4.0 input + v4.0 adapter ⇒ v4.1 canonical columns still present but empty
- v4.1 input → auto-detected version = "v4.1" (higher-version preference rule)
- Empty DataFrame input ⇒ empty canonical-shaped output
- Unknown input → `v4.0` fallback with WARNING log
- Explicit version override respected: `adapt(df_v41, version="v4.0")` does NOT map v4.1-specific fields
- `_coerce_bool_flag` correctly handles Y/N, 1/0, True/False, and NaN-like inputs

---

### A.13 Stochastic MPC Scheduler (§5.5)

#### A.13.1 Motivation

The static optimiser produces a morning schedule; real-time disruption (urgent arrivals, cancellations, overruns) was previously handled by the deterministic squeeze-in heuristic. §5.5 replaces that with a receding-horizon controller that reacts optimally to same-day uncertainty.

#### A.13.2 MDP formulation

$$
S_t = \left( \text{time}_t, \{\text{chair}_c^{(t)}\}_{c=1}^C, Q_t, \text{stats}_t \right)
$$

**Chair state**: `IDLE` or `(patient_id, remaining_minutes)`.

**Queue**: each `QueuedPatient` carries `(priority, expected_duration, no_show_prob, cancel_prob, arrival_time_min, is_urgent)`.

**Action**: per-chair assignment from queue, or `wait`. Action space factorises by chair.

**Transition**: during $[t, t+\Delta t)$ (Poisson-Bernoulli joint):
- Busy chairs decrement `remaining_minutes`, become IDLE at 0
- Urgent arrivals fire according to Poisson($\lambda(t) \cdot \Delta t$)
- Scheduled patients no-show with prob $\pi_p$ or cancel with prob $\gamma_p$

**Reward** (per §S-1.4):

$$
R_t = \underbrace{\sum_c \mathbb{1}_{\text{complete}_c} \cdot (6 - \text{prio}_p) \cdot b_{\text{complete}}}_{\text{completion credit}} - \underbrace{\sum_{p \in Q_t} w_{\text{wait}} \min(\text{wait}_p / 60, 2)}_{\text{waiting cost}} - \underbrace{\sum_c \mathbb{1}_{\text{idle \& queue non-empty}} \cdot w_{\text{idle}}}_{\text{idle penalty}}
$$

with $b_{\text{complete}} = $ `DEFAULT_PRIORITY_COMPLETE_BASE` and $w_{\text{idle}} = $ `DEFAULT_IDLE_PENALTY`.  The multiplier is $(6 - \text{prio}_p)$ rather than $(5 - \text{prio}_p)$ — applied to $\text{prio}_p \in \{1,\dots,5\}$ this gives priority-1 (most urgent) the largest credit ($5 b_{\text{complete}}$) and priority-5 a positive but minimal credit ($1 \cdot b_{\text{complete}}$).  The +1 floor matters clinically: every completed chemotherapy session must be rewarded, even when the patient was lowest priority, otherwise the optimiser has no incentive to ever schedule routine cycles.  When a chair was already `OCCUPIED` at day-start (no captured priority), the formula falls back to $\text{prio}_p = 3$ (mid).

`prio_p` is recovered from `ChairState.priority_at_assignment`, the field that preserves the assigned patient's priority across the chair's `OCCUPIED→IDLE` transition.[^prio-stash] The planner stamps it onto the chair at `_apply_action` time and keeps the value live through one decision step so the reward function can credit a completion by the priority of the patient who finished, not by some downstream proxy.

[^prio-stash]: Without this stashed field the reward code could only see whichever patient was queued for the chair *next*, since the completed patient's queue entry has already been removed by the time `compute_immediate_reward` runs.  Earlier revisions defaulted to a hard-coded mid-priority constant, which made the multi-objective formula insensitive to who finished — a priority-1 completion produced the same credit as a priority-5 completion.

**Terminal penalty** at $t = H$:

$$
R_H = - \sum_{p \in Q_H} \max(5 - \text{prio}_p,\ 1) \cdot u
$$

with $u = $ `DEFAULT_TERMINAL_UNSCHEDULED_PENALTY`.  The $\max(\cdot, 1)$ floor mirrors the $(6 - \text{prio})$ choice in $R_t$: an unscheduled priority-5 patient still incurs a non-zero terminal penalty so the planner cannot park them indefinitely.

#### A.13.3 MPC with scenario rollout

$K$ scenarios sampled from the posterior $\lambda(t)$ and per-patient $(\pi, \gamma)$:

```
for each candidate action a in A(S_t):
    for each scenario k in 1..K:
        apply a to S_t; advance state with greedy fill until t+tau
        accumulate R over the horizon; add V_hat(S_{t+tau})
    r_a = mean(scenario rewards)
action* = argmax_a r_a
```

Tie-break: deterministic on `(-n_assignments, config_name)` so replays are byte-identical.

#### A.13.4 Terminal value function

Linear feature combiner:

$$
\widehat V(S) = b + \mathbf{w}^\top \mathbf{x}(S)
$$

with $\mathbf{x}(S) = (n_{\text{queue,high}}, n_{\text{queue,low}}, n_{\text{idle}}, t/H, \text{cumwait}_{\text{h}}, \text{remaining\_cap})$ and default weights `[-4, -1, +2, -5, -1.5, +0.01]`. Operator can retrain offline on simulated days and swap in an MLP via JSON persistence.

#### A.13.5 Bayesian urgent arrival model (Gamma-Poisson)

Prior: $\lambda \sim \text{Gamma}(\alpha_0, \beta_0)$. Observed $n$ arrivals in $\Delta$ minutes → posterior Gamma$(\alpha_0 + n, \beta_0 + \Delta)$. Posterior mean rate = $\alpha / \beta$ arrivals/min (60α/β per hour).

```python
class UrgentArrivalModel:
    def __init__(self): self.alpha = 1.0; self.beta = 1.0
    def update(self, arrivals, minutes): self.alpha += arrivals; self.beta += minutes
    def predict_rate(self): return self.alpha / self.beta
```

Persisted JSON at `data_cache/mpc_scheduler/arrival_model.json`.

#### A.13.6 Fail-safe fallback

When total scenario-evaluation time exceeds `total_timeout_s` (default 500 ms) OR no candidate actions exist, controller falls back to the injected `fallback_fn` (squeeze-in handler in production) or priority-first greedy fill. The `MPCDecision` records `used_fallback=True` + `fallback_reason` so audits distinguish optimised vs degraded decisions.

#### A.13.7 Integration

| Event                                | What happens                                              |
|--------------------------------------|-----------------------------------------------------------|
| `POST /api/mpc/decide`               | On-demand decision on current schedule                    |
| `POST /api/mpc/simulate`             | Full-day rollout vs greedy / static baselines             |
| `POST /api/mpc/arrival/update`       | Gamma-Poisson conjugate update                            |
| `GET  /api/mpc/arrival/rate`         | Current posterior rate                                    |
| `GET  /api/mpc/status`               | Full config + arrival-model + last decision               |
| `POST /api/mpc/config`               | Retune K, τ, timeout                                      |
| `run_optimization()`                 | Attaches `mpc_last_decision` to results                   |

JSONL logs at `data_cache/mpc_scheduler/{events,decisions,simulations}.jsonl`. §34 of `dissertation_analysis.R` reads them. No UI panel.

#### A.13.8 Invariants (enforced by `tests/test_stochastic_mpc_scheduler.py`)

- Prior `UrgentArrivalModel(α=1,β=1)` ⇒ rate = 1.0 arrivals/min
- `update(n, Δ)` ⇒ `α ← α+n`, `β ← β+Δ` (conjugate Gamma update)
- Save/load roundtrips preserve (α, β)
- Empty queue ⇒ MPC decision = wait on all chairs
- No idle chairs ⇒ MPC decision = wait on all chairs
- Tight `total_timeout_s` ⇒ `used_fallback=True` + `fallback_reason` set
- **Budget–latency consistency** (`TestBudgetLatencyInvariants`, 3 tests): `decision_latency_ms > total_timeout_s × 1000 ⇒ used_fallback=True`; the latency field is always finite + non-negative; the R-side budget-status helper (`"within" / "slightly above"`) matches `p95 ≤ budget`.  Regression for the §4.5.20 prose that asserted "p95 = 505.7 ms ... well under the 500 ms budget" while lumping the 31 % fallback rate into the same "under 500 ms" clause.  The fix reframes `total_timeout_s` as a fallback **trigger** (not an SLA), emits `\mpcBudgetMs`, `\mpcPninetyfiveBudgetStatus`, `\mpcPninetyfiveOvershootMs` from live data, and adds a loud R warning when `p95 > budget` AND `fallback_rate = 0` (the only case where the operational invariant would actually be broken).
- Same RNG seed ⇒ reproducible scenarios
- Priority-1 patient in queue ⇒ assigned to first idle chair under greedy fill
- `evaluate_action` returns finite reward + valid final state
- `simulate_day` returns per-policy metrics with `0 ≤ urgent_acceptance_rate ≤ 1` and `utilisation ≤ 1`
- **Priority-weighted completion credit:** completing a priority-1 patient yields strictly more reward than completing a priority-5 patient, monotone across $\text{prio} \in \{1,\dots,5\}$ (`TestPriorityWeightedReward.test_priority_multiplier_is_monotone`)
- `ChairState.priority_at_assignment` is captured at `RolloutPlanner._apply_action` time and copied through `ChairState.copy()` (`TestPriorityWeightedReward.test_chairstate_copy_preserves_priority`)
- A chair `OCCUPIED` at day-start with no captured priority falls back to mid-priority (3) and still produces a positive completion reward (`TestPriorityWeightedReward.test_missing_priority_defaults_to_mid`)

---

## 10. Survival Analysis

### 10.1 Cox Proportional Hazards Model

The Cox PH model estimates the hazard (instantaneous risk) of no-show as a function of covariates, without assuming a specific baseline hazard distribution.

#### Hazard Function

$$h(t|X) = h_0(t) \cdot \exp(\beta^T X)$$

Where:
- $h(t|X)$ = hazard at time $t$ given covariates $X$
- $h_0(t)$ = baseline hazard function (unspecified)
- $\beta$ = coefficient vector
- $X$ = covariate vector

#### Partial Likelihood

The model is fitted by maximizing the partial likelihood:

$$L(\beta) = \prod_{i: \delta_i = 1} \frac{\exp(\beta^T X_i)}{\sum_{j \in R(t_i)} \exp(\beta^T X_j)}$$

Where:
- $\delta_i = 1$ if event (no-show) occurred
- $R(t_i)$ = risk set at time $t_i$ (patients still at risk)

#### Log Partial Likelihood

$$\ell(\beta) = \sum_{i: \delta_i = 1} \left[ \beta^T X_i - \log\left(\sum_{j \in R(t_i)} \exp(\beta^T X_j)\right) \right]$$

#### Gradient (Score Function)

$$\frac{\partial \ell}{\partial \beta} = \sum_{i: \delta_i = 1} \left[ X_i - \frac{\sum_{j \in R(t_i)} X_j \exp(\beta^T X_j)}{\sum_{j \in R(t_i)} \exp(\beta^T X_j)} \right]$$

#### Newton-Raphson Update

$$\beta^{(k+1)} = \beta^{(k)} - H^{-1} \nabla \ell(\beta^{(k)})$$

Where $H$ is the Hessian matrix of second derivatives.

### 10.2 Survival Function

The survival function (probability of attending by time $t$):

$$S(t|X) = \exp\left(-\int_0^t h(u|X) du\right) = S_0(t)^{\exp(\beta^T X)}$$

### 10.3 Hazard Ratios

The hazard ratio for a one-unit increase in covariate $X_k$:

$$HR_k = \exp(\beta_k)$$

Interpretation:
- $HR > 1$: Increased risk of no-show
- $HR < 1$: Decreased risk of no-show
- $HR = 1$: No effect

### 10.4 C-Index (Concordance)

Model discrimination is measured by the concordance index:

$$C = \frac{\text{concordant pairs}}{\text{concordant pairs} + \text{discordant pairs}}$$

A pair $(i, j)$ is concordant if $t_i < t_j$ and $\hat{h}_i > \hat{h}_j$.

### 10.5 Implementation

```python
from ml.survival_model import SurvivalAnalysisModel, CoxProportionalHazards

model = SurvivalAnalysisModel()
model.fit(X, event_times, event_indicators)

# Predict survival probability
survival_probs = model.predict_survival(X_new, times=[7, 14, 21])

# Get hazard ratios
hazard_ratios = model.get_hazard_ratios()
```

---

## 11. Uplift Modeling

Uplift modeling estimates the **Conditional Average Treatment Effect (CATE)** - how much a treatment (intervention) affects each individual patient.

### 11.1 Individual Treatment Effect (ITE)

$$\tau(x) = E[Y(1) - Y(0) | X = x]$$

Where:
- $Y(1)$ = outcome under treatment (e.g., with reminder)
- $Y(0)$ = outcome under control (no reminder)
- $X$ = patient features

Since we cannot observe both potential outcomes for the same patient (fundamental problem of causal inference), we estimate $\tau(x)$ using meta-learners.

### 11.2 S-Learner (Single Model)

Train a single model on treatment as a feature:

$$\hat{\mu}(x, t) = \hat{E}[Y | X = x, T = t]$$

$$\hat{\tau}_{S}(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)$$

**Algorithm:**
```
1. Create feature matrix [X, T] including treatment indicator
2. Train single model: f(X, T) → Y
3. For prediction: τ(x) = f(x, T=1) - f(x, T=0)
```

**Advantages:** Simple, uses all data for training
**Disadvantages:** May underestimate heterogeneity if treatment effect is small

### 11.3 T-Learner (Two Models)

Train separate models for treatment and control groups:

$$\hat{\mu}_0(x) = \hat{E}[Y | X = x, T = 0]$$
$$\hat{\mu}_1(x) = \hat{E}[Y | X = x, T = 1]$$

$$\hat{\tau}_{T}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$

**Algorithm:**
```
1. Split data by treatment: D_0 = {(x,y): T=0}, D_1 = {(x,y): T=1}
2. Train control model: f_0(X) → Y on D_0
3. Train treatment model: f_1(X) → Y on D_1
4. For prediction: τ(x) = f_1(x) - f_0(x)
```

**Advantages:** Captures heterogeneous effects well
**Disadvantages:** Requires sufficient data in both groups

### 11.4 Intervention Types

| Intervention | Description | Expected Effect |
|--------------|-------------|-----------------|
| `none` | No intervention | Baseline |
| `sms_reminder` | SMS reminder 24-48h before | -5% to -10% no-show |
| `phone_call` | Personal phone call | -8% to -15% no-show |
| `transport_assistance` | Arranged transport | -10% to -20% for distant patients |
| `care_coordinator` | Dedicated coordinator contact | -12% to -18% no-show |

### 11.5 Uplift Evaluation Metrics

**Area Under Uplift Curve (AUUC):**

Sort patients by predicted uplift $\hat{\tau}(x)$ and compute cumulative gain:

$$AUUC = \int_0^1 \text{Uplift}(q) \, dq$$

**Qini Coefficient:**

$$Qini = \frac{AUUC - AUUC_{random}}{AUUC_{perfect} - AUUC_{random}}$$

### 11.6 Intervention Recommendation

For a new patient, recommend the intervention with highest predicted uplift:

$$T^* = \arg\max_{t \in \mathcal{T}} \hat{\tau}_t(x)$$

Subject to cost constraints:

$$\sum_{i} cost(T_i^*) \leq Budget$$

### 11.7 Implementation

```python
from ml.uplift_model import UpliftModel, recommend_intervention

model = UpliftModel()
model.fit(X, y, treatment)

# Predict uplift for each intervention
uplifts = model.predict_uplift(patient_features)

# Get recommendation
recommendation = recommend_intervention(
    patient_data=patient,
    available_interventions=['sms_reminder', 'phone_call', 'transport_assistance']
)
# Returns: {'intervention': 'phone_call', 'expected_reduction': 0.12, 'confidence': 0.85}
```

---

## 12. Multi-Task Learning

### 12.1 Architecture

Multi-task learning jointly predicts no-show probability and treatment duration using a shared representation:

```
Input Features (20)
       │
       ▼
┌──────────────────┐
│  Shared Layers   │
│  Linear(20→128)  │
│  ReLU + Dropout  │
│  Linear(128→64)  │
│  ReLU + Dropout  │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│NoShow │ │Duration│
│ Head  │ │ Head   │
│32→16→1│ │32→16→1│
│Sigmoid│ │ ReLU   │
└───────┘ └───────┘
```

### 12.2 Loss Function

The total loss combines both tasks:

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{noshow} + (1 - \alpha) \cdot \mathcal{L}_{duration}$$

Where:
- $\alpha = 0.5$ (default, equal weighting)
- $\mathcal{L}_{noshow}$ = Binary Cross-Entropy loss
- $\mathcal{L}_{duration}$ = Mean Squared Error loss

**Binary Cross-Entropy (No-Show):**

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]$$

**Mean Squared Error (Duration):**

$$\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N} (d_i - \hat{d}_i)^2$$

### 12.3 Shared Representation Learning

The key insight is that no-show and duration share underlying factors:
- Patient reliability affects both attendance and on-time arrival
- Clinical complexity affects both attendance barriers and treatment time
- Geographic factors affect both travel feasibility and delays

The shared layers learn these common representations:

$$h_{shared} = f_{shared}(X; \theta_{shared})$$

$$\hat{p}_{noshow} = f_{noshow}(h_{shared}; \theta_{noshow})$$

$$\hat{d}_{duration} = f_{duration}(h_{shared}; \theta_{duration})$$

### 12.4 Gradient Balancing

To prevent one task from dominating training, gradients are balanced:

$$g_{balanced} = \frac{g_{noshow}}{||g_{noshow}||} + \frac{g_{duration}}{||g_{duration}||}$$

### 12.5 Implementation

```python
from ml.multitask_model import MultiTaskModel

model = MultiTaskModel(input_dim=20, task_weight=0.5)
model.fit(X_train, y_noshow, y_duration, epochs=100)

# Joint prediction
prediction = model.predict(patient_features)
# Returns: MultiTaskPrediction(noshow_prob=0.18, duration=142.5, ...)
```

---

## 13. Quantile Regression Forest

### 13.1 Standard Random Forest Limitation

Standard Random Forest provides point predictions:

$$\hat{y} = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

But does not naturally provide prediction intervals.

### 13.2 Quantile Regression Forest

QRF estimates the full conditional distribution $F(y|X=x)$ by keeping all training observations in each leaf:

$$\hat{F}(y|X=x) = \sum_{i=1}^{n} w_i(x) \cdot \mathbb{1}(Y_i \leq y)$$

Where the weights are:

$$w_i(x) = \frac{1}{B}\sum_{b=1}^{B} \frac{\mathbb{1}(X_i \in L_b(x))}{|L_b(x)|}$$

- $L_b(x)$ = leaf node containing $x$ in tree $b$
- $|L_b(x)|$ = number of training samples in that leaf

### 13.3 Quantile Estimation

The $\tau$-quantile is estimated as:

$$\hat{Q}_\tau(x) = \inf\{y : \hat{F}(y|X=x) \geq \tau\}$$

### 13.4 Prediction Intervals

For a $(1-\alpha)$ prediction interval:

$$PI_{1-\alpha}(x) = \left[\hat{Q}_{\alpha/2}(x), \hat{Q}_{1-\alpha/2}(x)\right]$$

**Example (90% interval):**

$$PI_{90\%}(x) = \left[\hat{Q}_{0.05}(x), \hat{Q}_{0.95}(x)\right]$$

### 13.5 Duration Prediction with Uncertainty

For duration prediction:

| Quantile | Interpretation |
|----------|----------------|
| $Q_{0.10}$ | Optimistic estimate (10% chance of being shorter) |
| $Q_{0.50}$ | Median prediction |
| $Q_{0.90}$ | Conservative estimate (90% chance of being shorter) |

**Scheduling Application:**
- Use $Q_{0.50}$ for expected duration
- Use $Q_{0.90}$ for buffer allocation

### 13.6 Implementation

```python
from ml.quantile_forest import QuantileForestDurationModel

model = QuantileForestDurationModel(n_estimators=100, quantiles=[0.1, 0.5, 0.9])
model.fit(X_train, y_duration)

prediction = model.predict(patient_features)
# Returns: QuantilePrediction(
#     median=145.0,
#     lower_90=118.0,
#     upper_90=178.0,
#     interval_width=60.0
# )
```

---

## 14. Hierarchical Bayesian Model

### 14.1 Model Structure

The hierarchical model accounts for patient-level and site-level variation:

**Observation Level:**

$$y_{ij} \sim N(\mu + \alpha_i + X_{ij}\beta, \sigma^2)$$

**Patient Random Effects:**

$$\alpha_i \sim N(0, \tau^2)$$

Where:
- $y_{ij}$ = duration for patient $i$, appointment $j$
- $\mu$ = grand mean
- $\alpha_i$ = patient-specific random effect
- $\beta$ = fixed effect coefficients
- $\sigma^2$ = within-patient variance
- $\tau^2$ = between-patient variance

### 14.2 Priors (PyMC Implementation)

$$\mu \sim N(120, 50^2)$$
$$\tau \sim \text{HalfNormal}(20)$$
$$\sigma \sim \text{HalfNormal}(30)$$
$$\beta_k \sim N(0, 10^2)$$

### 14.3 Intraclass Correlation Coefficient (ICC)

The proportion of variance attributable to patient-level differences:

$$ICC = \frac{\tau^2}{\tau^2 + \sigma^2}$$

Interpretation:
- $ICC \approx 0$: All variance is within-patient (no patient effect)
- $ICC \approx 1$: All variance is between-patient

### 14.4 Shrinkage Estimation

For a patient with few observations, the random effect is shrunk toward zero:

$$\hat{\alpha}_i = \frac{n_i \tau^2}{n_i \tau^2 + \sigma^2} \cdot (\bar{y}_i - \mu)$$

Where $n_i$ = number of observations for patient $i$.

- Many observations ($n_i$ large): $\hat{\alpha}_i \approx \bar{y}_i - \mu$ (trust the data)
- Few observations ($n_i$ small): $\hat{\alpha}_i \approx 0$ (shrink to population mean)

### 14.5 Posterior Prediction

For a new appointment for patient $i$:

$$\hat{y}_{i,new} = \mu + \hat{\alpha}_i + X_{new}\hat{\beta}$$

With credible interval from posterior samples.

### 14.6 MCMC Sampling

The model is fitted using NUTS (No-U-Turn Sampler) in PyMC:

```python
with pm.Model() as hierarchical_model:
    # Priors
    mu = pm.Normal('mu', mu=120, sigma=50)
    tau = pm.HalfNormal('tau', sigma=20)
    sigma = pm.HalfNormal('sigma', sigma=30)

    # Patient random effects
    alpha = pm.Normal('alpha', mu=0, sigma=tau, shape=n_patients)

    # Expected value
    mu_y = mu + alpha[patient_idx] + X @ beta

    # Likelihood
    y_obs = pm.Normal('y', mu=mu_y, sigma=sigma, observed=y)

    # Sample
    trace = pm.sample(1000, tune=1000, cores=2)
```

### 14.7 Implementation

```python
from ml.hierarchical_model import HierarchicalBayesianModel

model = HierarchicalBayesianModel()
model.fit(X, y_duration, patient_ids)

# Predict for specific patient
prediction = model.predict(patient_id='P12345', features=X_new)
# Returns: HierarchicalPrediction(
#     mean=152.3,
#     credible_interval=(128.5, 176.1),
#     patient_effect=12.3,
#     shrinkage_factor=0.72
# )
```

---

## 15. Causal Inference

### 15.1 Causal DAG (Directed Acyclic Graph)

The causal relationships in SACT scheduling:

```
            ┌─────────────┐
            │   WEATHER   │
            │  SEVERITY   │
            └──────┬──────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
   ┌──────────┐ ┌──────────┐ │
   │ TRAFFIC  │ │ PATIENT  │ │
   │  DELAY   │ │   MOOD   │ │
   └────┬─────┘ └────┬─────┘ │
        │            │       │
        └──────┬─────┘       │
               ▼             │
         ┌──────────┐        │
         │  NO-SHOW │◀───────┘
         └──────────┘
               ▲
               │
         ┌──────────┐
         │ PREVIOUS │
         │ NO-SHOWS │
         └──────────┘
```

### 15.2 Structural Causal Model (SCM)

$$U_{weather} \sim P(U_{weather})$$
$$Weather = f_{weather}(U_{weather})$$
$$Traffic = f_{traffic}(Weather, U_{traffic})$$
$$NoShow = f_{noshow}(Weather, Traffic, PrevNoShows, U_{noshow})$$

### 15.3 do-Calculus

The **do-operator** represents intervention (setting a variable to a value):

$$P(Y | do(X = x))$$

This is different from conditional probability $P(Y | X = x)$ which includes confounding.

**Adjustment Formula:**

If $Z$ is a sufficient adjustment set (blocks all backdoor paths):

$$P(Y | do(X = x)) = \sum_z P(Y | X = x, Z = z) \cdot P(Z = z)$$

### 15.4 Backdoor Criterion

A set $Z$ satisfies the backdoor criterion relative to $(X, Y)$ if:
1. No node in $Z$ is a descendant of $X$
2. $Z$ blocks every path between $X$ and $Y$ that contains an arrow into $X$

### 15.5 Causal Effect Estimation

**Average Treatment Effect (ATE):**

$$ATE = E[Y | do(T = 1)] - E[Y | do(T = 0)]$$

**Conditional Average Treatment Effect (CATE):**

$$CATE(x) = E[Y | do(T = 1), X = x] - E[Y | do(T = 0), X = x]$$

### 15.6 Counterfactual Queries

"What would the no-show rate have been if we had sent reminders?"

$$E[Y_{T=1} | T = 0, X = x]$$

### 15.7 Counterfactual Explanations

Find the minimum feature change to flip a patient's risk prediction:

$$CF(x) = \arg\min_{x'} \|x' - x\|_2 \quad \text{s.t.} \quad f(x') \neq f(x)$$

Where:
- $x$ = current patient features
- $x'$ = counterfactual features (closest alternative)
- $f$ = prediction function (no-show classifier)
- $\|x' - x\|_2$ = Euclidean distance (normalised by feature range)

**Example:** "What is the smallest change that would make this high-risk patient low-risk?"

| Feature | Current | Counterfactual | Change |
|---------|---------|---------------|--------|
| Previous_NoShows | 3 | 1 | -2 (fewer no-shows) |

The search uses greedy perturbation over mutable features only (not age, gender):
- Previous_NoShows, Previous_Cancellations
- Travel_Distance_KM, Travel_Time_Min
- Days_Booked_In_Advance, Cycle_Number
- Weather_Severity, Performance_Status

### 15.8 Causal Validation Framework

Validates causal estimates using three approaches:

**Placebo Tests** — interventions that CANNOT causally affect the outcome should show zero effect:

$$\hat{\tau}_{\text{placebo}} = E[Y | do(X_{\text{shuffled}})] - E[Y] \approx 0$$

Tests:
- Weather effect on clear days (severity < 0.05) → should be 0
- Shuffled weather → no-show correlation → should be 0
- Day-of-week effect after controlling for traffic → should be small

**Falsification Tests** — known non-causal relationships should show zero effect:

$$\text{Corr}(X_{\text{non-causal}}, Y) \approx 0$$

Tests:
- Chair number → no-show (chair is arbitrary assignment, not causal)
- Patient ID → duration (ID is arbitrary identifier)
- Appointment hour → weather (scheduling can't cause weather)

**Sensitivity Analysis (Rosenbaum Gamma):**

$$\Gamma = \frac{\hat{\tau}}{SE(\hat{\tau})}$$

If $\Gamma > 1$, the estimate is robust to moderate unmeasured confounding. An unmeasured confounder would need to change odds by at least $\Gamma$-fold to explain away the observed effect.

**Scope of this validation (§4.6.1 regression).**  These seven tests run against a synthetic frame generated from the same scheduling DAG they verify against (see `tests/test_causal_model.py::_validation_data`), so passing them confirms the estimator correctly recovers the injected effect — NOT that the recovered ATE (e.g. $\hat\tau_{\text{weather}} = 0.084$) identifies a population-level causal effect in the real Velindre cohort.  The distinction matters because the external review flagged the earlier wording "all seven tests pass confirms the ATE is robust" as implicitly claiming real-world validation.  `dissertation_analysis.R §21` now emits the `\causalValidationScope` macro, defaulting to "synthetic DAG recovery (implementation-verification only)" and flipping to a real-cohort label if the JSONL source ever upgrades to a non-simulated trial.  Two regression tests in `tests/test_causal_model.py::TestCausalValidationScope` guard the scope label against silent drift (fixture must not leak real-Velindre columns; scope-label constant starts with "synthetic" until a real observational cohort is wired).

### 15.9 Implementation

```python
from ml.causal_model import SchedulingCausalModel, compute_intervention_effect

model = SchedulingCausalModel()
model.fit(historical_data)

# Compute causal effect
effect = compute_intervention_effect(
    model=model,
    intervention='reminder_sent',
    outcome='no_show'
)
# Returns: CausalEffect(ate=-0.12, ci=(-0.15, -0.09), p_value=0.001)

# Answer counterfactual
result = model.answer_counterfactual(
    patient_data={'reminder_sent': 0, 'no_show': 1},
    intervention={'reminder_sent': 1}
)
# Returns: CounterfactualResult(predicted_outcome=0, confidence=0.78)
```

---

## 15b. Sensitivity Analysis

### 15b.1 Local Sensitivity

For a prediction function $\hat{y} = f(\mathbf{x})$, the local sensitivity of feature $x_i$ at point $\mathbf{x}^*$ is:

$$S_i = \frac{\partial \hat{y}}{\partial x_i} \bigg|_{\mathbf{x}=\mathbf{x}^*}$$

Computed via central finite differences for numerical stability:

$$S_i \approx \frac{f(\mathbf{x}^* + h \cdot \mathbf{e}_i) - f(\mathbf{x}^* - h \cdot \mathbf{e}_i)}{2h}$$

Where $h = \epsilon \cdot \max(|x_i|, 1)$ with default $\epsilon = 0.01$ (1% perturbation).

### 15b.2 Global Importance

Aggregated across $n$ patients in the population:

$$I_j = \frac{1}{n} \sum_{i=1}^{n} |S_{ij}|$$

This is the **Mean Absolute Sensitivity** — features with high $I_j$ are consistently influential across the entire patient population, not just for one individual.

Standard deviation of importance:

$$\sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (|S_{ij}| - I_j)^2}$$

### 15b.3 Elasticity

Normalized sensitivity measuring percentage change in output per percentage change in input:

$$E_i = S_i \cdot \frac{x_i}{\hat{y}}$$

This is unit-free, enabling comparison across features with different scales (e.g., age in years vs. distance in km).

### 15b.4 Feature Categories

Sensitivities are grouped into clinical categories:
- **Patient History**: prev_noshow_rate, total_appointments, is_new_patient, ...
- **Demographics**: age, age_band_*, ...
- **Temporal**: hour, day_of_week, booking_lead_days, ...
- **Geographic**: distance_km, travel_time_min, is_local, ...
- **Treatment**: priority_level, expected_duration_min, cycle_number, ...
- **External**: weather_severity, traffic_severity, event_severity, ...

### 15b.5 Implementation

```python
from ml.sensitivity_analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(perturbation=0.01)

# Local: S_i for one patient
local = analyzer.local_sensitivity(model, x_patient, feature_names, 'noshow')
# Returns: top_positive, top_negative features

# Global: I_j across population
global_imp = analyzer.global_importance(model, X, feature_names, 'noshow', max_samples=200)
# Returns: importance_rank, feature_categories

# Elasticity: E_i = S_i * (x_i / y_hat)
elasticity = analyzer.elasticity(model, x_patient, feature_names, 'noshow')
```

---

## 15c. Model Cards for Transparency

Following Mitchell et al. (2019), each ML model has a **Model Card** documenting:

### 15c.1 Card Structure

| Section | Content |
|---------|---------|
| **Intended Use** | Primary purpose, intended users, out-of-scope uses |
| **Performance** | Overall metrics + breakdown by subgroup (age, gender, site, priority) |
| **Limitations** | Known failure modes, data gaps, distributional assumptions |
| **Ethical Considerations** | Fairness, bias risks, patient safety, NHS compliance |
| **Maintenance** | Update frequency, drift monitoring, contact |

### 15c.2 Subgroup Performance Analysis

For each protected characteristic $g \in \{$Age, Gender, Site, Priority$\}$:

$$\text{NoShow Rate}_g = \frac{|\{i \in g : y_i = 1\}|}{|g|}$$

### 15c.3 Fairness Metrics on Model Cards

**Four-Fifths Rule** (Disparate Impact):

$$\text{Ratio} = \frac{\min_g \text{Rate}_g}{\max_g \text{Rate}_g} \geq 0.8$$

**Max Disparity**:

$$\text{Disparity} = \max_g \text{Rate}_g - \min_g \text{Rate}_g$$

### 15c.4 Models with Cards

| Model | Type | Key Ethical Concern |
|-------|------|---------------------|
| No-Show Ensemble | RF+GB+XGB stacking | Must not disadvantage deprived-area patients |
| Duration Model | Ensemble + protocol variance | Must not pressure rushed treatments |
| Causal Model | DAG + IV + DML | Claims require validation before policy changes |
| RL Scheduler | Q-Learning | Advisory only, must not deprioritize groups |

### 15c.5 NHS AI Ethics Compliance

Model Cards align with:
- **NHS AI Ethics Framework** — transparency, accountability, fairness
- **Equality Act 2010** — no discrimination by protected characteristics
- **GDPR Article 22** — right to explanation for automated decisions
- **NICE Evidence Standards** — effectiveness, safety, clinical relevance

---

## 16. Instrumental Variables

### 16.1 The Endogeneity Problem

When estimating the effect of $T$ on $Y$, naive regression fails if there are **unobserved confounders** $U$:

$$Y = \beta T + \epsilon$$

If $Cov(T, \epsilon) \neq 0$ due to $U$, then $\hat{\beta}_{OLS}$ is biased.

### 16.2 Instrumental Variable Solution

An instrument $Z$ satisfies:
1. **Relevance:** $Cov(Z, T) \neq 0$ (instrument affects treatment)
2. **Exclusion:** $Z$ affects $Y$ only through $T$ (no direct effect)
3. **Independence:** $Z \perp U$ (instrument is exogenous)

### 16.3 Two-Stage Least Squares (2SLS)

**Stage 1: Predict Treatment from Instrument**

$$\hat{T} = \gamma_0 + \gamma_1 Z + X\gamma_2 + \nu$$

**Stage 2: Regress Outcome on Predicted Treatment**

$$Y = \beta_0 + \beta_1 \hat{T} + X\beta_2 + \epsilon$$

The coefficient $\beta_1$ is the **Local Average Treatment Effect (LATE)**.

### 16.4 IV in SACT Scheduling

**Causal Chain:** Weather → Traffic Delay → No-Show

- **Instrument (Z):** Weather severity (exogenous)
- **Treatment (T):** Traffic delay (endogenous - affected by unmeasured factors)
- **Outcome (Y):** No-show

Weather affects no-show only through traffic (exclusion restriction).

### 16.5 First-Stage F-Statistic

The strength of the instrument is measured by the F-statistic:

$$F = \frac{(\hat{\gamma}_1)^2}{Var(\hat{\gamma}_1)}$$

**Rule of Thumb:**
- $F > 10$: Strong instrument
- $F < 10$: Weak instrument (biased estimates)

### 16.6 2SLS Formulas

**Stage 1:**

$$\hat{\gamma} = (Z'Z)^{-1}Z'T$$

**Stage 2:**

$$\hat{\beta}_{2SLS} = (\hat{T}'\hat{T})^{-1}\hat{T}'Y$$

**Standard Error (Robust):**

$$SE(\hat{\beta}_{2SLS}) = \sqrt{\frac{\hat{\sigma}^2}{T'\hat{P}_Z T}}$$

Where $\hat{P}_Z = Z(Z'Z)^{-1}Z'$ is the projection matrix.

### 16.7 Implementation

```python
from ml.causal_model import InstrumentalVariablesEstimator, estimate_iv_effect

estimator = InstrumentalVariablesEstimator()
result = estimate_iv_effect(
    data=historical_data,
    instrument='weather_severity',
    treatment='traffic_delay_minutes',
    outcome='no_show',
    covariates=['age', 'distance_km', 'previous_noshows']
)

# Returns: IVEstimationResult(
#     causal_effect=0.0095,      # 0.95% increase per 10 min delay
#     standard_error=0.003,
#     first_stage_f_stat=2134,   # Strong instrument
#     confidence_interval=(0.004, 0.015),
#     weak_instrument=False
# )
```

---

## 17. Double Machine Learning

### 17.1 Motivation

Traditional ML methods for causal inference suffer from **regularization bias**. DML uses cross-fitting to achieve:
- $\sqrt{n}$-consistency
- Asymptotic normality
- Valid confidence intervals

### 17.2 Partially Linear Model

$$Y = \theta T + g(X) + \epsilon$$

Where:
- $Y$ = outcome (no-show)
- $T$ = binary treatment (reminder sent)
- $X$ = high-dimensional confounders
- $\theta$ = causal effect of interest
- $g(X)$ = nuisance function (confounding effects)

### 17.3 Neyman-Orthogonal Score

The DML estimator uses the **orthogonal/debiased score**:

$$\psi(W; \theta, \eta) = (Y - g(X) - \theta T)(T - m(X))$$

Where:
- $g(X) = E[Y | X]$ (outcome nuisance)
- $m(X) = E[T | X] = P(T=1 | X)$ (propensity score)

### 17.4 Cross-Fitting Algorithm

```
Algorithm: Double Machine Learning

1. Split data into K folds: D_1, ..., D_K

2. For each fold k:
   a. Train ĝ(X) on D_{-k} (all folds except k)
   b. Train m̂(X) on D_{-k}
   c. Compute residuals on D_k:
      - Y_residual = Y - ĝ(X)
      - T_residual = T - m̂(X)

3. Pool all residuals and estimate θ:
   θ̂ = (Σᵢ T_residual,i × Y_residual,i) / (Σᵢ T_residual,i²)

4. Compute standard error using influence functions
```

### 17.5 DML Estimator Formula

$$\hat{\theta}_{DML} = \frac{\frac{1}{n}\sum_{i=1}^{n} (Y_i - \hat{g}(X_i))(T_i - \hat{m}(X_i))}{\frac{1}{n}\sum_{i=1}^{n} (T_i - \hat{m}(X_i))^2}$$

### 17.6 Variance Estimation

$$\hat{V}(\hat{\theta}) = \frac{\frac{1}{n}\sum_{i=1}^{n} \hat{\psi}_i^2}{\left(\frac{1}{n}\sum_{i=1}^{n}(T_i - \hat{m}(X_i))^2\right)^2}$$

Where:

$$\hat{\psi}_i = (Y_i - \hat{g}(X_i) - \hat{\theta}T_i)(T_i - \hat{m}(X_i))$$

### 17.7 Confidence Interval

$$CI_{1-\alpha} = \hat{\theta} \pm z_{1-\alpha/2} \cdot \sqrt{\hat{V}(\hat{\theta})/n}$$

### 17.8 Model Choices

| Nuisance Function | Model Used | Rationale |
|-------------------|------------|-----------|
| $\hat{g}(X)$ (outcome) | Gradient Boosting Regressor | Handles non-linearity, interactions |
| $\hat{m}(X)$ (propensity) | Random Forest Classifier | Robust probability estimation |

### 17.9 Implementation

```python
from ml.causal_model import DoubleMachineLearning, estimate_dml_effect

dml = DoubleMachineLearning(n_folds=5)
result = dml.fit(
    data=historical_data,
    treatment='reminder_sent',
    outcome='no_show',
    covariates=['age', 'distance_km', 'weather_severity', 'previous_noshows', ...]
)

# Returns: DMLResult(
#     treatment_effect=-0.085,    # 8.5% reduction in no-shows
#     standard_error=0.012,
#     confidence_interval=(-0.109, -0.061),
#     t_statistic=-7.08,
#     p_value=0.0000,
#     outcome_model_r2=0.72,
#     propensity_auc=0.81
# )
```

### 17.b RCT Randomisation as Bayesian Prior (§2.4)

#### 17.b.1 Motivation

DML cleans up observational endogeneity via cross-fitting, but it cannot *identify* a treatment effect that is confounded with a clinician decision.  The §2.4 brief closes this by running a small randomised controlled trial in parallel with the observational flow: 5 % of bookings are flagged and uniformly allocated to one of four reminder arms. The resulting ATE estimate is unbiased by construction and acts as a Bayesian prior on the DML posterior.

#### 17.b.2 Deterministic hash-based assignment

Arm allocation uses a BLAKE2b hash over `(seed, tag, patient_id, appointment_id)` so the same booking always gets the same arm across restarts:

```
inclusion:  hash1(seed, "select", pid, appt) mod 10^5 < r * 10^5
arm:        hash2(seed, "arm",    pid, appt) mod |arms|
```

With `r = 0.05` and 4 arms, each arm collects ≈1.25 % of bookings.

#### 17.b.3 ATE estimation (Wald CI)

For treatment arm `t` vs control `c`:

```
ATE     = π_t − π_c
SE      = sqrt( π_t(1-π_t)/n_t + π_c(1-π_c)/n_c )
CI_95   = ATE ± 1.96 · SE
```

An arm is **under-powered** (and therefore excluded from the DML prior) when `n_t < min_n_per_arm` or `n_c < min_n_per_arm`. Default `min_n_per_arm = 30`.

#### 17.b.4 Precision-weighted Gaussian shrinkage into DML

Given observational DML estimate `(θ̂_dml, σ_dml)` and RCT estimate `(θ̂_rct, σ_rct)`, the posterior is:

```
θ̂_post  = (θ̂_dml / σ²_dml + θ̂_rct / σ²_rct) / (1/σ²_dml + 1/σ²_rct)
σ²_post = 1 / (1/σ²_dml + 1/σ²_rct)
w_rct   = σ⁻²_rct / (σ⁻²_dml + σ⁻²_rct)  ∈ [0, 1]
```

- `σ_rct → ∞` → posterior equals DML (legacy behaviour)
- `σ_rct → 0`  → posterior equals RCT (gold standard)
- `σ_post ≤ min(σ_dml, σ_rct)` always — guaranteed tighter than either input

#### 17.b.5 Integration

| File | Purpose |
|------|---------|
| `ml/rct_randomization.py` | `TrialAssigner` with hash-based arm allocation, JSONL persistence, ATE + CI helpers, `apply_rct_prior()` shrinkage function |
| `data_cache/trial_assignments.jsonl` | One row per flagged booking |
| `data_cache/trial_outcomes.jsonl` | One row per recorded attendance outcome |
| `data_cache/trial_ate_history.jsonl` | Snapshots of running ATEs (consumed by `dissertation_analysis.R` §21) |
| `data_cache/trial_config.json` | Persisted trial rate / arms |

Endpoints:

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/trial/status`           | GET  | Config + assignment/outcome counts + live ATEs |
| `/api/trial/randomize`        | POST | Flag the current schedule (or a supplied list) |
| `/api/trial/record_outcome`   | POST | Log attended/no-show for a trialled booking |
| `/api/trial/ate`              | GET  | Wald-CI ATE for every non-control arm |
| `/api/trial/config`           | POST | Retune trial rate / arm list on the fly |
| `/api/ml/dml/estimate`        | POST | When the matching arm is powered, response gains an `rct_prior` block with posterior ATE/SE/CI and `w_rct` |

No UI panel — feature runs invisibly; when the trial has insufficient outcomes, the DML endpoint's response is unchanged.

---

## 18. Event Impact Model

### 18.1 Overview

The Event Impact Model predicts how external events (weather, traffic, local events) affect no-show rates.

### 18.2 Event Types

| Type | Examples | Base Impact |
|------|----------|-------------|
| WEATHER | Storm, heavy rain, snow | High |
| TRAFFIC | Road closure, accident | Medium-High |
| SPORTS | Rugby match, football | Medium |
| CONCERT | Large venue events | Low-Medium |
| PUBLIC_HOLIDAY | Bank holidays | Medium |
| ROADWORKS | Major construction | Low-Medium |

### 18.3 Sentiment Analysis

Event descriptions are analyzed using VADER sentiment:

$$sentiment = VADER(text) \in [-1, 1]$$

- Negative sentiment indicates disruptive event
- Keywords boost/reduce impact (e.g., "cancelled", "delayed", "severe")

### 18.4 Severity Estimation

Event severity is computed from sentiment and keywords:

$$severity = f(sentiment, keywords, event\_type)$$

Mapping to discrete levels:

| Severity Level | Value | Description |
|----------------|-------|-------------|
| MINIMAL | 1 | Little to no impact |
| LOW | 2 | Minor disruption |
| MODERATE | 3 | Noticeable impact |
| HIGH | 4 | Significant disruption |
| SEVERE | 5 | Major impact expected |

### 18.5 Distance-Weighted Impact

Impact decreases with distance from the event:

$$impact_i = severity \cdot f_{decay}(d_i)$$

**Decay Functions:**

**Exponential Decay:**
$$f_{decay}(d) = \exp\left(-\frac{d}{\lambda}\right)$$

**Inverse Distance:**
$$f_{decay}(d) = \frac{1}{1 + d/\lambda}$$

Where $\lambda$ = decay parameter (default: 10 km for local events).

### 18.6 Combined Event Impact

For multiple simultaneous events:

$$\Delta P_{noshow} = \sum_{e \in Events} w_e \cdot impact_e$$

Where $w_e$ = weight for event type $e$.

### 18.7 Impact on No-Show Rate

$$P_{noshow,adjusted} = P_{noshow,baseline} + \Delta P_{noshow}$$

Clamped to $[0, 1]$.

### 18.8 Weather Impact Coefficients

Learned from historical data:

| Weather Condition | Coefficient |
|-------------------|-------------|
| Clear/Cloudy | 0.0 |
| Light rain | +2% |
| Heavy rain | +5% |
| Snow | +8% |
| Storm | +12% |

### 18.9 Implementation

```python
from ml.event_impact_model import EventImpactModel, analyze_event_impact

model = EventImpactModel()
model.fit(historical_data)

# Analyze event impact
event = model.create_event(
    title="M4 Major Accident",
    description="Multi-vehicle accident causing severe delays on M4 near Cardiff",
    event_type=EventType.TRAFFIC
)

prediction = model.predict_impact([event])
# Returns: EventImpactPrediction(
#     baseline_noshow_rate=0.12,
#     predicted_noshow_rate=0.17,
#     absolute_increase=0.05,
#     relative_increase=41.7%,
#     recommendations=['Consider proactive patient calls', ...]
# )
```

---

## 19. Conformal Prediction

### 19.1 Coverage Guarantee

Conformal prediction provides **distribution-free** prediction intervals with finite-sample coverage guarantee:

$$P(Y_{n+1} \in \hat{C}(X_{n+1})) \geq 1 - \alpha$$

This holds regardless of the underlying distribution, with only the assumption of exchangeability.

### 19.2 Split Conformal Prediction

**Algorithm:**

```
1. Split data into training (D_train) and calibration (D_cal)

2. Train model on D_train:
   f̂(X) → Ŷ

3. Compute non-conformity scores on D_cal:
   s_i = |Y_i - f̂(X_i)|   for regression
   s_i = 1 - f̂(X_i)_Y_i   for classification

4. Compute quantile:
   q̂ = Quantile_{(1-α)(1 + 1/|D_cal|)}(s_1, ..., s_|D_cal|)

5. For new point X_{n+1}:
   Ĉ(X_{n+1}) = {y : |y - f̂(X_{n+1})| ≤ q̂}
              = [f̂(X_{n+1}) - q̂, f̂(X_{n+1}) + q̂]
```

### 19.3 Non-Conformity Scores

**For Regression (Duration):**

$$s(x, y) = |y - \hat{f}(x)|$$

**For Classification (No-Show):**

$$s(x, y) = 1 - \hat{f}(x)_y$$

Where $\hat{f}(x)_y$ is the predicted probability for class $y$.

### 19.4 Prediction Interval (Regression)

$$\hat{C}_{1-\alpha}(X) = [\hat{y} - \hat{q}_{1-\alpha}, \hat{y} + \hat{q}_{1-\alpha}]$$

### 19.5 Prediction Set (Classification)

$$\hat{C}_{1-\alpha}(X) = \{y : \hat{f}(X)_y \geq 1 - \hat{q}_{1-\alpha}\}$$

If threshold is high, the set may contain multiple classes (uncertainty).

### 19.6 Conformalized Quantile Regression (CQR)

CQR provides **adaptive** intervals that are wider when uncertainty is higher:

**Non-Conformity Score:**

$$s_i = \max\{\hat{q}_{\alpha/2}(X_i) - Y_i, Y_i - \hat{q}_{1-\alpha/2}(X_i)\}$$

**Prediction Interval:**

$$\hat{C}(X) = [\hat{q}_{\alpha/2}(X) - \hat{q}, \hat{q}_{1-\alpha/2}(X) + \hat{q}]$$

Where $\hat{q}_\tau(X)$ are quantile regression predictions.

### 19.7 Coverage Calibration

The empirical coverage should match the nominal level:

$$\text{Coverage} = \frac{1}{n_{test}}\sum_{i=1}^{n_{test}} \mathbb{1}(Y_i \in \hat{C}(X_i))$$

**Calibration Plot:** Plot nominal coverage (x-axis) vs empirical coverage (y-axis). Perfect calibration lies on the diagonal.

### 19.8 Interval Efficiency

Narrower intervals are better, measured by average width:

$$\text{Efficiency} = \frac{1}{n}\sum_{i=1}^{n} |\hat{C}(X_i)|$$

**Trade-off:** Higher coverage requires wider intervals.

### 19.9 Implementation

```python
from ml.conformal_prediction import (
    ConformalDurationPredictor,
    ConformalNoShowPredictor,
    ConformizedQuantileRegression
)

# Duration prediction with guaranteed 90% coverage
duration_predictor = ConformalDurationPredictor(alpha=0.1, use_cqr=True)
duration_predictor.fit(X_train, y_duration)

prediction = duration_predictor.predict(patient_features)
# Returns: ConformalPrediction(
#     point_estimate=145.0,
#     lower_bound=118.0,
#     upper_bound=172.0,
#     coverage_guarantee=0.90
# )

# No-show prediction with prediction sets
noshow_predictor = ConformalNoShowPredictor(alpha=0.1)
noshow_predictor.fit(X_train, y_noshow)

result = noshow_predictor.predict(patient_features)
# Returns: {
#     'noshow_probability': 0.25,
#     'prediction_set': [0, 1],  # Uncertain - both outcomes possible
#     'confident': False
# }
# Or for confident prediction:
# Returns: {
#     'noshow_probability': 0.08,
#     'prediction_set': [0],    # Confident - will attend
#     'confident': True
# }
```

### 19.10 Risk-Adaptive α

#### 19.10.1 Motivation

A fixed α = 0.10 is too blunt for scheduling: a routine patient on a quiet morning does not need the same wide prediction interval as a high-risk patient on a peak-occupancy day. The adaptive policy makes α a function of patient risk and live operational state.

#### 19.10.2 Formula

```
α(p, o) = clamp(α_base + β_1 · P_noshow(p) + β_2 · occupancy, α_floor, α_ceil)
```

- `P_noshow(p)` — DFL-calibrated no-show probability from §3.6
- `occupancy` — `app_state['metrics']['chair_utilization'] / 100`
- `α_base = 0.10`, `β_noshow = 0.15`, `β_occupancy = 0.08`
- `α_floor = 0.01` (coverage ≤ 99 %), `α_ceil = 0.20` (coverage ≥ 80 %)

Module-level defaults live in `ml/adaptive_alpha.py`; operator-settable via `POST /api/ml/conformal/adaptive/config`.

#### 19.10.3 Validity

Conformal validity is preserved because α(p, o) depends only on the features (not on the test label). The calibration-score set is still exchangeable with the test point, so the marginal coverage guarantee at each chosen α still holds (Angelopoulos & Bates 2022). The clamp keeps the per-row quantile index in [0, 1] so no corner case can produce an invalid interval. With β_1 = β_2 = 0 the policy is exactly the legacy fixed-α pipeline.

#### 19.10.4 Per-patient quantile lookup

Calibration scores are retained in `self.calibration_scores`. At predict time, when `alpha_adaptive` is supplied, the per-row quantile is `np.quantile(self.calibration_scores, (n_cal+1)·(1-α_i)/n_cal)` — O(log N) per patient.

#### 19.10.5 Integration

| File | Change |
|------|--------|
| `ml/adaptive_alpha.py` | New `AdaptiveAlphaPolicy` singleton with JSON persistence |
| `ml/conformal_prediction.py` | `ConformalPredictor.predict()`, `ConformizedQuantileRegression.predict()`, `ConformalNoShowPredictor.predict()` — all gained optional `alpha_adaptive` kwarg |
| `flask_app.py` | `/api/ml/conformal/duration` and `/api/ml/conformal/noshow` compute α per request; new diagnostic endpoints `/api/ml/conformal/adaptive/status` and `/api/ml/conformal/adaptive/config` |

Deliberately **no UI panel** — the feature runs invisibly in the prediction pipeline; sharper intervals for routine cases and wider intervals for risky ones show up everywhere conformal predictions are consumed.

---

## 20. Monte Carlo Dropout

### 20.1 Bayesian Approximation via Dropout

MC Dropout (Gal & Ghahramani, 2016) reinterprets dropout as approximate variational inference in a Bayesian neural network. By keeping dropout **active during inference** and performing $T$ stochastic forward passes, we sample from the approximate posterior:

$$p(y|x, \mathcal{D}) \approx \frac{1}{T}\sum_{t=1}^{T} p(y|x, \hat{W}_t)$$

Where $\hat{W}_t$ are the weights with dropout mask $t$ applied (i.e., random neurons zeroed out).

### 20.2 Predictive Mean and Variance

**Predictive Mean (point estimate):**

$$\hat{\mu}(x) = \frac{1}{T}\sum_{t=1}^{T} f(x; \hat{W}_t)$$

**Predictive Variance (total uncertainty):**

$$\hat{\sigma}^2(x) = \frac{1}{T}\sum_{t=1}^{T} f(x; \hat{W}_t)^2 - \hat{\mu}(x)^2$$

### 20.3 Uncertainty Decomposition

Total predictive uncertainty decomposes into two independent components:

$$\underbrace{Var[y|x]}_{\text{Total}} = \underbrace{E_W[Var[y|x,W]]}_{\text{Aleatoric}} + \underbrace{Var_W[E[y|x,W]]}_{\text{Epistemic}}$$

| Component | Name | Meaning | Reducible? |
|-----------|------|---------|------------|
| $Var_W[E[y \mid x, W]]$ | **Epistemic** | Model uncertainty due to limited data | Yes (more data reduces it) |
| $E_W[Var[y \mid x, W]]$ | **Aleatoric** | Inherent noise in the data | No (irreducible) |

### 20.4 Estimation from Samples

**Epistemic Uncertainty (model disagreement across forward passes):**

$$\hat{\sigma}^2_{epistemic} = \frac{1}{T}\sum_{t=1}^{T} \left(f(x; \hat{W}_t) - \hat{\mu}(x)\right)^2$$

**Aleatoric Uncertainty (for Bernoulli no-show):**

$$\hat{\sigma}^2_{aleatoric} = \hat{\mu}(x) \cdot (1 - \hat{\mu}(x))$$

### 20.5 Credible Intervals

From the $T$ samples, compute empirical quantiles:

$$CI_{1-\alpha} = \left[Q_{\alpha/2}(\{f_t\}), Q_{1-\alpha/2}(\{f_t\})\right]$$

Where $Q_\tau$ is the $\tau$-quantile of the $T$ forward pass outputs.

### 20.6 Connection to Variational Inference

Dropout training minimises the KL divergence between the approximate posterior $q(W)$ and the true posterior $p(W|\mathcal{D})$:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log p(y_i | f(x_i; W)) + \frac{1}{N} KL(q(W) || p(W))$$

Where:
- $q(W)$ = Bernoulli variational distribution (dropout mask)
- $p(W)$ = Prior (implicit $L_2$ regularization)
- The $KL$ term corresponds to weight decay

### 20.7 Dropout Rate as Prior Precision

The dropout rate $p$ and weight decay $\lambda$ together determine the model's prior:

$$\tau = \frac{p \cdot l^2}{2N\lambda}$$

Where:
- $l$ = prior length-scale
- $N$ = training set size
- Higher dropout → more uncertainty → wider intervals

### 20.8 Application to SACT Scheduling

**No-Show Prediction:**
- $T = 100$ forward passes through MultiTaskNetwork
- Each pass applies different dropout mask
- High epistemic uncertainty → model needs more data for this patient type
- High aleatoric uncertainty → inherent unpredictability

**Duration Prediction:**
- Same $T$ passes give duration distribution
- Use $Q_{0.90}$ for conservative scheduling buffer

### 20.9 Practical Configuration

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| $T$ (forward passes) | 100 | 50-200 | More = better but slower |
| $p$ (dropout rate) | 0.3 | 0.1-0.5 | From trained model |
| Confidence level | 0.95 | 0.90-0.99 | For credible intervals |

### 20.10 Implementation

```python
from ml.mc_dropout import MonteCarloDropout, predict_with_mc_dropout

# Method 1: Using class
mc = MonteCarloDropout(n_forward_passes=100, confidence_level=0.95)
result = mc.predict_multitask(
    network=multitask_model.network,
    patient_features=features,
    patient_id='P12345'
)

# Method 2: Convenience function
result = predict_with_mc_dropout(
    model=multitask_model.network,
    patient_features=features,
    n_iter=100,
    confidence=0.95
)

# Result:
# MCDropoutResult(
#     noshow_mean=0.18,
#     noshow_std=0.04,           # Epistemic spread
#     noshow_ci_lower=0.11,
#     noshow_ci_upper=0.26,
#     duration_mean=152.3,
#     duration_std=8.7,
#     epistemic_uncertainty=0.042,
#     aleatoric_uncertainty=0.148,
#     total_uncertainty=0.154,
#     interpretation="Moderate confidence in no-show prediction (18% +/- 4%)"
# )
```

---

## 22. Auto-Learning Pipeline

### 22.1 Three-Tier Data Architecture

The system ingests data from three tiers of increasing specificity:

| Tier | Source | Availability | Update | Recalibration Level |
|------|--------|-------------|--------|---------------------|
| **1** | NHS Open Data (CWT Jan 2026, NHSBSA SCMD-IP Jan 2026) | Now | Monthly/Quarterly | Level 1-2 |
| **2** | Local Hospital Data (ChemoCare exports) | On access | Ad-hoc | Level 2-3 |
| **3** | SACT v4.0 Patient-Level Submissions | **August 2026** (first complete) | Monthly | Level 3 |

> **SACT v4.0 NOTE:** Collection commenced 1 April 2026.
> Rollout period April–June 2026 (partial trust submissions) — **usable for Level 1-2 recalibration** (quality=`preliminary`).
> Full conformance July 2026 — **Level 2 recalibration** (quality=`conformance`).
> First complete dataset August 2026 — **Level 3 full retrain** (quality=`complete`).
> Auto-detection via `datasets/nhs_open_data/sact_v4/` directory. API: `POST /api/data/sact-v4/check`.

### 22.2 Recalibration Levels

| Level | Trigger | What Changes | Downtime |
|-------|---------|--------------|----------|
| **0** | Real-time event data | Weather/traffic coefficients | None |
| **1** | Monthly open data | No-show base rates, seasonal factors | None |
| **2** | Quarterly data or moderate drift | Feature weights, demographic adjustments | <1 min |
| **3** | Significant drift or new data tier | Full model retrain, hyperparameter tuning | 2-5 min |

> **Floor rule (v4.3):** `recommended_recalibration_level` always returns at least **1** — never `None`.
> During the SACT v4.0 rollout (April–June 2026), when no local CSV files are present, the ingester
> defaults to Level 1 so models always recalibrate on synthetic data rather than stalling.

### 22.3 Baseline Recalibration from CWT Data

Monthly update of no-show baseline rates using Cancer Waiting Times:

$$P_{base}^{(new)} = \alpha \cdot P_{observed} + (1 - \alpha) \cdot P_{base}^{(old)}$$

Where $\alpha = 0.2$ (exponential moving average learning rate).

### 22.4 Population Prior Update (Hierarchical Model)

When national data updates the population mean:

$$\mu_{prior}^{(new)} = \frac{n_{local} \cdot \bar{y}_{local} + n_{national} \cdot \bar{y}_{national}}{n_{local} + n_{national}}$$

### 22.5 Adaptive Optimization Weights

Optimizer weights adjust based on observed performance gaps:

$$w_{overbooking}^{(new)} = w_{overbooking}^{(old)} \cdot \frac{P_{noshow}^{observed}}{P_{noshow}^{predicted}}$$

If observed no-show rate exceeds predictions, overbooking threshold increases.

### 22.6 Online Learning Formulas

For real-time model updates as each new appointment outcome arrives:

#### Stochastic Gradient Descent (SGD)

For the no-show classifier and duration regressor, weights update after each observation:

$$\theta_{t+1} = \theta_t - \eta_t \cdot \nabla L(\theta_t; x_t, y_t)$$

Where:
- $\theta_t$ = model parameters at time $t$
- $\eta_t$ = learning rate (decaying: $\eta_t = \eta_0 / t^{0.25}$)
- $\nabla L$ = gradient of loss function (log-loss for classification, MSE for regression)
- $(x_t, y_t)$ = new observation (features, outcome)

Implementation uses `sklearn.SGDClassifier` and `SGDRegressor` with `partial_fit()` for incremental updates.

#### Bayesian Online Updating

For the no-show rate posterior (Beta-Bernoulli conjugate):

$$P(\theta | D_{1:t}) \propto P(D_t | \theta) \cdot P(\theta | D_{1:t-1})$$

Specifically:
- **Prior:** $\theta \sim \text{Beta}(\alpha, \beta)$
- **Likelihood:** $x_t \sim \text{Bernoulli}(\theta)$
- **Posterior:** $\theta | x_t \sim \text{Beta}(\alpha + x_t, \beta + 1 - x_t)$

For the duration mean (Normal-Normal conjugate):

$$\mu_n = \frac{n_0 \cdot \mu_0 + x_t}{n_0 + 1}, \quad \sigma^2_n = \frac{\sigma^2_0 \cdot n_0}{n_0 + 1}$$

Each new observation shrinks the posterior variance, making estimates more precise.

#### Exponential Moving Average (EMA)

For fast baseline tracking:

$$\mu_{t+1} = \alpha \cdot x_t + (1 - \alpha) \cdot \mu_t$$

Where $\alpha = 0.1$ (default). Applied to:
- No-show rate baseline
- Average treatment duration
- Weather impact coefficient

#### When Each Method Is Used

| Method | Speed | Use Case | Updated Parameter |
|--------|-------|----------|-------------------|
| **EMA** | Instant | Baselines | No-show rate, avg duration |
| **Bayesian** | Instant | Posteriors | Rate distributions with uncertainty |
| **SGD** | ~1ms | Classifiers | No-show/duration model weights |
| **Level 1-3** | Seconds-Minutes | Full models | All 12 models (batch retrain) |

---

### 22.7 Reinforcement Learning for Dynamic Scheduling

Q-Learning agent that learns optimal chair assignment policies from experience:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

Where:
- $s$ = state (hour, chairs occupied, queue size, avg no-show risk, urgent count)
- $a$ = action (assign to chair, delay, double-book, reject)
- $r$ = reward = $-\text{waiting\_time} - \lambda \cdot \text{noshow\_risk} + \text{utilization} + \text{priority}$
- $\alpha = 0.1$ (learning rate)
- $\gamma = 0.95$ (discount factor)
- $\epsilon = 0.1$ (exploration rate for $\epsilon$-greedy policy)

**State Space:** hour(10) × occupancy(7) × queue(6) × risk(5) × urgent(4) = 8,400 discrete states

**Action Space:** 4 types (assign, delay, double-book, reject) × 19 chairs = 76 actions

**Reward Function:**

$$r = -\frac{t_{wait}}{60} - \lambda \cdot P_{noshow} \cdot 10 + U \cdot 2 + (5 - p) \cdot 0.5 + r_{outcome}$$

Where $t_{wait}$ = waiting time, $\lambda = 0.3$, $U$ = utilization, $p$ = priority (1-4), $r_{outcome}$ = +1 if attended, -2 if no-show.

**Training:** Offline on historical appointment data, then online updates as new outcomes arrive.

### 22.8 Multi-Agent RL for Chair Assignment (MARL)

Extends the single-agent approach: each chair $c$ is an **independent agent** with its own Q-table, coordinated via shared reward.

#### Environment Design

**Per-chair state** $s_c$:

$$s_c = (\text{hour}, \text{occupied}_c, \text{priority}_c, \text{remaining}_c, \text{gap}_c, \text{queue})$$

State space per agent: $10 \times 2 \times 5 \times 5 \times 7 \times 6 = 21{,}000$ discrete states.

**Per-chair action** $a_c \in \{\text{accept}, \text{reject}, \text{delay}\}$:
- **accept**: take next patient from global queue
- **reject**: pass patient to another chair agent
- **delay**: wait for a higher-priority patient

**Shared reward** (team objective — all agents receive the same R):

$$R = w_{\text{util}} \cdot U - w_{\text{wait}} \cdot \frac{T_{\text{wait}}}{60} - w_{\text{noshow}} \cdot \pi \cdot 10 + w_{\text{fair}} \cdot (1 - \text{Var}[\text{priority}])$$

Where $w_{\text{util}} = 2.0$, $w_{\text{wait}} = 1.0$, $w_{\text{noshow}} = 0.5$, $w_{\text{fair}} = 0.3$.

#### Independent Q-Learning per Chair

Each agent $c$ updates independently:

$$Q_c(s_c, a_c) \leftarrow Q_c(s_c, a_c) + \alpha \left[ R + \gamma \max_{a'_c} Q_c(s'_c, a'_c) - Q_c(s_c, a_c) \right]$$

#### Coordination Mechanism

Agents coordinate through:
1. **Shared reward**: all agents benefit from good team outcomes
2. **Q-value voting**: when assigning a patient, all agents propose actions — the agent with highest $Q(s_c, \text{accept})$ gets the assignment
3. **Fallback**: if no agent accepts, assign to least-occupied chair

#### MARL vs Single-Agent vs CP-SAT

| Property | Single-Agent Q | MARL (per-chair) | CP-SAT |
|----------|---------------|-------------------|--------|
| Scalability | $O(8400 \times 76)$ | $O(19 \times 21000 \times 3)$ | NP-hard |
| Coordination | Centralized | Decentralized + shared reward | Global optimal |
| Online learning | Yes | Yes (per agent) | No |
| Computation | Instant | Instant | 5-10s |
| Fairness | Reward term | Reward term + voting | Hard constraint |

#### Implementation

```python
from ml.rl_scheduler import MultiAgentChairScheduler, ChairState

marl = MultiAgentChairScheduler(n_chairs=19)
marl.train_on_historical(records, n_epochs=5)

# Each chair agent votes on patient assignment
assignments = marl.assign_patients(patients, chair_states)
# Returns: [{patient_id, chair_id, action, q_value}, ...]
```

---

## 23. Drift Detection

### 23.1 Population Stability Index (PSI)

For detecting feature distribution shift between reference (training) data and new incoming data:

$$PSI = \sum_{i=1}^{B} (P_i^{new} - P_i^{ref}) \cdot \ln\left(\frac{P_i^{new}}{P_i^{ref}}\right)$$

Where:
- $P_i^{new}$ = proportion of new data in bin $i$
- $P_i^{ref}$ = proportion of reference data in bin $i$
- $B$ = number of bins (default 10)

**Decision thresholds:**

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | No significant shift | No action |
| 0.10 - 0.25 | Moderate shift | Level 1-2 recalibration |
| > 0.25 | Significant shift | Level 3 full retrain |

### 23.2 Kolmogorov-Smirnov Test

Two-sample KS test for distribution comparison:

$$D_{n,m} = \sup_x |F_n^{ref}(x) - F_m^{new}(x)|$$

Reject $H_0$ (same distribution) if $p < 0.05$.

### 23.3 CUSUM (Cumulative Sum) for Gradual Drift

For detecting slow changes in model performance over time:

$$S_t^+ = \max(0, S_{t-1}^+ + z_t - k)$$
$$S_t^- = \max(0, S_{t-1}^- - z_t - k)$$

Where:
- $z_t = (x_t - \mu_0) / \sigma_0$ (standardized residual)
- $k = 0.5$ (allowance parameter)
- Alarm when $S_t^+$ or $S_t^-$ exceeds threshold $h = 5$

### 23.4 Performance Decay Monitoring

Track key metrics over a sliding window:

$$\Delta AUC = AUC_{baseline} - AUC_{current}$$

| Metric | Threshold | Action |
|--------|-----------|--------|
| AUC-ROC drop > 3% | Moderate | Level 2 recalibration |
| MAE increase > 15% | Moderate | Level 2 recalibration |
| Either exceeds 2x threshold | Severe | Level 3 retrain |

---

## 24. SACT v4.0 Data Standard

### 24.1 Overview

SACT v4.0 is the NHS England national standard for reporting systemic anti-cancer therapy data. Data collection commenced 1 April 2026 with a 3-month rollout period. It defines 60 data items across 7 sections.

> **Data Availability:** SACT v4.0 collection commenced 1 April 2026. The rollout period runs April–June 2026 (partial trust submissions only). Full conformance required from 1 July 2026. **First complete monthly dataset expected August 2026.** This dissertation uses synthetic data structured to the v4.0 schema, enabling seamless future integration.

### 24.2 Data Classification

| Classification | Code | Meaning |
|---------------|------|---------|
| Mandatory | M | Must be submitted; record rejected without it |
| Required | R | Must be included where available or applicable |
| Conditional | C | Required only if parent condition is met |

### 24.3 Submission Format

- **File format:** CSV only (`.csv`)
- **Date format:** `ccyy-mm-dd`
- **Text delimiter:** Double-quote for comma-containing values
- **File naming:** `UnitID-ccyymmdd-ccyymmdd.csv`
- **Frequency:** Monthly (2-month delayed schedule)
- **Connection:** Health and Social Care Network (HSCN)

### 24.4 Modification Reason Codes (SACT v4.0 Section 5)

| Code | Reason | System Mapping |
|------|--------|----------------|
| 0 | No modification | Attended_Status = 'Yes' |
| 1 | Patient choice | Cancellation_Reason = 'Patient' |
| 2 | Organisational issues | Cancellation_Reason = 'Scheduling' |
| 3 | Patient clinical factors | Cancellation_Reason = 'Medical' |
| 4 | Toxicity | Cancellation_Reason = 'Toxicity' |

### 24.5 Toxicity Grading (CTCAE v5.0)

| Grade | Severity | Impact on Scheduling |
|-------|----------|---------------------|
| 0 | No adverse event | Normal scheduling |
| 1 | Mild | Monitor, no change |
| 2 | Moderate | May delay next cycle |
| 3 | Severe | Likely delay or dose reduction |
| 4 | Life-threatening | Treatment hold |
| 5 | Death | End of regimen |

### 24.6 Performance Status (WHO Scale)

Used in both SACT v4.0 reporting and scheduling priority:

| PS | Description | Scheduling Priority | No-Show Risk |
|----|-------------|-------------------|-------------|
| 0 | Fully active | P3-P4 | Low |
| 1 | Restricted but ambulatory | P2-P3 | Low-Medium |
| 2 | Ambulatory, <50% in bed | P2 | Medium |
| 3 | Limited self-care, >50% in bed | P1-P2 | Medium-High |
| 4 | Completely disabled | P1 | High |

---

## 24b. Uncertainty-Aware Optimization (DRO)

### 24b.1 Wasserstein Ambiguity Set

Instead of optimizing under the empirical distribution $\hat{P}$, we optimize under the worst-case distribution within a Wasserstein ball:

$$\mathcal{P} = \{Q : W_2(\hat{P}, Q) \leq \varepsilon\}$$

where $W_2$ is the 2-Wasserstein distance and $\varepsilon$ controls robustness (larger = more conservative).

### 24b.2 DRO Objective

$$\min_{\mathbf{x}} \max_{Q \in \mathcal{P}} \mathbb{E}_Q[\text{loss}(\mathbf{x})]$$

For our scheduling problem, this reformulates to a tractable convex program.

### 24b.3 Worst-Case No-Show Probability

For patient $p$ with estimated no-show probability $\hat{\pi}_p$ and variance $\sigma_p^2$:

$$\pi_p^{\text{worst}} = \hat{\pi}_p + \varepsilon \sqrt{\sigma_p^2 + \varepsilon^2}$$

This is the maximum expected no-show rate over all distributions $Q$ within the Wasserstein ball (Mohajerin Esfahani & Kuhn, 2018).

### 24b.4 CVaR Duration Buffer

Conditional Value-at-Risk protects against the worst $(1-\alpha)$ fraction of duration outcomes:

$$\text{VaR}_\alpha = \mu_D + z_\alpha \sigma_D$$

$$\text{CVaR}_\alpha = \mu_D + \sigma_D \cdot \frac{\phi(z_\alpha)}{1 - \alpha}$$

where $\phi$ is the standard normal PDF, $z_\alpha$ is the $\alpha$-quantile, and $\alpha = 0.90$ (default).

### 24b.5 Schedule Robustness Evaluation

Given $N$ perturbation scenarios $\{Q_1, \ldots, Q_N\}$ sampled from the ambiguity set:

$$\text{Mean} = \frac{1}{N} \sum_{k=1}^{N} Z(Q_k)$$

$$\text{CVaR}_\alpha = \frac{1}{\lfloor N(1-\alpha) \rfloor} \sum_{k=1}^{\lfloor N(1-\alpha) \rfloor} Z_{(k)}$$

where $Z_{(k)}$ are the sorted (ascending) objective values.

### 24b.6 Integration with CP-SAT

The DRO worst-case penalties replace point estimates in the CP-SAT objective:

```python
# Before DRO: noshow_penalty = int(p.noshow_probability * 100)
# After DRO:  noshow_penalty = robust_penalties[p.patient_id]  # worst-case
```

Duration buffers from CVaR are applied to `patient.expected_duration` before solving.

### 24b.7 Full Scenario-Based DRO + CVaR in CP-SAT

#### Step 1: Scenario Generation within Wasserstein Ball

Generate $K = 10$ scenarios of (no-show, duration) realizations. Each scenario $k$ perturbs the empirical distribution within the Wasserstein ball:

$$(\pi_p^{(k)}, d_p^{(k)}) \sim Q_k, \quad W_2(\hat{P}, Q_k) \leq \varepsilon$$

$$\pi_p^{(k)} = \text{clip}(\hat{\pi}_p + \mathcal{N}(0, \varepsilon), 0.01, 0.95)$$

$$d_p^{(k)} = \max(15, \hat{d}_p + \mathcal{N}(0, 0.10 \cdot \hat{d}_p))$$

Then for each scenario, sample a Bernoulli realization:

$$\text{shows}_p^{(k)} \sim \text{Bernoulli}(1 - \pi_p^{(k)})$$

#### Step 2: Scenario Utility Variables

For each scenario $k$, define the scenario utility as a CP-SAT integer variable:

$$U_k = \sum_{p \in P} c_p^{(k)} \cdot x_p$$

where:
$$c_p^{(k)} = \begin{cases} (5 - \text{priority}_p) \times 100 & \text{if } \text{shows}_p^{(k)} = 1 \\ -\lfloor d_p^{(k)} / 2 \rfloor & \text{if } \text{shows}_p^{(k)} = 0 \end{cases}$$

#### Step 3: DRO Robust Counterpart

The worst-case scenario utility is tracked as a CP-SAT variable:

$$U_{\text{worst}} = \min_{k=1}^{K} U_k$$

This is enforced via `AddMinEquality`. The optimizer receives a bonus for improving the worst-case outcome, preventing schedules that collapse under any scenario within the ambiguity set.

#### Step 4: CVaR with Binary Indicators (Rockafellar & Uryasev, 2000)

Instead of just maximizing $\mathbb{E}[U]$, maximize:

$$\max_{\pi} \text{CVaR}_\alpha(U(\pi)) = \max_{\pi} \left( \frac{1}{\alpha} \int_0^\alpha F_U^{-1}(p) \, dp \right)$$

Linearized for CP-SAT via auxiliary variables:

$$\max \left\{ \eta - \frac{1}{n_{\text{worst}}} \sum_{k=1}^{K} z_k \right\}$$

Subject to:
- $z_k \geq \eta - U_k, \quad z_k \geq 0 \quad \forall k$ (shortfall slack)
- $w_k \in \{0, 1\} \quad \forall k$ (binary: identifies worst-case scenarios)
- $\sum_{k=1}^{K} w_k = n_{\text{worst}} = \lceil K(1-\alpha) \rceil$

Where:
- $\eta$ = VaR auxiliary variable (threshold)
- $z_k$ = shortfall of scenario $k$ below $\eta$
- $w_k$ = binary indicator: 1 if scenario $k$ is in the worst $(1-\alpha)$ tail
- $n_{\text{worst}} = \lceil K \cdot (1-\alpha) \rceil$ = number of tail scenarios

#### Step 5: Combined Objective

$$\max Z = \underbrace{\sum_i w_i \cdot Z_i}_{\text{6-objective E[U]}} + w_{\text{CVaR}} \cdot \underbrace{\left(\eta - \frac{1}{n_{\text{worst}}} \sum_k z_k\right)}_{\text{CVaR tail protection}} + \underbrace{U_{\text{worst}}}_{\text{DRO robust floor}}$$

Parameters: $w_{\text{CVaR}} = 3$, $\alpha = 0.90$, $K = 10$ scenarios, $\varepsilon = 0.05$.

This guarantees that the schedule performs well not just on average, but even under the worst 10% of no-show and duration realizations within the Wasserstein ambiguity set — critical for NHS operational reliability.

#### Empirical Comparison

| Metric | Standard $\mathbb{E}[U]$ | DRO + CVaR |
|--------|--------------------------|------------|
| Utilization | 0.444 | **0.536** (+21%) |
| No-show exposure | 0.242 | 0.242 |
| R(S) robustness | 0.500 | 0.500 |
| Critical slots | $> 0$ possible | **0** |
| Tight slots | $> 0$ possible | **0** |
| Solve time | 5.0s | 10.1s |

The CVaR formulation produces schedules with higher utilization and zero critical/tight slots, at the cost of approximately 2x solve time due to the additional binary variables and scenario constraints.

### 24b.8 Data Requirements for DRO

The uncertainty-aware optimization requires:

**Historical outcomes** (from `historical_appointments.xlsx`):
- `Attended_Status` — binary no-show outcomes for distribution estimation
- `Actual_Duration` — realized durations for variance calibration
- `Date` — timestamps for detecting seasonal shifts and trends

**Epsilon calibration** from historical distributional shifts:

$$\varepsilon = \max\left(\Delta_{\text{monthly}}^{\max}, 0.3 \cdot \text{CV}_D\right) \times \gamma_{\text{safety}}$$

Where:
- $\Delta_{\text{monthly}}^{\max} = \max_t \pi_t - \min_t \pi_t$ is the maximum month-to-month no-show rate shift
- $\text{CV}_D = \sigma_D / \mu_D$ is the coefficient of variation of actual durations
- $\gamma_{\text{safety}} = 1.2$ (20% safety margin)

**Disruption scenario profiles** calibrate $\varepsilon$ for different operating conditions:

| Scenario | $\varepsilon$ | Description |
|----------|--------------|-------------|
| Baseline | $0.5 \times \hat{\varepsilon}$ | Normal operations |
| Seasonal peak | $\hat{\varepsilon}$ | Winter flu / summer holidays |
| Moderate disruption | $1.5 \times \hat{\varepsilon}$ | NHS strikes, severe weather, transport failure |
| Major disruption | $3.0 \times \hat{\varepsilon}$ | Pandemic-level event (COVID-like) |

### 24b.9 Evaluation Metrics

**Worst-case utilization** — schedule performance under the most adverse scenario:

$$U_{\text{worst}} = \min_{k=1}^{K} \frac{\sum_{p \in P_k^{\text{show}}} d_p}{\sum_{c} T_c}$$

where $P_k^{\text{show}}$ are patients who attend in scenario $k$ and $T_c$ is chair capacity.

**Stability** — variance in performance across scenarios:

$$\text{CV}(U) = \frac{\sigma_U}{\mu_U} = \frac{\sqrt{\frac{1}{K}\sum_k (U_k - \bar{U})^2}}{\bar{U}}$$

Interpretation:
- $\text{CV} < 0.10$: Stable schedule
- $0.10 \leq \text{CV} < 0.20$: Moderate variability
- $\text{CV} \geq 0.20$: High variability — increase $\varepsilon$

**CVaR of utilization** — expected utilization in the worst $(1-\alpha)$ fraction:

$$\text{CVaR}_\alpha(U) = \frac{1}{\lceil K(1-\alpha) \rceil} \sum_{k=1}^{\lceil K(1-\alpha) \rceil} U_{(k)}$$

where $U_{(k)}$ are sorted utilizations (ascending).

**Tail probability** — risk of falling below acceptable utilization:

$$P(U < U_{\text{threshold}}) = \frac{|\{k : U_k < 0.5\}|}{K}$$

---

## 25. Model Summary Table

| Model | Section | Purpose | Key Output | Data Source |
|-------|---------|---------|------------|-------------|
| **Ensemble (RF+GB+XGB)** | 3-4 | No-show & duration prediction | Probability, minutes | Historical appointments |
| **Sequence (GRU)** | 3.6 | Patient history patterns | Probability adjustment | Patient sequences |
| **Survival (Cox PH)** | 10 | Time-to-event modeling | Hazard ratios, survival curves | Event times |
| **Uplift (S+T Learner)** | 11 | Intervention effectiveness | ITE, recommendations | Treatment assignments |
| **Multi-Task (NN)** | 12 | Joint prediction | Both outputs, shared learning | All features |
| **Quantile Forest** | 13 | Distribution-free intervals | Quantiles, prediction intervals | Duration data |
| **Hierarchical (PyMC)** | 14 | Patient random effects | Shrinkage estimates, credible intervals | Grouped data |
| **Causal DAG** | 15 | Causal structure | do-calculus effects | Observational data |
| **IV (2SLS)** | 16 | Endogeneity correction | LATE, F-statistic | Instrument + outcome |
| **DML** | 17 | High-dimensional causal | ATE with valid CI | Treatment + confounders |
| **Event Impact** | 18 | External event effects | No-show rate adjustment | Weather/traffic/news |
| **Conformal** | 19 | Coverage guarantee | Valid prediction intervals | Calibration set |
| **MC Dropout** | 20 | Bayesian uncertainty | Epistemic/aleatoric decomposition | Existing NN + dropout |
| **Sensitivity Analysis** | 15b | Feature importance | S_i (local), I_j (global), E_i (elasticity) | Trained model + data |
| **Model Cards** | 15c | ML transparency | Subgroup metrics, fairness, ethics | All models |
| **Causal Validation** | 15.8 | Model integrity | Placebo, falsification, sensitivity tests | Historical data |
| **Counterfactual Explanations** | 15.7 | Explainability | Minimal feature changes for different outcome | Patient features |
| **Online Learning** | 22.6 | Real-time updates | SGD + Bayesian posterior + EMA | Streaming outcomes |
| **RL Scheduler** | 22.7 | Dynamic scheduling | Q-Learning policy (8400 states, 76 actions) | Historical scheduling |
| **MARL Chair Agents** | 22.8 | Multi-agent chair assignment | 19 independent Q-learners, shared reward, Q-value voting | Historical scheduling |
| **DRO (Wasserstein)** | 24b | Distributionally robust optimization | Worst-case penalties, CVaR buffers | Ambiguity set around empirical |
| **Auto-Learning** | 22 | Continuous improvement | Recalibrated models | NHS open data |
| **Drift Detection** | 23 | Model monitoring | PSI, KS test, CUSUM alerts | Incoming data streams |

---

## 27. Algorithmic Complexity Guarantees

For every core optimisation component the system carries an explicit
complexity bound — both for capacity-planning (how big an instance can
we solve in the time budget?) and for falsifying performance regressions
(if a profile run blows past the bound, a regression has crept in).
Constants where given are measured on the synthetic Velindre dataset
($n=202$ patients, $m=45$ chairs) on a workstation-class CPU.

### 27.1 CP-SAT (monolithic) — §2

| Resource          | Bound                                                                                                                                                                                            |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Variables         | $n \cdot m$ binary $x_{p,c}$ (assignment) + $n$ integer start times                                                                                                                              |
| Constraints       | $O(m \cdot n^2)$ no-overlap edges over the chair-interval graph                                                                                                                                  |
| Time complexity   | NP-hard in general; OR-Tools clause learning solves typical instances ($n \leq 50$, $m = 45$) in $< 5$ s.  Hard wall-clock cap: 300 s default (`SOLVER_TIME_LIMIT_SECONDS = 300` in `config.py`); the auto-scaling cascade (§A.9) tightens this to a 5 / 2 / 1 / 0.5 s budget sequence in practice.                       |
| Space complexity  | $O(m \cdot n^2)$ for the constraint graph; learnt-clause memory bounded by OR-Tools' `--clauses_cleanup_limit`.                                                                                  |
| Optimality gap    | $\leq 1\%$ when the early-stopping rule fires (`abs(objective − best_bound) / abs(objective) ≤ 0.01`).  Status `STATUS_OPTIMAL` proves $0\%$ gap.                                                |
| Practical limit   | $n \approx 50$ at $m = 45$; beyond that, column generation (§27.3) takes over via `COLUMN_GEN_THRESHOLD = 50`.                                                                                   |

### 27.2 GNN feasibility pre-filter — §2.11

| Resource          | Bound                                                                                                                                                                                            |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Training          | Online, after every 5 solves; logistic-regression / random-forest fit over the rolling buffer (size 1000), $F = 44$ edge features.  Cost: $O(\text{buffer} \cdot F \log F)$ per refit.            |
| Inference         | $O(n \cdot m \cdot (d_P + d_C))$ with $d_P = d_C = 20$ embedding dims and a fixed $R = 2$ rounds of mean-pooled message passing.  Per-pair classifier eval is $O(F)$.                            |
| Space             | $O(n \cdot d_P + m \cdot d_C)$ embeddings + $O(\text{buffer} \cdot F)$ training cache.                                                                                                            |
| Prune rate        | Converges to 55–70 \% of $(p, c)$ pairs after $\geq 20$ training solves; below the convergence threshold the pruner is held in shadow mode (records but does not prune) for safety.              |
| Soundness         | Pinned by `tests/test_gnn_feasibility.py`: every $(p, c)$ pair that CP-SAT actually used in a baseline schedule must survive the pruner.                                                          |

### 27.3 Column generation — §2.12

| Resource           | Bound                                                                                                                                                                                            |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Master LP per iter | $O(K^3)$ where $K$ is the number of columns currently in the restricted master; $K$ grows to $\sim 2m$ at convergence.  Solved with GLOP.                                                        |
| Pricing subproblem | One CP-SAT solve per chair, $n$ binary variables each; per subproblem $\leq 5.0$ s (`CG_SUBPROBLEM_TIME_LIMIT = 5.0`).                                                                            |
| Iterations         | Typically 10–30 to LP optimality; capped at `CG_MAX_ITERATIONS = 100` with reduced-cost tolerance `CG_REDUCED_COST_TOLERANCE = 1e-4`.                                                            |
| Total time         | For $n = 100$, $m = 45$: $\approx 5$ s (vs. monolithic CP-SAT timeout $> 60$ s on the same instance — verified by `tests/test_column_generation.py` ground-truth comparison).                     |
| Space              | $O(K \cdot n)$ for the column matrix + per-chair CP-SAT models held warm.                                                                                                                       |
| Convergence        | Guaranteed LP-optimal when the best reduced cost $\leq$ `CG_REDUCED_COST_TOLERANCE`; integer rounding (branch-and-price-lite) produces a feasible integer solution within $\approx 1\%$ of LP.   |

### 27.4 Conformal prediction — §19

| Resource              | Bound                                                                                                                                                                                            |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Calibration           | One quantile sort over $n_{\text{cal}}$ calibration scores: $O(n_{\text{cal}} \log n_{\text{cal}})$ at fit time.                                                                                  |
| Inference (fixed $\alpha$) | $O(1)$ — quantile cached.                                                                                                                                                                  |
| Inference (adaptive $\alpha$, §19.10) | $O(\log n_{\text{cal}})$ per prediction (binary search on the sorted score array for the patient-specific quantile).                                                          |
| Space                 | $O(n_{\text{cal}})$ for the score array.                                                                                                                                                         |
| Coverage guarantee    | $\Pr\!\bigl(Y_{n+1} \in \widehat{C}(X_{n+1})\bigr) \geq 1 - \alpha$ in finite samples, under exchangeability of $(X_i, Y_i)_{i=1}^{n+1}$.                                                        |
| Adaptive bound        | The risk-adaptive policy still satisfies marginal coverage $\geq 1 - \bar{\alpha}$ where $\bar{\alpha} = \mathbb{E}[\alpha(\text{patient})]$ — pinned by `tests/test_adaptive_alpha.py`.          |

### 27.5 MPC rollout — §A.13

| Resource          | Bound                                                                                                                                                                                            |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| State size        | $O(m + \lvert Q \rvert)$, where $\lvert Q \rvert \leq n$.                                                                                                                                        |
| Action space      | $\lvert \mathcal{A} \rvert = m + 1$ per step (assign to any chair, or wait).                                                                                                                     |
| Rollout per call  | $K \cdot \lvert \mathcal{A} \rvert \cdot (\tau / \Delta t)$ steps, each $O(m \log m)$ for the greedy-fill heuristic.  With defaults $K = 10$, $\tau = 60$ min, $\Delta t = 5$ min, $m = 45$: $\approx 10 \cdot 46 \cdot 12 = 5{,}520$ steps.  Decision latency $< 500$ ms (`DEFAULT_TOTAL_TIMEOUT_S = 0.5$). |
| Space             | $O(K \cdot \tau / \Delta t)$ scenario buffers; one persistent `MPCDecision` row per call appended to `decisions.jsonl`.                                                                          |
| Fail-safe         | When the wall budget is exceeded, the controller records `used_fallback = True` and emits a deterministic priority-first greedy schedule.  Hard guarantee: a decision is always produced.        |

### 27.6 Distributionally Robust Fairness (DRO) — §A.5

| Resource          | Bound                                                                                                                                                                                            |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Scenarios         | $K = 10$ Wasserstein-perturbed empirical distributions (`DEFAULT_N_SCENARIOS`).                                                                                                                  |
| Worst-case eval   | $O(K \cdot G^2)$ per certify call, where $G = $ number of protected groups (typically $G \leq 5$ for age band, gender, postcode tertile).                                                        |
| CVaR buffer       | $O(n \log n)$ to sort patients by predicted no-show before building the upper-tail mean.                                                                                                         |
| Space             | $O(K \cdot n)$ scenario tables; certificate is a single JSONL row per `data_cache/dro_fairness/certificates.jsonl`.                                                                              |
| Certificate       | Worst-case demographic-parity gap $\leq \delta$ inside the Wasserstein ball of radius $\varepsilon$; finite-sample correction from §A.5.3 widens $\delta$ by $O\!\bigl(\sqrt{\log G / n}\bigr)$. |

---

## 28. Numerical Stability Notes

Many of the closed-form expressions in this document involve logarithms,
divisions, exponentials, or square roots that can underflow / overflow
when their argument approaches a degenerate value (zero probabilities,
empty groups, perfect-fit residuals).  The implementation applies the
following safeguards uniformly so a corner case in the input distribution
never propagates as a NaN or `-inf` to a CP-SAT objective term or a
calibration cost.  Each row is a verified live behaviour, citing
file:line.

| Operation / Site                                              | Safeguard                                                              | Rationale                                                                                              | Live in code                                              |
|---------------------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| $\log(p)$ in BMA log-likelihood                              | $\log(\max(p,\ 10^{-15}))$ via `epsilon = 1e-15`                       | Prevents $-\infty$ when an ensemble base learner predicts $p = 0$ on the BMA hold-out fold             | `ml/noshow_model.py:217`                                  |
| Softmax $w_i = \exp(\text{LL}_i / T) / \sum_j \exp(\text{LL}_j / T)$ over BMA log-likelihoods | Subtract $\max_j \text{LL}_j$ before exponentiation                    | Prevents `exp` overflow on log-likelihoods of order $10^3$ for large calibration sets                 | `ml/noshow_model.py:231–233`                              |
| CP-SAT no-show penalty term                                   | `int(p.noshow_probability * 100)` after upstream sigmoid clamp to $[0,1]$ | Keeps the integer objective coefficient bounded in $[0, 100]$; CP-SAT requires `int64` coefficients   | `optimization/optimizer.py:702`                           |
| Standard-error $\sqrt{p(1-p)/n}$ in DRO and RCT              | $\sqrt{\max(\cdot,\ 0)}$ (and $\sqrt{\max(\cdot,\ 10^{-12})}$ for RCT pooled SE) | Defends against tiny negative values from floating-point rounding when the empirical rate is 0 or 1   | `ml/dro_fairness.py:330–331`; `ml/rct_randomization.py:334` |
| $1/\pi_g$ in Wasserstein DRO worst-case bound                | $\pi_g \mapsto \max(\pi_g,\ 10^{-6})$ before reciprocation             | Prevents blow-up when a protected group is empty in the calibration sample                            | `ml/dro_fairness.py:323–324`                              |
| Hierarchical-Bayes posterior $\sigma$, $\tau$                | $\sqrt{\max(\sigma^2,\ 1)},\ \sqrt{\max(\tau^2,\ 1)}$                  | Floors variance components at 1 (one-minute resolution) to stop the duration sampler collapsing       | `ml/hierarchical_model.py:371, 376`                       |
| Probability outputs from QRF + sequence model                | `np.clip(preds, 0, 1)` post-prediction                                 | Quantile interpolation can fall a hair outside $[0,1]$; the clip keeps downstream consumers honest    | `ml/quantile_forest.py:548, 555`; `ml/sequence_model.py:763` |
| Conformal split-quantile fraction $(n_{\text{cal}}+1)\alpha / n_{\text{cal}}$ | `min(1.0, ...)` clamp                                                  | Stops the quantile lookup from sliding past the largest calibration score when $\alpha$ is near 0    | `ml/conformal_prediction.py:209–211`                      |
| Adaptive conformal $\alpha(p)$                               | `np.clip(raw_alpha, alpha_floor, alpha_ceil)` with $\alpha_{\text{floor}} = 0.01$, $\alpha_{\text{ceil}} = 0.20$ | Keeps the per-patient miscoverage in a range where the calibration-array index is well-defined        | `ml/adaptive_alpha.py:122, 124`                           |
| Inverse-RL MLE under L-BFGS-B                                | `bounds=[(0, None)]*6` non-negativity + L2 ridge $\lambda = 0.05$       | Guarantees non-negative weights and a strictly convex objective so the optimiser converges in $\leq 500$ iterations | `ml/inverse_rl_preferences.py:80, 471–472`                |

When debugging a NaN regression, the test in `tests/test_safe_loader.py`
plus the per-module property tests (e.g. `test_quantile_forest.py` checks
that all clipped probabilities land in $[0,1]$) are the first place to
look — they catch the symptom; this table indexes where to fix it.

---

## 29. Hyperparameter Selection Methodology

The system carries dozens of tunable parameters — $\varepsilon$, $\alpha$,
$\beta$, $\lambda$, $\tau$, $L$, $\delta$, plus every model
hyperparameter (max depth, learning rate, hidden size, …).  This section
records *how each one was chosen* so a future maintainer can reproduce
the choice or re-tune it against new data.  Methods are listed in order
of how often they apply in the live codebase.

### 29.1 Domain-driven defaults *(the default for the majority of constants)*

Used for: clinical thresholds (double-booking tiers, slack floors,
priority buckets), DRO defaults, fairness budgets, retention windows.

Source: NHS guidelines + Velindre operational rules.  Each constant is
declared once in `config.py` (or the per-module `DEFAULT_*` constant)
and pinned by the corresponding test.

| Parameter                              | Value                | Where pinned                               | Provenance                                                         |
|---------------------------------------|----------------------|--------------------------------------------|---------------------------------------------------------------------|
| `NOSHOW_THRESHOLDS = {very_high:0.5, high:0.3, medium:0.15, low:0.0}` | 4-tier risk band | `config.py:141`                            | Velindre operational double-booking rule (recalibrated v4.4)        |
| `DEFAULT_EPSILON` (DRO Wasserstein)   | $0.02$               | `ml/dro_fairness.py:93`                    | Operator-tunable; 2 % mass shift covers the historical drift band  |
| `DEFAULT_DELTA` (parity gap budget)   | $0.15$               | `ml/dro_fairness.py:94`                    | Equality Act 4/5ths-rule equivalent for demographic parity         |
| `DEFAULT_SUGGEST_THRESHOLD` (override) | $0.80$               | `ml/override_learning.py:80`               | §5.2 brief — "≥ 80 % override probability before we speak up"     |
| `L2_LAMBDA_DEFAULT` (IRL ridge)       | $0.05$               | `ml/inverse_rl_preferences.py:80`          | Standard L-BFGS-B ridge to keep MLE strictly convex                |
| `DEFAULT_PRIORITY_COMPLETE_BASE` (MPC) | reward base unit     | `ml/stochastic_mpc_scheduler.py:DEFAULT_*` | §S-1.4 — base credit per completion, multiplied by $(6 - \text{prio})$ |
| `critical_slack_floor`                 | $\geq 5$ min         | `ml/safety_guardrails.py:396`              | Mean chemo chair setup time — clinically derived                  |
| `SOLVER_TIME_LIMIT_SECONDS`            | $300$                | `config.py:60`                             | Operational budget — ChemoCare schedule deadline                   |
| `COLUMN_GEN_THRESHOLD`                 | $50$                 | `config.py:64`                             | Empirical knee where monolithic CP-SAT starts timing out            |
| `CG_MAX_ITERATIONS`                    | $100$                | `config.py:65`                             | Hard cap matching the §27.3 "10-30 typical iterations" bound       |
| `CG_REDUCED_COST_TOLERANCE`            | $10^{-4}$            | `config.py:66`                             | LP-solver standard convergence floor                                |
| `CG_SUBPROBLEM_TIME_LIMIT`             | $5.0$ s              | `config.py:67`                             | Per-chair budget; cascade total $\leq 100 \cdot 5 = 500$ s         |
| `EVENT_RETENTION_DAYS`                 | $30$                 | `.env.example` (T4.8)                      | NHS Digital minimum for derived operational cache data             |
| `AUDIT_RETENTION_DAYS`                 | $2557$ (7 years)     | `.env.example` (T4.8)                      | NHS Wales records-management guidance for clinical access logs      |

### 29.2 Cross-validated training (active for ML model training)

Used for: every base learner inside the no-show + duration ensembles.

Method:

- $k = 5$ stratified $K$-fold by default, switching to
  `TimeSeriesSplit(n_splits=5)` when `use_temporal_cv=True` (data
  evolves over time → train-on-past, validate-on-future).
- Scoring: `roc_auc` for no-show, `neg_mean_absolute_error` for
  duration.  Live in `ml/noshow_model.py:499, 588–596` and
  `ml/duration_model.py:252–256`.

Hyperparameters themselves (tree depth, learning rate, n_estimators)
are held at framework-recommended defaults — this is a v6.0 follow-up
to add a `RandomizedSearchCV` sweep + record the winners as new
`DEFAULT_*` constants.

### 29.3 Data-driven calibration (active for one parameter today)

Used for: the Wasserstein DRO radius $\varepsilon$.

Method: `dro.calibrate_epsilon(historical_appointments_df)` (live at
`flask_app.py:11432, 11453` via `/api/ml/uncertainty-optimization/*`)
estimates $\hat{\varepsilon}$ as the empirical Wasserstein distance
between the calibration window and the most recent rolling window of
the same length.  Returned as `epsilon_recommended` in the JSON
response and persisted on the `UncertaintyAwareOptimizer` instance.
Operator can accept the recommended value or stay with the default.

### 29.4 Live tuners — channel-gated (Grid + Random + Bayesian)

All three search-based tuners are now wired in the `tuning/` package.
Each produces an entry in a single shared manifest at
`data_cache/tuning/manifest.json`.  The manifest carries a
`data_channel` field (`"synthetic"` | `"real"`) and the boot path in
`flask_app.py` calls `tuning.manifest.load_overrides()` which returns
`{}` whenever `data_channel != "real"`.  This single gate is what
makes the tuners safe to run today against synthetic data while
guaranteeing no override leaks into the live prediction pipeline until
real Velindre data is wired through Channel 2.

| Tuner | Module | Search space | Tool | Live test |
|-------|--------|--------------|------|-----------|
| **Grid search** | `tuning/grid_search.py` | `PARETO_WEIGHT_SETS` (`config.py:48–54`) — six objective weights per profile | Pareto-frontier filter on $(\text{composite}, \text{scheduled\_fraction}, \text{utilisation}, \text{robustness}, -\text{waiting})$ | `tests/test_tuning.py::TestGridSearch` |
| **Random search** | `tuning/random_search.py` | `n_estimators ∈ {50,100,200,300}`, `max_depth ∈ {3,5,8,12,None}`, `min_samples_leaf`, `min_samples_split`/`learning_rate` | `sklearn.model_selection.RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)`, scoring `roc_auc` (no-show) / `neg_mean_absolute_error` (duration) | `tests/test_tuning.py::TestRandomSearch` |
| **Bayesian optimisation** | `tuning/bayes_opt.py` | DRO $\varepsilon \in [0.005, 0.20]$, CVaR $\alpha \in [0.05, 0.30]$, Lipschitz $L \in [0.5, 5.0]$ — scalar at a time | `skopt.gp_minimize` (30 initial random points + 20 EI acquisitions) over composite $0.5 \cdot \text{util} - 0.3 \cdot \text{wait\_norm} + 0.2 \cdot \text{fairness\_ratio}$ | `tests/test_tuning.py::TestBayesOpt` |

**Channel gate.** The manifest is the only state that crosses from
tuner to runtime:

```json
{
  "version": 1,
  "data_channel": "synthetic",
  "tuned_at": "2026-04-23T20:22:30+00:00",
  "git_sha": "...",
  "n_records": 1900,
  "results": {
    "noshow_model": {"method":"RandomizedSearchCV", "best_params":{...}, "best_score":0.9449, ...},
    "duration_model": {...},
    "cpsat_weights": {"method":"grid_search", "winner":{"name":"balanced", ...}, "pareto_frontier":[...]},
    "dro_epsilon": {"method":"skopt.gp_minimize", "best_value":0.063, "best_objective":0.78, ...}
  }
}
```

`channel="synthetic"` → boot logs *"manifest is in 'synthetic' mode;
overrides are NOT applied"* and the live runtime stays on its
domain-default constants.

`channel="real"` (set the first time the tuner runs after Channel 2
data lands) → boot logs *"applying N override(s) from real-channel
manifest"* and the named hyperparameters flow into the next process
restart.  No mid-flight mutation; all changes happen at boot for
reproducibility.

**Endpoints (status-only diagnostics — no UI panel).**

| Route                       | Purpose                                                                                       |
|-----------------------------|------------------------------------------------------------------------------------------------|
| `GET  /api/tuning/status`   | Manifest summary: channel, tuners present, `overrides_active` flag, manifest path             |
| `POST /api/tuning/run`      | Body `{"tuner":"random_search"\|"grid_search"\|"bayes_opt", ...}`; runs synchronously, writes manifest, returns the new summary |

**CLI** (`python -m tuning.run --tuner=random_search [--channel=auto]`)
auto-detects the channel from `SACT_CHANNEL` env var or the presence of
`datasets/real_data/historical_appointments.xlsx`.

**Channel 2 cutover plan**:

1. Velindre drops `patients.xlsx` + `historical_appointments.xlsx`
   into `datasets/real_data/` (caught by `monitoring/channel2_watcher.py`).
2. Operator sets `SACT_CHANNEL=real` in the deployment env.
3. Operator runs `POST /api/tuning/run` once per tuner (or
   `python -m tuning.run --tuner=random_search` from the host).
4. Operator inspects `GET /api/tuning/status` — confirms
   `data_channel == "real"` and `overrides_active == true`.
5. Operator restarts the Flask process; boot picks up the manifest
   and applies the tuned values.
6. The `tests/test_tuning.py` suite + the existing 961-test suite are
   re-run to confirm no regression.

All tuned values still flow through `config.py` (or per-module
`DEFAULT_*`) at the constant level and re-flow into the auto-generated
Model Cards (§15c) on the next `model_cards.generate_all_cards()` call.

**Synthetic-data smoke run already executed**:
`data_cache/tuning/manifest.json` now contains a `data_channel="synthetic"`
entry from a real run (1,900 rows, AUC 0.9449 on the synthetic-generator
distribution).  The high AUC is the well-documented synthetic-overfit
signal — it is exactly *why* the channel gate exists and *why* the boot
path refuses to apply it.

---

## 30. Validation Invariants (Mathematical Tests)

The test suite enforces every formula in this document via property-style
invariants — not just point-value regression checks.  When a formula
changes, the corresponding invariant either still holds (safe edit) or
fails loudly (the change broke a guarantee the rest of the system
depends on).

| Invariant                                          | Mathematical statement                                                                                                                            | Test location                                                                          |
|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| PSI attribution sum                                | $\sum_j \text{share}_j = 1$                                                                                                                       | `tests/test_drift_attribution.py::TestAttribution::test_shares_sum_to_one`             |
| Per-feature PSI consistency                        | $\text{PSI}_j = \sum_i \delta_{j,i}$                                                                                                              | `tests/test_drift_attribution.py::TestAttribution::test_per_bin_decomposition`         |
| DRO bound monotonicity                             | $\text{worst\_case} \geq \text{empirical\_gap}$, with equality at $\varepsilon = 0$                                                               | `tests/test_dro_fairness.py::TestPairCertify::test_zero_epsilon_collapses_to_empirical`|
| Lipschitz violation monotonicity                   | $L_1 \leq L_2 \Rightarrow \text{violations}(L_1) \geq \text{violations}(L_2)$                                                                     | `tests/test_individual_fairness.py::TestCertify::test_violations_monotone_in_L`        |
| CVaR ≤ mean                                        | $\text{CVaR}_\alpha(U) \leq \mathbb{E}[U]$ for all $\alpha \in (0, 1)$                                                                            | `tests/test_uncertainty_optimization.py::TestCVaR::test_cvar_le_mean`                  |
| Conformal coverage (empirical)                     | $\frac{1}{n_\text{test}}\sum_i \mathbb{1}\bigl[y_i \in \widehat{C}(x_i)\bigr] \geq 1 - \alpha \pm \delta$                                         | `tests/test_adaptive_alpha.py` (clamping invariants); `ml/conformal_prediction.py` is import-tested by every consumer |
| Inverse-RL convexity                               | Hessian of negative log-likelihood is positive semi-definite                                                                                      | `tests/test_irl_preferences.py::TestFit::test_objective_strictly_convex_with_ridge`    |
| Warm-start fingerprint determinism                 | Same fingerprint $\Rightarrow$ same hint set                                                                                                      | `tests/test_optimization.py::TestColumnGeneration` + `optimization/optimizer.py` cache key    |
| Priority-weighted MPC reward monotonicity (T2.1)   | $\text{prio}_1 < \text{prio}_2 \Rightarrow R_t(\text{prio}_1) > R_t(\text{prio}_2)$                                                               | `tests/test_stochastic_mpc_scheduler.py::TestPriorityWeightedReward::test_priority_multiplier_is_monotone` |
| Auto-scaling weight isolation (T2.2)               | Concurrent solves never observe weights other than their own                                                                                       | `tests/test_auto_scaling_optimizer.py::TestParallelRaceWeightIsolation`                |
| SHA-256 sidecar integrity (T2.3)                   | `safe_load(p)` raises `UnsafeLoadError` if `sha256(p) ≠ sidecar`                                                                                  | `tests/test_safe_loader.py::TestSafeLoad::test_sidecar_mismatch_raises`                |
| Tuning channel gate (T5)                           | `data_channel == "synthetic"` $\Rightarrow$ `load_overrides() == {}`                                                                              | `tests/test_tuning.py::TestManifest::test_overrides_blocked_in_synthetic_mode`         |

When debugging a regression, find the broken invariant first; the
relevant section's narrative will tell you which formula it pins.

---

## 31. Pseudocode Style Guide

Every algorithm box in this document follows the same convention so a
reader can skim them all without re-learning the syntax.

```
Algorithm: <Name>
Input:  <typed list>
Output: <description>

1.  Initialise <data structures>
2.  for <condition>:
3.      if <guard>:
4.          <operation>
5.      else:
6.          <alternative>
7.  return <result>
```

Conventions:

- `←` is assignment (not `=`); equality stays as `=` or `==`.
- `# ...` is a comment.  No multi-line comments; split across lines.
- Indentation is two spaces inside loops / branches.
- Loop / branch headers always end with `:` then newline + indent.
- Set comprehensions: `{ x | x ∈ S, P(x) }`.  Sums / products: `Σ_i`, `∏_i`.
- Function calls: `f(x)`; method calls: `obj.method(x)`.
- Variables in `code_font`; mathematical objects in $\mathit{italic}$.

Worked example (CP-SAT warm-start hint injection, §2.10):

```
Algorithm: WarmStartHintInjection
Input:  patients P, chairs C, cache (fingerprint → past assignment)
Output: cp_model with assignment hints attached

1.  model ← CpModel()
2.  fp    ← fingerprint(P, C)        # day-of-week, |P|, priority distribution, ...
3.  if fp ∈ cache:
4.      prior ← cache[fp]
5.      for each p ∈ P:
6.          model.AddHint(assigned_var[p], 1)
7.          model.AddHint(start_var[p], prior[p].start)
8.          for each c ∈ C:
9.              model.AddHint(chair_var[p, c], 1 if c == prior[p].chair else 0)
10. return model
```

Worked example (Bayesian-opt loss for §29.4 / `tuning/bayes_opt.py`):

```
Algorithm: ScalarObjective
Input:  candidate value v ∈ [v_lo, v_hi], composite weights w
Output: skopt loss (the more negative, the better)

1.  metrics ← evaluate(v)
2.  util    ← metrics.utilisation
3.  wait    ← min(metrics.avg_waiting_days / 14, 1)   # normalise to [0, 1]
4.  fair    ← metrics.fairness_ratio
5.  score   ← w_util · util  −  w_wait · wait  +  w_fair · fair
6.  return -score                                      # skopt minimises
```

---

## 32. Known Limitations per Model

A world-class document acknowledges where each formulation can break.
Each subsection here lists the limit, the symptom, and the live
mitigation in the codebase.

### 32.1 CP-SAT (monolithic, §2)

- **Limit.** $n \cdot m > 5{,}000$ binary $x_{p,c}$ exhausts the branch-and-cut search budget; OR-Tools either times out or returns a poor incumbent.
- **Symptom.** `STATUS_FEASIBLE` rather than `STATUS_OPTIMAL`, with `solve_time` near `SOLVER_TIME_LIMIT_SECONDS`.
- **Live mitigation.** `COLUMN_GEN_THRESHOLD = 50` (`config.py:64`) auto-routes any instance with $n > 50$ to the column-generation solver (§2.12, $\approx 5$ s for $n = 100$); `AutoScalingOptimizer` (§A.9) further wraps the call with a cascading 5 / 2 / 1 / 0.5-second budget + greedy fallback so a decision is always returned.

### 32.2 No-show ensemble (§3)

- **Limit.** The sequence-model (RNN) component (§3.6) needs $\geq 5$ prior appointments per patient to materially lift AUC.
- **Symptom.** AUC drops by $\approx 0.04$–$0.06$ on patients with $\leq 2$ prior appointments compared to those with full sequence history.
- **Live mitigation.** `min_sequence_length = 5` gate inside the ensemble routes short-history patients through the static features only; the RNN is invoked only when history is long enough to be informative (`ml/sequence_model.py`).  `cold_start_prior` returns the population baseline (default $0.10$) until $\geq$ `min_events_for_fit` real outcomes have been observed.

### 32.3 Decision-Focused Learning calibration (§3.7)

- **Limit.** The two-parameter calibration head $g(p) = \sigma(a \cdot \text{logit}(p) + b)$ assumes the underlying ensemble is at least roughly monotone in true risk.
- **Symptom.** $a$ goes negative or near-zero, flipping or flattening the ranking.
- **Live mitigation.** Non-negativity constraint `a ≥ 0` plus L2 prior pulling $(a, b)$ toward $(1, 0)$ keeps the head close to identity in the small-data regime (`ml/decision_focused_learning.py`).

### 32.4 Conformal prediction — split (§19)

- **Limit.** Coverage $\Pr(Y \in \widehat{C}(X)) \geq 1 - \alpha$ is **marginal** over the joint $(X, Y)$ distribution; it does NOT guarantee conditional coverage $\Pr(Y \in \widehat{C}(X) \mid X = x)$ at every $x$.
- **Symptom.** Intervals can be too narrow on hard subgroups (e.g., elderly patients with multiple cycles) and too wide on easy ones.
- **Live mitigation.** Risk-adaptive $\alpha$ policy (§19.10) tightens or loosens $\alpha(p) \in [\alpha_{\text{floor}}, \alpha_{\text{ceil}}] = [0.01, 0.20]$ per patient, lifting conditional coverage in exchange for slightly wider average intervals.

### 32.5 DRO Wasserstein certificate (§A.5)

- **Limit.** The closed-form upper bound assumes $Y \in \{0, 1\}$ and a 0-1 cost; it is not valid for continuous outcomes.
- **Symptom.** A user mistakenly applying the certificate to a duration-regression target would get a meaningless inflation term.
- **Live mitigation.** The certificate is wired only to fairness-gap callers (`ml/dro_fairness.py:certify_pair`) — never to the duration model or the no-show *probability* itself.  The 1/π_g term is also clipped (`max(π_g, 1e-6)`) to stop the bound from blowing up on empty groups (see §28).

### 32.6 Inverse-RL preference learner (§2.13)

- **Limit.** Reliable estimation needs $\geq 20$ real overrides; with fewer, the bootstrap prior dominates and the learned weights barely move from $\boldsymbol{\theta}^{*}_{\text{boot}}$.
- **Symptom.** Cross-validated agreement above 90 % on the bootstrap distribution but no ability to recover a held-out clinician's preferences.
- **Live mitigation.** `min_events_for_fit = 20` gate (`ml/inverse_rl_preferences.py`) keeps the learner on the bootstrap prior until enough real overrides land.  Ridge $\lambda = 0.05$ (`L2_LAMBDA_DEFAULT`, §29.1) keeps the L-BFGS-B objective strictly convex even with sparse data.

### 32.7 MPC + Gamma-Poisson arrivals (§A.13)

- **Limit.** The Gamma-Poisson model assumes stationary Poisson arrivals over the short horizon $\tau$; it does not capture self-exciting (bursty) patterns.
- **Symptom.** Under a real-world burst (e.g., emergency-department referral spike on Monday morning), the posterior mean rate underestimates the next-step intensity for a few minutes.
- **Live mitigation.** Conjugate update on every observed window keeps the posterior responsive ($\alpha \mapsto \alpha + n$, $\beta \mapsto \beta + \Delta$); for production-grade burst modelling the natural upgrade is a Hawkes-process arrival model — flagged as future work in the dissertation §5.5 limitations subsection.

### 32.8 Tuning manifest (§29.4)

- **Limit.** Tuning runs against synthetic data overfit the generator's quirks; the resulting hyperparameters are statistically noisy with respect to real distributions.
- **Symptom.** Synthetic-channel manifests routinely show no-show AUC above 0.90 — substantially higher than the AUC any real-data deployment will see.
- **Live mitigation.** The boot path applies overrides ONLY when `data_channel == "real"`; synthetic-channel manifests are loud reminders ("manifest is in 'synthetic' mode; overrides NOT applied") rather than silent leaks.  Pinned by `tests/test_tuning.py::TestManifest::test_overrides_blocked_in_synthetic_mode`.

---

## References

### Peer-Reviewed Literature for Advanced Methods (cross-reference)

Quick-lookup table — each row maps a method used somewhere in this
document to its canonical academic source.  The right column is the
entry number in the bibliography that follows.  All other equations
(CP-SAT objective construction, fairness constraints, simple aggregates,
etc.) are original to this work or derived from standard operations-
research / machine-learning textbooks already cited below.

| Method                                                       | Canonical citation                                                  | Bibliography entry |
|--------------------------------------------------------------|---------------------------------------------------------------------|--------------------|
| Wasserstein DRO                                              | Mohajerin Esfahani & Kuhn (2018) *Math. Program.*                   | #38                |
| CVaR in LP                                                   | Rockafellar & Uryasev (2000) *J. Risk*                              | #37                |
| Conformal prediction (foundational)                          | Vovk, Gammerman & Shafer (2005)                                     | #13                |
| Conformal prediction (modern self-contained treatment)       | Angelopoulos & Bates (2022) *FnT ML*                                | #43                |
| Conformalised Quantile Regression (CQR)                      | Romano, Patterson & Candès (2019) *NeurIPS*                         | #15                |
| Monte-Carlo Dropout (Bayesian approximation)                 | Gal & Ghahramani (2016) *ICML*                                      | #14                |
| Double / Debiased Machine Learning                           | Chernozhukov et al. (2018) *Econom. J.*                             | #11                |
| Instrumental variables (textbook)                            | Angrist & Pischke (2008)                                            | #10                |
| Uplift meta-learners                                         | Künzel et al. (2019) *PNAS*                                         | #16                |
| Model Cards for ML transparency                              | Mitchell et al. (2019) *FAT\* Conf.*                                | #28                |
| Lipschitz / individual fairness                              | Dwork et al. (2012) *ITCS*                                          | #44                |
| Column generation (master + pricing)                         | Desaulniers, Desrosiers & Solomon (2005)                            | #45                |
| Branch-and-Price (integer-rounded CG)                        | Barnhart et al. (1998) *Operations Research*                        | #46                |
| Temporal Fusion Transformer                                  | Lim et al. (2021) *Int. J. Forecast.*                               | #47                |
| SPO+ — Smart Predict-then-Optimise                           | Elmachtoub & Grigas (2022) *Management Science*                     | #41                |
| Decision-Focused Learning (blackbox solver gradient)         | Wilder, Dilkina & Tambe (2019) *AAAI*                               | #42                |
| Constrained Model Predictive Control (stability)             | Mayne, Rawlings, Rao & Scokaert (2000) *Automatica*                 | #39                |
| Bayesian Model Averaging (BMA)                               | Hoeting, Madigan, Raftery & Volinsky (1999) *Statist. Sci.*         | #40                |

### Core Scheduling & Optimization
1. Google OR-Tools CP-SAT Solver Documentation
2. Baptiste, P., Le Pape, C. & Nuijten, W. (2001). Constraint-Based Scheduling
3. Pinedo, M. (2016). Scheduling: Theory, Algorithms, and Systems

### Machine Learning
4. Scikit-learn Ensemble Methods Documentation
5. XGBoost: Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*

### Statistical & Bayesian Methods
6. Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*
7. Gelman, A. & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical Models
8. Meinshausen, N. (2006). Quantile Regression Forests. *JMLR*

### Causal Inference
9. Pearl, J. (2009). Causality: Models, Reasoning, and Inference
10. Angrist, J. D. & Pischke, J.-S. (2008). Mostly Harmless Econometrics
11. Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning. *Econometrics Journal*
12. Hernan, M. A. & Robins, J. M. (2020). Causal Inference: What If

### Uncertainty Quantification
13. Vovk, V., Gammerman, A. & Shafer, G. (2005). Algorithmic Learning in a Random World
14. Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*
15. Romano, Y., Patterson, E. & Candes, E. (2019). Conformalized Quantile Regression. *NeurIPS*

### Treatment Effect Estimation
16. Kunzel, S. R., et al. (2019). Metalearners for Estimating Heterogeneous Treatment Effects. *PNAS*
17. Caruana, R. (1997). Multitask Learning. *Machine Learning*

### Healthcare & NHS
18. NHS Digital (2025). SACT Data Set v4.0 Technical Guidance. NDRS
19. NHS England (2026). Cancer Waiting Times Statistics
20. Royal College of Radiologists (2023). The SACT Capacity Crisis in the NHS
21. System C Healthcare (2025). ChemoCare: Electronic Chemotherapy Prescribing
22. Ahmad Hamdan, A.F. & Abu Bakar, A. (2023). Machine learning predictions on outpatient no-show appointments in a Malaysia major tertiary hospital. *Malaysian Journal of Medical Sciences*, 30(5), pp.169–180. doi:10.21315/mjms2023.30.5.14
23. Hadid, M., et al. (2022). Clustering and stochastic simulation optimisation for outpatient chemotherapy appointment planning and scheduling. *International Journal of Environmental Research and Public Health*, 19(23), p.15539. doi:10.3390/ijerph192315539

### Drift Detection & Monitoring
24. Siddiqi, N. (2006). Credit Risk Scorecards: PSI and CSI
25. Page, E. S. (1954). Continuous Inspection Schemes. *Biometrika*

### Sensitivity Analysis
26. Saltelli, A., et al. (2008). Global Sensitivity Analysis: The Primer. *Wiley*
27. Sobol', I. M. (1993). Sensitivity Estimates for Nonlinear Mathematical Models. *MMCE*

### Model Transparency & Fairness
28. Mitchell, M., et al. (2019). Model Cards for Model Reporting. *FAT\* Conference*
29. NHS England (2023). A Buyer's Guide to AI in Health and Care
30. Barocas, S. & Selbst, A. (2016). Big Data's Disparate Impact. *California Law Review*
31. Equality Act 2010 (UK). Protected characteristics and discrimination
32. Chen, R.J., et al. (2023). Algorithmic fairness in artificial intelligence for medicine and healthcare. *Nature Biomedical Engineering*, 7(6), pp.719–742. doi:10.1038/s41551-023-01056-8

### Reinforcement Learning
33. Watkins, C. J. C. H. & Dayan, P. (1992). Q-Learning. *Machine Learning*
34. Sutton, R. S. & Barto, A. G. (2018). Reinforcement Learning: An Introduction

### Causal Validation
35. Imbens, G. W. & Rubin, D. B. (2015). Causal Inference for Statistics
36. Rosenbaum, P. R. (2002). Observational Studies. *Springer*

### Distributionally Robust Optimisation, CVaR, MPC, BMA, SPO+
37. Rockafellar, R. T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2(3), pp.21–41.  *— CVaR derivation cited in §24b.7.*
38. Mohajerin Esfahani, P. & Kuhn, D. (2018). Data-driven distributionally robust optimization using the Wasserstein metric: performance guarantees and tractable reformulations. *Mathematical Programming*, 171(1–2), pp.115–166. doi:10.1007/s10107-017-1172-1  *— Wasserstein-DRO closed-form bound cited in §A.5 + §24b.*
39. Mayne, D. Q., Rawlings, J. B., Rao, C. V. & Scokaert, P. O. M. (2000). Constrained model predictive control: stability and optimality. *Automatica*, 36(6), pp.789–814. doi:10.1016/S0005-1098(99)00214-9  *— Receding-horizon control foundations cited in §A.13.*
40. Hoeting, J. A., Madigan, D., Raftery, A. E. & Volinsky, C. T. (1999). Bayesian Model Averaging: A Tutorial. *Statistical Science*, 14(4), pp.382–417.  *— BMA weighting in the no-show ensemble (§3.1).*
41. Elmachtoub, A. N. & Grigas, P. (2022). Smart "Predict, then Optimize". *Management Science*, 68(1), pp.9–26. doi:10.1287/mnsc.2020.3922  *— SPO+ loss formulation cited in §3.7.*
42. Wilder, B., Dilkina, B. & Tambe, M. (2019). Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization. *AAAI*, 33(1), pp.1658–1665. doi:10.1609/aaai.v33i01.33011658  *— Blackbox-solver-gradient route cited in §3.7.*

### Conformal Prediction, Lipschitz Fairness, Column Generation, TFT
43. Angelopoulos, A. N. & Bates, S. (2022). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. *Foundations and Trends in Machine Learning*, 16(4), pp.494–591. doi:10.1561/2200000101  *— Modern self-contained treatment cited in §19 / §27.4.*
44. Dwork, C., Hardt, M., Pitassi, T., Reingold, O. & Zemel, R. (2012). Fairness through Awareness. In: *3rd Innovations in Theoretical Computer Science Conference (ITCS)*, pp.214–226. doi:10.1145/2090236.2090255  *— Lipschitz / individual-fairness condition cited in §A.6.*
45. Desaulniers, G., Desrosiers, J. & Solomon, M. M. (eds.) (2005). *Column Generation*. New York: Springer. doi:10.1007/b135457  *— Master-pricing decomposition cited in §2.12.*
46. Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W. P. & Vance, P. H. (1998). Branch-and-Price: Column Generation for Solving Huge Integer Programs. *Operations Research*, 46(3), pp.316–329. doi:10.1287/opre.46.3.316  *— Branch-and-price-lite integer rounding cited in §2.12.7.*
47. Lim, B., Arık, S. Ö., Loeff, N. & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), pp.1748–1764. doi:10.1016/j.ijforecast.2021.03.012  *— TFT architecture cited in §3.8.*

---

*Document updated for MSc Data Science Dissertation, Cardiff University*
*SACT Scheduling System v4.0*
*March 2026*
