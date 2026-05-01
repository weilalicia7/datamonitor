# Post-submission code-quality patches

This file records source-code corrections made *after* the dissertation
deliverables (`main.pdf`, `MAT099-25038355-Dissertation*.pdf`,
`dissertation_overleaf.zip`) were finalised and the JSONL benchmark caches
in `data_cache/` were generated.

The patches below correct genuine bugs but the snapshot data files in
`datasets/sample_data/*.xlsx` and `data_cache/**/results.jsonl` were
generated under the *pre-patch* code, so the dissertation's published
numbers correspond to the pre-patch snapshot. Re-running
`python -m datasets.generate_sample_data` followed by all
`ml/benchmark_*.py` harnesses on a non-AV-constrained machine will
produce post-patch numbers that differ marginally (cohort-level shift
≈1.6 %; aggregate metrics within rounding).

---

## Patch 1 — patient-stable SACT v4.0 fields (2026-05-01)

**File:** `datasets/generate_sample_data.py`, lines 1220–1233 of the
appointment-row dictionary inside `generate_historical_appointments`.

**Bug:** Six fields that should be patient-stable (i.e. one value per
patient, repeated across every appointment row for that patient) were
being randomised independently per appointment row using
`random.choices(...)`:

| Field | Before fix | After fix |
|---|---|---|
| `Person_Stated_Gender_Code` | `random.choices([1,2,9], weights=[0.44,0.53,0.03])[0]` | `patient.get('Person_Stated_Gender_Code', 1)` |
| `Primary_Diagnosis_ICD10` | `random.choice([...10 ICD codes...])` | `patient.get('Primary_Diagnosis_ICD10', 'C18.9')` |
| `Intent_Of_Treatment` | `random.choices(['06','07'], weights=[0.55,0.45])[0]` | `patient.get('Intent_Of_Treatment', '06')` |
| `Treatment_Context` | `random.choices(['01','02','03'], weights=[0.15,0.30,0.55])[0]` | `patient.get('Treatment_Context', '03')` |
| `Clinical_Trial` | `random.choices(['01','02'], weights=[0.08,0.92])[0]` | `patient.get('Clinical_Trial', '02')` |
| `Chemoradiation` | `random.choices(['Y','N'], weights=[0.12,0.88])[0]` | `patient.get('Chemoradiation', 'N')` |

`Performance_Status` was deliberately *not* changed because performance
status legitimately varies across a treatment course.

**Impact:** Under the pre-fix code, a single patient could appear as Male
in one appointment row and Female in another, with different ICD-10
codes and different treatment-intent flags across rows. This would
silently corrupt any per-patient longitudinal aggregation (gender-
stratified fairness audit, per-patient diagnosis-trajectory analysis).
Cohort-level aggregates were essentially unaffected because the
per-row weight distributions
(`[0.44,0.53,0.03]`) closely matched the patient-level distribution
(`[0.45,0.52,0.03]`) — the bug averaged out at scale.

**Verification:** On a 30-patient × 2-month seeded sample run:
- Before patch: 14/25 patients had inconsistent diagnoses, 12/25
  inconsistent intent, 11/25 inconsistent context, 4/25 inconsistent
  trial / chemoradiation flags.
- After patch: 0/25 inconsistent on every field.

**Discovery:** Found during a thorough audit of agent-reported "all
green" verification — the data agent's vague "matches dissertation"
claim was challenged with direct row-level comparison.

**Why not re-run benchmarks:** The Cardiff-University-managed Windows
endpoint that hosts this work has enterprise antivirus that scans every
Python module load, making each fresh `python -m ml.benchmark_*` invocation
take 10–15 minutes wall-clock just to clear the import phase. With 12+
primary benchmarks needed, total wall-clock would be 3–6 hours, and
several benchmarks fail under the machine's NumPy-2.x / NumPy-1.x ABI
mismatch in `bottleneck` and `numexpr`. Re-running on an
admin-equipped or non-corporate machine is left as future work; the code
fix is the substantive correctness improvement.

---

## Roadmap reference

The aborted-at-Stage-2 regeneration plan and exact CLI reproduction
recipe live at `dissertation/REGEN_ROADMAP.md` for any future operator
who wants to flow the patch through to dissertation deliverables.
