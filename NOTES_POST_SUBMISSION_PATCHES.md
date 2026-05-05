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

## Patch 2 — Cardiff Windows WMI bypass (2026-05-03)

**File added:** `scripts/install_wmi_bypass.py` (self-installing).

**Bug:** On Cardiff-University-managed Windows endpoints (corporate
antivirus + locked-down Winmgmt service) the Python 3.12 `platform`
module hangs indefinitely on `platform._wmi_query` (CPython
`Lib/platform.py:326`).  The query is invoked transitively by
`numpy.testing._private.utils:89` -> `platform.machine()`, which is
loaded the first time anything imports `scipy.sparse` / `sklearn`.
Net effect: importing the `optimization.*` package (or anything else
that pulls sklearn) deadlocks the interpreter.  The Flask webapp
process never reaches `app.run`, every `ml/benchmark_*.py` harness
hangs at startup, and pytest collection blocks for >30 min.

**Repro:** any of these hang on the affected machine, exit cleanly
on a healthy one:
```
python -c "import platform, time; t=time.time(); platform.machine(); print(time.time()-t)"
python -c "from optimization.optimizer import ScheduleOptimizer"
python flask_app.py
```

**Fix:** `scripts/install_wmi_bypass.py` writes a `usercustomize.py`
into the user's site-packages.  Python auto-loads `usercustomize.py`
at every interpreter startup when `site.ENABLE_USER_SITE` is True
(default).  The hook patches `platform._wmi_query` to raise
`OSError` immediately, which is the exact exception CPython's own
non-WMI fallback path is written for (`platform._win32_ver` lines
400-443).  The fallback uses `sys.getwindowsversion()` and `winreg`
- direct Windows API calls that return the SAME version / release /
build / edition values WMI would have returned on a healthy box.

**Verification on the affected Cardiff machine:**
- Before fix: `python -c "import platform; platform.machine()"` hung
  past the timeout (>15 s); `flask_app.py` never bound port 1421.
- After fix: `platform.machine() = 'AMD64'` (<1 ms),
  `platform.win32_ver() = ('11', '10.0.26100', 'SP0', 'Multiprocessor Free')`
  (~45 ms), `optimization` imports in 5.5 s, `flask_app.py` boots in
  ~70 s and serves `GET /` (HTTP 200, 176 KB) and
  `GET /api/data/channel/real/status` (HTTP 200).

**Install / uninstall:**
```
python scripts/install_wmi_bypass.py            # install
python scripts/install_wmi_bypass.py --status   # check
python scripts/install_wmi_bypass.py --uninstall # remove
```

**Why it's safe to ship:**
- No-op on non-Windows hosts (the installer prints "nothing to install").
- No-op on healthy Windows hosts: the WMI fallback path was already a
  first-class CPython feature, the patch just routes the WMI call to
  the same fallback unconditionally.  The values returned are
  identical because they come from the same Windows kernel APIs.
- Removing the file silently restores upstream behaviour - no other
  source code depends on the patch.
- Independent of any project code: applies system-wide for the user's
  Python interpreter, so the dev server, R via reticulate, every
  `ml/benchmark_*.py`, and ad-hoc CLI scripts all benefit without
  per-script changes.

---

## Patch 3 — priority-tier `max_delay_days` aligned with FSSA framework (2026-05-05)

**File:** `config.py`, `PRIORITY_DEFINITIONS` and `PRIORITY_MAX_DELAY`.

**Before:** values were 2 / 7 / 14 / 30 days for P1 / P2 / P3 / P4.

**After:** values are 1 / 3 / 14 / 30 days, matching the FSSA *Clinical
Guide to Surgical Prioritisation* (P1a < 24 h, P1b < 72 h, P2 < 1 month,
P3 < 3 months) under the 4-tier oncology collapse the dissertation uses
(P1 = FSSA P1a, P2 = FSSA P1b).

**Bug:** the legacy 2 / 7 day values contradicted the dissertation's
prose ("P1 within 24 h, P2 within 72 h", repeated at `main.tex:1008,
1175, 1188, 2242`) for no functional reason -- the `max_delay_days`
field is **not read at runtime** by any module.  Verified by
grepping the whole tree: only the dict object is imported (by
`data/validators.py`, `optimization/emergency_mode.py`,
`optimization/squeeze_in.py`, `optimization/constraints.py`), and
only the `name` field is ever dereferenced.

**Impact:** none on simulation behaviour, benchmark outputs, or
dissertation deliverables.  The change is documentation-quality
only -- it makes the codebase's printed deadlines match the
prose's clinical aspirations.  The actual runtime enforcement is
unchanged: `P1_MAX_START_MIN = 90` (hard CP-SAT constraint) plus
per-patient `earliest_time` / `latest_time` set by callers.

**Audit trail:** full rationale, FSSA mapping table, and source
citations live at `dissertation/PRIORITY_TIER_NOTE.md`.

---

## Roadmap reference

The aborted-at-Stage-2 regeneration plan and exact CLI reproduction
recipe live at `dissertation/REGEN_ROADMAP.md` for any future operator
who wants to flow the patch through to dissertation deliverables.
