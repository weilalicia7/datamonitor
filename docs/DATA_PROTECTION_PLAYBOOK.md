# Data Protection Playbook

> Scope: the SACT Scheduler when deployed against **real** Velindre Cancer
> Centre patient data.  This playbook is the operational runbook that
> implements the controls promised in `SECURITY.md`, the
> ``UK GDPR`` framework, and NHS Wales Information Governance (IG)
> expectations.  Non-production installs using synthetic data inherit
> the same controls by default; operators may relax individual knobs
> (documented inline) when they can prove no real PII is in scope.

---

## 1. Personal data inventory

The system encounters the following categories of personal data during
normal operation.  Any field here is PII under the UK GDPR.

| Field | Source | Where it lives | Lawful basis |
|---|---|---|---|
| NHS number (10 digits) | SACT feed | pandas DataFrame in process memory only | Article 9(2)(h) — health care provision |
| Patient name | SACT feed | memory only | 9(2)(h) |
| Date of birth | SACT feed | memory only | 9(2)(h) |
| Postcode (partial, sector) | SACT feed | memory only; sector-only retained for travel-time model | 9(2)(h) |
| Appointment history | SACT feed | memory + `data_cache/events/*.jsonl` cache | 9(2)(h) |
| No-show rate, attendance labels | derived | in-process ML features only | 9(2)(h) |

We do **NOT** collect, persist, or transmit:

- Full postcode (truncated to sector on ingest)
- Free-text clinical notes
- GP identifiers
- Ethnicity / gender beyond what SACT v4.x requires for fairness audits
- Consent status (handled upstream by the electronic-records system)

## 2. Data minimisation on ingest

Enforced in `datasets/_nhs_calibration.py` and `ml/feature_engineering.py`:

- NHS number is hashed with SHA-256 + per-deployment salt before any
  field leaves the initial adapter.  Downstream ML + optimisation code
  sees the hash, not the raw number.
- Postcodes truncated to sector (first two / three characters) inside
  `datasets/_nhs_calibration._anonymise()`.  Full postcodes never reach
  a model fit call.
- Name + DOB are used only for the `patient_id` lookup and are dropped
  before the DataFrame is handed to the ML + optimiser layers.
- Logs scrub any remaining patient IDs via `logging_config.PatientIdRedactor`
  (T4.4) — `[REDACTED]` replaces `patient_id=…`, `Patient_ID: …`,
  10-digit NHS numbers, and UK postcodes.

## 3. Retention schedule

Configured in `.env` via per-category env vars, enforced by
`scripts/retention_enforcer.py` (cron job expected once per day).

| Category | Path | Default TTL | Env var |
|---|---|---|---|
| Event cache | `data_cache/events/*.jsonl` | 30 days | `EVENT_RETENTION_DAYS` |
| Audit trail | `data_cache/audit/*.jsonl` | 2557 days (7 years) | `AUDIT_RETENTION_DAYS` |
| Model artefacts | `models/*.pkl`, `models/*.pt` | overwritten on retrain; never shipped | — |
| Aggregate metrics (Prometheus) | time-series store | 90 days | (store-level) |
| Logs (stdout only) | container log collector | 30 days (collector-level) | — |

Retention choices rationale:

- 30 days on the event cache follows NHS Digital minimum for derived
  operational cache data when the source-of-truth (SACT) retains the
  full record.
- 7 years on the audit trail matches NHS Wales records-management
  guidance for clinical-system access logs.
- Model pickles are regenerated on every retrain and are never written
  to a persistent volume in production — they live on ephemeral pod
  disk inside the container (deleted with the pod).

## 4. Right-to-erasure procedure

A patient requests deletion via the clinical team (the clinical record
is the source of truth; we are a derivative read-only consumer in
production).

Operator steps:

1. Confirm the requester's identity via the standard Velindre IG
   process.  Record the DSAR / Article 17 request ID.
2. Obtain the **salted SHA-256 hash** of the patient's NHS number
   from the clinical-records system using the current deployment salt.
3. Run:
   ```bash
   python scripts/retention_enforcer.py --erase <hash>
   ```
   The script scans every JSONL file under `data_cache/` + any
   persisted ML feature stores and rewrites them to remove records
   with the matching hash.  (Append-only audit files are preserved;
   GDPR permits retention for legal-obligation accountability — the
   audit row records the erasure itself.)
4. Restart the app so in-memory caches are dropped:
   ```bash
   kubectl rollout restart deployment/sact-scheduler
   ```
5. Record the erasure in the compliance ticket system, including the
   `action: data.erasure, target: <hash-prefix>` audit-trail entry
   that the command above writes.

## 5. Encryption posture

| Data at rest | Control |
|---|---|
| Container volume (`data_cache/`) | Backing k8s PV with LUKS / AWS EBS AES-256 |
| Named volume `audit` (docker-compose) | Docker default aufs; host disk should be encrypted (BitLocker / FileVault / LUKS) |
| `.env` files | LOCAL DEV ONLY; gitignored; pre-commit blocked |
| Production secrets | AWS Secrets Manager / Vault — never on filesystem |

| Data in transit | Control |
|---|---|
| Browser → nginx | TLS 1.2/1.3 (T4.6 `nginx/nginx.conf`) |
| nginx → app | internal docker network; TLS optional |
| App → TomTom API | HTTPS enforced by `requests>=2.32.0` |
| OTel export (optional) | OTLP/HTTP; operator should terminate at a private collector |

## 6. Access controls

See also `docs/PRODUCTION_READINESS_PLAN.md` T4.1.

- Three roles: `admin ⊇ operator ⊇ viewer`.
- PBKDF2-HMAC-SHA256 password hashing (200k iterations, 32-byte salt).
- API keys: 32-byte URL-safe random tokens; SHA-256 digest stored
  alongside the role.
- Password seeding only at first boot; rotation per
  `docs/SECRETS_ROTATION.md`.

## 7. Incident response

If an information-security incident is suspected:

1. **Contain** — kick the suspect pod (`kubectl delete pod ...`) and
   rotate affected credentials per `docs/SECRETS_ROTATION.md § 5`.
2. **Preserve** — copy the affected `data_cache/audit/*.jsonl` files
   to a write-once store (S3 object-lock / equivalent).
3. **Notify** — contact Cardiff IG (`infogov@cardiff.ac.uk`) within
   24 hours of detection.  For UK GDPR Article 33 obligations, a
   personal-data breach triggers the 72-hour ICO notification clock.
4. **Remediate** — patch, deploy, verify, and document.
5. **Post-mortem** — file an incident report in the compliance tracker;
   update this playbook if the root cause reveals a gap.

## 8. DPIA

The per-deployment Data Protection Impact Assessment lives in
`docs/DPIA.md`.  It must be updated any time:

- A new personal-data field is ingested.
- A new external service receives any PII.
- Retention windows are shortened or extended.
- A new role / API client is granted read access.
- A tuning run is executed against real Channel-2 data
  (`SACT_CHANNEL=real`).  The tuning subsystem (`tuning/` package)
  reads `historical_appointments.xlsx` end-to-end and writes a manifest
  that overrides ML hyperparameters on the next boot.  Until the
  governance gate (Caldicott + DSA + SMREC + DSPT) clears, only
  synthetic-channel runs are permitted; the channel discriminator
  in `data_cache/tuning/manifest.json` and the boot-time
  `load_overrides()` gate enforce this in code.  See
  `docs/MATH_LOGIC.md §29.4` for the full Channel-2 cutover plan.

Reviewed annually or on change; countersigned by the Cardiff DPO
and Velindre IG lead.

## 9. Audit checklist (operator self-review every 90 days)

- [ ] Retention script runs on schedule (`cron.yaml` or systemd timer)
  and logs at least one delete per day.
- [ ] Pre-commit hook (`scripts/pre-commit.sh`) is installed on every
  developer's clone (`scripts/install-hooks.sh`).
- [ ] `.env` files are absent from every laptop running as a prod-style
  deployment (`SACT_PROD_MODE=true` or `FLASK_ENV=production`).
- [ ] TLS cert validity >60 days remaining; rotation calendared.
- [ ] Audit trail shows no unexplained `auth.login: failure` spikes.
- [ ] `pip-audit` (CI `deps` job) has no outstanding high-severity CVEs.
- [ ] Secrets rotated per cadence in `docs/SECRETS_ROTATION.md § 1`.

---

**Last reviewed:** 2026-04-22
**Owner:** SACT Scheduler maintainer
**Escalation:** Cardiff IG (`infogov@cardiff.ac.uk`),
Velindre Cancer Centre IG lead.
