# Data Protection Impact Assessment

> Template DPIA for the SACT Scheduler.  Required under UK GDPR Article
> 35 because this system carries out "systematic and extensive"
> processing of health data that drives operational decisions (chair
> allocation, reminder dispatch).  This document is the dissertation's
> template; a per-deployment copy must be filled, signed, and retained
> by the deploying NHS organisation before any real-patient-data run.

---

## 1. System description

| Field | Value |
|---|---|
| System name | SACT Intelligent Scheduler |
| Version | (commit SHA of the deployment release) |
| Controller | Velindre University NHS Trust |
| Processor | (deploying unit — typically Velindre IT) |
| Purpose | Scheduling of SACT chemotherapy sessions across chairs + reminder/intervention recommendation |
| Lawful basis | UK GDPR Art. 6(1)(e) — public task; Art. 9(2)(h) — health care |
| Anticipated run-rate | ~500 appointments/day, single site |

## 2. Data flow summary

```
[ SACT clinical record (source of truth) ]
            │
            │  one-way read; pseudonymised
            ▼
[ SACT Scheduler — this application ]
   │    │       │
   │    │       └──► Prometheus / Grafana (no PII — aggregate only)
   │    └────────►  OTel (optional; pseudonymised spans only)
   │
   └──► data_cache/events/ (hashed patient ID + features, 30-day TTL)
   └──► data_cache/audit/  (actor/action/outcome, 7-year retention)
```

No PII is transmitted outside the trust.  External APIs (TomTom traffic,
feedparser RSS) are consulted only with depersonalised inputs (town
names, postcode sectors).

## 3. Necessity + proportionality

| Question | Answer |
|---|---|
| Is a DPIA legally required? | Yes — Art. 35, large-scale processing of health data. |
| Could we achieve the aim with less data? | No — chair scheduling needs appointment duration, priority, and attendance history.  We drop name / DOB / full postcode on ingest. |
| Have we minimised retention? | Yes — see `DATA_PROTECTION_PLAYBOOK.md § 3`. |
| Have we given data subjects fair information? | Handled via the Velindre patient privacy notice, which covers derivative systems. |

## 4. Risk register

| # | Risk | Likelihood | Impact | Controls | Residual |
|---|---|---|---|---|---|
| 1 | Accidental commit of real data to public git | Medium | Very high | `.gitignore` + pre-commit hook + `datasets/real_data/.gitkeep` | Low |
| 2 | Session hijack via stolen cookie | Low | High | T4.3 SameSite=Strict + Secure + HTTPOnly + 30-min TTL | Very low |
| 3 | CSRF on browser form endpoint | Low | Medium | T4.3 CSRFProtect default-on; JSON API exempted | Low |
| 4 | Unauthenticated access to `/api/*` | Medium | High | T4.1 `AUTH_ENABLED=true` + role cascade | Very low |
| 5 | Log line leaks patient ID | Medium | Medium | T4.4 `PatientIdRedactor` active by default | Low |
| 6 | Secret accidentally committed | Low | Very high | `.gitignore` + pre-commit scanner + `docs/SECRETS_ROTATION.md § 5` | Very low |
| 7 | Right-to-erasure request unfulfilled | Low | High | `scripts/retention_enforcer.py --erase <hash>` | Low |
| 8 | Pickle-deserialisation RCE from untrusted model | Low | Very high | `SECURITY.md` known-limitation; T2.3 planned switch to joblib + SHA verify | Medium |
| 9 | DoS via huge JSON payload | Medium | Medium | T4.2 `MAX_CONTENT_LENGTH=16 MB` + Flask-Limiter 60/min | Very low |
| 10 | Audit trail tampering by compromised app | Low | High | Append-only JSONL with `fsync`; WORM backup in § 3 playbook | Medium |

## 5. Mitigation progress

| Tier | Status |
|---|---|
| T1 Public-repo hygiene | ✅ Complete |
| T4.1 Authentication | ✅ Complete |
| T4.2 Input caps + rate limits | ✅ Complete |
| T4.3 CSRF + session hardening | ✅ Complete |
| T4.4 Structured logging + audit trail | ✅ Complete |
| T4.5 Observability | ✅ Complete |
| T4.6 Deployment assets | ✅ Complete |
| T4.7 Secrets management | ✅ Complete |
| T4.8 Data-protection playbook | 🟡 This doc + retention script in progress |
| T4.9 Pentest readiness | ⏳ Planned |

## 6. Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| System owner | | | |
| Cardiff DPO | | | |
| Velindre IG | | | |
| Clinical safety officer | | | |

---

**Last reviewed:** 2026-04-22
**Next review:** on system change or annually, whichever sooner
