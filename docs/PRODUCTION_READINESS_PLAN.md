# Production-Readiness Plan

Living reference document — tracks the journey from "MSc dissertation quality" to "production clinical deployment ready."
Each step is checked off as the corresponding commit lands.  The plan is executed in the order below; checkboxes updated in commits named `Plan: complete T<tier>.<step>`.

---

## Tier 1 — Public-repo hygiene  *(must complete first; blocks every later tier)*

- [ ] T1.1 — Strengthen `.gitignore` (`*.log`, `data_cache/`, `*.pkl`, `*.pt`)
- [ ] T1.2 — Scrub git history of `sact_scheduler.log`, `data_cache/**/*.jsonl`, `data_cache/_*.json`, `models/**/*.pkl`, `data_cache/feature_store/online.pkl`
- [ ] T1.3 — Replace hardcoded `app.secret_key` with `os.environ.get('FLASK_SECRET_KEY') or os.urandom(32).hex()`
- [ ] T1.4 — Add `scripts/pre-commit.sh` blocking `*.log`, `data_cache/`, `*.pkl`, and `datasets/real_data/*.{xlsx,csv,json,parquet}`
- [ ] T1.5 — Add `SECURITY.md` with disclosure contact + threat model
- [ ] T1.6 — Verify: fresh clone → `pytest -q` green, no leaked artefacts

**Exit criteria**: `git ls-files | grep -E '\.log$|\.pkl$|^data_cache'` → empty.  Force-push main + baseline.

---

## Tier 2 — Correctness fixes  *(blocks dissertation submission)*

- [ ] T2.1 — Fix MPC reward priority multiplier (`ml/stochastic_mpc_scheduler.py:419-427`). Add `priority_at_assignment` to `ChairState`.  Add regression test.  Rerun `dissertation_analysis.R §34` + update `main.tex` macros.
- [ ] T2.2 — Fix auto-scaling "parallel" race condition (`ml/auto_scaling_optimizer.py:_parallel_race`).  Pass weights per-call via `optimize(..., weights=...)` kwarg.  Add race-detection test.  Rerun §30.
- [ ] T2.3 — Replace unsafe `pickle.load()` with SHA256-verified `joblib.load()` in 7 modules: `duration_model.py`, `sequence_model.py`, `temporal_fusion_transformer.py`, `feature_store.py`, `decision_focused_learning.py`, `inverse_rl_preferences.py`, `noshow_model.py`
- [ ] T2.4 — Full suite green (441 + ~3 new tests → 444)

**Exit criteria**: three clean bug-fix commits; dissertation numbers regenerated; `pytest -q` → all green.

---

## Tier 3 — Test coverage  *(every untested module gets real tests)*

Execution protocol per module: (a) Read source, (b) write failure-mode checklist, (c) write + iterate tests, (d) verify pass, (e) run full suite, (f) commit plain-message.

### Wave 3.1 — ML core  *(6 modules, ~6 h)*
- [ ] 3.1.1 — `ml/noshow_model.py` (10 tests)
- [ ] 3.1.2 — `ml/feature_engineering.py` (8 tests)
- [ ] 3.1.3 — `ml/duration_model.py` (8 tests)
- [ ] 3.1.4 — `ml/sequence_model.py` (6 tests)
- [ ] 3.1.5 — `ml/multitask_model.py` (6 tests)
- [ ] 3.1.6 — `ml/quantile_forest.py` (6 tests)

### Wave 3.2 — Advanced ML  *(8 modules, ~6 h)*
- [ ] 3.2.1 — `ml/hierarchical_model.py` (6 tests)
- [ ] 3.2.2 — `ml/mc_dropout.py` (6 tests)
- [ ] 3.2.3 — `ml/survival_model.py` (6 tests)
- [ ] 3.2.4 — `ml/uplift_model.py` (6 tests)
- [ ] 3.2.5 — `ml/event_impact_model.py` (5 tests)
- [ ] 3.2.6 — `ml/online_learning.py` (5 tests)
- [ ] 3.2.7 — `ml/causal_model.py` + `ml/causal_validation.py` (8 tests)
- [ ] 3.2.8 — `ml/rl_scheduler.py` (8 tests)

### Wave 3.3 — Research / legacy ML  *(5 modules, ~3 h)*
- [ ] 3.3.1 — `ml/fairness_audit.py` (6 tests)
- [ ] 3.3.2 — `ml/sensitivity_analysis.py` (4 tests)
- [ ] 3.3.3 — `ml/auto_recalibration.py` (4 tests)
- [ ] 3.3.4 — `ml/model_cards.py` (3 tests)
- [ ] 3.3.5 — `ml/train.py` (8 tests)

### Wave 3.4 — Optimization core  *(4 modules, ~4 h)*
- [ ] 3.4.1 — `optimization/column_generation.py` (8 tests, incl vs-CPSAT ground truth)
- [ ] 3.4.2 — `optimization/gnn_feasibility.py` (6 tests, incl soundness)
- [ ] 3.4.3 — `optimization/emergency_mode.py` (5 tests)
- [ ] 3.4.4 — `optimization/uncertainty_optimization.py` (6 tests)

### Wave 3.5 — Monitoring  *(5 modules, ~3 h)*
- [ ] 3.5.1 — `monitoring/alert_manager.py` (5 tests)
- [ ] 3.5.2 — `monitoring/event_aggregator.py` (5 tests)
- [ ] 3.5.3 — `monitoring/news_monitor.py` (4 tests, HTTP-mocked)
- [ ] 3.5.4 — `monitoring/traffic_monitor.py` (5 tests)
- [ ] 3.5.5 — `monitoring/weather_monitor.py` (4 tests)

### Wave 3.6 — Data generation  *(2 modules, ~2 h)*
- [ ] 3.6.1 — `datasets/generate_sample_data.py` (6 tests)
- [ ] 3.6.2 — `datasets/_nhs_calibration.py` (4 tests)

### Wave 3.7 — Viz + root  *(5 modules, ~3 h)*
- [ ] 3.7.1 — `visualization/charts.py` (4 tests)
- [ ] 3.7.2 — `visualization/dashboard.py` (3 tests)
- [ ] 3.7.3 — `visualization/maps.py` (3 tests)
- [ ] 3.7.4 — `config.py` (5 tests)
- [ ] 3.7.5 — `flask_app.py` split into 8 route-group suites (~40 tests)

### Wave 3.8 — Error-path tests for §5.x modules  *(~2 h)*
- [ ] 3.8.1 — `TestEdgeCases` for `test_stochastic_mpc_scheduler.py`
- [ ] 3.8.2 — `TestEdgeCases` for `test_sact_version_adapter.py`
- [ ] 3.8.3 — `TestEdgeCases` for `test_override_learning.py`
- [ ] 3.8.4 — `TestEdgeCases` for `test_rejection_explainer.py`
- [ ] 3.8.5 — expand `test_auto_scaling_optimizer.py` error paths

**Exit criteria**: `pytest -q` → ~656 passed.  `docs/TEST_COVERAGE.md` shows module → test mapping.

---

## Tier 4 — Production hardening  *(~30 h, can ship to real users after)*

### T4.1 — Authentication + authorisation  *(~6 h)*
- [ ] 4.1.1 — Flask-Login session auth for browser users
- [ ] 4.1.2 — `@api_key_required` decorator for machine-to-machine
- [ ] 4.1.3 — Three roles: `viewer`, `operator`, `admin`
- [ ] 4.1.4 — Gate every mutating endpoint (`/api/optimize*`, `/api/mpc/*`, `/api/fairness/*/config`, `/api/sact/adapt`)
- [ ] 4.1.5 — Tests covering 401/403 on protected routes

### T4.2 — Input caps + rate limits  *(~3 h)*
- [ ] 4.2.1 — `Flask-Limiter` default `60/min`
- [ ] 4.2.2 — Tight caps on heavy endpoints (`/api/optimize` 5/min, `/api/twin/evaluate` 10/min, `/api/mpc/simulate` 5/min)
- [ ] 4.2.3 — Numeric-param bounds at route level (reject `400`, not `500`): `horizon_days ≤ 365`, `total_minutes ≤ 1440`, `n_scenarios ≤ 500`, `limit ≤ 1000`
- [ ] 4.2.4 — `app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024`
- [ ] 4.2.5 — Whitelist DML `treatment` / `outcome` column names

### T4.3 — CSRF + session hardening  *(~2 h)*
- [ ] 4.3.1 — `CSRFProtect(app)` for browser flows
- [ ] 4.3.2 — `SESSION_COOKIE_SAMESITE='Strict'`, `HTTPONLY=True`, `SECURE=True`
- [ ] 4.3.3 — `PERMANENT_SESSION_LIFETIME=1800`
- [ ] 4.3.4 — `@csrf.exempt` on JSON API endpoints that require API key

### T4.4 — Structured logging + audit trail  *(~4 h)*
- [ ] 4.4.1 — Replace raw `logger.info(f"...{patient_id}...")` with structured logger that redacts IDs outside debug mode
- [ ] 4.4.2 — JSON log format to stdout (for container log collection)
- [ ] 4.4.3 — `data_cache/audit/*.jsonl` append-only compliance trail
- [ ] 4.4.4 — `request_id` middleware: `request.headers.get('X-Request-ID') or uuid4().hex`
- [ ] 4.4.5 — Every log line tagged with `request_id`

### T4.5 — Observability  *(~4 h)*
- [ ] 4.5.1 — `/metrics` Prometheus endpoint (request_count, latency histogram, optimisation_solve_time, ML prediction latency)
- [ ] 4.5.2 — `/health/live` + `/health/ready`
- [ ] 4.5.3 — OpenTelemetry traces on CP-SAT + MPC hot paths
- [ ] 4.5.4 — Sample Grafana dashboard JSON in `docs/observability/`

### T4.6 — Deployment assets  *(~4 h)*
- [ ] 4.6.1 — Multi-stage `Dockerfile` (Python 3.12, non-root user, pinned deps)
- [ ] 4.6.2 — `docker-compose.yml` (app + nginx + TLS)
- [ ] 4.6.3 — GitHub Actions CI: `pytest` + `ruff` + `mypy` + `bandit` + `pip-audit` on every push; block merge on failure
- [ ] 4.6.4 — Pin deps: `numpy>=1.26.4`, `requests>=2.32.0`, `streamlit>=1.39.0`, `Flask>=3.0`, `urllib3>=2.0`

### T4.7 — Secrets management  *(~2 h)*
- [ ] 4.7.1 — `.env.example` template (no real values)
- [ ] 4.7.2 — `python-dotenv` integration
- [ ] 4.7.3 — Document production-secrets backends (AWS Secrets Manager / Vault / K8s Secrets)

### T4.8 — Data-protection playbook  *(~3 h, NHS-specific)*
- [ ] 4.8.1 — `docs/DATA_PROTECTION.md` covering lawful basis, DSPT cite, data-sharing agreement slot
- [ ] 4.8.2 — Data-retention policy + auto-delete for events older than N days
- [ ] 4.8.3 — Right-to-erasure flow (cascade delete across caches + models + deletion log)
- [ ] 4.8.4 — Pre-export anonymisation check (DRO + identifier-leak scan)

### T4.9 — Pentest readiness  *(~2 h)*
- [ ] 4.9.1 — `bandit -r . -ll` → 0 HIGH
- [ ] 4.9.2 — `semgrep --config p/python` → 0 HIGH
- [ ] 4.9.3 — `pip-audit` → 0 known CVE
- [ ] 4.9.4 — OWASP ZAP baseline against live Flask → no HIGH alerts
- [ ] 4.9.5 — Report in `docs/security/PENTEST_READINESS.md`

**Exit criteria**: `docker compose up` → HTTPS Flask behind nginx; auth required; `bandit` + `pip-audit` + CI green; `DATA_PROTECTION.md` signed off.

---

## Cumulative effort

| Tier | Effort | Calendar | Git commits |
|------|--------|----------|-------------|
| T1 | ~90 min | 0.5 day | 4-5 |
| T2 | ~1 day | 1 day | 3-4 |
| T3 | ~22 h | 3 days | 40+ |
| T4 | ~30 h | 4 days | 20+ |
| **Total** | **~62 h** | **~9 days** | **~70 commits** |

Every commit pushed to `main` on `github.com/weilalicia7/datamonitor` — local and remote kept 100% synced.
