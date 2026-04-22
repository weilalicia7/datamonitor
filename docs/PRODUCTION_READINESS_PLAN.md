# Production-Readiness Plan

Living reference document — tracks the journey from "MSc dissertation quality" to "production clinical deployment ready."
Each step is checked off as the corresponding commit lands.  The plan is executed in the order below; checkboxes updated in commits named `Plan: complete T<tier>.<step>`.

---

## Tier 1 — Public-repo hygiene  *(must complete first; blocks every later tier)*

- [x] T1.1 — Strengthen `.gitignore` (`*.log`, `data_cache/`, `*.pkl`, `*.pt`, smoke-test strays, real-data paths) — commit `f3f15f2`
- [x] T1.2 — Scrub git history of `sact_scheduler.log`, `data_cache/**`, `models/*.pkl`, `models/*.pt` via `git filter-branch`; dropped empty `baseline` branch (orphan, was unusable with `/ultrareview`); GC'd 44 orphaned objects — force-pushed over `f5df823` to `f3f15f2`
- [x] T1.3 — Replace hardcoded `app.secret_key` with `os.environ.get('FLASK_SECRET_KEY') or os.urandom(32).hex()` + warning-log when env var missing — `flask_app.py:91-99`
- [x] T1.4 — Add `scripts/pre-commit.sh` blocking `*.log`, `data_cache/`, `*.pkl`, `*.pt`, `.env`, real-data files; `scripts/install-hooks.sh` for idempotent install; hook self-tested (correctly blocked a `.log` stage attempt)
- [x] T1.5 — Add `SECURITY.md` with Cardiff Information Governance contact + explicit threat model + pickle-deserialisation known limitation
- [x] T1.6 — Verified: local = origin at `f3f15f2` (159 tracked files, 0 sensitive); `pytest -q` → 441/441 still green

**Exit criteria**: `git ls-files | grep -E '\.log$|\.pkl$|^data_cache'` → empty. ✅  Force-push main. ✅

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

### T4.1 — Authentication + authorisation  *(~6 h)* ✅ **complete**
- [x] 4.1.1 — Flask-Login session auth wired via `init_login_manager(app)` + `/auth/login` / `/auth/logout` / `/auth/whoami` endpoints (`auth.py` + `flask_app.py:103-219`)
- [x] 4.1.2 — `@api_key_required` decorator + `X-API-Key` header recognised by `_current_identity()` (`auth.py:219-253`)
- [x] 4.1.3 — Three roles with cascade (admin ⊇ operator ⊇ viewer) via `_has_role()` + `role_required(...)` decorator (`auth.py:55-72, 201-219`)
- [x] 4.1.4 — `before_request` gate in `flask_app.py:174-219` applies role rules across all ~60 endpoints via a cascade: `/auth/*` + `/health/*` + `/metrics` open; `POST .../config` = admin; all other POST/PUT/PATCH/DELETE = operator; GET = viewer. No need to touch individual routes.
- [x] 4.1.5 — `tests/test_auth.py` — 39 tests: password hashing, role hierarchy, registration, env-seeding, Flask endpoints, route-role mapping, before_request enforcement. Full suite 441 → 480 green.
- [x] 4.1.6 — `.env.example` + `requirements.txt` updated with Flask-Login / Flask-WTF / Flask-Limiter / python-dotenv

### T4.2 — Input caps + rate limits  *(~3 h)*
- [x] 4.2.1 — `Flask-Limiter` factory in `validators.py:init_rate_limiter()`; env-gated by `RATE_LIMIT_ENABLED` (default `False` so tests don't trip 429); defaults `60/min + 600/hr` per remote IP when active; `storage_uri` via `RATE_LIMIT_STORAGE_URI` (defaults `memory://`) — `flask_app.py:156-160`
- [x] 4.2.2 — Per-route tight caps deferred to production via `@limiter.limit(...)` — limiter instance exposed; caps applied at-need without code changes when `RATE_LIMIT_ENABLED=true`
- [x] 4.2.3 — Numeric-param bounds enforced at route level via `_clamp_int()` / `_clamp_float()`: applied to `/api/twin/evaluate` (horizon_days, step_hours, rng_seed), `/api/twin/compare` (same), `/api/twin/evaluations` (limit), `/api/irl/overrides` (limit), `/api/mpc/simulate` (total_minutes, rng_seed, policies ≤10), `/api/mpc/config` (n_scenarios, lookahead_minutes, total_timeout_s), `/api/ml/uncertainty-optimization/evaluate` + `/metrics` (n_scenarios, epsilon, alpha); caps centralised in `validators.py:ENDPOINT_BOUNDS` with per-key `VALIDATOR_CAP_<NAME>` env override
- [x] 4.2.4 — `app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024` via `DEFAULT_MAX_CONTENT_LENGTH` constant — `flask_app.py:153`
- [x] 4.2.5 — DML column whitelists: `DML_TREATMENT_ALLOWED` (9 names), `DML_OUTCOME_ALLOWED` (5), `DML_COVARIATES_ALLOWED` (16); applied to `/api/ml/dml/estimate` + `/api/ml/dml/compare` via `_validate_whitelist()` / `_validate_whitelist_many()`
- [x] 4.2.6 — `tests/test_validators.py` — 40 tests covering clamp_int/clamp_float, ENDPOINT_BOUNDS + env override, whitelist single + many, error-response shape, rate-limiter factory, MAX_CONTENT_LENGTH via Flask test client, end-to-end errorhandler. Full suite 480 → 520 green.

### T4.3 — CSRF + session hardening  *(~2 h)*
- [x] 4.3.1 — `session_config.py:init_csrf()` instantiates `CSRFProtect(app)` for browser flows (fail-safe default); env-gated via `CSRF_ENABLED` — `flask_app.py:184-189`
- [x] 4.3.2 — `session_config.py:apply_session_cookie_config()` sets `SESSION_COOKIE_SAMESITE='Strict'`, `HTTPONLY=True`, `SECURE=false` (dev default; prod must set `SESSION_COOKIE_SECURE=true` in env); plus Chromium fix-up when SameSite=None without Secure — `flask_app.py:176-182`
- [x] 4.3.3 — `PERMANENT_SESSION_LIFETIME=1800` (30 min idle) via `SESSION_LIFETIME_SECONDS` env override
- [x] 4.3.4 — `exempt_json_api_routes()` called post-registration (end of `flask_app.py`) auto-exempts every view under `/api/*`, `/auth/*`, `/health`, `/metrics`; browser routes (`/`, `/dashboard`) stay protected
- [x] 4.3.5 — `tests/test_session_config.py` — 19 tests across 5 classes: cookie defaults + env overrides, SameSite/Secure fixup, CSRF init gating, exempt-prefix matching (positive + negative), end-to-end Flask client verifying `/api/*` allowed without token and `/form/*` blocked (400). Full suite 520 → 539 green.
- [x] 4.3.6 — `.env.example` updated: `CSRF_ENABLED`, `SESSION_COOKIE_*`, `SESSION_LIFETIME_SECONDS`, plus T4.2 rate-limit + validator-cap examples

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
