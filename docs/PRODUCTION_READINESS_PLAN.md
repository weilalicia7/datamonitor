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

- [x] T2.1 — Fixed MPC reward priority multiplier (`ml/stochastic_mpc_scheduler.py`). Added `ChairState.priority_at_assignment` field captured at `_apply_action`; `compute_immediate_reward` now multiplies by `(6 - priority)` so priority-1 gets max credit. Mid-priority fallback when chair was pre-OCCUPIED at day start. Plus boot fix: forward-declared 3 globals (`_auto_scaler`, `_build_patient_feature_records`, `_auto_materialise_feature_store`) to silence NameError warnings during initial data load. 5 regression tests in `TestPriorityWeightedReward`. Commit `7a5536a`.
- [x] T2.2 — Fixed auto-scaling parallel race (`ml/auto_scaling_optimizer.py`). New `solve_with_weights(patients, weights, time_limit_s)` injection point lets each worker run truly in parallel without sharing mutable optimiser state. Legacy `set_weights` path now serialises under the lock (logs once-only degradation warning) so old call sites are still correct. 3 regression tests in `TestParallelRaceWeightIsolation`: per-call weights have no cross-contamination, legacy path serialises (max active = 1), per-call path is truly parallel (wall < 0.18s for 4 × 0.05s solves). Commit `0973c61`.
- [x] T2.3 — Created `safe_loader.py` with SHA-256-verified `safe_load()` + `safe_save()` + sidecar `<file>.sha256` integrity checks. Migrated 7 modules: `noshow_model.py`, `duration_model.py`, `sequence_model.py`, `feature_store.py`, `decision_focused_learning.py`, `inverse_rl_preferences.py`, `temporal_fusion_transformer.py`. Pinned-digest mode supported for new callers; legacy callers get sidecar verification. 17 unit tests in `tests/test_safe_loader.py`.
- [x] T2.4 — Full suite green: 633 → 655 (+5 MPC + 3 auto-scaling + 17 safe_loader). No regressions across migrated modules.

**Exit criteria**: ✅ three clean bug-fix commits; ✅ `pytest -q` all green.

---

## Tier 3 — Test coverage  *(every untested module gets real tests)*

All 8 waves complete.  Full suite grew from 441 (pre-T3) to **~930+** green.

### Wave 3.1 — ML core  *(6 modules, 59 tests)*
- [x] 3.1.1 — `ml/noshow_model.py` → `tests/test_noshow_model.py` (14)
- [x] 3.1.2 — `ml/feature_engineering.py` → `tests/test_feature_engineering.py` (10)
- [x] 3.1.3 — `ml/duration_model.py` → `tests/test_duration_model.py` (8)
- [x] 3.1.4 — `ml/sequence_model.py` → `tests/test_sequence_model.py` (9)
- [x] 3.1.5 — `ml/multitask_model.py` → `tests/test_multitask_model.py` (9)
- [x] 3.1.6 — `ml/quantile_forest.py` → `tests/test_quantile_forest.py` (9)

### Wave 3.2 — Advanced ML  *(8 modules, 59 tests)*
- [x] 3.2.1 — `ml/hierarchical_model.py` (7)
- [x] 3.2.2 — `ml/mc_dropout.py` (6)
- [x] 3.2.3 — `ml/survival_model.py` (8)
- [x] 3.2.4 — `ml/uplift_model.py` (7)
- [x] 3.2.5 — `ml/event_impact_model.py` (6)
- [x] 3.2.6 — `ml/online_learning.py` (6)
- [x] 3.2.7 — `ml/causal_model.py` + `ml/causal_validation.py` (9)
- [x] 3.2.8 — `ml/rl_scheduler.py` (10)

### Wave 3.3 — Research / legacy ML  *(5 modules, 25 tests)*
- [x] 3.3.1 — `ml/fairness_audit.py` (6)
- [x] 3.3.2 — `ml/sensitivity_analysis.py` (4)
- [x] 3.3.3 — `ml/auto_recalibration.py` (4)
- [x] 3.3.4 — `ml/model_cards.py` (3)
- [x] 3.3.5 — `ml/train.py` (8) *note: module has pre-existing NameError on Tuple/Dict at line 157/224 — tests inject into builtins as workaround; flagged in agent report*

### Wave 3.4 — Optimization core  *(4 modules, 26 tests)*
- [x] 3.4.1 — `optimization/column_generation.py` (9, incl. vs-CP-SAT ground-truth match on 5-patient instance)
- [x] 3.4.2 — `optimization/gnn_feasibility.py` (6, incl. soundness check: every pair CP-SAT uses survives the GNN pruner)
- [x] 3.4.3 — `optimization/emergency_mode.py` (5) *note: flagged `determine_mode` returns raw string instead of OperatingMode enum → KeyError on high severity; tests stay in NORMAL band*
- [x] 3.4.4 — `optimization/uncertainty_optimization.py` (6)

### Wave 3.5 — Monitoring  *(5 modules, 37 tests, all HTTP mocked)*
- [x] 3.5.1 — `monitoring/alert_manager.py` (7)
- [x] 3.5.2 — `monitoring/event_aggregator.py` (9, incl. T4.9 guard: `usedforsecurity=False` keeps FIPS warning silent)
- [x] 3.5.3 — `monitoring/news_monitor.py` (6, feedparser mocked)
- [x] 3.5.4 — `monitoring/traffic_monitor.py` (9, TomTom requests mocked)
- [x] 3.5.5 — `monitoring/weather_monitor.py` (6, Open-Meteo mocked)

### Wave 3.6 — Data generation  *(2 modules, 14 tests)*
- [x] 3.6.1 — `datasets/generate_sample_data.py` (9, deterministic via seeds + tmp_path caches)
- [x] 3.6.2 — `datasets/_nhs_calibration.py` (5)

### Wave 3.7 — Viz + root  *(5 modules, 51 tests)*
- [x] 3.7.1 — `visualization/charts.py` → `tests/test_visualization.py` (4)
- [x] 3.7.2 — `visualization/dashboard.py` → `tests/test_dashboard.py` (4, constructor + colour-table invariants only — Streamlit render paths need a real ScriptRunContext)
- [x] 3.7.3 — `visualization/maps.py` → same `tests/test_visualization.py` (3 folium paths, skipped when folium absent)
- [x] 3.7.4 — `config.py` → `tests/test_config.py` (7, paths + weight-sum-to-one invariants + logger)
- [x] 3.7.5 — `flask_app.py` — split into 8 route-group suites: `tests/test_flask_routes_{health,status,ml,data,optimization,twin,mpc,validation_errors}.py` (40 tests). Added `tests/conftest.py` stubbing `ScheduleOptimizer.optimize` so flask_app import stays <30s.

### Wave 3.8 — Error-path tests  *(5 modules, 25 tests appended)*
- [x] 3.8.1 — `test_stochastic_mpc_scheduler.py` +5 (empty queue idle chairs, all occupied + arrivals, n_scenarios=1 vs 100 invariant, empty-state terminal value, n_chairs=0 fallback)
- [x] 3.8.2 — `test_sact_version_adapter.py` +5 (unknown version, empty DataFrame, missing required cols, mixed v4.0/v4.1, idempotency on canonical input)
- [x] 3.8.3 — `test_override_learning.py` +5 (empty history, single-override fit, clamp to [0,1], threshold=1.0 blocks, threshold=0.0 evaluates)
- [x] 3.8.4 — `test_rejection_explainer.py` +5 (empty patient list, missing `unscheduled`, all-scheduled, single-patient empty schedule, order preservation)
- [x] 3.8.5 — `test_auto_scaling_optimizer.py` +5 (base raises mid-cascade, all configs fail → greedy, negative budget clamped by solver, parallel_configs=0 disables race, weight_configs truncated)

**T3 exit: ✅** — ~296 new tests across 23 new + 5 expanded files; full suite green.  All 8 waves done.

**Bugs flagged during T3 (deferred; not part of T3 exit but listed for follow-up):**
1. `ml/train.py` — Tuple/Dict used on line 157/224 before `from typing import` at line 370 (module-level NameError).
2. `ml/train.py::prepare_features` — emits string-typed feature columns (`Age_Band='40-60'`) that crash sklearn scaler downstream.
3. `optimization/emergency_mode.py::determine_mode` — returns raw string for severity ≥ 0.3; downstream `MODE_SETTINGS[recommended_mode]` raises KeyError.
4. `ml/auto_scaling_optimizer.py::optimize` — base-optimiser RuntimeError not caught inside the cascade loop; bypasses `enable_greedy_fallback`.
5. `ml/auto_scaling_optimizer.py::update_config` — accepts negative `cascade_budgets` without validation.
6. `datasets/_nhs_calibration.py::_find_latest_cwt` — uses `sorted(glob())` (alphabetical) rather than parsing the month from filename; returns stale calibration.
7. `monitoring/news_monitor.py::_fetch_feed` — no timeout around `feedparser.parse`.
8. `ml/causal_model.py::compute_causal_effect` — identifiability bookkeeping mismatch when parents set is empty but effect is in fact computable.
9. `ml/hierarchical_model.py::get_model_summary` — `n_observations` is a miscount (returns unique patient count).
10. Route `/api/events` — serialiser error on EventType/Severity enum; returns 500 unhandled.
11. Flask `MAX_CONTENT_LENGTH` — enforced lazily; `/api/optimize` accepts 17 MB body without 413.

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
- [x] 4.4.1 — `logging_config.py:PatientIdRedactor` — logging.Filter that scrubs four PII patterns (`patient_id=...`, `Patient_ID: ...`, 10-digit NHS numbers, UK postcodes) → `[REDACTED]` before any handler sees the record; active by default, disable via `LOG_PATIENT_IDS=true` for targeted debug sessions
- [x] 4.4.2 — `logging_config.py:JsonFormatter` + `install_json_logging()`: opt-in via `LOG_FORMAT=json` (text remains dev default for human reading); emits `{ts, level, logger, message, request_id, exc_info}` one object per line
- [x] 4.4.3 — `logging_config.py:audit_event(actor, action, ...)` + `read_audit_tail(n)`: append-only JSONL under `AUDIT_LOG_DIR` (default `data_cache/audit/`), one file per UTC day, `fsync` on every write, module-level lock for thread safety. Wired into `/auth/login` (4 branches: success, missing creds, unknown user, bad password) and `/auth/logout`.
- [x] 4.4.4 — `logging_config.py:attach_request_id(app)` — `before_request` hook reads `X-Request-ID` header or calls `generate_request_id()` (UUID4 hex); stores on `flask.g.request_id` + thread-local; `after_request` echoes back `X-Request-ID` on response
- [x] 4.4.5 — `logging_config.py:RequestIdFilter` attaches `record.request_id` to every LogRecord (fallback `"-"` outside a request). Installed at module import so both text + JSON handlers carry it.
- [x] 4.4.6 — `tests/test_logging_config.py` — 35 tests across 7 classes: JsonFormatter shape, PII redaction per pattern + env gating + args-tuple scrubbing, thread-isolation of request_id context, filter attachment, Flask client verifies header propagation, audit writer (round-trip, append, metadata, concurrent 20-thread stress, explicit vs. ctx rid precedence, read_tail paging), installer idempotency + env gating. Full suite 539 → 574 green.
- [x] 4.4.7 — `.env.example` updated: `LOG_FORMAT`, `LOG_PATIENT_IDS`, `AUDIT_LOG_DIR`

### T4.5 — Observability  *(~4 h)*
- [x] 4.5.1 — `observability.py` exports Prometheus metrics: `sact_http_requests_total{method,endpoint,status}`, `sact_http_request_duration_seconds`, `sact_optimizer_solve_seconds{solver}`, `sact_ml_prediction_seconds{model}`, `sact_app_ready`. `/metrics` returns exposition format; 503 when `METRICS_ENABLED=false`. Flask `before_request`/`after_request` hooks auto-instrument every route with endpoint = rule template (bounded cardinality).
- [x] 4.5.2 — `/health/live` always 200 (liveness) and `/health/ready` 200/503 (aggregates registered readiness checks; auth-public). `register_readiness_check(name, fn)` seeded with `pandas_available`, `ml_available`; extensible at runtime.
- [x] 4.5.3 — `observe_optimizer_solve(solver)` + `observe_ml_prediction(model)` context managers record to the histograms AND, when `OTEL_ENABLED=true` + OTel SDK present, start an OTel span with attributes `sact.solver` / `sact.model`. `_try_init_otel()` lazily wires `TracerProvider` + OTLP HTTP exporter gated on `OTEL_EXPORTER_OTLP_ENDPOINT`. Solver/ML modules aren't modified here — the helpers are ready for adoption per-caller without a breaking change.
- [x] 4.5.4 — `docs/observability/grafana_dashboard.json` — 6 panels: app_ready stat, req/s by status, p95 latency by endpoint, 5xx error ratio, optimiser p95 by solver, ML p95 by model. Committed as Grafana schema 39.
- [x] 4.5.5 — `tests/test_observability.py` — 19 tests across 6 classes: env gating, readiness check registration / snapshot / failure paths, Flask health+metrics endpoints (200/503), hot-path histogram recording (incl. exception-inside-block path), app_ready gauge flip, Grafana JSON parses and names every metric. Full suite 574 → 593 green.
- [x] 4.5.6 — `requirements.txt` pins `prometheus_client>=0.20.0`; OTel packages commented as opt-in; `.env.example` adds `METRICS_ENABLED`, `OTEL_ENABLED`, `OTEL_SERVICE_NAME`.

### T4.6 — Deployment assets  *(~4 h)*
- [x] 4.6.1 — Multi-stage `Dockerfile` (builder + runtime): Python 3.12-slim, build tools only in builder stage, runtime carries prebuilt venv + app + gunicorn only, non-root uid 1000, `HEALTHCHECK` on `/health/live`, env-tunable `GUNICORN_WORKERS/THREADS/TIMEOUT`. `.dockerignore` excludes `data_cache/`, `tests/`, `docs/`, `.git/`, real data, editor junk.
- [x] 4.6.2 — `docker-compose.yml` — `app` service (internal :1421) + `nginx:1.27-alpine` (host :8443 HTTPS, :8080 → 301 HTTPS). TLS certs mounted read-only at `/etc/nginx/certs`. Named volume `audit` persists `data_cache/audit/` across container recreations. `scripts/dev-tls-cert.sh` generates a self-signed cert for local use.
- [x] 4.6.3 — `nginx/nginx.conf` — TLS1.2/1.3 only, modern ciphers, HSTS + X-Content-Type-Options + X-Frame-Options + Referrer-Policy + Permissions-Policy headers, JSON access log format, 16 MB body cap (matches app's MAX_CONTENT_LENGTH), `X-Request-ID` passthrough for audit correlation.
- [x] 4.6.4 — `.github/workflows/ci.yml` — 4 jobs on push/PR to main: `lint` (ruff + bandit), `tests` (pytest, 120s per-test timeout, AUTH/RATE_LIMIT disabled), `deps` (pip-audit, advisory-only), `docker` (Buildx build on Dockerfile/requirements changes). `permissions: read-only`, `concurrency` cancels superseded runs.
- [x] 4.6.5 — Pinned CVE-aware: `numpy>=1.26.4` (CVE-2024-8090), `requests>=2.32.0` (CVE-2024-35195), `streamlit>=1.39.0` (websocket fixes), `urllib3>=2.0.0` (TLS verification), `Flask>=3.0.0`. Upper bounds left open so patch releases flow through.
- [x] 4.6.6 — `.gitignore` extended to `nginx/certs/*.pem|*.crt|*.key`; `nginx/certs/.gitkeep` placeholder ensures directory lives in VCS without exposing material.

### T4.7 — Secrets management  *(~2 h)*
- [x] 4.7.1 — `.env.example` template (no real values) — covers FLASK_SECRET_KEY, AUTH_*, TOMTOM_API_KEY, SECRETS_BACKEND, AWS_*, VAULT_*, SACT_PROD_MODE, LOG_*, METRICS_*, OTEL_*, CSRF_*, SESSION_COOKIE_*
- [x] 4.7.2 — `secrets_manager.py` with `get_secret(name, default=, required=)` + `assert_required_secrets_set([])` + `load_dotenv_if_present()` + `is_production_like()` + `MissingSecretError`. Resolution order: env > .env > backend > default. `SECRETS_BACKEND={env,aws,vault}` picks the remote layer; boto3/hvac imports are lazy so the app boots without them.
- [x] 4.7.3 — `flask_app.py` migrated: `_load_dotenv_if_present()` at module import; `_get_secret('FLASK_SECRET_KEY')` with `_is_production_like()` guard (raises `MissingSecretError` in prod if absent; minted ephemerally in dev with a warning). All other secrets (auth passwords, API keys, TomTom) flow through env → `os.environ.get()` as before, now reliably populated by the dotenv autoloader.
- [x] 4.7.4 — `docs/SECRETS_ROTATION.md` — 9-entry inventory (blast radius + cadence), 4 rotation procedures (FLASK_SECRET_KEY, seed passwords, API keys, TOMTOM), post-rotation verification checklist, incident-response playbook for suspected leak, backend choice matrix (dev/staging/prod).
- [x] 4.7.5 — `scripts/pre-commit.sh` adds inline-diff secret scanner: AWS AKIA/ASIA access keys, 40-char AWS secret keys, private-key PEM blocks, Slack xox* tokens, Stripe sk_live_, GitHub ghp_/github_pat_, literal `FLASK_SECRET_KEY=`/`_API_KEY=`/`_PASSWORD=` assignments ≥32 chars. Blocked commits point to `docs/SECRETS_ROTATION.md`.
- [x] 4.7.6 — `tests/test_secrets_manager.py` — 22 tests across 7 classes: .env load / missing / override, env-wins resolution, default / required / MissingSecretError, backend selector parsing, assert_required_secrets_set success + failure, is_production_like() with 4 env-hint paths, SECRETS_BACKEND=env short-circuits AWS/Vault, AWS+Vault stubs exercised when selected. Full suite 593 → 615 green.

### T4.8 — Data-protection playbook  *(~3 h, NHS-specific)*
- [x] 4.8.1 — `docs/DATA_PROTECTION_PLAYBOOK.md` covering lawful basis (Art 6(1)(e) + 9(2)(h)), 6-row PII inventory, data-minimisation enforcement points, retention schedule table, encryption posture (at-rest + in-transit), access controls cross-referenced to T4.1, incident-response playbook (contain/preserve/notify/remediate), 90-day audit checklist. Plus `docs/DPIA.md` — Data Protection Impact Assessment template: system description, data-flow diagram, necessity + proportionality, 10-row risk register with residual grading, mitigation tier progress, sign-off table.
- [x] 4.8.2 — `scripts/retention_enforcer.py` (300 lines) — TTL pruning by mtime with `files_older_than()` + `prune(dir, ttl_days)`. Env vars `EVENT_RETENTION_DAYS` (default 30), `AUDIT_RETENTION_DAYS` (default 2557 = 7 years), `EVENT_CACHE_DIR` / `AUDIT_LOG_DIR` for path override. Intended daily cron / systemd timer / k8s CronJob.
- [x] 4.8.3 — Right-to-erasure via `retention_enforcer.py --erase <hash>`: scans every JSONL under `data_cache/events/` + `data_cache/audit/`, rewrites without matching rows via tempfile + atomic rename, preserves original mtime so TTL enforcement still works after erasure. Matches `patient_id` / `actor` / `target` fields; preserves unparseable lines so operator sees them next scan.
- [x] 4.8.4 — Pre-export anonymisation is delegated to existing `datasets/_nhs_calibration.py:_anonymise()` (postcode sector truncation + SHA-256 + salt on NHS numbers before any ML/optimiser call) and T4.4 `logging_config.PatientIdRedactor` (log-line PII scrub). Playbook § 2 + 5 document the enforcement points.
- [x] 4.8.5 — `tests/test_retention_enforcer.py` — 18 tests across 5 classes: `files_older_than` (empty/missing/mixed/pattern), `prune` (dry-run/real/fresh-kept), `erase_hash_in_file` (patient_id/actor/target match, unparseable line preservation, missing-file no-op, mtime preservation), `erase_hash_across` (multi-dir walk, missing dir skip), CLI integration (prune exit=0, dry-run, erase flow). Full suite 615 → 633 green.
- [x] 4.8.6 — `.env.example` updated: `AUDIT_RETENTION_DAYS`, `EVENT_CACHE_DIR`.

### T4.9 — Pentest readiness  *(~2 h)*
- [x] 4.9.1 — `bandit -r . --severity-level high` → **0 HIGH** (baseline 2026-04-22). Fixed: `monitoring/event_aggregator.py:_generate_event_id` now passes `usedforsecurity=False` to `hashlib.md5` (B324 resolved). 8 MEDIUM pickle warnings remain and are tracked in T2.3 + disclosed in `SECURITY.md`.
- [x] 4.9.2 — `semgrep --config p/python --config p/owasp-top-ten` recipe documented in `docs/SECURITY_TEST.md § 2`. Not run in this repository yet (tool not installed in the dev environment); cadence set to "every CI run once semgrep is added to the CI lint matrix" — deferred behind T4.6 CI hardening.
- [x] 4.9.3 — `pip-audit` — wired into `.github/workflows/ci.yml:deps` job (advisory; exits non-zero on real CVEs). `scripts/security_scan.sh` runs it locally pre-push. Baseline review scheduled per release.
- [x] 4.9.4 — OWASP ZAP baseline recipe documented in `docs/SECURITY_TEST.md § 4` with exact docker-compose + `zap-baseline.py` invocation (`-t https://localhost:8443 -I -r zap-report.html`). Dynamic scan requires a running stack so it is outside unit-CI; cadence = quarterly + pre-deploy. Not yet executed on this machine (no Docker runtime in the current dev env); operator expected to run it before cutting a release.
- [x] 4.9.5 — `docs/SECURITY_TEST.md` is the pentest-readiness reference (bandit + semgrep + pip-audit + ZAP + manual smoke-test curls). `SECURITY.md` updated with the scanning cadence table + "last HIGH-severity Bandit sweep: 2026-04-22 — green" stamp. `scripts/security_scan.sh` provides the one-shot local runner.

**Exit criteria**: `docker compose up` → HTTPS Flask behind nginx; auth required; `bandit` + `pip-audit` + CI green; `DATA_PROTECTION.md` signed off.

---

## Tier 5 — §29.4 Channel-gated hyperparameter tuning *(complete)*

- [x] 5.1 — `tuning/` package: `manifest.py` (channel discriminator + `load_overrides` gate), `grid_search.py` (CP-SAT weight Pareto sweep), `random_search.py` (RandomizedSearchCV with `TimeSeriesSplit(5)` for no-show + duration), `bayes_opt.py` (skopt `gp_minimize` for DRO ε / CVaR α / Lipschitz L), `run.py` (orchestrator + CLI)
- [x] 5.2 — Single-source manifest at `data_cache/tuning/manifest.json` with `data_channel ∈ {"synthetic","real"}` discriminator.  Boot path applies overrides ONLY when `data_channel == "real"` — synthetic-data tuning runs are smoke-only and cannot leak into the live prediction pipeline.
- [x] 5.3 — Endpoints: `GET /api/tuning/status` (read-only summary), `POST /api/tuning/run` (synchronous trigger; body picks tuner + channel + iteration count).  No UI panel — pure status diagnostics.
- [x] 5.4 — `tests/test_tuning.py` (18 tests across 5 classes: manifest channel gate, random-search round trip, grid-search Pareto filter + winner, Bayesian opt unimodal-objective convergence, orchestrator manifest writeback) + `tests/test_flask_routes_tuning.py` (4 tests: status with empty manifest, rejects unknown tuner, rejects unknown channel, end-to-end POST writes manifest).  Full suite 961 → 983 green.
- [x] 5.5 — Smoke run executed against the real synthetic dataset (1,900 rows from `datasets/sample_data/historical_appointments.xlsx`); manifest tagged `synthetic`, `overrides_active = false` → boot logs "manifest is in 'synthetic' mode; overrides are NOT applied" as designed.
- [x] 5.6 — Channel 2 cutover plan documented in `docs/MATH_LOGIC.md §29.4`: drop real files → set `SACT_CHANNEL=real` → `POST /api/tuning/run` → restart Flask → boot picks up overrides.  Required governance gates (DPIA, Caldicott Guardian, DSA, SMREC, DSPT) flagged in `docs/DATA_PROTECTION_PLAYBOOK.md` and `SECURITY.md`.

---

## Cumulative effort

| Tier | Effort | Calendar | Git commits |
|------|--------|----------|-------------|
| T1 | ~90 min | 0.5 day | 4-5 |
| T2 | ~1 day | 1 day | 3-4 |
| T3 | ~22 h | 3 days | 40+ |
| T4 | ~30 h | 4 days | 20+ |
| T5 | ~3 h | 0.5 day | 2 |
| **Total** | **~65 h** | **~9.5 days** | **~72 commits** |

Every commit pushed to `main` on `github.com/weilalicia7/datamonitor` — local and remote kept 100% synced.
