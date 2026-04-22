# Security Test Runbook

Scope: the scan procedures a human or CI job should run **before every
deploy to production**.  The aim is to keep four metrics at zero
whenever main is green:

| Tool | Cadence | Target |
|---|---|---|
| Bandit  (static AST) | every CI run | 0 HIGH |
| Semgrep (rule-based) | every CI run | 0 HIGH on `p/python` + `p/owasp-top-ten` |
| pip-audit (dep CVEs) | nightly + every CI run | 0 open HIGH/CRITICAL CVEs with fix available |
| OWASP ZAP (dynamic)  | weekly + pre-deploy | 0 HIGH findings on `http://localhost:1421/api/*` |

All four are wired into `.github/workflows/ci.yml` except ZAP (dynamic
scan requires a running instance — not part of unit CI).  The ZAP recipe
below can run against the docker-compose stack.

---

## 1. Bandit — static Python security linter

Install (one-time):

```bash
pip install --user bandit
```

Run:

```bash
# Fail on any HIGH severity; ignore test fixtures and runtime caches.
python -m bandit -r . --severity-level high \
    -x tests,data_cache,.venv,node_modules,docs
```

Exit code `0` means zero HIGH issues.  Medium-severity issues are
tolerated (currently 8, all pickle-deserialisation warnings pending T2.3).
Each remaining medium is documented in `SECURITY.md § known limitations`.

## 2. Semgrep — rule-based semantic scan

```bash
pip install --user semgrep

semgrep --config p/python --config p/owasp-top-ten \
        --severity ERROR --error --quiet \
        --exclude tests --exclude data_cache
```

`--error` makes semgrep exit non-zero on any match of ERROR severity.

## 3. pip-audit — CVE scan of installed deps

```bash
pip install --user pip-audit

pip-audit --strict -r requirements.txt
```

`--strict` treats tool errors (transient network, malformed metadata)
as failures — a truly green run is necessary, not sufficient.

## 4. OWASP ZAP — dynamic application scan

Requires the docker-compose stack to be running on the local machine
(`docker compose up -d`).  ZAP runs in its own container and proxies
against the nginx endpoint.

```bash
# 1) Start the app stack
./scripts/dev-tls-cert.sh
docker compose up -d --build

# 2) Wait for healthy
until curl -sk https://localhost:8443/health/live >/dev/null; do
    sleep 2
done

# 3) Baseline scan (passive only; no exploits)
docker run --rm --network=host \
    -v "$(pwd)/reports":/zap/wrk/:rw \
    -t zaproxy/zap-stable \
    zap-baseline.py \
        -t https://localhost:8443 \
        -r zap-report.html \
        -I          # ignore TLS cert warnings (self-signed in dev)

# 4) Inspect reports/zap-report.html
# 0 HIGH findings should be baseline expectation before prod deploy.
```

For a **full active scan** (simulates an attacker — only run against a
throwaway container), substitute `zap-full-scan.py`.  NEVER run an
active scan against a live clinical system.

## 5. Manual smoke tests

Commands the operator runs to sanity-check the security controls once
the stack is up:

```bash
# --- T4.1 auth ---
# Without auth, viewer GETs are 200
curl -sk -o /dev/null -w "%{http_code}\n" https://localhost:8443/api/status
# With AUTH_ENABLED=true, same call returns 401 unless we send a token
AUTH_ENABLED=true docker compose up -d --build app
curl -sk -o /dev/null -w "%{http_code}\n" https://localhost:8443/api/status     # expect 401
curl -sk -H "X-API-Key: $AUTH_VIEWER_API_KEY" \
    -o /dev/null -w "%{http_code}\n" https://localhost:8443/api/status         # expect 200

# --- T4.2 input cap ---
# 16 MB is tolerated; 17 MB is rejected pre-handler.
dd if=/dev/zero of=/tmp/big.json bs=1M count=17
curl -sk -o /dev/null -w "%{http_code}\n" \
    -X POST -H "Content-Type: application/json" --data-binary @/tmp/big.json \
    https://localhost:8443/api/mpc/simulate                                    # expect 413

# --- T4.3 session cookie ---
curl -skI https://localhost:8443/ | grep -i "set-cookie"                       # HttpOnly; SameSite=Strict

# --- T4.4 request-id propagation ---
curl -sk -H "X-Request-ID: ZAP-smoke-123" \
    -D - https://localhost:8443/health/live | grep -i "X-Request-ID"

# --- T4.5 metrics ---
curl -sk https://localhost:8443/metrics | head -5                              # Prometheus exposition

# --- T4.6 nginx security headers ---
curl -skI https://localhost:8443/ | grep -iE \
    "strict-transport-security|x-content-type-options|x-frame-options"
```

## 6. Reporting cadence

| When | Who | Where |
|---|---|---|
| Every PR | CI (`lint` + `tests` + `deps`) | GitHub PR check |
| Weekly | Operator | `docs/security/scans-YYYY-MM-DD.md` (gitignored if noisy) |
| Pre-deploy | Operator | Attach scan summary to change-advisory ticket |
| Quarterly | Security-aware engineer | Run full ZAP scan + update risk register in `DPIA.md § 4` |

## 7. Known exceptions

Any tool finding that is accepted rather than fixed must be documented
in `SECURITY.md § Known limitations` **with a remediation owner and ETA**.
Silent suppression (`#nosec`, `# noqa`) is not allowed without a matching
entry in that file.
