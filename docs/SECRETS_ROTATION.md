# Secrets Rotation Runbook

Operator-facing procedure for every credential / token the SACT Scheduler
relies on.  Applies to production deployments; development installs
generally set values once in `.env` and rotate only when something leaks.

All secrets are resolved via `secrets_manager.get_secret(name, required=...)`
(see `secrets_manager.py`).  Resolution order: process env →
python-dotenv `.env` → configured remote backend (`SECRETS_BACKEND=aws|vault`).

> **Key rule:** **never commit a rotated secret** to git.  Update your
> infrastructure secret store (AWS Secrets Manager / Vault / Kubernetes
> Secret) and restart / rolling-update the app — secrets are loaded at
> process start.

---

## 1. Inventory

| Secret | Purpose | Rotation cadence | Blast radius if leaked |
|---|---|---|---|
| `FLASK_SECRET_KEY` | Signs session cookies + CSRF tokens | 90 days in prod; immediately on leak | Session forgery; auth bypass for active sessions |
| `AUTH_SEED_ADMIN_PASSWORD` | Bootstraps the admin user at first boot | 90 days; immediately on leak | Full admin access |
| `AUTH_SEED_OPERATOR_PASSWORD` | Bootstraps operator identity | 90 days | Schedule write access |
| `AUTH_SEED_VIEWER_PASSWORD` | Read-only dashboard access | 180 days | Read access to schedules + patient IDs (redacted in logs) |
| `AUTH_ADMIN_API_KEY` | Machine-to-machine admin access (X-API-Key) | 90 days | Full admin access |
| `AUTH_OPERATOR_API_KEY` | M2M operator access | 90 days | Schedule write access |
| `AUTH_VIEWER_API_KEY` | M2M read access | 180 days | Read access |
| `TOMTOM_API_KEY` | Traffic API calls | Per TomTom ToS (12 months) | Billing fraud (free tier capped; paid tier abuseable) |
| `AUDIT_LOG_DIR` signing key (future) | Audit-log tamper detection | 365 days | Ability to rewrite history undetected |

Additional operational credentials (DB password, object-store keys, OTLP
auth) follow the same backend pattern but aren't consumed by the current
codebase.

---

## 2. Pre-rotation checklist

1. **Announce** the rotation window in the team channel (30 min for
   API keys; 60 min for the Flask secret — session invalidation follows).
2. **Verify current audit trail** is green: no unexplained failed logins
   in the past 24 h.  See `/admin/audit` (T4.4) or tail
   `data_cache/audit/*.jsonl`.
3. **Warm standby**: confirm you have a second admin credential available
   so you can log back in after rotating the first.

---

## 3. Rotation steps

### 3.1  FLASK_SECRET_KEY

```bash
# 1. Generate a new key
NEW_KEY="$(python -c "import secrets; print(secrets.token_hex(32))")"

# 2. Update the secret store
#    AWS:   aws secretsmanager update-secret --secret-id sact/FLASK_SECRET_KEY ...
#    Vault: vault kv put secret/sact/FLASK_SECRET_KEY value="$NEW_KEY"
#    K8s:   kubectl create secret generic sact-flask --from-literal=FLASK_SECRET_KEY="$NEW_KEY" --dry-run=client -oyaml | kubectl apply -f -

# 3. Rolling restart — new pods pick up the new secret
kubectl rollout restart deployment/sact-scheduler    # (or: docker compose restart app)

# 4. All existing sessions are now invalid.  Operators re-authenticate.
```

### 3.2  AUTH_SEED_* passwords

Seed passwords only fire on **first boot** — once the user exists in
memory, changing the env var does nothing to the running process.

```bash
# 1. Pick a new password
NEW_PW="$(python -c "import secrets; print(secrets.token_urlsafe(24))")"

# 2. Update the secret store (same three patterns as above)

# 3. Rolling restart — the new seed is applied only when the
#    in-memory user store is empty.  In production, the persistent
#    user DB rotation happens via an admin API call (not via env var).
```

### 3.3  AUTH_*_API_KEY

```bash
# 1. Issue a replacement
NEW_KEY="$(python -c "import secrets; print(secrets.token_urlsafe(32))")"

# 2. ADD the new key to the secret store alongside the old one
#    (requires a short overlap where both keys validate — implement via
#    a second env var AUTH_ADMIN_API_KEY_NEXT and switch auth.py to
#    accept either; once clients update, remove the old value).

# 3. Notify API consumers; cut over; remove the retired key.
```

### 3.4  TOMTOM_API_KEY

```bash
# 1. Provision a new key from the TomTom developer portal.
# 2. Update the secret store.
# 3. Rolling restart.
# 4. Retire the old key in the TomTom dashboard (stops ghost-billing if
#    leak is suspected).
```

---

## 4. Post-rotation verification

1. **Smoke test**: log in with a fresh session, verify `/auth/whoami`
   returns the expected role.
2. **Audit trail**: the rotation event should appear in
   `data_cache/audit/<today>.jsonl` as `action: auth.login, outcome: success`
   for the re-login.
3. **Monitor 5xx**: watch the Grafana error-rate panel for 5 minutes;
   an unexpected spike can indicate a stale client cache.
4. **Document**: log the rotation in your secrets-management ticket
   system (what was rotated, by whom, at what UTC time).  Retain 7 years
   under NHS records-management guidance.

---

## 5. Incident: suspected leak

If a secret is posted publicly (GitHub, Pastebin, Slack leak):

1. **Immediately** revoke / rotate per § 3 above.
2. Pull the git history of the repo where the leak happened with
   `git log -p --all -S '<secret-fragment>'` to scope the exposure.
3. Invalidate every session (just rotate `FLASK_SECRET_KEY`).
4. File a Data-Protection-Impact-Assessment (DPIA) entry; Cardiff IG
   contact: `infogov@cardiff.ac.uk` (via `SECURITY.md`).
5. Review the pre-commit hook + `.gitignore` to confirm the offending
   file is now blocked from future commits.

---

## 6. Backend choice

| Env | Reasonable backend | Why |
|---|---|---|
| Local dev | `SECRETS_BACKEND=env` + `.env` file | Zero-friction; `.env` gitignored + pre-commit blocks it anyway. |
| Staging | `SECRETS_BACKEND=aws` (AWS Secrets Manager) with rotation Lambda | Matches prod; surfaces misconfig early. |
| Prod | `SECRETS_BACKEND=aws` or `vault` | Audit logs per-secret access; quorum-rotation support. |

Never use `SECRETS_BACKEND=env` in production.  Lean on the infra layer
to inject values via IAM role / service-account binding; the app should
never see plain-text secrets in deployment manifests.
