#!/usr/bin/env bash
#
# Pre-commit hook — block runtime artefacts and real-patient-data files from
# ever reaching a commit.  Install via:
#
#     cp scripts/pre-commit.sh .git/hooks/pre-commit
#     chmod +x .git/hooks/pre-commit
#
# (Or run  scripts/install-hooks.sh  if provided.)
#
# The checks here are intentionally duplicative of .gitignore: a developer
# can always override .gitignore with `git add -f`, but this hook fires
# unconditionally on every `git commit` attempt.

set -euo pipefail

# Compute the list of paths the developer is asking to stage.
STAGED="$(git diff --cached --name-only --diff-filter=ACM)"

if [ -z "$STAGED" ]; then
    exit 0
fi

BLOCKED_LINES=""
block() {
    local pattern="$1"
    local reason="$2"
    local hits
    hits="$(echo "$STAGED" | grep -E "$pattern" || true)"
    if [ -n "$hits" ]; then
        BLOCKED_LINES+="  - ${reason}:\n$(echo "$hits" | sed 's/^/      /')\n"
    fi
}

# 1. Real patient data file types under real_data/
block '^datasets/real_data/.+\.(xlsx|csv|json|parquet)$' \
      'real-patient-data files must never be committed'

# 2. Log files anywhere
block '\.log(\.[0-9]+)?$'          'log files may carry patient IDs'

# 3. Pickled models / binaries
block '\.(pkl|pt|pth|joblib|onnx)$' 'model binaries are opaque + code-exec risk'

# 4. Runtime-generated event caches
block '^data_cache/'                'data_cache/ is a runtime output directory'

# 5. Smoke-test scratch files
block '^data_cache/_|^_smoketest_' 'scratch / smoke-test leftovers'

# 6. .env secrets (but allow .env.example / .env.sample / .env.template which
# are documented templates carrying no real secrets)
block '^\.env$|^\.env\.(local|production|staging|development|dev|prod|stage)$' \
      '.env files may contain API keys (templates ending .example/.sample/.template are allowed)'

# 7. Config locals
block '^config_local\.py$|^secrets\.ya?ml$|^credentials\.json$' \
      'local config / credential files'

# 8. Secret scanner — grep staged diff for known credential patterns.
#    These patterns are high-signal; false positives are rare.
SECRET_HITS=""
scan_diff_for() {
    local pattern="$1"
    local label="$2"
    local hits
    hits="$(git diff --cached -U0 | grep -E "^\+" | grep -E "$pattern" | head -20 || true)"
    if [ -n "$hits" ]; then
        SECRET_HITS+="  - ${label}:\n$(echo "$hits" | sed 's/^/      /')\n"
    fi
}
# AWS access key IDs (AKIA... / ASIA...)
scan_diff_for '\b(AKIA|ASIA)[0-9A-Z]{16}\b'              'AWS Access Key ID'
# AWS secret access key — 40 base64-ish chars preceded by aws_secret
scan_diff_for '(?i)aws_secret_access_key\s*[:=]\s*[A-Za-z0-9/+=]{40}' 'AWS Secret Access Key'
# Generic private keys
scan_diff_for '-----BEGIN (RSA|EC|DSA|OPENSSH|PRIVATE) KEY-----'      'private key material'
# Slack tokens
scan_diff_for '\bxox[abprs]-[0-9A-Za-z-]{10,}\b'         'Slack token'
# Stripe secrets
scan_diff_for '\bsk_live_[0-9a-zA-Z]{20,}\b'             'Stripe live key'
# GitHub token prefixes
scan_diff_for '\bghp_[A-Za-z0-9]{30,}\b'                 'GitHub personal access token'
scan_diff_for '\bgithub_pat_[A-Za-z0-9_]{60,}\b'         'GitHub fine-grained token'
# High-entropy env literals — only flag obvious FLASK_SECRET_KEY / API_KEY assignments
scan_diff_for '(FLASK_SECRET_KEY|_API_KEY|_PASSWORD)\s*=\s*[A-Za-z0-9]{32,}' \
              'literal secret assignment'

if [ -n "$BLOCKED_LINES" ] || [ -n "$SECRET_HITS" ]; then
    echo "pre-commit: BLOCKED — the following staged content violates the"
    echo "            dissertation project's data-protection policy"
    echo "            (see SECURITY.md + docs/PRODUCTION_READINESS_PLAN.md)."
    echo ""
    if [ -n "$BLOCKED_LINES" ]; then
        echo "File-path blocklist:"
        echo -e "$BLOCKED_LINES"
    fi
    if [ -n "$SECRET_HITS" ]; then
        echo "Secret scanner hits (inline credentials detected):"
        echo -e "$SECRET_HITS"
        echo "Rotate the leaked credential and remove it from the diff."
        echo "See docs/SECRETS_ROTATION.md for the rotation runbook."
    fi
    echo ""
    echo "If you genuinely need to commit one of the above, unstage it first:"
    echo "    git reset HEAD <path>"
    echo ""
    echo "For real patient data or secrets, this commit MUST be blocked."
    exit 1
fi

exit 0
