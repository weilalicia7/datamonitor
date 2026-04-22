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

if [ -n "$BLOCKED_LINES" ]; then
    echo "pre-commit: BLOCKED — the following staged paths violate the"
    echo "            dissertation project's data-protection policy"
    echo "            (see SECURITY.md + docs/PRODUCTION_READINESS_PLAN.md)."
    echo ""
    echo -e "$BLOCKED_LINES"
    echo ""
    echo "If you genuinely need to commit one of the above, unstage it first:"
    echo "    git reset HEAD <path>"
    echo ""
    echo "For real patient data specifically, this commit MUST be blocked."
    exit 1
fi

exit 0
