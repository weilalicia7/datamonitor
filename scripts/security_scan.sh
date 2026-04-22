#!/usr/bin/env bash
#
# One-shot security scan wrapper — runs Bandit (HIGH) + pip-audit
# (strict) and exits non-zero if either finds a blocker.  Intended for
# local pre-push use; CI runs the same tools independently.
#
# Does NOT run Semgrep or OWASP ZAP; those have heavier install /
# runtime requirements.  See docs/SECURITY_TEST.md for the full recipe.

set -uo pipefail

FAIL=0

echo "=== bandit (HIGH-severity only) ==="
if command -v bandit >/dev/null 2>&1; then
    python -m bandit -r . --severity-level high \
        -x tests,data_cache,.venv,node_modules,docs || FAIL=1
else
    echo "bandit not installed — pip install --user bandit"
    FAIL=1
fi

echo ""
echo "=== pip-audit (strict) ==="
if command -v pip-audit >/dev/null 2>&1; then
    pip-audit --strict -r requirements.txt || FAIL=1
else
    echo "pip-audit not installed — pip install --user pip-audit"
    FAIL=1
fi

echo ""
if [ "$FAIL" -ne 0 ]; then
    echo "SECURITY SCAN: FAIL (see output above)"
    exit 1
fi
echo "SECURITY SCAN: PASS"
exit 0
