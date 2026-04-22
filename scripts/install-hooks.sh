#!/usr/bin/env bash
#
# Install the project's git hooks into this working copy's .git/hooks dir.
# Idempotent — safe to rerun.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="${ROOT}/scripts"
HOOKS_DST="${ROOT}/.git/hooks"

install_hook() {
    local name="$1"
    local src="${HOOKS_SRC}/${name}.sh"
    local dst="${HOOKS_DST}/${name}"

    if [ ! -f "$src" ]; then
        echo "install-hooks: ${src} not found, skipping" >&2
        return 0
    fi

    cp "$src" "$dst"
    chmod +x "$dst"
    echo "install-hooks: ${name} installed at ${dst}"
}

install_hook pre-commit

echo ""
echo "install-hooks: done.  Test with:"
echo "    echo 'secret' > .env && git add -f .env && git commit -m 'test' || echo 'hook blocked (expected)'"
