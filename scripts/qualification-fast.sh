#!/usr/bin/env bash
# LGAE qualification-fast: unit + integration, parallel, no crash/recovery.
#
# For development iteration only. NOT a release qualification.
# Excludes meta-tests (self-referential) and slow tests (crash recovery,
# cross-process determinism, subprocess isolation).
#
# Usage: ./scripts/qualification-fast.sh

set -e
cd "$(dirname "$0")/.."

echo "=== LGAE qualification-fast (development) ==="
echo "Excluding: meta, crash_recovery (crash/recovery, cross-process)"
echo ""

python -m pytest -q -n auto -m "not meta and not crash_recovery" "$@"

echo ""
echo "=== Fast qualification complete ==="
echo "NOTE: This is NOT a release qualification."
echo "Run scripts/qualification-release.sh for full release qualification."
