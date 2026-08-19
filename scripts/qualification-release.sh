#!/usr/bin/env bash
# LGAE qualification-release: full suite, parallel, crash/recovery included.
#
# This is the authoritative release qualification path.
# Excludes only meta-tests (self-referential tests that re-run other tests).
# Includes all crash/recovery, cross-process, and subprocess isolation tests.
#
# Generates:
#   - release_verification.json
#   - qualification_summary.json
#   - BUILD_REPORT.md (updated test summary)
#   - MANIFEST.sha256.json
#
# Usage: ./scripts/qualification-release.sh

set -e
cd "$(dirname "$0")/.."

echo "=== LGAE qualification-release (full) ==="
echo "Including: all substantive tests (crash/recovery, cross-process)"
echo "Excluding: meta (self-referential)"
echo ""

# Run qualification metadata generation (includes full pytest).
python scripts/generate_release_metadata.py

# Generate manifest.
python scripts/generate_manifest.py
python scripts/generate_manifest.py --check

echo ""
echo "=== Release qualification complete ==="
echo "Artifacts:"
echo "  - release_verification.json"
echo "  - qualification_summary.json"
echo "  - BUILD_REPORT.md"
echo "  - MANIFEST.sha256.json"
