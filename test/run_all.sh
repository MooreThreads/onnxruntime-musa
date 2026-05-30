#!/usr/bin/env bash
# Run all test suites under test/.
# Usage: bash test/run_all.sh [extra pytest args]
#
# The script discovers every sub-directory that contains at least one test_*.py
# file and passes them all to a single pytest invocation, so results are
# collected in one report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Collect sub-directories that contain test files.
TEST_DIRS=()
while IFS= read -r -d '' dir; do
    TEST_DIRS+=("$dir")
done < <(find "$SCRIPT_DIR" -mindepth 1 -maxdepth 1 -type d \
    -exec sh -c 'ls "$1"/test_*.py >/dev/null 2>&1' _ {} \; -print0 | sort -z)

if [[ ${#TEST_DIRS[@]} -eq 0 ]]; then
    echo "No test directories found under $SCRIPT_DIR" >&2
    exit 1
fi

echo "=== Running tests in: ${TEST_DIRS[*]} ==="
python -m pytest "${TEST_DIRS[@]}" "$@"
