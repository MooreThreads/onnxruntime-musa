#!/usr/bin/env bash
# Run the standard test suites under test/.
# Usage: bash test/run_all.sh [extra pytest args]
#
# The script intentionally runs the stable suites explicitly instead of
# discovering every sub-directory under test/. This keeps temporary experiments
# out of the default run while ensuring both op and fusion coverage are included.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITES=(ops fusion)

TEST_DIRS=()
for suite in "${SUITES[@]}"; do
    dir="$SCRIPT_DIR/$suite"
    if compgen -G "$dir/test_*.py" >/dev/null; then
        TEST_DIRS+=("$dir")
    fi
done

if [[ ${#TEST_DIRS[@]} -eq 0 ]]; then
    echo "No standard test suites with test_*.py found under $SCRIPT_DIR" >&2
    exit 1
fi

echo "=== Running test suites: ${TEST_DIRS[*]} ==="
python -m pytest "${TEST_DIRS[@]}" "$@"
