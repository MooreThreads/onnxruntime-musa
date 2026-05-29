#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python musa/build_and_test.py --build-wheel "$@"
