#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "clang-format is required but was not found on PATH."
  exit 1
fi

mapfile -t files < <(find src -type f \( \
  -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o \
  -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \
\) | sort)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No C/C++ files found."
  exit 0
fi

clang-format -style=file -i "${files[@]}"
echo "Formatted ${#files[@]} files."
