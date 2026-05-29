#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${script_dir}/hooks" ]]; then
  echo "Error: hooks directory not found. Run this script from the repo root."
  exit 1
fi

"${script_dir}/hooks/install-hooks.sh"

echo "Hook setup complete."
echo "Run scripts/format.sh to format C/C++ sources manually."
