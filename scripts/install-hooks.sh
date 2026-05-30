#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "${script_dir}")"

if [[ ! -d "${repo_root}/hooks" ]]; then
  echo "Error: hooks directory not found at ${repo_root}/hooks."
  exit 1
fi

"${repo_root}/hooks/install-hooks.sh"

echo "Hook setup complete."
echo "Run scripts/format.sh to format C/C++ sources manually."
