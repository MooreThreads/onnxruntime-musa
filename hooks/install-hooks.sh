#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "${script_dir}")"
git_hooks_dir="${repo_root}/.git/hooks"

if [[ ! -d "${repo_root}/.git" ]]; then
  echo "Error: ${repo_root} is not a Git repository."
  exit 1
fi

mkdir -p "${git_hooks_dir}"

for hook in pre-commit commit-msg; do
  if [[ -f "${script_dir}/${hook}" ]]; then
    cp "${script_dir}/${hook}" "${git_hooks_dir}/${hook}"
    chmod +x "${git_hooks_dir}/${hook}"
    echo "Installed ${hook} hook."
  fi
done

if command -v pre-commit >/dev/null 2>&1; then
  (cd "${repo_root}" && pre-commit install --hook-type pre-commit --hook-type commit-msg)
  echo "Installed pre-commit framework hooks."
else
  echo "pre-commit is not installed; installed the lightweight pre-commit hook only."
  echo "Install pre-commit to enable .pre-commit-config.yaml: pip install pre-commit"
fi
