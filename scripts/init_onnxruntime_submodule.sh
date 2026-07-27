#!/usr/bin/env bash
set -euo pipefail

readonly ORT_SUBMODULE_PATH="third_party/onnxruntime"

repo_root="$(git rev-parse --show-toplevel)"
submodule_update_args=(submodule update --init --recommend-shallow)

# Newer Git versions can combine a partial clone with submodule initialization.
# Keep the fallback compatible with Git 2.34, where shallow=true in .gitmodules
# still prevents the complete ONNX Runtime history from being downloaded.
submodule_help="$(git -C "${repo_root}" submodule update -h 2>&1 || true)"
if [[ "${submodule_help}" == *"--filter"* ]]; then
  submodule_update_args+=(--filter=blob:none)
fi

git -C "${repo_root}" "${submodule_update_args[@]}" -- "${ORT_SUBMODULE_PATH}"
git -C "${repo_root}/${ORT_SUBMODULE_PATH}" sparse-checkout init --cone
git -C "${repo_root}/${ORT_SUBMODULE_PATH}" sparse-checkout set include/onnxruntime

ort_version="$(<"${repo_root}/${ORT_SUBMODULE_PATH}/VERSION_NUMBER")"
ort_commit="$(git -C "${repo_root}/${ORT_SUBMODULE_PATH}" rev-parse --short HEAD)"
printf 'ONNX Runtime %s (%s) initialized at %s\n' \
  "${ort_version}" "${ort_commit}" "${ORT_SUBMODULE_PATH}"
