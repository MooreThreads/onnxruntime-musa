#!/usr/bin/env bash
# Prepare the matching ORT C++ SDK, install the built MUSA Plugin EP, and run
# the Python-independent C++ inference example. The SDK archive is downloaded only once.
#
# Usage:
#   bash scripts/run_cpp_inference_example.sh
#   bash scripts/run_cpp_inference_example.sh /absolute/path/model.onnx
#
# Without MODEL, the example registers the plugin and enumerates visible devices.
# With MODEL, it also creates a MUSA-only session; it does not run inference.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "${script_dir}")"
cd "${repo_root}"

if [[ $# -gt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,10p' "$0"
  exit $(( $# > 1 ? 2 : 0 ))
fi

for required_command in cmake tar wget; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "Error: ${required_command} is required but was not found on PATH." >&2
    exit 1
  fi
done

build_config="Release"
build_dir="${repo_root}/build/${build_config}"
example_build_dir="${repo_root}/build/cxx-probe"
package_version="$(tr -d '[:space:]' < "${repo_root}/VERSION_NUMBER")"
ort_version="$(tr -d '[:space:]' < "${repo_root}/third_party/onnxruntime/VERSION_NUMBER")"
ort_sdk_name="onnxruntime-linux-x64-${ort_version}"
ort_sdk_parent="${repo_root}/build/ort-sdk"
ort_sdk_root="${ort_sdk_parent}/${ort_sdk_name}"
ort_sdk_archive="${ort_sdk_parent}/${ort_sdk_name}.tgz"
ort_sdk_url="https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/${ort_sdk_name}.tgz"
ep_install_root="${repo_root}/build/install/onnxruntime-ep-musa/${package_version}"
plugin_path="${ep_install_root}/lib/libonnxruntime_providers_musa_plugin.so"
example_path="${example_build_dir}/ort_musa_ep_probe"
repo_ort_header="${repo_root}/third_party/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h"
sdk_ort_header="${ort_sdk_root}/include/onnxruntime_c_api.h"
jobs="$(nproc 2>/dev/null || echo 4)"

if [[ ! -f "${build_dir}/libonnxruntime_providers_musa_plugin.so" ]]; then
  echo "Error: the Release plugin has not been built: ${build_dir}" >&2
  echo "Run bash build.sh first, then rerun this script." >&2
  exit 1
fi

mkdir -p "${ort_sdk_parent}"
if [[ -f "${ort_sdk_root}/include/onnxruntime_cxx_api.h" && \
      -f "${ort_sdk_root}/lib/libonnxruntime.so" ]]; then
  echo "==> Reusing ORT C++ SDK ${ort_sdk_root}"
else
  if [[ -e "${ort_sdk_root}" ]]; then
    echo "Error: incomplete ORT SDK directory: ${ort_sdk_root}" >&2
    echo "Move it aside and rerun this script." >&2
    exit 1
  fi

  if [[ -f "${ort_sdk_archive}" ]]; then
    echo "==> Reusing downloaded SDK archive ${ort_sdk_archive}"
  else
    echo "==> Downloading ONNX Runtime C++ SDK ${ort_version} to ${ort_sdk_archive}"
    wget -O "${ort_sdk_archive}" "${ort_sdk_url}"
  fi

  echo "==> Extracting ${ort_sdk_archive}"
  tar -xzf "${ort_sdk_archive}" -C "${ort_sdk_parent}"
fi

if [[ ! -f "${ort_sdk_root}/include/onnxruntime_cxx_api.h" || \
      ! -f "${ort_sdk_root}/lib/libonnxruntime.so" || \
      ! -f "${sdk_ort_header}" ]]; then
  echo "Error: extracted SDK is missing the required headers or libonnxruntime.so." >&2
  exit 1
fi

repo_api_version="$(awk '/^#define ORT_API_VERSION / { print $3; exit }' "${repo_ort_header}")"
sdk_api_version="$(awk '/^#define ORT_API_VERSION / { print $3; exit }' "${sdk_ort_header}")"
if [[ -z "${repo_api_version}" || "${repo_api_version}" != "${sdk_api_version}" ]]; then
  echo "Error: ORT C API version mismatch (plugin=${repo_api_version:-unknown}, SDK=${sdk_api_version:-unknown})." >&2
  exit 1
fi

echo "==> Installing MUSA Plugin EP to ${ep_install_root}"
cmake --install "${build_dir}" --config "${build_config}" --prefix "${ep_install_root}"
if [[ ! -f "${plugin_path}" ]]; then
  echo "Error: installed plugin was not found: ${plugin_path}" >&2
  exit 1
fi

echo "==> Configuring the C++ example"
cmake -S "${repo_root}/examples/cpp" -B "${example_build_dir}" \
  -DORT_ROOT="${ort_sdk_root}"

echo "==> Building the C++ example with ${jobs} jobs"
cmake --build "${example_build_dir}" --config "${build_config}" -j "${jobs}"

echo "==> Running the C++ example"
probe_args=("${plugin_path}")
if [[ $# -eq 1 ]]; then
  probe_args+=("$1")
fi

runtime_library_path="${ort_sdk_root}/lib:${ep_install_root}/lib:/usr/local/musa/lib:/usr/local/musa/lib64"
LD_LIBRARY_PATH="${runtime_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${example_path}" "${probe_args[@]}"
