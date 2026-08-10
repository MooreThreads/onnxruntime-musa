#!/usr/bin/env bash
# Incremental build of the ONNX Runtime MUSA Plugin EP and provider-only wheel.
# The final summary also points to the Python-independent C++ inference example.
#
# Usage:
#   ./build.sh                                  # incremental build .so + build wheel (Release)
#   ./build.sh --clean                          # remove build output, then rebuild .so + wheel
#   ./build.sh --no-wheel                       # incrementally build only the plugin .so
#   ./build.sh --config Debug                   # use Debug config
#   ./build.sh --package-name onnxruntime-musa  # build the legacy distribution name
#   ./build.sh -- -DMUSA_HOME=/opt/musa         # extra args after `--` go to CMake
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="Release"
PACKAGE_NAME="onnxruntime-ep-musa"
BUILD_WHEEL=1
CLEAN_BUILD=0
CMAKE_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --package-name) PACKAGE_NAME="$2"; shift 2 ;;
    --clean) CLEAN_BUILD=1; shift ;;
    --no-wheel) BUILD_WHEEL=0; shift ;;
    --) shift; CMAKE_EXTRA_ARGS+=("$@"); break ;;
    -h|--help)
      sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

BUILD_DIR="build/${CONFIG}"
DIST_DIR="dist"
JOBS="$(nproc 2>/dev/null || echo 4)"
VERSION="$(cat VERSION_NUMBER)"
ORT_HOST_VERSION="$(tr -d '[:space:]' < third_party/onnxruntime/VERSION_NUMBER)"
SO_PATH="${BUILD_DIR}/libonnxruntime_providers_musa_plugin.so"
PACKAGE_WHEEL_PREFIX="${PACKAGE_NAME//-/_}"
PACKAGE_WHEEL_PREFIX="${PACKAGE_WHEEL_PREFIX//./_}"

# Pick a Python that satisfies the wheel's requires-python (>=3.11):
# prefer the in-repo .venv, then python3.12 / python3.11, then $PYTHON, then python3.
pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then echo "${PYTHON}"; return; fi
  if [[ -x ".venv/bin/python" ]]; then echo "$(pwd)/.venv/bin/python"; return; fi
  for c in python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then echo "$c"; return; fi
  done
  echo "python3"
}
PYTHON_BIN="$(pick_python)"

if [[ "${CLEAN_BUILD}" -eq 1 ]]; then
  echo "==> Cleaning ${BUILD_DIR} and ${DIST_DIR}"
  rm -rf "${BUILD_DIR}" "${DIST_DIR}"
else
  echo "==> Reusing ${BUILD_DIR} for an incremental build"
fi

echo "==> Configuring (${CONFIG})"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${CONFIG}" "${CMAKE_EXTRA_ARGS[@]}"

echo "==> Building plugin with ${JOBS} jobs"
cmake --build "${BUILD_DIR}" --config "${CONFIG}" -j "${JOBS}"

echo "==> Built ${SO_PATH}"
ls -lh "${SO_PATH}"

if [[ "${BUILD_WHEEL}" -eq 1 ]]; then
  echo "==> Building wheel ${PACKAGE_NAME}==${VERSION} (python: ${PYTHON_BIN})"
  "${PYTHON_BIN}" musa/ep/python/build_wheel.py \
    --binary_dir "${BUILD_DIR}" \
    --version "${VERSION}" \
    --package_name "${PACKAGE_NAME}" \
    --output_dir "${DIST_DIR}"

  echo "==> Built wheel(s) for ${PACKAGE_NAME}:"
  ls -lh "${DIST_DIR}/${PACKAGE_WHEEL_PREFIX}"-*.whl

  echo ""
  echo "How to install in a Python environment:"
  echo "=========================================="
  echo "Provider-only wheel (install one ORT host separately):"
  for whl in "${DIST_DIR}/${PACKAGE_WHEEL_PREFIX}"-*.whl; do
    echo "  pip install onnxruntime==${ORT_HOST_VERSION} ${whl} --no-deps"
  done
  echo "=========================================="
fi

echo ""
echo "How to use in a C++ environment:"
echo "=========================================="
echo "C++ example (Python-independent):"
echo "  bash scripts/run_cpp_inference_example.sh"
echo ""
echo "Run with a custom ONNX model:"
echo "  bash scripts/run_cpp_inference_example.sh examples/cpp/single_gemm_opset19.onnx"
echo "=========================================="
