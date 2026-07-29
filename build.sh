#!/usr/bin/env bash
# Incremental build of the onnxruntime-musa plugin and Python wheel.
#
# Usage:
#   ./build.sh                                  # incremental build .so + build wheel (Release)
#   ./build.sh --clean                          # remove build output, then rebuild .so + wheel
#   ./build.sh --no-wheel                       # incrementally build only the plugin .so
#   ./build.sh --config Debug                   # use Debug config
#   ./build.sh --package-name onnxruntime-musa  # override wheel distribution name
#   ./build.sh -- -DMUSA_HOME=/opt/musa         # extra args after `--` go to CMake
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="Release"
PACKAGE_NAME="onnxruntime-musa"
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
SO_PATH="${BUILD_DIR}/libonnxruntime_providers_musa_plugin.so"

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

  echo "==> Wheel(s) under ${DIST_DIR}:"
  ls -lh "${DIST_DIR}"/*.whl

  echo ""
  echo "=========================================="
  echo "Install with:"
  for whl in "${DIST_DIR}"/*.whl; do
    echo "  pip install ${whl} --no-deps"
  done
  echo "=========================================="
fi
