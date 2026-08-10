cmake_minimum_required(VERSION 3.20)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION_NUMBER" ONNXRUNTIME_MUSA_VERSION LIMIT_COUNT 1)
string(STRIP "${ONNXRUNTIME_MUSA_VERSION}" ONNXRUNTIME_MUSA_VERSION)

project(onnxruntime_musa_plugin VERSION ${ONNXRUNTIME_MUSA_VERSION} LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MUSA_HOME "/usr/local/musa" CACHE PATH "Path to MUSA toolkit")

# ONNX Runtime is pinned as a submodule under third_party/onnxruntime (tag v1.26.0).
# Only include/onnxruntime is materialized by the repository's sparse-checkout helper.
set(ONNXRUNTIME_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include/onnxruntime"
    CACHE PATH "ONNX Runtime public include directory")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/core/session/onnxruntime_c_api.h")
  message(FATAL_ERROR
    "ONNX Runtime submodule headers not found at ${ONNXRUNTIME_INCLUDE_DIR}. "
    "Run ./scripts/init_onnxruntime_submodule.sh first.")
endif()

file(STRINGS
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/VERSION_NUMBER"
  ONNXRUNTIME_VERSION
  LIMIT_COUNT 1
)
string(STRIP "${ONNXRUNTIME_VERSION}" ONNXRUNTIME_VERSION)

file(STRINGS
  "${ONNXRUNTIME_INCLUDE_DIR}/core/session/onnxruntime_c_api.h"
  _ONNXRUNTIME_API_VERSION_DEFINE
  REGEX "^#define ORT_API_VERSION [0-9]+$"
  LIMIT_COUNT 1
)
if(NOT _ONNXRUNTIME_API_VERSION_DEFINE)
  message(FATAL_ERROR "Could not determine ORT_API_VERSION from ONNX Runtime headers")
endif()
string(REGEX REPLACE
  "^#define ORT_API_VERSION ([0-9]+)$" "\\1"
  ONNXRUNTIME_API_VERSION
  "${_ONNXRUNTIME_API_VERSION_DEFINE}"
)

add_subdirectory(musa/ep)
