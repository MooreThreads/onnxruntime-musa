cmake_minimum_required(VERSION 3.20)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION_NUMBER" ONNXRUNTIME_MUSA_VERSION LIMIT_COUNT 1)
string(STRIP "${ONNXRUNTIME_MUSA_VERSION}" ONNXRUNTIME_MUSA_VERSION)

project(onnxruntime_musa_plugin VERSION ${ONNXRUNTIME_MUSA_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ONNXRUNTIME_MUSA_ENABLE_SAMPLE_KERNELS "Build CPU-backed sample kernels for plugin smoke tests" ON)

set(MUSA_HOME "/usr/local/musa" CACHE PATH "Path to MUSA toolkit")

# ONNX Runtime public headers are vendored under third_party/onnxruntime (tag v1.26.0).
# See third_party/onnxruntime/VERSION and third_party/onnxruntime/LICENSE.
set(ONNXRUNTIME_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include/onnxruntime"
    CACHE PATH "ONNX Runtime public include directory")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/core/session/onnxruntime_c_api.h")
  message(FATAL_ERROR
    "Vendored ONNX Runtime headers not found at ${ONNXRUNTIME_INCLUDE_DIR}.")
endif()

add_subdirectory(musa/ep)
