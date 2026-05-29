cmake_minimum_required(VERSION 3.20)

project(onnxruntime_musa_plugin_ep LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ORT_MUSA_FETCH_ORT_HEADERS "Download ONNX Runtime C API headers if ONNXRUNTIME_ROOT is not set" ON)
set(ORT_MUSA_ORT_REF "main" CACHE STRING "ONNX Runtime git ref used when downloading headers")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime install or build tree")
set(MUSA_PATH "/usr/local/musa-4.3.5" CACHE PATH "Path to MUSA installation")

function(download_ort_header relative_path)
  set(dst "${CMAKE_BINARY_DIR}/onnxruntime_headers/include/onnxruntime/core/session/${relative_path}")
  if(NOT EXISTS "${dst}")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/onnxruntime_headers/include/onnxruntime/core/session")
    file(DOWNLOAD
      "https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_MUSA_ORT_REF}/include/onnxruntime/core/session/${relative_path}"
      "${dst}"
      STATUS download_status
      TLS_VERIFY ON
      TIMEOUT 60)
    list(GET download_status 0 download_code)
    list(GET download_status 1 download_message)
    if(NOT download_code EQUAL 0)
      message(FATAL_ERROR "Failed to download ${relative_path}: ${download_message}")
    endif()
  endif()
endfunction()

set(ORT_INCLUDE_CANDIDATES)
if(ONNXRUNTIME_ROOT)
  list(APPEND ORT_INCLUDE_CANDIDATES
    "${ONNXRUNTIME_ROOT}/include"
    "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
endif()
list(APPEND ORT_INCLUDE_CANDIDATES
  "/usr/local/include"
  "/usr/include")

set(ORT_INCLUDE_DIR "")
foreach(candidate ${ORT_INCLUDE_CANDIDATES})
  if(EXISTS "${candidate}/onnxruntime_cxx_api.h" OR
     EXISTS "${candidate}/onnxruntime/core/session/onnxruntime_cxx_api.h")
    set(ORT_INCLUDE_DIR "${candidate}")
    break()
  endif()
endforeach()

if(NOT ORT_INCLUDE_DIR)
  if(NOT ORT_MUSA_FETCH_ORT_HEADERS)
    message(FATAL_ERROR "ONNX Runtime headers were not found. Set ONNXRUNTIME_ROOT or enable ORT_MUSA_FETCH_ORT_HEADERS.")
  endif()
  download_ort_header("onnxruntime_c_api.h")
  download_ort_header("onnxruntime_cxx_api.h")
  download_ort_header("onnxruntime_cxx_inline.h")
  download_ort_header("onnxruntime_ep_c_api.h")
  download_ort_header("onnxruntime_float16.h")
  set(ORT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/onnxruntime_headers/include/onnxruntime/core/session")
endif()

find_path(MUSA_INCLUDE_DIR musa_runtime.h
  HINTS "${MUSA_PATH}/include" "/usr/local/musa/include" "/usr/local/musa-4.3.5/include"
  NO_DEFAULT_PATH)
find_library(MUSA_RUNTIME_LIBRARY NAMES musart musa
  HINTS "${MUSA_PATH}/lib" "${MUSA_PATH}/lib64" "/usr/local/musa/lib" "/usr/local/musa/lib64" "/usr/local/musa-4.3.5/lib" "/usr/local/musa-4.3.5/lib64"
  NO_DEFAULT_PATH)

if(MUSA_INCLUDE_DIR AND NOT MUSA_RUNTIME_LIBRARY)
  file(GLOB MUSA_RUNTIME_CANDIDATES
    "${MUSA_PATH}/lib/libmusart.so*"
    "${MUSA_PATH}/lib64/libmusart.so*"
    "/usr/local/musa/lib/libmusart.so*"
    "/usr/local/musa/lib64/libmusart.so*"
    "/usr/local/musa-4.3.5/lib/libmusart.so*"
    "/usr/local/musa-4.3.5/lib64/libmusart.so*")
  list(SORT MUSA_RUNTIME_CANDIDATES)
  list(GET MUSA_RUNTIME_CANDIDATES 0 MUSA_RUNTIME_LIBRARY)
endif()

if(MUSA_INCLUDE_DIR AND MUSA_RUNTIME_LIBRARY)
  set(ORT_MUSA_HAS_MUSA 1)
else()
  set(ORT_MUSA_HAS_MUSA 0)
  message(WARNING "MUSA runtime was not found. Building a loadable host-fallback plugin EP only.")
endif()

add_library(onnxruntime_musa_ep SHARED
  src/ep.cc
  src/ep_factory.cc
  src/ep_kernel_registration.cc
  src/ep_lib_entry.cc
  src/kernels/add.cc
  src/kernels/matmul.cc
  src/musa/musa_backend.cc)

target_include_directories(onnxruntime_musa_ep PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${ORT_INCLUDE_DIR}")

if(ORT_MUSA_HAS_MUSA)
  target_include_directories(onnxruntime_musa_ep PRIVATE "${MUSA_INCLUDE_DIR}")
  target_link_libraries(onnxruntime_musa_ep PRIVATE "${MUSA_RUNTIME_LIBRARY}")
  target_compile_definitions(onnxruntime_musa_ep PRIVATE ORT_MUSA_HAS_MUSA=1)
else()
  target_compile_definitions(onnxruntime_musa_ep PRIVATE ORT_MUSA_HAS_MUSA=0)
endif()

set_target_properties(onnxruntime_musa_ep PROPERTIES
  OUTPUT_NAME "onnxruntime_musa_ep"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  BUILD_RPATH "${MUSA_PATH}/lib;${MUSA_PATH}/lib64")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(onnxruntime_musa_ep PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS onnxruntime_musa_ep LIBRARY DESTINATION lib)
