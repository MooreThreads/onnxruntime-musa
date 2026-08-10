cmake_minimum_required(VERSION 3.20)
project(ort_musa_ep_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_ROOT "" CACHE PATH "Root of an ONNX Runtime C++ SDK")

if(NOT ORT_INCLUDE_DIR AND ORT_ROOT)
  find_path(ORT_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ORT_ROOT}"
    PATH_SUFFIXES include include/onnxruntime/core/session
    NO_DEFAULT_PATH
    DOC "Directory containing onnxruntime_cxx_api.h"
  )
endif()
if(NOT ORT_LIBRARY AND ORT_ROOT)
  find_library(ORT_LIBRARY
    NAMES onnxruntime
    HINTS "${ORT_ROOT}"
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
    DOC "Path to libonnxruntime.so or platform equivalent"
  )
endif()

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR
    "Set ORT_ROOT to an ONNX Runtime C++ SDK, or pass both "
    "ORT_INCLUDE_DIR and ORT_LIBRARY explicitly"
  )
endif()
if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ORT_INCLUDE_DIR does not contain onnxruntime_cxx_api.h: ${ORT_INCLUDE_DIR}"
  )
endif()
if(NOT EXISTS "${ORT_LIBRARY}")
  message(FATAL_ERROR "ORT_LIBRARY does not exist: ${ORT_LIBRARY}")
endif()

add_executable(ort_musa_ep_probe ort_musa_ep_probe.cc)
target_include_directories(ort_musa_ep_probe PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(ort_musa_ep_probe PRIVATE "${ORT_LIBRARY}")

get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY}" DIRECTORY)
set_target_properties(ort_musa_ep_probe PROPERTIES
  BUILD_RPATH "${ORT_LIBRARY_DIR}"
)
