cmake_minimum_required(VERSION 3.20)
project(onnxruntime_musa_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ONNXRUNTIME_MUSA_ENABLE_SAMPLE_KERNELS "Build CPU-backed sample kernels for plugin smoke tests" ON)

set(ORT_ROOT "/home/workspace/onnxruntime" CACHE PATH "Path to an ONNX Runtime source checkout with public headers")
set(ORT_BUILD_ROOT "/home/workspace/onnxruntime/build_musa_wheel_py311/Release" CACHE PATH "Path to an ONNX Runtime build tree for third-party headers")
set(MUSA_HOME "/usr/local/musa" CACHE PATH "Path to MUSA toolkit")

add_subdirectory(musa/ep)
