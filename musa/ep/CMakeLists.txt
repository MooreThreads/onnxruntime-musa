set(MUSA_PLUGIN_TARGET onnxruntime_providers_musa_plugin)

set(MUSA_PLUGIN_SOURCES
  src/ep_lib_entry.cc
  src/ep_factory.cc
  src/ep.cc
  src/ep_kernel_registration.cc
  src/ep_data_transfer.cc
  src/ep_profiling.cc
  src/kernels/basic_ops.cc
  src/kernels/matmul.cc
)

add_library(${MUSA_PLUGIN_TARGET} SHARED ${MUSA_PLUGIN_SOURCES})

target_include_directories(${MUSA_PLUGIN_TARGET}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${ORT_ROOT}/include/onnxruntime
    ${ORT_ROOT}/include/onnxruntime/core/session
    ${ORT_BUILD_ROOT}/_deps/gsl-src/include
    ${MUSA_HOME}/include
)

target_link_directories(${MUSA_PLUGIN_TARGET} PRIVATE ${MUSA_HOME}/lib)
target_link_libraries(${MUSA_PLUGIN_TARGET} PRIVATE musart mublas)

set_target_properties(${MUSA_PLUGIN_TARGET}
  PROPERTIES
    OUTPUT_NAME "onnxruntime_providers_musa_plugin"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

if(UNIX AND NOT APPLE)
  target_link_options(${MUSA_PLUGIN_TARGET} PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/ep_lib.lds"
  )
endif()
