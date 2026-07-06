set(MUSA_PLUGIN_TARGET onnxruntime_providers_musa_plugin)

# Core EP sources are listed explicitly; kernel implementations under
# src/kernels/**/*.cc and device implementations under src/kernels/**/*.mu are
# auto-discovered so adding a new operator follows the CUDA EP-style directory
# layout without touching this file.
set(MUSA_PLUGIN_SOURCES
  src/ep_lib_entry.cc
  src/ep_factory.cc
  src/ep.cc
  src/ep_kernel_registration.cc
  src/ep_data_transfer.cc
  src/ep_profiling.cc
  src/ep_stream.cc
  src/graph_mermaid_dump.cc
)

file(GLOB_RECURSE MUSA_PLUGIN_KERNEL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cc")
file(GLOB_RECURSE MUSA_PLUGIN_KERNEL_MU_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.mu")
list(APPEND MUSA_PLUGIN_SOURCES
  ${MUSA_PLUGIN_KERNEL_SOURCES}
  ${MUSA_PLUGIN_KERNEL_MU_SOURCES}
)

file(GLOB MUSA_PLUGIN_FUSION_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/fusion/*.cc")
list(APPEND MUSA_PLUGIN_SOURCES ${MUSA_PLUGIN_FUSION_SOURCES})

list(APPEND CMAKE_MODULE_PATH "${MUSA_HOME}/cmake")
set(MUSA_TOOLKIT_ROOT_DIR "${MUSA_HOME}")
find_package(MUSA REQUIRED)
list(APPEND MUSA_MCC_FLAGS -std=c++17)

set_source_files_properties(
  ${MUSA_PLUGIN_KERNEL_MU_SOURCES}
  PROPERTIES
    MUSA_SOURCE_PROPERTY_FORMAT OBJ
)

musa_add_library(${MUSA_PLUGIN_TARGET} SHARED ${MUSA_PLUGIN_SOURCES})

target_include_directories(${MUSA_PLUGIN_TARGET}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}/core/session
    ${MUSA_HOME}/include
    ${MUSA_INCLUDE_DIRS}
)

target_link_directories(${MUSA_PLUGIN_TARGET} PRIVATE ${MUSA_HOME}/lib)
target_link_libraries(${MUSA_PLUGIN_TARGET} musart mublas mudnncxx mudnn)

target_compile_definitions(${MUSA_PLUGIN_TARGET} PRIVATE
  ONNXRUNTIME_MUSA_VERSION="${ONNXRUNTIME_MUSA_VERSION}"
)

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

# Regenerate musa/docs/supported_ops.md after every build by scanning the
# kernel registration macros. Always runs (the generator itself is cheap and
# overwrites the file in place).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  set(_GEN_SUPPORTED_OPS "${CMAKE_SOURCE_DIR}/scripts/gen_supported_ops.py")
  set(_GEN_FUSION_DOCS "${CMAKE_SOURCE_DIR}/scripts/gen_fusion_docs.py")
  add_custom_command(TARGET ${MUSA_PLUGIN_TARGET} POST_BUILD
    COMMAND ${Python3_EXECUTABLE} ${_GEN_SUPPORTED_OPS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Regenerating musa/docs/supported_ops.md"
    VERBATIM
  )
  add_custom_command(TARGET ${MUSA_PLUGIN_TARGET} POST_BUILD
    COMMAND ${Python3_EXECUTABLE} ${_GEN_FUSION_DOCS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Regenerating musa/docs/fusion/*.md"
    VERBATIM
  )
endif()
