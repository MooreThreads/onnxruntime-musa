# MUSA Plugin EP C++ probe

This example is a Python-free C++ host test. It links an ONNX Runtime C++ SDK and
loads the MUSA Plugin EP at runtime. The application does not link the plugin library
directly.

The Python companion example lives in [`examples/python/`](../python/README.md).

## 1. Download the ORT C++ SDK

The repository plugin is built against ONNX Runtime v1.26.0 and
`ORT_API_VERSION=26`. On Linux x86_64, download the CPU C++ SDK asset from the
[ONNX Runtime v1.26.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0):

[onnxruntime-linux-x64-1.26.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz)

```bash
ORT_SDK_PARENT="$PWD/build/ort-sdk"
ORT_SDK_ARCHIVE="$ORT_SDK_PARENT/onnxruntime-linux-x64-1.26.0.tgz"
mkdir -p "$ORT_SDK_PARENT"
wget -O "$ORT_SDK_ARCHIVE" \
  https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz
tar -xzf "$ORT_SDK_ARCHIVE" -C "$ORT_SDK_PARENT"
export ORT_ROOT="$ORT_SDK_PARENT/onnxruntime-linux-x64-1.26.0"
```

This keeps downloaded binary dependencies under the ignored `build/` directory instead of
committing them to `third_party/`. The extracted release SDK is expected to look like:

```text
$ORT_ROOT/
  include/
    onnxruntime_c_api.h
    onnxruntime_cxx_api.h
  lib/
    libonnxruntime.so
    libonnxruntime.so.1
    libonnxruntime.so.1.26.0
```

Verify that the extracted SDK has both the C++ header and the linkable runtime:

```bash
test -f "$ORT_ROOT/include/onnxruntime_cxx_api.h"
test -f "$ORT_ROOT/lib/libonnxruntime.so"
grep '^#define ORT_API_VERSION ' "$ORT_ROOT/include/onnxruntime_c_api.h"
```

The API check must report version `26`.

## 2. Reuse the already-built MUSA plugin

Run these commands from the repository root:

```bash
test -f build/Release/libonnxruntime_providers_musa_plugin.so
export MUSA_PLUGIN_SO="$PWD/build/Release/libonnxruntime_providers_musa_plugin.so"
```

This reuses the plugin produced by the main `build/Release` build. It does not compile the MUSA
EP again and does not run `cmake --install`. Installation is only needed for a package-style
deployment directory; the ORT runtime still comes from `ORT_ROOT`.

## 3. Build this C++ probe

The preferred form lets CMake discover the standard ORT SDK layout:

```bash
cmake -S examples/cpp -B build/cxx-probe \
  -DORT_ROOT="$ORT_ROOT"
cmake --build build/cxx-probe -j
```

The probe links only `libonnxruntime.so`. It never links
`libonnxruntime_providers_musa_plugin.so`.

## 4. Reference ORT from your own CMake application

Use the downloaded SDK headers and runtime library as normal compile/link inputs:

```cmake
set(ORT_ROOT "" CACHE PATH "Extracted ONNX Runtime C++ SDK")
find_path(ORT_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS "${ORT_ROOT}"
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)
find_library(ORT_LIBRARY
  NAMES onnxruntime
  HINTS "${ORT_ROOT}"
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH
)
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR "ORT_ROOT must contain include/ and lib/ or lib64/")
endif()

add_executable(my_app main.cc)
target_include_directories(my_app PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(my_app PRIVATE "${ORT_LIBRARY}")

get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY}" DIRECTORY)
set_target_properties(my_app PROPERTIES
  BUILD_RPATH "${ORT_LIBRARY_DIR}"
  INSTALL_RPATH "${ORT_LIBRARY_DIR}"
)
```

Configure that application with:

```bash
cmake -S /path/to/app -B /path/to/app/build -DORT_ROOT="$ORT_ROOT"
cmake --build /path/to/app/build -j
```

The MUSA plugin is not a `target_link_libraries()` input. Pass its absolute path to
`Ort::Env::RegisterExecutionProviderLibrary()` when the application starts.

## 5. Run and register the plugin

The runtime linker must find ORT, the MUSA plugin directory, and the MUSA toolkit:

```bash
MUSA_APP_LIBRARY_PATH="$ORT_ROOT/lib:/usr/local/musa/lib:/usr/local/musa/lib64"
export LD_LIBRARY_PATH="$MUSA_APP_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
build/cxx-probe/ort_musa_ep_probe "$MUSA_PLUGIN_SO"
```

With an ONNX model, the probe also creates a Session with CPU fallback disabled:

```bash
build/cxx-probe/ort_musa_ep_probe "$MUSA_PLUGIN_SO" "$PWD/examples/cpp/single_gemm_opset19.onnx"
```

The current probe validates plugin registration, visible MUSA devices, and model/session
creation. It does not call `Session::Run()` because a generic ONNX model does not provide input
values. Use a model-specific example when you need end-to-end execution with inputs and output
comparison.

Internally the application performs:

```cpp
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ort-musa-cxx-probe"};
env.RegisterExecutionProviderLibrary("MUSAExecutionProvider", plugin_path);
```

If the process reports `CreatePlatform failed!` or no visible MUSA devices, the host
runtime/plugin link path has been reached but the current environment has no usable MUSA
device. Run the final device test in an environment where the MUSA driver is visible.
