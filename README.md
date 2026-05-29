# ONNX Runtime MUSA Plugin EP

This repository contains a minimal ONNX Runtime plugin Execution Provider named `MusaExecutionProvider`.

The first implementation focuses on the standard plugin EP shape: a shared library exporting `CreateEpFactories` and `ReleaseEpFactory`, an `OrtEpFactory`, an `OrtEp`, and a kernel registry with a small float operator set. It registers host-memory kernels for `Add` and `MatMul` so the library can be loaded by ONNX Runtime and used in an inference session before a full MUSA allocator/data-transfer path is added.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMUSA_PATH=/usr/local/musa-4.3.5
cmake --build build -j
```

If ONNX Runtime development headers are installed, pass `-DONNXRUNTIME_ROOT=/path/to/onnxruntime`. Otherwise CMake downloads the required C API headers from the ONNX Runtime git ref in `ORT_MUSA_ORT_REF`.

The output library is:

```text
build/libonnxruntime_musa_ep.so
```

## Formatting

This repo uses the same clang-format/pre-commit based formatting flow as the TensorFlow MUSA extension.

Format C/C++ sources manually with:

```bash
./scripts/format.sh
```

Install local Git hooks with:

```bash
./install-hooks.sh
```

If `pre-commit` is installed, the hook uses `.pre-commit-config.yaml`; otherwise it falls back to formatting staged C/C++ files with `clang-format` directly.

## Use From ONNX Runtime

With ONNX Runtime 1.23+ plugin EP APIs, applications should:

1. Call `RegisterExecutionProviderLibrary` with `libonnxruntime_musa_ep.so`.
2. Query `GetEpDevices` and select `MusaExecutionProvider`.
3. Call `SessionOptionsAppendExecutionProvider_V2`.
4. Create the session.

For ONNX Runtime provider tests, set:

```json
{
  "ep_library_registration_name": "onnxruntime_musa_ep",
  "ep_library_path": "/absolute/path/to/libonnxruntime_musa_ep.so",
  "selected_ep_name": "MusaExecutionProvider"
}
```
