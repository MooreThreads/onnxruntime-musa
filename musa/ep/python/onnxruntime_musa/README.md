# onnxruntime_musa

Python helper package shipped inside the `onnxruntime-ep-musa` provider wheel. It bundles the plugin
shared library (`libonnxruntime_providers_musa_plugin.so`) and exposes two helpers used
to register the MUSA Plugin Execution Provider into a compatible ONNX Runtime host.

The provider wheel intentionally does not depend on `onnxruntime` or `onnxruntime-gpu`.
Install exactly one compatible ORT host flavor before installing this wheel.

## Usage

```python
import onnxruntime as ort
import onnxruntime_musa as musa_ep

ep_name = musa_ep.get_ep_name()         # "MUSAExecutionProvider"
lib_path = musa_ep.get_library_path()   # absolute path to the bundled .so / .dll
manifest = musa_ep.get_manifest()       # ORT API/build compatibility metadata

ort.register_execution_provider_library(ep_name, lib_path)

# pick the MUSA device and create a session bound to it
musa_device = next(d for d in ort.get_ep_devices() if d.ep_name == ep_name)
so = ort.SessionOptions()
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
so.add_provider_for_devices([musa_device], {})
session = ort.InferenceSession("model.onnx", sess_options=so)
```

External MUSA compute streams are supported through provider options. The stream is
borrowed: the caller owns its lifetime and must not destroy it while the session can
run on it.

```python
import ctypes

stream = ctypes.c_void_p(...)  # musaStream_t created by the caller
so.add_provider_for_devices(
    [musa_device],
    musa_ep.make_provider_options(user_compute_stream=stream),
)
```

A runnable end-to-end MatMul example lives in the repository as
[`test/ops/test_matmul.py`](../../../../test/ops/test_matmul.py). It uses
`test/ops/op_test_utils.py` to register the plugin EP, disable CPU fallback for the
MUSA run, and compare the MUSA output against the CPU reference.

## Requirements

- Python >= 3.11
- One compatible ORT host flavor installed separately. The current plugin is built and tested
  with ORT 1.26.0 / `ORT_API_VERSION=26`; inspect `get_manifest()` for packaged metadata.
- MUSA toolkit runtime libraries reachable by the dynamic linker. See the repository
  developer guide for environment setup.
