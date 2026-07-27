# onnxruntime_musa

Python helper package shipped inside the `onnxruntime-musa` wheel. It bundles the plugin
shared library (`libonnxruntime_providers_musa_plugin.so`) and exposes two helpers used
to register the MUSA Plugin Execution Provider into a stock ONNX Runtime install.

## Usage

```python
import onnxruntime as ort
import onnxruntime_musa as musa_ep

ep_name = musa_ep.get_ep_name()         # "MUSAExecutionProvider"
lib_path = musa_ep.get_library_path()   # absolute path to the bundled .so / .dll

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

A runnable end-to-end smoke test (1-op MatMul model, prints the device that ran the node)
lives in the repository as [`run_matmul.py`](../../../../run_matmul.py) / `./run.sh`.

## Requirements

- Python >= 3.11
- `onnxruntime` matching the wheel's `Requires-Dist` constraint (auto-derived from the
  pinned ORT submodule; currently `~=1.26.0`)
- MUSA toolkit runtime libraries reachable by the dynamic linker. See the repository
  developer guide for environment setup.
