# onnxruntime-musa

Independent ONNX Runtime Plugin Execution Provider for **Moore Threads MUSA** GPUs.

This repo follows the `onnxruntime-qnn` product skeleton, with the provider implementation written
against ONNX Runtime's kernel-registry **Plugin EP** C API. The plugin builds out-of-tree as a
single shared library, gets packaged into a Python wheel, and is registered into a stock
`pip install onnxruntime` at runtime — no ORT source build is required.

---

## Repository layout

```
CMakeLists.txt              # top-level CMake, sets C++20 and points at the ORT submodule headers
build.sh                    # one-shot clean build + wheel
musa/ep/                    # plugin EP sources, kernels, Python packaging
  src/                      # C++ implementation (ep_factory, ep, kernels/, ...)
  python/                   # build_wheel.py + onnxruntime_musa package
third_party/onnxruntime/    # ORT submodule pinned to tag v1.26.0
```

See [musa/docs/architecture.md](musa/docs/architecture.md) for design notes. See
[musa/docs/developer_guide.md](musa/docs/developer_guide.md) for environment variables and developer switches.

---

## Dependencies

Build-time (only two):

| Component | Version | Notes |
|---|---|---|
| **MUSA toolkit** | **5.1.0** | Defaults to `/usr/local/musa`. Override with `-DMUSA_HOME=...` or `./build.sh -- -DMUSA_HOME=/opt/musa`. Links the MUSA runtime, muBLAS, and muDNN libraries. |
| **C++ compiler** | C++20 | GCC 11+ / Clang 14+. Required for `std::span`. |

ONNX Runtime is a Git submodule pinned to tag **v1.26.0** (commit `8c546c37`). The build uses
only [third_party/onnxruntime/include/onnxruntime/](third_party/onnxruntime/include/onnxruntime/);
the initialization helper configures a cone-mode sparse checkout for that directory. Initialize it
after cloning the repository:

```bash
./scripts/init_onnxruntime_submodule.sh
```

No `FetchContent` or ORT build tree is needed to compile the plugin. See
[third_party/README.md](third_party/README.md) for submodule details and the ORT upgrade procedure.

GSL is **not** used; `std::span` (C++20) replaced `gsl::span` everywhere.

Runtime:

- Python **>=3.11**.
- `pip install -r requirements.txt` — pins `onnxruntime==1.26.0`, `onnx==1.21.0`, `numpy`,
  plus `pytest>=7.0` for the op tests under `test/`.
  The wheel itself declares `onnxruntime~=1.26.0` (auto-derived from
  `third_party/onnxruntime/VERSION_NUMBER`), because the plugin's C ABI is locked to the
  pinned ORT headers. Bumping to a newer ORT release requires updating the submodule gitlink
  (see [third_party/README.md](third_party/README.md)).

---

## Build

### One-shot: plugin `.so` + wheel

```bash
./build.sh                            # clean rebuild + wheel (Release)
./build.sh --config Debug             # Debug build
./build.sh --no-wheel                 # only build the .so, skip wheel
./build.sh -- -DMUSA_HOME=/opt/musa   # forward extra args to CMake after `--`
```

What it does:

1. Cleans `build/<Config>/` and `dist/`.
2. Picks a Python that satisfies `requires-python>=3.11` — prefers `./.venv/bin/python`, then
   `python3.12` / `python3.11`, then `$PYTHON` / `python3`.
3. `cmake -S . -B build/<Config> -DCMAKE_BUILD_TYPE=<Config>` → `cmake --build`.
4. Runs `musa/ep/python/build_wheel.py` to stage the `.so` into the `onnxruntime_musa` package
   and `pip wheel` it.

Artifacts:

- Plugin: `build/<Config>/libonnxruntime_providers_musa_plugin.so`
- Wheel:  `dist/onnxruntime_musa-<version>-py3-none-linux_x86_64.whl`

### Manual / step by step

```bash
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release -j

# (optional) package as a wheel
.venv/bin/python musa/ep/python/build_wheel.py \
    --binary_dir build/Release \
    --version "$(cat VERSION_NUMBER)" \
    --package_name onnxruntime-musa \
    --output_dir dist
```

---

## Install & run

Python **3.11+** is required (enforced by the wheel's `requires-python`). Pinned
runtime dependencies live in [requirements.txt](requirements.txt):

```
onnxruntime==1.26.0
onnx==1.21.0
numpy
pytest>=7.0
setuptools<70
wheel
pip
cmake
```

Install everything into a fresh venv:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install dist/onnxruntime_musa-*.whl
```

See [test/ops/test_matmul.py](test/ops/test_matmul.py) for the minimal MatMul end-to-end
example. It uses the shared helpers in [test/ops/op_test_utils.py](test/ops/op_test_utils.py)
to register the MUSA plugin EP, disable CPU fallback for the MUSA run, and compare the
MUSA output against the CPU reference.

---

## Op tests

End-to-end per-op tests live under [test/ops/](test/ops). Each `test_<op>.py` builds a
single-node ONNX model, runs it on the stock CPU EP and on the MUSA EP, and asserts the
outputs match (`test/ops/op_test_utils.py` holds the shared helpers).

The MUSA session is created with `session.disable_cpu_ep_fallback=1` and refuses to run
if no MUSA device is present, so an op/dtype the EP does **not** support fails loudly
instead of silently falling back to CPU (which would degrade into a meaningless
CPU-vs-CPU comparison). On a machine with no MUSA device the suite is skipped via
[test/ops/conftest.py](test/ops/conftest.py).

Run the standard suites with the one-shot script (`ops`, `fusion`, and `multi_stream`):

```bash
cd test
bash run_all.sh                 # = python -m pytest ops/ fusion/ multi_stream/
bash run_all.sh -v -k div       # extra args are forwarded to pytest
```

Or invoke pytest directly:

```bash
python -m pytest test/ops/
```

---

## How it fits together

```
┌──────────────────────────────────────────────────────────────────────┐
│  pip install onnxruntime  →  libonnxruntime.so (stock, ORT 1.26.0)   │
│                                                                      │
│  pip install onnxruntime_musa-*.whl                                  │
│      └── onnxruntime_musa/libonnxruntime_providers_musa_plugin.so    │
│                                                                      │
│  Python:                                                             │
│    ort.register_execution_provider_library(name, lib_path)           │
│       └─▶ dlopen(.so) → CreateEpFactories  (ep_lib_entry.cc)         │
│              └─▶ MUSAEpFactory → MUSAEp (kernel registry)            │
│                     └─▶ kernels/{math, activation, tensor, logical,  │
│                                  reduction, nn}                       │
│                            └─▶ MUSA runtime libraries on MUSA 5.1.0   │
└──────────────────────────────────────────────────────────────────────┘
```

Key entry points:

- [musa/ep/src/ep_lib_entry.cc](musa/ep/src/ep_lib_entry.cc) exports `CreateEpFactories` /
  `ReleaseEpFactory`; the export list is pinned by [musa/ep/src/ep_lib.lds](musa/ep/src/ep_lib.lds).
- [musa/ep/src/ep_factory.cc](musa/ep/src/ep_factory.cc) advertises the EP's `OrtEpDevice`(s).
- [musa/ep/src/ep.cc](musa/ep/src/ep.cc) — graph partitioning + kernel registration.
- [musa/ep/src/kernels/](musa/ep/src/kernels/) — current operator coverage is listed in [musa/docs/supported_ops.md](musa/docs/supported_ops.md).
- [musa/ep/python/onnxruntime_musa/__init__.py](musa/ep/python/onnxruntime_musa/__init__.py)
  exposes `get_library_path()` and `get_ep_name()` for registration from Python.

For the current operator coverage, see [musa/docs/supported_ops.md](musa/docs/supported_ops.md).
Fusion documentation is generated the same way: [musa/docs/fusion_priority.md](musa/docs/fusion_priority.md)
lists the GetCapability/Compile priority order, and [musa/docs/fusion/](musa/docs/fusion/)
contains the per-fusion graph notes generated from the current C++ fusion sources. For the
current fusion development workflow, see [musa/docs/fusion_development.md](musa/docs/fusion_development.md).
