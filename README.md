# onnxruntime-musa

Independent ONNX Runtime Plugin Execution Provider for **Moore Threads MUSA** GPUs.

This repo follows the `onnxruntime-qnn` product skeleton, with the provider implementation written
against ONNX Runtime's kernel-registry **Plugin EP** C API. The plugin builds out-of-tree as a
single shared library, gets packaged into a Python wheel, and is registered into a stock
ONNX Runtime host at runtime. Python and C++ applications use the same plugin binary;
the provider package does not bundle or link `libonnxruntime.so`.

---

## Repository layout

```
CMakeLists.txt              # top-level CMake, sets C++20 and points at the ORT submodule headers
build.sh                    # incremental plugin build + wheel; --clean for a fresh rebuild
musa/ep/                    # plugin EP sources, kernels, Python packaging
  src/                      # C++ implementation (ep_factory, ep, kernels/, ...)
  python/                   # provider-only wheel + onnxruntime_musa helper package
examples/cpp/               # Python-free C++ host/plugin registration probe
examples/python/            # Python host/plugin registration probe
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

Development/test runtime:

- Python **>=3.11**.
- `pip install -r requirements.txt` — pins `onnxruntime==1.26.0`, `onnx==1.21.0`, `numpy`,
  plus `pytest>=7.0` for the op tests under `test/`. These are repository development
  requirements, not dependencies embedded in the provider-only wheel.

Application runtime:

- Python applications install exactly one compatible ORT host flavor (`onnxruntime` or
  `onnxruntime-gpu`) separately.
- C++ applications provide a compatible ORT C++ SDK/runtime separately and link
  `libonnxruntime.so`.
- Both load the same MUSA plugin at runtime. The package manifest records the build/test ORT
  version and `ORT_API_VERSION`.

---

## Build

### Build plugin `.so` + wheel

```bash
./build.sh                            # incremental build + wheel (Release)
./build.sh --clean                    # delete build output, then rebuild + wheel
./build.sh --config Debug             # Debug build
./build.sh --no-wheel                 # incrementally build only the .so, skip wheel
./build.sh -- -DMUSA_HOME=/opt/musa   # forward extra args to CMake after `--`
```

What it does:

1. Reuses `build/<Config>/` and `dist/` by default; `--clean` removes both before configuring.
2. Picks a Python that satisfies `requires-python>=3.11` — prefers `./.venv/bin/python`, then
   `python3.12` / `python3.11`, then `$PYTHON` / `python3`.
3. `cmake -S . -B build/<Config> -DCMAKE_BUILD_TYPE=<Config>` → `cmake --build`.
4. Runs `musa/ep/python/build_wheel.py` to stage the `.so` into the `onnxruntime_musa` package
   and `pip wheel` it.

Artifacts:

- Plugin: `build/<Config>/libonnxruntime_providers_musa_plugin.so`
- Manifest: `build/<Config>/onnxruntime_musa_ep_manifest.json`
- Wheel:  `dist/onnxruntime_ep_musa-<version>-py3-none-linux_x86_64.whl`

### Manual / step by step

```bash
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release -j

# (optional) package as a wheel
.venv/bin/python musa/ep/python/build_wheel.py \
    --binary_dir build/Release \
    --version "$(cat VERSION_NUMBER)" \
    --package_name onnxruntime-ep-musa \
    --output_dir dist
```

The legacy distribution name can still be produced explicitly with
`./build.sh --package-name onnxruntime-musa` during migration. The Python import name remains
`onnxruntime_musa` in both distributions.

---

## Code formatting

Format all C/C++ sources under `musa/ep/src/` with the repository's
[`.clang-format`](.clang-format) configuration:

```bash
scripts/format.sh
```

The script requires `clang-format` to be available on `PATH` and edits files in place.
To have staged C/C++ changes formatted automatically at commit time, install the repository hooks:

```bash
scripts/install-hooks.sh
```

---

## Install & run

### Install for Python

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

Install the test host and provider wheel into a fresh venv:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install dist/onnxruntime_ep_musa-*.whl --no-deps
```

The wheel deliberately has no `Requires-Dist: onnxruntime`: `onnxruntime` and
`onnxruntime-gpu` are alternative host flavors that share the same Python import namespace.
Production users should install one compatible host first, then install the provider wheel
with `--no-deps`.

### Install for C++

The C++ application needs two independent artifacts:

1. an ONNX Runtime C++ host SDK/runtime (`onnxruntime_cxx_api.h` plus
   `libonnxruntime.so`);
2. the MUSA provider plugin from this repository.

Do not download a Python wheel for the C++ application, and do not put
`libonnxruntime.so` into the repository `third_party` directory.

#### 1. Download the matching ONNX Runtime C++ SDK

This repository is pinned to ONNX Runtime **v1.26.0** and `ORT_API_VERSION=26`.
For a Linux x86_64 machine, download the CPU C++ package:

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

Check the exact files before configuring the application:

```bash
test -f "$ORT_ROOT/include/onnxruntime_cxx_api.h"
test -f "$ORT_ROOT/lib/libonnxruntime.so"
grep '^#define ORT_API_VERSION ' \
  "$ORT_ROOT/include/onnxruntime_c_api.h"
```

The last command must report API version `26`.

#### 2. Build and install the MUSA provider

```bash
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --target onnxruntime_providers_musa_plugin -j
cmake --install build/Release \
  --prefix "$PWD/build/install/onnxruntime-ep-musa/1.0.0"
```

This installs the plugin, compatibility manifest, licenses, and a CMake package. It does not
install ORT core.

#### 3. Build the Python-free C++ probe

```bash
cmake -S examples/cpp -B build/cxx-probe \
  -DORT_ROOT="$ORT_ROOT"
cmake --build build/cxx-probe -j
```

If the SDK uses a non-standard layout, pass the two files explicitly:

```bash
cmake -S examples/cpp -B build/cxx-probe \
  -DORT_INCLUDE_DIR="$ORT_ROOT/include" \
  -DORT_LIBRARY="$ORT_ROOT/lib/libonnxruntime.so"
```

#### 4. Register the plugin from C++

```bash
MUSA_EP_ROOT="$PWD/build/install/onnxruntime-ep-musa/1.0.0"
MUSA_APP_LIBRARY_PATH="$ORT_ROOT/lib:$MUSA_EP_ROOT/lib:/usr/local/musa/lib:/usr/local/musa/lib64"
export LD_LIBRARY_PATH="$MUSA_APP_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"

build/cxx-probe/ort_musa_ep_probe \
  "$MUSA_EP_ROOT/lib/libonnxruntime_providers_musa_plugin.so"
```

To create a MUSA-only session for a model:

```bash
build/cxx-probe/ort_musa_ep_probe \
  "$MUSA_EP_ROOT/lib/libonnxruntime_providers_musa_plugin.so" \
  /absolute/path/model.onnx
```

The application links only `libonnxruntime.so`. It passes the plugin path to
`Ort::Env::RegisterExecutionProviderLibrary()` at runtime; do not link the application
against `libonnxruntime_providers_musa_plugin.so`.

See [examples/cpp/README.md](examples/cpp/README.md) for the same procedure in a standalone
C++-focused document.

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
│  Host (choose one):                                                  │
│    Python: onnxruntime / onnxruntime-gpu                             │
│    C++:    ORT C++ SDK + libonnxruntime.so                           │
│                                                                      │
│  Provider: onnxruntime_ep_musa-*.whl or CMake-installed plugin       │
│      └── onnxruntime_musa/libonnxruntime_providers_musa_plugin.so    │
│                                                                      │
│  Python:                                                             │
│    ort.register_execution_provider_library(name, lib_path)           │
│       └─▶ dlopen(.so) → CreateEpFactories  (ep_lib_entry.cc)         │
│              └─▶ MUSAEpFactory → MUSAEp (kernel registry)            │
│                     └─▶ kernels/{math, activation, tensor, logical,  │
│                                  reduction, nn}                      │
│                            └─▶ MUSA runtime libraries on MUSA 5.1.0  │
└──────────────────────────────────────────────────────────────────────┘
```

Key entry points:

- [musa/ep/src/ep_lib_entry.cc](musa/ep/src/ep_lib_entry.cc) exports `CreateEpFactories` /
  `ReleaseEpFactory`; the export list is pinned by [musa/ep/src/ep_lib.lds](musa/ep/src/ep_lib.lds).
- [musa/ep/src/ep_factory.cc](musa/ep/src/ep_factory.cc) advertises the EP's `OrtEpDevice`(s).
- [musa/ep/src/ep.cc](musa/ep/src/ep.cc) — graph partitioning + kernel registration.
- [musa/ep/src/kernels/](musa/ep/src/kernels/) — current operator coverage is listed in [musa/docs/supported_ops.md](musa/docs/supported_ops.md).
- [musa/ep/python/onnxruntime_musa/__init__.py](musa/ep/python/onnxruntime_musa/__init__.py)
  exposes `get_library_path()`, `get_ep_name()`, and compatibility-manifest helpers.

For the current operator coverage, see [musa/docs/supported_ops.md](musa/docs/supported_ops.md).
Fusion documentation is generated the same way: [musa/docs/fusion_priority.md](musa/docs/fusion_priority.md)
lists the GetCapability/Compile priority order, and [musa/docs/fusion/](musa/docs/fusion/)
contains the per-fusion graph notes generated from the current C++ fusion sources. For the
current fusion development workflow, see [musa/docs/fusion_development.md](musa/docs/fusion_development.md).
