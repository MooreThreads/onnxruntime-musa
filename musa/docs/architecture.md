# Architecture

`onnxruntime-musa` ships an out-of-tree **Plugin Execution Provider** for ONNX Runtime
targeting Moore Threads (MThreads) MUSA GPUs. The plugin is a single shared library that a
stock `pip install onnxruntime` loads on demand via
`ort.register_execution_provider_library(...)`. No ORT source build is required.

## Layout

```
musa/
  ep/
    src/                     C++ implementation of the plugin EP
      ep_lib_entry.cc        C entry points: CreateEpFactories / ReleaseEpFactory
      ep_lib.lds             Linker version script (only the entry points are exported)
      ep_factory.{h,cc}      OrtEpFactory: name/vendor/version, OrtEpDevice advertising,
                             allocator, data-transfer, sync-stream factory
      ep.{h,cc}              OrtEp: graph partitioning, hands the cached kernel registry
                             back to ORT
      ep_kernel_registration.{h,cc}
                             Builds the kernel registry from the list of
                             ONNX_OPERATOR_VERSIONED_KERNEL_EX macros
      ep_data_transfer.{h,cc}
                             Host <-> MUSA memcpy via musart
      ep_stream.{h,cc}       OrtSyncStream backed by a musaStream_t
      ep_allocator.h         GPU and pinned-host allocators on top of musart
      musa_arena.{h,cc}      Optional stream-ordered BFC device arena
      ep_profiling.{h,cc}    Plugin-side profiling event hooks
      ep_options.{h,cc}      Parses session-level provider options
      plugin_ep_utils.h      ABI helper macros (status returns, exception bridge)
      kernels/
        utils.h              KernelDefBuilder helpers + macro that emits a kernel class
        math/                Elementwise math, Gemm/MatMul, Softmax and math device impls
        activation/          Activation-style unary operator entry points
        tensor/              Cast/Gather/Slice/Transpose/shape transform entry points
        logical/             Bool/logical operator entry points and device impls
        reduction/           Reduction entry points, helpers and device impls
        nn/                  Neural-network kernels such as BatchNormalization
        shared_inc/          Shared kernel/device helper types
      graph/
        README.md            Reserved for graph-level fusion/capability logic (no sources yet)
      runtime/
        musa_runtime.h       Thin wrappers + error-string helpers around musart
    python/
      build_wheel.py         Stages the .so, renders pyproject.toml from .in, runs pip wheel
      pyproject.toml.in      Template; @onnxruntime_version@ is auto-derived from
                             third_party/onnxruntime/VERSION
      onnxruntime_musa/      Installed Python package (get_library_path / get_ep_name)
  docs/                      This directory, plus the auto-generated supported_ops.md
```

## Runtime call graph

```
Python:
  ort.register_execution_provider_library("MUSAExecutionProvider", lib_path)
     |
     v  dlopen(.so)
  CreateEpFactories                       (ep_lib_entry.cc)
     |
     v
  MusaEpFactory                           (ep_factory.{h,cc})
     |--- GetSupportedDevices -> OrtEpDevice(s) advertised to ORT
     |--- CreateAllocator    -> cache limit == 0: CustomAllocator
     |                         cache limit > 0: MusaArenaAllocator
     |                           -> raw CustomAllocator -> musart
     |--- CreateDataTransfer -> ep_data_transfer.{h,cc}
     |--- CreateSyncStream   -> ep_stream.{h,cc}
     |--- CreateEp           -> MusaEp
                                  |
                                  v
                              kernel registry  (ep_kernel_registration.cc)
                                  |
                                  v
                              kernels/{math, activation, tensor, logical,
                                       reduction, nn}
                                  |
                                  v
                              musart / mublas (MUSA 5.1.0)
```

## Stream-ordered device arena

When `ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB` is a positive integer,
`MusaEpFactory` wraps the raw MUSA device allocator with
`MusaArenaAllocator`. The value is the arena reservation hard limit. Logical
`Free` returns a chunk to the arena but does not immediately release its
underlying MUSA region.

Each free chunk retains its logical `OrtSyncStream`. It can be reused by that
same stream, or after stream release clears ownership. ORT sync-id records
stream synchronization events, but without an allocator-owned event at
logical `Free` it cannot prove that a wait covers the chunk's final device
access; therefore an owned chunk is never reused directly by another stream.
Every chunk belongs to at most one stream ownership set.

`OnSessionRunEnd` does not clear ownership because host-side Run completion is
not a device-completion fence. When ORT releases the `MusaSyncStream`
implementation, its destructor first synchronizes that one MUSA stream, then
clears ownership; adjacent unowned chunks can then coalesce. This avoids a
device-wide synchronization while ensuring release never exposes unfinished
device work to another stream.
Setting the environment variable to `0` or leaving it unset keeps the original
direct allocator path.

Graph partitioning is kernel-registry driven: ORT asks the EP which nodes it can run, and
the EP answers based on the (op, domain, opset, type-constraints) tuples registered via the
`ONNX_OPERATOR_VERSIONED_KERNEL_EX` macros in `kernels/`.

## Host-Assisted Metadata Paths

Some kernels intentionally use host-visible metadata for ONNX control inputs or
shape-only values. Examples include `OrtMemTypeCPUInput` for axes, split sizes,
`TopK`'s scalar `K`, or shape tensors, and `CopyToHost(...)` for reading small
values that are part of the op's metadata contract.

These paths are not a hidden CPU fallback for the main tensor computation. The
operator's real data path still runs on MUSA when the kernel is supported; the
host work is limited to metadata parsing, shape construction, or other ORT
control inputs that are already defined as CPU-visible by the operator schema.

The complete, build-time-generated list of registered kernels and their dtype constraints
lives in [supported_ops.md](supported_ops.md) (regenerated by
`scripts/gen_supported_ops.py` via a CMake `POST_BUILD` hook).

## ABI surface

The shared library exports exactly two symbols, pinned by
[`ep_lib.lds`](../ep/src/ep_lib.lds):

- `CreateEpFactories`
- `ReleaseEpFactory`

Everything else has hidden visibility, so the plugin can evolve without breaking ORT.

## Dependencies

- **Build time**: only the MUSA toolkit (`musart`, `mublas`) and the vendored ORT public
  headers under [`third_party/onnxruntime/include/`](../../third_party/onnxruntime/). No
  GSL; `std::span` (C++20) is used instead.
- **Run time**: any `onnxruntime` matching the wheel's `onnxruntime~=<vendored-version>`
  constraint, plus the MUSA toolkit shared libraries. See
  [developer_guide.md](developer_guide.md) for environment setup.
