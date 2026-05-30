# runtime/

Thin wrappers around the MUSA C runtime (`musart`) shared by allocator, data-transfer,
stream, and kernel code.

- [`musa_runtime.h`](musa_runtime.h) — small helpers such as `MusaErrorString(musaError_t)`
  used to turn driver/runtime errors into ORT statuses.

Further helpers ported from the legacy `onnx_musa` provider (stream pools, scratch
allocators, kernel-launch wrappers, etc.) will be added here as more kernels land.
