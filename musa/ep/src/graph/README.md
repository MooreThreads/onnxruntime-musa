# graph/

Graph-level capability and fusion logic for the MUSA Plugin EP.

The Plugin EP partitions graphs through the kernel registry (`ep_kernel_registration.cc`),
so the bulk of "which nodes can we run" is implicit in the list of registered
`ONNX_OPERATOR_VERSIONED_KERNEL_EX` entries. This directory holds the free-standing
predicates and (future) fusion patterns that the EP needs in addition to the registry:

- [`supported_ops.h`](supported_ops.h) / [`supported_ops.cc`](supported_ops.cc) — small
  predicates the EP can call without going through the registry (e.g. to short-circuit
  partitioning decisions or to gate experimental ops).

Planned: fusion-pattern matchers that emit `com.microsoft`-domain fused kernels
(`FusedGemm`, `FusedMatMul`, …) registered from `kernels/`.
