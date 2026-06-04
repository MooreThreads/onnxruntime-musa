# graph/

Graph-level capability and fusion logic for the MUSA Plugin EP.

## Why this directory exists

The Plugin EP partitions graphs in [`../ep.cc`](../ep.cc): for every node it checks
an op-type allow-list and then confirms a matching kernel is registered through the
kernel registry (`../ep_kernel_registration.cc`). As a result, "which nodes can we
run" is today fully derived from the registered
`ONNX_OPERATOR_VERSIONED_KERNEL_EX` entries under [`../kernels/`](../kernels) — no
extra graph-level code is required, so this directory currently holds no sources.

It is kept as the designated home for graph-level logic that does **not** fit the
per-kernel registry model and will be added as the EP grows:

- **Fusion-pattern matchers** that rewrite subgraphs into `com.microsoft`-domain
  fused kernels (e.g. `FusedGemm`, `FusedMatMul`) before partitioning.
- **Free-standing capability predicates** for decisions the registry cannot
  express on its own, such as gating a node on attribute values, input shapes, or
  experimental flags.

When such logic lands, its `.h`/`.cc` files go here and must be added explicitly to
[`../../CMakeLists.txt`](../../CMakeLists.txt) (`src/kernels/**/*.cc` is
auto-globbed recursively; sources in this directory are not).
