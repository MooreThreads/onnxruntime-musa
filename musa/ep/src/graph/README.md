# graph/

Graph-level capability and fusion logic for the MUSA Plugin EP.

## Why this directory exists

The Plugin EP partitions graphs in [`../ep.cc`](../ep.cc): for every node it confirms
a matching kernel is registered through the kernel registry
(`../ep_kernel_registration.cc`), and then asks the per-operator predicates here
whether that kernel can actually run the node. "Which nodes can we run" is therefore
derived from the registered `ONNX_OPERATOR_VERSIONED_KERNEL_EX` entries under
[`../kernels/`](../kernels), narrowed by the graph-level logic that does **not** fit
the per-kernel registry model:

- **Fusion-pattern matchers** that rewrite subgraphs into `com.microsoft`-domain
  fused kernels (e.g. `FusedGemm`, `FusedMatMul`) before partitioning.
- **Free-standing capability predicates** ([`capability_predicates.cc`](capability_predicates.cc))
  for decisions the registry cannot express on its own, such as gating a node on
  attribute values, input shapes, or experimental flags. A predicate returns false
  for nodes the matching kernel cannot run, and those nodes stay on the CPU EP
  instead of failing when the kernel runs. The per-operator rules themselves live
  next to the kernels they gate (e.g. `../kernels/tensor/resize_support.h`), so the
  capability decision and the kernel's runtime guard share one definition.
- **CPU-preferred node selection** ([`cpu_preferred_nodes.cc`](cpu_preferred_nodes.cc))
  for shape/metadata arithmetic that is cheaper on the CPU EP.

Such `.h`/`.cc` files live here and must be added explicitly to
[`../../CMakeLists.txt`](../../CMakeLists.txt) (`src/kernels/**/*.cc` is
auto-globbed recursively; sources in this directory are not).
