# Architecture

`musa/` follows the `qcom/` product layout from `onnxruntime-qnn`, but the EP implementation is kernel
registry based rather than compile/subgraph based.

- `musa/ep/src`: Plugin EP ABI, kernel registry, allocator/data transfer, runtime, kernels, and graph support.
- `musa/ep/python`: Python wheel helper package.
- `musa/model_test`: model-level validation scripts.
- `musa/samples`: small user-facing examples.
