# Migration From onnx_musa

Current in-tree source:

- `/home/workspace/onnx_musa/onnxruntime/core/providers/musa`
- `/home/workspace/onnx_musa/onnxruntime/contrib_ops/musa`
- `/home/workspace/onnx_musa/onnxruntime/core/optimizer/musa_operator_fusion.*`

Target layout:

- runtime helpers -> `musa/ep/src/runtime`
- standard kernels -> `musa/ep/src/kernels`
- fused kernels -> `musa/ep/src/kernels/fused`
- supported op and fusion capability -> `musa/ep/src/graph`

In-tree `musa_execution_provider.*` and ORT core optimizer/schema edits are not copied directly.
