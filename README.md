# onnxruntime-musa

Independent ONNX Runtime MUSA Plugin Execution Provider.

This repository uses the `onnxruntime-qnn` style product skeleton while the provider implementation follows
ONNX Runtime's kernel registry Plugin EP model. The first milestone is a loadable Plugin EP skeleton; MUSA
runtime, kernels, and fused kernels are migrated from the current in-tree `onnx_musa` repository in later
milestones.
