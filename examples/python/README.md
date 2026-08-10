# MUSA Plugin EP Python probe

This example uses Python + ONNX Runtime to load the MUSA Plugin EP at runtime
and run the `examples/cpp/single_gemm_opset19.onnx` model.

It exercises two inference paths:

1. single-stream inference with one MUSA session run;
2. multi-stream inference with concurrent `session.run()` calls on the same session.

## Requirements

- `onnxruntime==1.26.0`
- `onnxruntime_musa` from this repository's provider wheel
- a visible MUSA device

## Run

From the repository root:

```bash
./.venv/bin/python examples/python/ort_musa_ep_probe.py
```

Single-stream only:

```bash
./.venv/bin/python examples/python/ort_musa_ep_probe.py --mode single
```

Multi-stream only:

```bash
./.venv/bin/python examples/python/ort_musa_ep_probe.py --mode multi --workers 4 --repeats 16
```

The script registers `MUSAExecutionProvider`, selects the first visible MUSA
device, compares the MUSA output against a CPU reference, and prints the
verified outputs.
