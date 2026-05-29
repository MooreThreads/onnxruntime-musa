from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import onnxruntime_musa as musa_ep


def make_model(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        "musa_matmul_smoke",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 3]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="musa/model_test/smoke_matmul.onnx")
    args = parser.parse_args()

    model_path = Path(args.model)
    make_model(model_path)

    ort.register_execution_provider_library(musa_ep.get_ep_name(), musa_ep.get_library_path())
    devices = [device for device in ort.get_ep_devices() if device.ep_name == musa_ep.get_ep_name()]
    if not devices:
        raise RuntimeError(f"{musa_ep.get_ep_name()} is not visible")

    session_options = ort.SessionOptions()
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session_options.add_provider_for_devices(devices[:1], {})

    session = ort.InferenceSession(str(model_path), sess_options=session_options)
    rng = np.random.default_rng(2026)
    a = rng.standard_normal((4, 8), dtype=np.float32)
    b = rng.standard_normal((8, 3), dtype=np.float32)
    y = session.run(["Y"], {"A": a, "B": b})[0]
    ref = a @ b
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)
    print({"status": "ok", "providers": session.get_providers(), "max_abs_err": float(np.max(np.abs(y - ref)))})


if __name__ == "__main__":
    main()
