"""End-to-end coverage for multi-slice QKV projection + MHA + output projection."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_model(feeds, branches, input_dim, hidden, output_dim, heads):
    starts = [0, 1, 3]
    ends = [1, 3, 5]
    nodes = []
    initializers = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes0"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="step1"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), name="batch0"),
        numpy_helper.from_array(
            np.array([hidden, hidden, hidden], dtype=np.int64), name="split_sizes"
        ),
    ]
    q_names, k_names, v_names = [], [], []
    for i in range(branches):
        start_name = f"start_{i}"
        end_name = f"end_{i}"
        initializers.extend(
            [
                numpy_helper.from_array(np.array([starts[i]], dtype=np.int64), name=start_name),
                numpy_helper.from_array(np.array([ends[i]], dtype=np.int64), name=end_name),
            ]
        )
        slice_out = f"slice_{i}"
        proj_out = f"proj_{i}"
        split_names = [f"q_{i}", f"k_{i}", f"v_{i}"]
        nodes.extend(
            [
                helper.make_node(
                    "Slice", ["X", start_name, end_name, "axes0", "step1"], [slice_out]
                ),
                helper.make_node(
                    "MatMul", [slice_out, f"qkv_weight_{i}"], [proj_out]
                ),
                helper.make_node(
                    "Split", [proj_out, "split_sizes"], split_names, axis=-1
                ),
            ]
        )
        q_names.append(split_names[0])
        k_names.append(split_names[1])
        v_names.append(split_names[2])

    nodes.extend(
        [
            helper.make_node("Concat", q_names, ["q_concat"], axis=0),
            helper.make_node("Concat", k_names, ["k_concat"], axis=0),
            helper.make_node("Concat", v_names, ["v_concat"], axis=0),
            helper.make_node("Unsqueeze", ["q_concat", "axes0"], ["q3"]),
            helper.make_node("Unsqueeze", ["k_concat", "axes0"], ["k3"]),
            helper.make_node("Unsqueeze", ["v_concat", "axes0"], ["v3"]),
            helper.make_node(
                "MultiHeadAttention",
                ["q3", "k3", "v3", "mha_bias", "mask"],
                ["mha_out"],
                domain="com.microsoft",
                num_heads=heads,
            ),
            helper.make_node("Gather", ["mha_out", "batch0"], ["attn"], axis=0),
        ]
    )

    output_names = []
    for i in range(branches):
        slice_out = f"out_slice_{i}"
        gemm_out = f"out_gemm_{i}"
        output_names.append(gemm_out)
        nodes.extend(
            [
                helper.make_node(
                    "Slice", ["attn", f"start_{i}", f"end_{i}", "axes0", "step1"],
                    [slice_out],
                ),
                helper.make_node(
                    "Gemm",
                    [slice_out, f"out_weight_{i}", f"out_bias_{i}"],
                    [gemm_out],
                    transB=1,
                ),
            ]
        )
    nodes.append(helper.make_node("Concat", output_names, ["Y"], axis=0))

    initializers.extend(
        [
            numpy_helper.from_array(
                np.zeros((3 * hidden,), dtype=np.float32), name="mha_bias"
            ),
        ]
    )
    for i in range(branches):
        initializers.extend(
            [
                numpy_helper.from_array(feeds[f"qkv_weight_{i}"], name=f"qkv_weight_{i}"),
                numpy_helper.from_array(feeds[f"out_weight_{i}"], name=f"out_weight_{i}"),
                numpy_helper.from_array(feeds[f"out_bias_{i}"], name=f"out_bias_{i}"),
            ]
        )

    graph = helper.make_graph(
        nodes,
        "multi_kqv_mha_output_projection_graph",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [5, input_dim]),
            helper.make_tensor_value_info("mask", TensorProto.INT32, [1, 5, 5]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5, output_dim])],
        initializer=initializers,
        value_info=[
            helper.make_tensor_value_info("q3", TensorProto.FLOAT, [1, 5, hidden]),
            helper.make_tensor_value_info("k3", TensorProto.FLOAT, [1, 5, hidden]),
            helper.make_tensor_value_info("v3", TensorProto.FLOAT, [1, 5, hidden]),
            helper.make_tensor_value_info("mha_out", TensorProto.FLOAT, [1, 5, hidden]),
            helper.make_tensor_value_info("attn", TensorProto.FLOAT, [5, hidden]),
            *[
                helper.make_tensor_value_info(
                    f"out_slice_{i}", TensorProto.FLOAT, [ends[i] - starts[i], hidden]
                )
                for i in range(branches)
            ],
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_multi_kqv_mha_output_projection_fuses_dynamic_slices(tmp_path):
    rng = np.random.default_rng(20260803)
    branches, input_dim, hidden, output_dim, heads = 3, 6, 8, 5, 2
    feeds = {
        "X": rng.standard_normal((5, input_dim)).astype(np.float32),
        "mask": np.ones((1, 5, 5), dtype=np.int32),
    }
    feeds["mask"][:, :, -1] = 0
    for i in range(branches):
        feeds[f"qkv_weight_{i}"] = (
            rng.standard_normal((input_dim, 3 * hidden)).astype(np.float32) / 7
        )
        feeds[f"out_weight_{i}"] = (
            rng.standard_normal((output_dim, hidden)).astype(np.float32) / 7
        )
        feeds[f"out_bias_{i}"] = rng.standard_normal(output_dim).astype(np.float32) / 9

    model = _build_model(feeds, branches, input_dim, hidden, output_dim, heads)
    run_model_and_compare(model, {"X": feeds["X"], "mask": feeds["mask"]}, rtol=3e-4, atol=3e-4)

    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "multi_kqv_mha")
    options.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=options)
    session.run(None, {"X": feeds["X"], "mask": feeds["mask"]})
    profile_path = session.end_profiling()
    try:
        with open(profile_path, encoding="utf-8") as profile_file:
            events = json.load(profile_file)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)

    musa_ops = {
        event.get("args", {}).get("op_name")
        for event in events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "MUSAExecutionProvider"
    }
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert "MultiHeadAttention" not in musa_ops
    assert "MatMul" not in musa_ops
    assert "Gemm" not in musa_ops
