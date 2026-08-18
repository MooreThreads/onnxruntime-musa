# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""End-to-end test for the LogBucketizeGather fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import (
    TensorProto,
    float32_to_bfloat16_bits,
    musa_devices,
    run_model_and_compare,
    run_with_iobinding,
)


def _const(name, array):
    return helper.make_node(
        "Constant", [], [name], value=numpy_helper.from_array(np.asarray(array), name=f"{name}_value")
    )


def _build_model(table_dtype=TensorProto.FLOAT, with_tail_cast=False) -> bytes:
    if table_dtype == TensorProto.BFLOAT16:
        table_f32 = np.arange(128 * 4, dtype=np.float32).reshape(128, 4)
        table = numpy_helper.from_array(float32_to_bfloat16_bits(table_f32), name="table")
        table.data_type = TensorProto.BFLOAT16
        output_dtype = TensorProto.BFLOAT16
    else:
        table = numpy_helper.from_array(
            np.arange(128 * 4, dtype=np.float32).reshape(128, 4), name="table"
        )
        output_dtype = TensorProto.FLOAT
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    zero_value = helper.make_tensor("zero_shape_value", TensorProto.INT64, [1], [0])
    high_value = helper.make_tensor("high_shape_value", TensorProto.INT64, [1], [126])
    nodes = [
        _const("reshape_shape", np.array([-1], dtype=np.int64)),
        _const("diff_min", np.array(0, dtype=np.int64)),
        _const("time_scale", np.array(60000.0, dtype=np.float32)),
        _const("value_min", np.array(1.0, dtype=np.float32)),
        _const("log_base", np.array(np.log(2.0), dtype=np.float32)),
        _const("bucket_start", np.array(11.0, dtype=np.float32)),
        _const("bucket_span", np.array(9.0, dtype=np.float32)),
        _const("bucket_scale", np.array(126.0, dtype=np.float32)),
        _const("bucket_offset", np.array(1, dtype=np.int64)),
        _const("bucket_min", np.array(1, dtype=np.int64)),
        _const("bucket_max", np.array(126, dtype=np.int64)),
        _const("lower_threshold", np.array(11.0, dtype=np.float32)),
        _const("upper_threshold", np.array(20.0, dtype=np.float32)),
        helper.make_node("Reshape", ["SeqTs", "reshape_shape"], ["SeqFlat"], name="Reshape_8"),
        helper.make_node("Cast", ["SeqFlat"], ["SeqFlatI64"], to=TensorProto.INT64, name="Cast_61"),
        helper.make_node("Sub", ["CurrentTs", "SeqFlatI64"], ["Diff"], name="Sub_12"),
        helper.make_node("Clip", ["Diff", "diff_min", ""], ["DiffClipped"], name="Clip_18"),
        helper.make_node("Cast", ["DiffClipped"], ["DiffF"], to=TensorProto.FLOAT, name="Cast_64"),
        helper.make_node("Div", ["DiffF", "time_scale"], ["MinutesRaw"], name="Div_18"),
        helper.make_node("Clip", ["MinutesRaw", "value_min", ""], ["Minutes"], name="Clip_19"),
        helper.make_node("Log", ["Minutes"], ["Logged"], name="Log_6"),
        helper.make_node("Div", ["Logged", "log_base"], ["LogBucket"], name="Div_19"),
        helper.make_node("Sub", ["LogBucket", "bucket_start"], ["BucketShift"], name="Sub_13"),
        helper.make_node("Div", ["BucketShift", "bucket_span"], ["BucketNorm"], name="Div_20"),
        helper.make_node("Mul", ["BucketNorm", "bucket_scale"], ["BucketScaled"], name="Mul_6"),
        helper.make_node("Floor", ["BucketScaled"], ["BucketFloor"], name="Floor_6"),
        helper.make_node("Cast", ["BucketFloor"], ["BucketInt"], to=TensorProto.INT64, name="Cast_67"),
        helper.make_node("Add", ["BucketInt", "bucket_offset"], ["BucketPlus"], name="Add_6"),
        helper.make_node("Clip", ["BucketPlus", "bucket_min", "bucket_max"], ["BucketClipped"], name="Clip_20"),
        helper.make_node("Shape", ["BucketClipped"], ["BucketShape"], name="Shape_13"),
        helper.make_node(
            "ConstantOfShape",
            ["BucketShape"],
            ["ZeroBuckets"],
            value=zero_value,
            name="ConstantOfShape_12",
        ),
        helper.make_node(
            "ConstantOfShape",
            ["BucketShape"],
            ["HighBuckets"],
            value=high_value,
            name="ConstantOfShape_13",
        ),
        helper.make_node("GreaterOrEqual", ["LogBucket", "upper_threshold"], ["IsHigh"], name="GreaterOrEqual_6"),
        helper.make_node("Where", ["IsHigh", "HighBuckets", "BucketClipped"], ["UpperApplied"], name="Where_12"),
        helper.make_node("LessOrEqual", ["LogBucket", "lower_threshold"], ["IsLow"], name="LessOrEqual_6"),
        helper.make_node("Where", ["IsLow", "ZeroBuckets", "UpperApplied"], ["GatherIds"], name="Where_13"),
        helper.make_node("Gather", ["table", "GatherIds"], ["GatherOut"], axis=0, name="time_diff_embedding_6/Gather"),
        helper.make_node("Unsqueeze", ["GatherOut", "axes"], ["Unsqueezed"], name="Unsqueeze_6"),
    ]
    output_name = "Unsqueezed"
    if with_tail_cast:
        nodes.append(
            helper.make_node(
                "Cast", ["Unsqueezed"], ["Y"], to=table_dtype, name="Cast_70"
            )
        )
        output_name = "Y"
    graph = helper.make_graph(
        nodes,
        "log_bucketize_gather_fusion",
        [
            helper.make_tensor_value_info("CurrentTs", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("SeqTs", TensorProto.INT64, ["N"]),
        ],
        [helper.make_tensor_value_info(output_name, output_dtype, None)],
        initializer=[table, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _expected_indices(current, seq):
    diff = np.maximum(current.reshape(-1)[0] - seq.reshape(-1), 0).astype(np.float32)
    minutes = np.maximum(diff / np.float32(60000.0), np.float32(1.0))
    log_bucket = np.log(minutes) / np.float32(np.log(2.0))
    scaled = np.floor(((log_bucket - 11.0) / 9.0) * 126.0).astype(np.int64) + 1
    clipped = np.clip(scaled, 1, 126)
    ids = np.where(log_bucket >= 20.0, 126, clipped)
    ids = np.where(log_bucket <= 11.0, 0, ids)
    return ids.astype(np.int64)


def _profile_ops(model, feeds, tmp_path, output_dtype=TensorProto.FLOAT, output_shape=None):
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "log_bucketize_gather_fusion")
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    if output_shape is None:
        session.run(None, feeds)
    else:
        output = np.empty(output_shape, dtype=np.uint16)
        io_binding = session.io_binding()
        for name, value in feeds.items():
            io_binding.bind_input(
                name,
                "cpu",
                0,
                TensorProto.INT64,
                value.shape,
                value.ctypes.data,
            )
        io_binding.bind_output(
            "Y", "cpu", 0, output_dtype, output.shape, output.ctypes.data
        )
        session.run_with_iobinding(io_binding)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)
    node_events = [
        e
        for e in events
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
    ]
    return {e.get("args", {}).get("op_name") for e in node_events}


def test_log_bucketize_gather_fusion_float32(tmp_path):
    model = _build_model()
    current = np.array([2**20 * 60000], dtype=np.int64)
    seq = current[0] - np.array(
        [
            -60000,
            0,
            2**11 * 60000,
            2**12 * 60000,
            # Keep this below 2**31 ms. ORT CPU EP's vectorized int64 Clip
            # currently clips values in [2**31, 2**32) to zero, which would
            # make the CPU reference disagree with the ONNX arithmetic this
            # fusion implements.
            2**15 * 60000,
            2**20 * 60000,
            2**21 * 60000,
        ],
        dtype=np.int64,
    )
    feeds = {"CurrentTs": current, "SeqTs": seq}

    (actual,) = run_model_and_compare(model, feeds, rtol=0, atol=0)
    table = np.arange(128 * 4, dtype=np.float32).reshape(128, 4)
    expected = np.expand_dims(table[_expected_indices(current, seq)], axis=1)
    np.testing.assert_array_equal(actual, expected)

    op_names = _profile_ops(model, feeds, tmp_path)
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not (
        {
            "Reshape",
            "Sub",
            "Clip",
            "Cast",
            "Div",
            "Log",
            "Floor",
            "Where",
            "Gather",
            "Unsqueeze",
        }
        & op_names
    )


def test_log_bucketize_gather_fusion_bfloat16_tail_cast(tmp_path):
    model = _build_model(TensorProto.BFLOAT16, with_tail_cast=True)
    current = np.array([2**20 * 60000], dtype=np.int64)
    seq = current[0] - np.array(
        [0, 2**11 * 60000, 2**15 * 60000, 2**20 * 60000, 2**21 * 60000],
        dtype=np.int64,
    )
    feeds = {"CurrentTs": current, "SeqTs": seq}

    (actual,) = run_with_iobinding(
        model,
        feeds,
        {},
        [("Y", TensorProto.BFLOAT16, (5, 1, 4))],
        use_musa=True,
    )
    table_bits = float32_to_bfloat16_bits(
        np.arange(128 * 4, dtype=np.float32).reshape(128, 4)
    )
    expected = np.expand_dims(table_bits[_expected_indices(current, seq)], axis=1)
    np.testing.assert_array_equal(actual, expected)

    op_names = _profile_ops(
        model, feeds, tmp_path, TensorProto.BFLOAT16, output_shape=(5, 1, 4)
    )
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not ({"Log", "Floor", "Where", "Gather", "Unsqueeze", "Cast"} & op_names)
