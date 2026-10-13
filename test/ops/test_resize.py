# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end CPU-vs-MUSA tests for the supported Resize routes."""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto as OnnxTensorProto, helper

from op_test_utils import (
    TensorProto,
    build_graph_model,
    musa_devices,
    run,
    run_and_compare,
)

_RESIZE_NODE = "Resize_node"


def _resize_model(x, attrs, *, scales=None, sizes=None, opset=17):
    """Build a single-node Resize model and its feed dict.

    Unlike ``build_model`` this lets an optional input be genuinely absent (an
    empty node-input name) instead of an empty tensor, which is what the zoo
    graphs do and what ONNX shape inference requires when ``sizes`` is used.
    """
    node_inputs = ["X", "", "", ""]
    feeds = {"X": x}
    input_vis = [
        helper.make_tensor_value_info(
            "X", helper.np_dtype_to_tensor_dtype(x.dtype), list(x.shape)
        )
    ]
    if scales is not None:
        node_inputs[2] = "scales"
        feeds["scales"] = scales
        input_vis.append(
            helper.make_tensor_value_info(
                "scales", OnnxTensorProto.FLOAT, list(scales.shape)
            )
        )
    if sizes is not None:
        node_inputs[3] = "sizes"
        feeds["sizes"] = sizes
        input_vis.append(
            helper.make_tensor_value_info(
                "sizes", OnnxTensorProto.INT64, list(sizes.shape)
            )
        )

    node = helper.make_node("Resize", node_inputs, ["Y"], **attrs)
    graph = helper.make_graph(
        [node],
        "Resize_graph",
        input_vis,
        [helper.make_tensor_value_info("Y", OnnxTensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), feeds


def _initialized_resize_model(x, attrs, *, scales=None, sizes=None, roi=None):
    """Single-node Resize whose optional inputs are constant initializers.

    The graph-level capability predicate can only inspect the value of a
    constant, so the cases that depend on the roi/scales/sizes contents use this
    builder instead of feeding those tensors at run time.
    """
    node_inputs = ["X", "", "", ""]
    initializers = []
    if roi is not None:
        node_inputs[1] = "roi"
        initializers.append(
            helper.make_tensor(
                "roi",
                OnnxTensorProto.FLOAT,
                list(roi.shape),
                [float(value) for value in roi],
            )
        )
    if scales is not None:
        node_inputs[2] = "scales"
        initializers.append(
            helper.make_tensor(
                "scales",
                OnnxTensorProto.FLOAT,
                list(scales.shape),
                [float(value) for value in scales],
            )
        )
    if sizes is not None:
        node_inputs[3] = "sizes"
        initializers.append(
            helper.make_tensor(
                "sizes",
                OnnxTensorProto.INT64,
                list(sizes.shape),
                [int(value) for value in sizes],
            )
        )

    node = helper.make_node("Resize", node_inputs, ["Y"], name=_RESIZE_NODE, **attrs)
    model_bytes = build_graph_model(
        [node],
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=initializers,
        opset=17,
        name="Resize_graph",
    )
    return model_bytes, {"X": x}


def _musa_session(model_bytes):
    """Create a MUSA session that cannot silently fall back to CPU.

    Both fallbacks are disabled: the graph-level one (an unplaced node makes
    session creation fail) and the runtime one (an EP failure at run time would
    otherwise make ORT rebuild the whole session as CPU-only and hand back a
    perfect CPU result).
    """
    devices = musa_devices()
    if not devices:
        raise RuntimeError(
            "No MUSA device available; refusing to build a CPU-only session "
            "for the MUSA run (would compare CPU against CPU)."
        )
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model_bytes, sess_options=so)
    session.disable_fallback()
    return session


def _musa_session_with_assignment_report(model_bytes):
    """MUSA session that reports which nodes it claimed and hides no failure.

    The graph-level CPU fallback stays enabled, because a node the MUSA EP
    declines has to be able to run on the CPU EP; the run-time EP-error fallback
    is disabled so a node the MUSA EP claimed and then failed inside Compute()
    raises instead of silently re-running the whole session on the CPU (which
    would make the output comparison meaningless). ``dtype_diagnostics`` makes
    the EP print one ``MUSA_DTYPE_NODE`` line per node with ``assigned=1`` when
    it registered that node and ``assigned=0`` when it declined it.
    """
    devices = musa_devices()
    if not devices:
        raise RuntimeError(
            "No MUSA device available; refusing to build a CPU-only session "
            "for the MUSA run (would compare CPU against CPU)."
        )
    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {"dtype_diagnostics": "1"})
    session = ort.InferenceSession(model_bytes, sess_options=so)
    session.disable_fallback()
    return session


def _musa_assignment(diagnostics):
    """Map node name -> claimed-by-MUSA from the EP's ``MUSA_DTYPE_NODE`` lines."""
    assert "MUSA_DTYPE_REPORT_BEGIN" in diagnostics, (
        "the MUSA EP did not emit its dtype/assignment report:\n" + diagnostics
    )
    assignment = {}
    for line in diagnostics.splitlines():
        if "MUSA_DTYPE_NODE" not in line:
            continue
        fields = dict(
            field.split("=", 1) for field in line.split() if "=" in field
        )
        if "name" in fields:
            assignment[fields["name"]] = fields["assigned"] == "1"
    return assignment


def _run_resize(x, attrs, *, scales=None, sizes=None):
    """Run the Resize model on the CPU EP and the MUSA EP, then compare."""
    model_bytes, feeds = _resize_model(x, attrs, scales=scales, sizes=sizes)
    cpu_outputs = run(model_bytes, feeds, use_musa=False)
    musa_outputs = _musa_session(model_bytes).run(None, dict(feeds))

    assert len(cpu_outputs) == len(musa_outputs)
    for cpu_output, musa_output in zip(cpu_outputs, musa_outputs):
        assert cpu_output.shape == musa_output.shape
        assert cpu_output.dtype == musa_output.dtype
        np.testing.assert_allclose(
            musa_output, cpu_output, rtol=1e-3, atol=1e-4
        )
    return musa_outputs


def test_resize_opset17_float_nchw_nearest_asymmetric_floor_scales():
    x = np.arange(1 * 3 * 5 * 7, dtype=np.float32).reshape(1, 3, 5, 7)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    run_and_compare(
        "Resize",
        inputs={"X": x, "roi": np.empty((0,), dtype=np.float32), "scales": scales},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        opset=17,
        rtol=0,
        atol=0,
    )


def test_resize_opset17_float_nchw_nearest_asymmetric_floor_sizes():
    x = np.arange(1 * 3 * 9 * 13, dtype=np.float32).reshape(1, 3, 9, 13)
    sizes = np.array([1, 3, 3, 4], dtype=np.int64)
    _run_resize(
        x,
        {
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        sizes=sizes,
    )


def test_resize_opset17_float_nchw_nearest_asymmetric_floor_exact_ratio_sizes():
    # 6/2 == 3 and 15/3 == 5 give coordinates that are integral in float32 but
    # just below the integer in double, so a wider quotient would truncate one
    # sample short (regression guard for the nearest index computation).
    x = np.arange(1 * 2 * 6 * 15, dtype=np.float32).reshape(1, 2, 6, 15)
    sizes = np.array([1, 2, 2, 3], dtype=np.int64)
    _run_resize(
        x,
        {
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        sizes=sizes,
    )


def test_resize_opset17_float_nchw_linear_half_pixel_scales():
    x = np.arange(1 * 3 * 5 * 7, dtype=np.float32).reshape(1, 3, 5, 7)
    scales = np.array([1.0, 1.0, 1.7, 1.4], dtype=np.float32)
    _run_resize(
        x,
        {"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        scales=scales,
    )


def test_resize_opset17_float_nchw_linear_half_pixel_sizes():
    x = np.arange(2 * 3 * 5 * 7, dtype=np.float32).reshape(2, 3, 5, 7) - 21.0
    sizes = np.array([2, 3, 9, 13], dtype=np.int64)
    _run_resize(
        x,
        {"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        sizes=sizes,
    )


def test_resize_opset17_float_nchw_linear_half_pixel_downsample_sizes():
    x = np.arange(1 * 3 * 9 * 13, dtype=np.float32).reshape(1, 3, 9, 13)
    sizes = np.array([1, 3, 4, 5], dtype=np.int64)
    _run_resize(
        x,
        {"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        sizes=sizes,
    )


def test_resize_opset17_float_nchw_linear_pytorch_half_pixel_sizes():
    x = np.arange(1 * 2 * 5 * 7, dtype=np.float32).reshape(1, 2, 5, 7)
    sizes = np.array([1, 2, 3, 3], dtype=np.int64)
    _run_resize(
        x,
        {"mode": "linear", "coordinate_transformation_mode": "pytorch_half_pixel"},
        sizes=sizes,
    )


def test_resize_opset17_float_nchw_linear_align_corners_sizes():
    x = np.arange(1 * 2 * 5 * 7, dtype=np.float32).reshape(1, 2, 5, 7)
    sizes = np.array([1, 2, 9, 13], dtype=np.int64)
    _run_resize(
        x,
        {"mode": "linear", "coordinate_transformation_mode": "align_corners"},
        sizes=sizes,
    )


def test_resize_opset17_float_nchw_linear_asymmetric_scales():
    x = np.arange(1 * 2 * 5 * 7, dtype=np.float32).reshape(1, 2, 5, 7)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    _run_resize(
        x,
        {"mode": "linear", "coordinate_transformation_mode": "asymmetric"},
        scales=scales,
    )


def test_resize_opset17_float_ncdhw_linear_half_pixel_scales():
    x = np.arange(1 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 2, 3, 4)
    scales = np.array([1.0, 1.0, 2.0, 1.5, 2.0], dtype=np.float32)
    _run_resize(
        x,
        {"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        scales=scales,
    )


def test_resize_opset17_float_ncdhw_linear_half_pixel_sizes():
    x = np.arange(1 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 2, 3, 4)
    sizes = np.array([1, 2, 4, 5, 9], dtype=np.int64)
    _run_resize(
        x,
        {"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        sizes=sizes,
    )


# Configurations the MUSA route cannot run but the CPU EP can: the route only
# implements nearest/asymmetric/floor and the linear coordinate transformations,
# on rank-4 NCHW and rank-5 NCDHW tensors, with an empty roi and with the batch
# and channel axes left alone. Each case below trips exactly one rule, and all
# of them carry the values the rule needs as constant initializers so the
# graph-level gate can see them.
_UNSUPPORTED_RESIZE_CASES = {
    "cubic_sizes": dict(
        attrs={"mode": "cubic", "coordinate_transformation_mode": "half_pixel"},
        x_shape=(1, 2, 5, 7),
        sizes=np.array([1, 2, 9, 13], dtype=np.int64),
    ),
    "cubic_scales": dict(
        attrs={"mode": "cubic", "coordinate_transformation_mode": "half_pixel"},
        x_shape=(1, 2, 5, 7),
        scales=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
    ),
    "nearest_half_pixel_scales": dict(
        attrs={"mode": "nearest", "coordinate_transformation_mode": "half_pixel"},
        x_shape=(1, 2, 5, 7),
        scales=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
    ),
    "nearest_round_prefer_floor_scales": dict(
        attrs={
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "round_prefer_floor",
        },
        x_shape=(1, 2, 5, 7),
        scales=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
    ),
    "nearest_rank5_scales": dict(
        attrs={
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        x_shape=(1, 1, 2, 3, 4),
        scales=np.array([1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32),
    ),
    "nearest_nonempty_roi_scales": dict(
        attrs={
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        x_shape=(1, 2, 5, 7),
        roi=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        scales=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
    ),
    "nearest_channel_scales": dict(
        attrs={
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        x_shape=(1, 2, 5, 7),
        scales=np.array([1.0, 2.0, 2.0, 2.0], dtype=np.float32),
    ),
    "nearest_batch_sizes": dict(
        attrs={
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        x_shape=(1, 2, 5, 7),
        sizes=np.array([2, 2, 10, 14], dtype=np.int64),
    ),
    "linear_rank3_sizes": dict(
        attrs={"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        x_shape=(1, 5, 7),
        sizes=np.array([1, 9, 13], dtype=np.int64),
    ),
    "tf_crop_and_resize_sizes": dict(
        attrs={"mode": "linear", "coordinate_transformation_mode": "tf_crop_and_resize"},
        x_shape=(1, 2, 5, 7),
        roi=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.6, 0.6], dtype=np.float32),
        sizes=np.array([1, 2, 10, 14], dtype=np.int64),
    ),
}


def _unsupported_resize_model(case):
    spec = _UNSUPPORTED_RESIZE_CASES[case]
    x = np.arange(
        int(np.prod(spec["x_shape"])), dtype=np.float32
    ).reshape(spec["x_shape"])
    optional = {
        key: value
        for key, value in spec.items()
        if key not in ("attrs", "x_shape")
    }
    return _initialized_resize_model(x, spec["attrs"], **optional)


@pytest.mark.parametrize("case", list(_UNSUPPORTED_RESIZE_CASES))
def test_resize_unsupported_configuration_stays_on_cpu(case, capfd):
    """Configurations the MUSA kernel cannot run must be left to the CPU EP.

    Each of these is a valid ONNX Resize the CPU EP implements, but not one the
    MUSA route does, so the EP has to decline the node during GetCapability
    instead of claiming it and failing inside Compute(). The session keeps the
    graph-level CPU fallback (that is where the node must land) but disables the
    run-time EP-error fallback, so a node the MUSA EP claimed and then failed
    would raise here rather than silently re-running on the CPU, and the EP's
    own assignment report then shows ``assigned=0`` for the node.
    """
    model_bytes, feeds = _unsupported_resize_model(case)
    cpu_outputs = run(model_bytes, feeds, use_musa=False)

    session = _musa_session_with_assignment_report(model_bytes)
    musa_outputs = session.run(None, dict(feeds))

    assert len(cpu_outputs) == len(musa_outputs)
    for cpu_output, musa_output in zip(cpu_outputs, musa_outputs):
        assert cpu_output.shape == musa_output.shape
        assert cpu_output.dtype == musa_output.dtype
        np.testing.assert_allclose(musa_output, cpu_output, rtol=0, atol=0)

    assignment = _musa_assignment(capfd.readouterr().err)
    assert assignment[_RESIZE_NODE] is False


# Configurations the MUSA route does run, with the values that decide them
# present as constant initializers, so the gate has every chance to decline them
# and must still claim the node.
_SUPPORTED_RESIZE_CASES = {
    "nearest_scales": dict(
        attrs={
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        x_shape=(1, 3, 5, 7),
        scales=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
    ),
    "linear_sizes": dict(
        attrs={"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        x_shape=(2, 3, 5, 7),
        sizes=np.array([2, 3, 9, 13], dtype=np.int64),
    ),
    "linear_rank5_scales": dict(
        attrs={"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        x_shape=(1, 2, 2, 3, 4),
        scales=np.array([1.0, 1.0, 2.0, 1.5, 2.0], dtype=np.float32),
    ),
    # Paddle exports drive Resize through `sizes` and pass an empty roi and an
    # empty scales initializer next to it. Those mean "not provided" - the
    # kernel reads a zero-element input as absent - so they must not be mistaken
    # for a second, conflicting input and stay on the MUSA EP.
    "paddle_empty_roi_scales_linear_sizes": dict(
        attrs={"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        x_shape=(1, 3, 5, 7),
        roi=np.empty((0,), dtype=np.float32),
        scales=np.empty((0,), dtype=np.float32),
        sizes=np.array([1, 3, 9, 13], dtype=np.int64),
    ),
    "paddle_empty_roi_scales_nearest_sizes": dict(
        attrs={
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        x_shape=(1, 3, 5, 7),
        roi=np.empty((0,), dtype=np.float32),
        scales=np.empty((0,), dtype=np.float32),
        sizes=np.array([1, 3, 9, 13], dtype=np.int64),
    ),
}


@pytest.mark.parametrize("case", list(_SUPPORTED_RESIZE_CASES))
def test_resize_supported_configuration_is_claimed_by_musa(case, capfd):
    """The supported route keeps the node on the MUSA EP, and computes it there.

    The assignment report proves the EP claimed the node (``assigned=1``), and
    the run-time EP-error fallback is disabled, so the matching output also
    proves the MUSA kernel computed it instead of the CPU EP.
    """
    spec = _SUPPORTED_RESIZE_CASES[case]
    x = np.arange(
        int(np.prod(spec["x_shape"])), dtype=np.float32
    ).reshape(spec["x_shape"])
    optional = {
        key: value
        for key, value in spec.items()
        if key not in ("attrs", "x_shape")
    }
    model_bytes, feeds = _initialized_resize_model(x, spec["attrs"], **optional)
    cpu_outputs = run(model_bytes, feeds, use_musa=False)

    session = _musa_session_with_assignment_report(model_bytes)
    musa_outputs = session.run(None, dict(feeds))

    assignment = _musa_assignment(capfd.readouterr().err)
    assert assignment[_RESIZE_NODE] is True

    assert len(cpu_outputs) == len(musa_outputs)
    for cpu_output, musa_output in zip(cpu_outputs, musa_outputs):
        assert cpu_output.shape == musa_output.shape
        assert cpu_output.dtype == musa_output.dtype
        np.testing.assert_allclose(musa_output, cpu_output, rtol=1e-3, atol=1e-4)


def test_resize_run_time_only_rejection_still_fails_loudly():
    """The kernel guard still rejects what the graph-level gate cannot see.

    ``scales`` is a run-time input here, so its values are unknown during
    GetCapability and the EP claims the node; the kernel must then reject the
    batch/channel scaling itself instead of running it. Both fallbacks are
    disabled by ``_musa_session``, so the rejection cannot be hidden by a
    CPU-only re-run of the session.
    """
    x = np.arange(1 * 2 * 5 * 7, dtype=np.float32).reshape(1, 2, 5, 7)
    scales = np.array([1.0, 2.0, 2.0, 2.0], dtype=np.float32)
    model_bytes, feeds = _resize_model(
        x,
        {
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        scales=scales,
    )
    with pytest.raises(Exception, match="batch and channel scales to be 1"):
        _musa_session(model_bytes).run(None, dict(feeds))
