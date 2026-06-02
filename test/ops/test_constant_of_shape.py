# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ConstantOfShape operator."""

import numpy as np
import onnx

from op_test_utils import TensorProto, build_model, musa_available, run


def _run_constant_of_shape(shape_vals, value_tensor=None):
    """Build and run a ConstantOfShape node with the given shape and value."""
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    shape_input = np.array(shape_vals, dtype=np.int64)
    attrs = {}
    if value_tensor is not None:
        attrs["value"] = value_tensor

    model_bytes = build_model(
        "ConstantOfShape",
        inputs={"shape": shape_input},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs=attrs if attrs else None,
    )
    cpu_out = run(model_bytes, {"shape": shape_input}, use_musa=False)
    if musa_available():
        musa_out = run(model_bytes, {"shape": shape_input}, use_musa=True)
        np.testing.assert_allclose(musa_out[0], cpu_out[0], rtol=1e-5, atol=1e-6)
    return cpu_out[0]


def test_constant_of_shape_default():
    # Default fill value is float 0.0
    out = _run_constant_of_shape([2, 3])
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, np.zeros((2, 3), dtype=np.float32))


def test_constant_of_shape_nonzero_fill():
    from onnx import numpy_helper

    val = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="val")
    out = _run_constant_of_shape([3, 4], value_tensor=val)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out, np.ones((3, 4), dtype=np.float32))


def test_constant_of_shape_1d():
    out = _run_constant_of_shape([8])
    assert out.shape == (8,)
