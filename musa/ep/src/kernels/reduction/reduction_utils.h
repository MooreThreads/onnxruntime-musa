// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "reduction/reduction_functions.h"
#include "shared_inc/op_kernel_common.h"

enum class ReduceMode { kProd, kSum, kMean, kSumSquare };

inline MusaReduceOp ToMusaReduceOp(ReduceMode mode) {
  if (mode == ReduceMode::kSum) return MusaReduceOp::Sum;
  if (mode == ReduceMode::kMean) return MusaReduceOp::Mean;
  if (mode == ReduceMode::kSumSquare) return MusaReduceOp::SumSquare;
  return MusaReduceOp::Prod;
}

inline MusaReduceParams MakeReduceParams(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape, const std::set<int64_t>& axes_set,
    bool keepdims) {
  auto input_strides = Strides(input_shape);
  auto output_strides = Strides(output_shape);
  MusaReduceParams params{};
  params.rank = static_cast<int32_t>(input_shape.size());
  params.reduce_axis = static_cast<int32_t>(*axes_set.begin());
  params.output_elements = NumElements(output_shape);
  params.reduce_dim = input_shape[static_cast<size_t>(params.reduce_axis)];
  params.reduction_elements = 1;

  size_t out_dim = 0;
  for (size_t dim = 0; dim < input_shape.size(); ++dim) {
    const bool is_reduce_axis = axes_set.count(static_cast<int64_t>(dim)) != 0;
    params.input_dims[dim] = input_shape[dim];
    params.input_strides[dim] = input_strides[dim];
    params.reduce_axes[dim] = is_reduce_axis ? 1 : 0;
    if (is_reduce_axis) {
      params.reduction_elements *= input_shape[dim];
      params.output_strides[dim] = 0;
    } else {
      params.output_strides[dim] =
          keepdims ? output_strides[dim] : output_strides[out_dim++];
    }
  }
  return params;
}

inline OrtStatus* ReduceCompute(Ort::KernelContext& ctx,
                                std::vector<int64_t> axes, bool keepdims,
                                ReduceMode mode) {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto input_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = input_info.GetElementType();
  auto input_shape = input_info.GetShape();
  if (ctx.GetInputCount() > 1) axes = ReadIntTensor(ctx, 1);
  auto axes_set = AxesSet(axes, input_shape.size());

  std::vector<int64_t> output_shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (axes_set.count(static_cast<int64_t>(i))) {
      if (keepdims) output_shape.push_back(1);
    } else {
      output_shape.push_back(input_shape[i]);
    }
  }
  if (output_shape.empty()) output_shape.push_back(1);

  if (input_shape.size() > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "reduce rank exceeds MUSA kernel limit");
  }
  if (!IsGpuMemory(input0.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "reduce requires MUSA input");
  }

  Ort::UnownedValue y = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "reduce requires MUSA output");
  }

  MusaReduceParams params =
      MakeReduceParams(input_shape, output_shape, axes_set, keepdims);
  MusaReduceOp reduce_op = ToMusaReduceOp(mode);
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return LaunchStatus(LaunchMusaReduceFloatKernel(
        input0.GetTensorData<float>(), y.GetTensorMutableData<float>(), params,
        reduce_op, nullptr));
  }
  if (mode == ReduceMode::kMean) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "integer ReduceMean is not supported");
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return LaunchStatus(LaunchMusaReduceInt32Kernel(
        input0.GetTensorData<int32_t>(), y.GetTensorMutableData<int32_t>(),
        params, reduce_op, nullptr));
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return LaunchStatus(LaunchMusaReduceInt64Kernel(
        input0.GetTensorData<int64_t>(), y.GetTensorMutableData<int64_t>(),
        params, reduce_op, nullptr));
  }
  std::string message = "unsupported reduce dtype: " +
                        std::to_string(static_cast<int>(elem_type));
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message.c_str());
}
