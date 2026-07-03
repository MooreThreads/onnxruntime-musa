// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "reduction/reduction_functions.h"
#include "shared_inc/op_kernel_common.h"

enum class ReduceMode { kProd, kSum, kMean, kSumSquare, kMax, kL2 };

inline MusaReduceOp ToMusaReduceOp(ReduceMode mode) {
  if (mode == ReduceMode::kSum) return MusaReduceOp::Sum;
  if (mode == ReduceMode::kMean) return MusaReduceOp::Mean;
  if (mode == ReduceMode::kSumSquare) return MusaReduceOp::SumSquare;
  if (mode == ReduceMode::kMax) return MusaReduceOp::Max;
  if (mode == ReduceMode::kL2) return MusaReduceOp::L2;
  return MusaReduceOp::Prod;
}

inline OrtStatus* UnsupportedReduceStatus(const char* message) {
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message);
}

inline MusaReduceParams MakeReduceParams(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape, const std::set<int64_t>& axes_set,
    bool keepdims) {
  auto input_strides = Strides(input_shape);
  auto output_strides = Strides(output_shape);
  MusaReduceParams params{};
  params.rank = static_cast<int32_t>(input_shape.size());
  params.reduce_axis =
      axes_set.empty() ? 0 : static_cast<int32_t>(*axes_set.begin());
  params.reduce_axes_count = static_cast<int32_t>(axes_set.size());
  params.output_elements = NumElements(output_shape);
  params.reduce_dim =
      input_shape.empty()
          ? 1
          : input_shape[static_cast<size_t>(params.reduce_axis)];
  params.reduction_elements = 1;
  params.inner_size = 1;
  for (size_t dim = static_cast<size_t>(params.reduce_axis) + 1;
       dim < input_shape.size(); ++dim) {
    params.inner_size *= input_shape[dim];
  }

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

// Shared device-side reduction used by ReduceProd / ReduceSum / ReduceMean /
// ReduceSumSquare.
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

  if (input_shape.size() > kMusaMaxBroadcastRank) {
    return UnsupportedReduceStatus("Reduce rank exceeds MUSA device limit");
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return UnsupportedReduceStatus("unsupported reduce dtype");
  }

  Ort::UnownedValue y = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedReduceStatus("Reduce requires MUSA device tensors");
  }
  musaStream_t stream = GetComputeStream(ctx);
  DeviceInputBuffer input_buffer;
  RETURN_IF_ERROR(input_buffer.Bind(input0, stream));

  if (mode == ReduceMode::kMean &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return UnsupportedReduceStatus("unsupported ReduceMean dtype");
  }
  if (mode == ReduceMode::kL2 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return UnsupportedReduceStatus("unsupported ReduceL2 dtype");
  }

  MusaReduceParams params =
      MakeReduceParams(input_shape, output_shape, axes_set, keepdims);
  return LaunchStatus(LaunchMusaReduceKernel(
      input_buffer.data(), y.GetTensorMutableRawData(), params,
      ToMusaReduceOp(mode), musa_elem_type, stream));
}
