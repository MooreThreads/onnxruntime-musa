// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "reduction/reduction_functions.h"
#include "shared_inc/blas_utils.h"
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

inline bool TryMudnnReduce(Ort::ConstValue input, Ort::UnownedValue output,
                           const std::vector<int64_t>& input_shape,
                           const std::vector<int64_t>& output_shape,
                           const std::set<int64_t>& axes_set,
                           ONNXTensorElementDataType elem_type,
                           ReduceMode mode) {
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      axes_set.size() != 1 || mode == ReduceMode::kSumSquare ||
      mode == ReduceMode::kL2 || !IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return false;
  }

  ::musa::dnn::Reduce::Mode mudnn_mode;
  switch (mode) {
    case ReduceMode::kProd:
      mudnn_mode = ::musa::dnn::Reduce::Mode::PROD;
      break;
    case ReduceMode::kSum:
      mudnn_mode = ::musa::dnn::Reduce::Mode::ADD;
      break;
    case ReduceMode::kMean:
      mudnn_mode = ::musa::dnn::Reduce::Mode::MEAN;
      break;
    case ReduceMode::kMax:
      mudnn_mode = ::musa::dnn::Reduce::Mode::MAX;
      break;
    default:
      return false;
  }

  std::vector<int> axes;
  axes.reserve(axes_set.size());
  for (int64_t axis : axes_set) {
    axes.push_back(static_cast<int>(axis));
  }

  std::vector<int64_t> mudnn_output_shape;
  mudnn_output_shape.reserve(input_shape.size());
  for (size_t dim = 0; dim < input_shape.size(); ++dim) {
    mudnn_output_shape.push_back(
        axes_set.count(static_cast<int64_t>(dim)) != 0 ? 1 : input_shape[dim]);
  }
  if (NumElements(mudnn_output_shape) != NumElements(output_shape)) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnFloatTensor(input_tensor, input.GetTensorRawData(),
                           input_shape) ||
      !SetMudnnFloatTensor(output_tensor, output.GetTensorMutableRawData(),
                           mudnn_output_shape)) {
    return false;
  }

  ::musa::dnn::Reduce op;
  if (op.SetMode(mudnn_mode) != ::musa::dnn::Status::SUCCESS ||
      op.SetDim(static_cast<int>(axes.size()), axes.data()) !=
          ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  size_t workspace_size = 0;
  if (op.GetWorkspaceSize(*handle, workspace_size, output_tensor,
                          input_tensor) != ::musa::dnn::Status::SUCCESS ||
      workspace_size != 0) {
    return false;
  }

  ::musa::dnn::MemoryMaintainer maintainer =
      [](size_t /*bytes*/) -> ::musa::dnn::MemoryHandler {
    return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
  };
  return op.Run(*handle, output_tensor, input_tensor, maintainer) ==
         ::musa::dnn::Status::SUCCESS;
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
  if (!IsGpuMemory(input0.GetTensorMemoryInfo()) ||
      !IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedReduceStatus("Reduce requires MUSA device tensors");
  }

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

  if (TryMudnnReduce(input0, y, input_shape, output_shape, axes_set, elem_type,
                     mode)) {
    return nullptr;
  }

  MusaReduceParams params =
      MakeReduceParams(input_shape, output_shape, axes_set, keepdims);
  return LaunchStatus(LaunchMusaReduceKernel(
      input0.GetTensorRawData(), y.GetTensorMutableRawData(), params,
      ToMusaReduceOp(mode), musa_elem_type, nullptr));
}
