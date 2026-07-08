// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "shared_inc/op_kernel_common.h"
#include "tensor/scatter_nd_impl.h"

namespace {
constexpr int32_t kScatterNDReductionNone = 0;
constexpr int32_t kScatterNDReductionAdd = 1;

bool IsScatterNDAddType(ONNXTensorElementDataType elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

class ScatterND : public OpKernelBase<ScatterND> {
 public:
  ScatterND(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    reduction_ = AttrOrDefault<std::string>(kernel_info, "reduction", "none");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::string reduction_;
};

OrtStatus* ValidateScatterNDShapes(const std::vector<int64_t>& input_shape,
                                   const std::vector<int64_t>& indices_shape,
                                   const std::vector<int64_t>& updates_shape) {
  if (indices_shape.empty()) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "ScatterND indices rank must be larger than 0");
  }
  for (int64_t dim : input_shape) {
    if (dim < 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "ScatterND requires concrete input shape");
    }
  }
  for (int64_t dim : indices_shape) {
    if (dim < 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "ScatterND requires concrete indices shape");
    }
  }
  for (int64_t dim : updates_shape) {
    if (dim < 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "ScatterND requires concrete updates shape");
    }
  }

  const int64_t last_index_dimension = indices_shape.back();
  if (last_index_dimension <= 0 ||
      last_index_dimension > static_cast<int64_t>(input_shape.size())) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "ScatterND last dimension of indices exceeds input rank");
  }

  std::vector<int64_t> expected_updates_shape(indices_shape.begin(),
                                              indices_shape.end() - 1);
  expected_updates_shape.insert(expected_updates_shape.end(),
                                input_shape.begin() + last_index_dimension,
                                input_shape.end());
  if (updates_shape != expected_updates_shape) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "ScatterND updates shape mismatch");
  }
  return nullptr;
}

OrtStatus* ScatterND::Compute(Ort::KernelContext& ctx) const {
  int32_t reduction = kScatterNDReductionNone;
  if (reduction_ == "add") {
    reduction = kScatterNDReductionAdd;
  } else if (reduction_ != "none") {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "ScatterND MUSA only supports reduction='none' or 'add'");
  }

  Ort::ConstValue input = ctx.GetInput(0);
  Ort::ConstValue indices = ctx.GetInput(1);
  Ort::ConstValue updates = ctx.GetInput(2);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto indices_info = indices.GetTensorTypeAndShapeInfo();
  auto updates_info = updates.GetTensorTypeAndShapeInfo();
  auto input_shape = input_info.GetShape();
  auto indices_shape = indices_info.GetShape();
  auto updates_shape = updates_info.GetShape();

  if (indices_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "ScatterND only supports int64 indices");
  }
  const auto elem_type = input_info.GetElementType();
  if (reduction == kScatterNDReductionAdd && !IsScatterNDAddType(elem_type)) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ScatterND reduction='add' unsupported dtype");
  }
  if (updates_info.GetElementType() != elem_type) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "ScatterND input dtypes must match");
  }
  const size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "ScatterND unsupported dtype");
  }
  if (input_shape.size() > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ScatterND rank exceeds MUSA device limit");
  }
  RETURN_IF_ERROR(
      ValidateScatterNDShapes(input_shape, indices_shape, updates_shape));

  Ort::UnownedValue output = ctx.GetOutput(0, input_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "ScatterND requires MUSA output tensor");
  }

  musaStream_t stream = GetComputeStream(ctx);
  RETURN_IF_ERROR(
      CopyRawTensor(input, output, input.GetTensorSizeInBytes(), stream));

  if (indices_shape.empty() || NumElements(indices_shape) == 0) {
    return nullptr;
  }

  DeviceInputBuffer indices_buffer;
  DeviceInputBuffer updates_buffer;
  RETURN_IF_ERROR(indices_buffer.Bind(indices, stream));
  RETURN_IF_ERROR(updates_buffer.Bind(updates, stream));

  MusaScatterNDParams params{};
  params.last_index_dimension = static_cast<int32_t>(indices_shape.back());
  params.reduction = reduction;
  params.num_indices = NumElements(indices_shape) / indices_shape.back();
  params.updates_slice_size = NumElements(std::vector<int64_t>(
      input_shape.begin() + params.last_index_dimension, input_shape.end()));
  std::vector<int64_t> input_strides = Strides(input_shape);
  for (int32_t i = 0; i < params.last_index_dimension; ++i) {
    params.input_strides[i] = input_strides[static_cast<size_t>(i)];
    params.input_dims[i] = input_shape[static_cast<size_t>(i)];
  }

  return LaunchStatus(LaunchMusaScatterNDKernel(
      output.GetTensorMutableRawData(),
      static_cast<const int64_t*>(indices_buffer.data()), updates_buffer.data(),
      static_cast<int32_t>(elem_size), static_cast<int32_t>(elem_type), params,
      stream));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterND, kOnnxDomain, 11, 12,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddTypeConstraint(
             "indices", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))),
    ScatterND)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterND, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddTypeConstraint(
             "indices", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))),
    ScatterND)
