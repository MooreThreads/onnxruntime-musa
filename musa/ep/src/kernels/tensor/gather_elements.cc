// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/gather_elements_impl.h"

namespace {

class GatherElements : public OpKernelBase<GatherElements> {
 public:
  GatherElements(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
};

OrtStatus* ValidateGatherElementsShapes(
    const std::vector<int64_t>& data_shape,
    const std::vector<int64_t>& indices_shape, int64_t axis) {
  if (data_shape.empty()) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "GatherElements cannot operate on scalar input");
  }
  if (data_shape.size() != indices_shape.size()) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "GatherElements data and indices ranks must match");
  }
  if (data_shape.size() > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "GatherElements rank exceeds MUSA device limit");
  }
  for (size_t i = 0; i < data_shape.size(); ++i) {
    if (data_shape[i] < 0 || indices_shape[i] < 0) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "GatherElements requires concrete shapes");
    }
    if (static_cast<int64_t>(i) != axis && indices_shape[i] > data_shape[i]) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "GatherElements indices shape is out of data shape bounds");
    }
  }
  return nullptr;
}

OrtStatus* GatherElements::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue data = ctx.GetInput(0);
  Ort::ConstValue indices = ctx.GetInput(1);
  auto data_info = data.GetTensorTypeAndShapeInfo();
  auto indices_info = indices.GetTensorTypeAndShapeInfo();
  auto data_shape = data_info.GetShape();
  auto indices_shape = indices_info.GetShape();

  if (data_shape.empty()) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "GatherElements cannot operate on scalar input");
  }
  int64_t axis = NormalizeAxis(axis_, data_shape.size());
  if (axis < 0 || axis >= static_cast<int64_t>(data_shape.size())) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "GatherElements axis out of range");
  }
  RETURN_IF_ERROR(
      ValidateGatherElementsShapes(data_shape, indices_shape, axis));

  auto indices_type = indices_info.GetElementType();
  if (indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "GatherElements only supports int32/int64 indices");
  }

  const size_t elem_size = ElementSize(data_info.GetElementType());
  const size_t index_elem_size = ElementSize(indices_type);
  if (elem_size == 0 || index_elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "GatherElements unsupported dtype");
  }

  Ort::UnownedValue output = ctx.GetOutput(0, indices_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "GatherElements requires MUSA output");
  }

  musaStream_t stream = GetComputeStream(ctx);
  DeviceInputBuffer data_buffer;
  DeviceInputBuffer indices_buffer;
  RETURN_IF_ERROR(data_buffer.Bind(data, stream));
  RETURN_IF_ERROR(indices_buffer.Bind(indices, stream));

  MusaGatherElementsParams params{};
  params.rank = static_cast<int32_t>(data_shape.size());
  params.axis = static_cast<int32_t>(axis);
  params.output_elements = NumElements(indices_shape);
  auto data_strides = Strides(data_shape);
  auto indices_strides = Strides(indices_shape);
  for (size_t i = 0; i < data_shape.size(); ++i) {
    params.data_dims[i] = data_shape[i];
    params.data_strides[i] = data_strides[i];
    params.indices_strides[i] = indices_strides[i];
  }

  return LaunchStatus(LaunchMusaGatherElementsKernel(
      data_buffer.data(), indices_buffer.data(),
      output.GetTensorMutableRawData(), static_cast<int32_t>(elem_size),
      static_cast<int32_t>(index_elem_size), params, stream));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GatherElements, kOnnxDomain, 11, 12,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddTypeConstraint("Tind", IntTensorTypes())),
    GatherElements)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GatherElements, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddTypeConstraint("Tind", IntTensorTypes())),
    GatherElements)
