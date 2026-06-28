// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "shared_inc/op_kernel_common.h"
#include "tensor/scatter_elements_impl.h"

namespace {

constexpr int32_t kScatterElementsReductionNone = 0;
constexpr int32_t kScatterElementsReductionAdd = 1;

bool IsScatterElementsAddType(ONNXTensorElementDataType elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

OrtStatus* ValidateScatterElementsShapes(
    const std::vector<int64_t>& data_shape,
    const std::vector<int64_t>& indices_shape,
    const std::vector<int64_t>& updates_shape, int64_t axis) {
  if (data_shape.empty()) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "ScatterElements data rank must be larger than 0");
  }
  if (data_shape.size() != indices_shape.size() ||
      data_shape.size() != updates_shape.size()) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "ScatterElements data, indices, and updates ranks must match");
  }
  if (data_shape.size() > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ScatterElements rank exceeds MUSA device limit");
  }
  for (size_t i = 0; i < data_shape.size(); ++i) {
    if (data_shape[i] < 0 || indices_shape[i] < 0 || updates_shape[i] < 0) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "ScatterElements requires concrete input shapes");
    }
    if (indices_shape[i] != updates_shape[i]) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "ScatterElements indices and updates shapes must match");
    }
    if (static_cast<int64_t>(i) != axis && indices_shape[i] > data_shape[i]) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "ScatterElements indices shape exceeds data shape");
    }
  }
  return nullptr;
}

class ScatterElements : public OpKernelBase<ScatterElements> {
 public:
  ScatterElements(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
    reduction_ = AttrOrDefault<std::string>(kernel_info, "reduction", "none");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
  std::string reduction_;
};

OrtStatus* ScatterElements::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue data = ctx.GetInput(0);
  Ort::ConstValue indices = ctx.GetInput(1);
  Ort::ConstValue updates = ctx.GetInput(2);
  auto data_info = data.GetTensorTypeAndShapeInfo();
  auto indices_info = indices.GetTensorTypeAndShapeInfo();
  auto updates_info = updates.GetTensorTypeAndShapeInfo();
  auto data_shape = data_info.GetShape();
  auto indices_shape = indices_info.GetShape();
  auto updates_shape = updates_info.GetShape();
  int64_t axis = NormalizeAxis(axis_, data_shape.size());

  const auto elem_type = data_info.GetElementType();
  if (updates_info.GetElementType() != elem_type) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "ScatterElements input dtypes must match");
  }
  const auto index_type = indices_info.GetElementType();
  const size_t index_elem_size = ElementSize(index_type);
  if (index_elem_size != sizeof(int32_t) &&
      index_elem_size != sizeof(int64_t)) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ScatterElements unsupported index dtype");
  }
  const size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "ScatterElements unsupported dtype");
  }

  int32_t reduction = kScatterElementsReductionNone;
  if (reduction_ == "add") {
    if (!IsScatterElementsAddType(elem_type)) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "ScatterElements reduction='add' unsupported dtype");
    }
    reduction = kScatterElementsReductionAdd;
  } else if (reduction_ != "none") {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "ScatterElements MUSA only supports reduction='none' or 'add'");
  }

  RETURN_IF_ERROR(ValidateScatterElementsShapes(data_shape, indices_shape,
                                                updates_shape, axis));

  Ort::UnownedValue output = ctx.GetOutput(0, data_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ScatterElements requires MUSA output tensor");
  }

  musaStream_t stream = GetComputeStream(ctx);
  RETURN_IF_ERROR(
      CopyRawTensor(data, output, data.GetTensorSizeInBytes(), stream));

  DeviceInputBuffer indices_buffer;
  DeviceInputBuffer updates_buffer;
  RETURN_IF_ERROR(indices_buffer.Bind(indices, stream));
  RETURN_IF_ERROR(updates_buffer.Bind(updates, stream));

  MusaScatterElementsParams params{};
  params.rank = static_cast<int32_t>(data_shape.size());
  params.axis = static_cast<int32_t>(axis);
  params.reduction = reduction;
  params.updates_elements = NumElements(updates_shape);
  auto data_strides = Strides(data_shape);
  auto updates_strides = Strides(updates_shape);
  for (size_t i = 0; i < data_shape.size(); ++i) {
    params.data_dims[i] = data_shape[i];
    params.data_strides[i] = data_strides[i];
    params.updates_strides[i] = updates_strides[i];
  }

  return LaunchStatus(LaunchMusaScatterElementsKernel(
      output.GetTensorMutableRawData(), indices_buffer.data(),
      updates_buffer.data(), static_cast<int32_t>(elem_size),
      static_cast<int32_t>(index_elem_size), static_cast<int32_t>(elem_type),
      params, stream));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterElements, kOnnxDomain, 11, 12,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddTypeConstraint("Tind", IntTensorTypes())),
    ScatterElements)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterElements, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddTypeConstraint("Tind", IntTensorTypes())),
    ScatterElements)
