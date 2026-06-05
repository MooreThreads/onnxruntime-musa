// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/where_impl.h"

namespace {

void FillBroadcastStrides(const std::vector<int64_t>& out_shape,
                          const std::vector<int64_t>& input_shape,
                          int64_t* dst) {
  std::fill(dst, dst + kMusaMaxBroadcastRank, 0);
  const size_t rank = out_shape.size();
  const size_t input_rank = input_shape.size();
  const size_t offset = rank - input_rank;
  auto input_strides = Strides(input_shape);
  for (size_t dim = 0; dim < rank; ++dim) {
    if (dim < offset) {
      dst[dim] = 0;
      continue;
    }
    const size_t input_dim = dim - offset;
    dst[dim] = input_shape[input_dim] == 1 ? 0 : input_strides[input_dim];
  }
}

MusaWhereParams MakeWhereParams(const std::vector<int64_t>& out_shape,
                                const std::vector<int64_t>& condition_shape,
                                const std::vector<int64_t>& x_shape,
                                const std::vector<int64_t>& y_shape) {
  MusaWhereParams params{};
  params.rank = static_cast<int32_t>(out_shape.size());
  params.total_elements = NumElements(out_shape);
  auto output_strides = Strides(out_shape);
  for (size_t dim = 0; dim < out_shape.size(); ++dim) {
    params.output_strides[dim] = output_strides[dim];
  }
  FillBroadcastStrides(out_shape, condition_shape, params.condition_strides);
  FillBroadcastStrides(out_shape, x_shape, params.x_strides);
  FillBroadcastStrides(out_shape, y_shape, params.y_strides);
  return params;
}

class Where : public OpKernelBase<Where> {
 public:
  Where(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Where::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue condition = ctx.GetInput(0);
  Ort::ConstValue x = ctx.GetInput(1);
  Ort::ConstValue y_value = ctx.GetInput(2);
  auto condition_info = condition.GetTensorTypeAndShapeInfo();
  auto x_info = x.GetTensorTypeAndShapeInfo();
  auto y_info = y_value.GetTensorTypeAndShapeInfo();
  if (condition_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL ||
      x_info.GetElementType() != y_info.GetElementType()) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Where requires bool condition and matching X/Y dtypes");
  }
  auto condition_shape = condition_info.GetShape();
  auto x_shape = x_info.GetShape();
  auto y_shape = y_info.GetShape();
  auto out_shape = BroadcastShape(BroadcastShape(condition_shape, x_shape),
                                  y_shape);
  if (out_shape.size() > kMusaMaxBroadcastRank ||
      condition_shape.size() > kMusaMaxBroadcastRank ||
      x_shape.size() > kMusaMaxBroadcastRank ||
      y_shape.size() > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Where rank exceeds MUSA kernel limit");
  }

  const size_t elem_size = ElementSize(x_info.GetElementType());
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Where unsupported dtype");
  }
  if (!IsGpuMemory(condition.GetTensorMemoryInfo()) ||
      !IsGpuMemory(x.GetTensorMemoryInfo()) ||
      !IsGpuMemory(y_value.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Where requires MUSA device inputs");
  }

  Ort::UnownedValue output = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Where requires MUSA device output");
  }

  return LaunchStatus(LaunchMusaWhereKernel(
      condition.GetTensorData<uint8_t>(), x.GetTensorRawData(),
      y_value.GetTensorRawData(), output.GetTensorMutableRawData(),
      static_cast<int32_t>(elem_size),
      MakeWhereParams(out_shape, condition_shape, x_shape, y_shape), nullptr));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Where, kOnnxDomain, 9, 15,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint(
             "B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("T", WhereOpset9TensorTypes())),
    Where)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Where, kOnnxDomain, 16, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint(
             "B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())),
    Where)
