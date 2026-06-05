// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/range_impl.h"
#include "shared_inc/op_kernel_common.h"

namespace {

template <typename T>
bool ReadScalar(Ort::ConstValue value, T& out) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1) {
    return false;
  }
  std::vector<T> data = ReadTyped<T>(value);
  if (data.size() != 1) {
    return false;
  }
  out = data[0];
  return true;
}

template <typename T>
OrtStatus* ComputeRangeTyped(Ort::KernelContext& ctx,
                             ONNXTensorElementDataType elem_type) {
  T start{};
  T limit{};
  T delta{};
  if (!ReadScalar<T>(ctx.GetInput(0), start) ||
      !ReadScalar<T>(ctx.GetInput(1), limit) ||
      !ReadScalar<T>(ctx.GetInput(2), delta)) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Range inputs must be scalar tensors");
  }
  if (delta == static_cast<T>(0)) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Range delta must not be zero");
  }

  double count_value =
      std::ceil((static_cast<double>(limit) - static_cast<double>(start)) /
                static_cast<double>(delta));
  int64_t count = count_value > 0.0 ? static_cast<int64_t>(count_value) : 0;
  Ort::UnownedValue output = ctx.GetOutput(0, {count});
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Range requires MUSA output");
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Range unsupported dtype");
  }
  MusaRangeParams params{};
  params.count = count;
  return LaunchStatus(LaunchMusaRangeKernel(
      output.GetTensorMutableRawData(), params, musa_elem_type,
      static_cast<double>(start), static_cast<double>(delta), nullptr));
}

class Range : public OpKernelBase<Range> {
 public:
  Range(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Range::Compute(Ort::KernelContext& ctx) const {
  auto elem_type = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetElementType();
  if (ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetElementType() !=
          elem_type ||
      ctx.GetInput(2).GetTensorTypeAndShapeInfo().GetElementType() !=
          elem_type) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Range input dtypes must match");
  }
  switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return ComputeRangeTyped<int16_t>(ctx, elem_type);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return ComputeRangeTyped<int32_t>(ctx, elem_type);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return ComputeRangeTyped<int64_t>(ctx, elem_type);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return ComputeRangeTyped<float>(ctx, elem_type);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return ComputeRangeTyped<double>(ctx, elem_type);
    default:
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "Range unsupported dtype");
  }
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Range, kOnnxDomain, 11, 19,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T",
                                                          RangeTensorTypes())
                                       .SetInputMemType(0, OrtMemTypeCPUInput)
                                       .SetInputMemType(1, OrtMemTypeCPUInput)
                                       .SetInputMemType(2, OrtMemTypeCPUInput)),
                                  Range)
