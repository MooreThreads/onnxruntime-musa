// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/onehot_impl.h"

namespace {

uint64_t ReadScalarBits(const std::vector<uint8_t>& bytes,
                        size_t element_size,
                        size_t index) {
  uint64_t bits = 0;
  std::memcpy(&bits, bytes.data() + index * element_size, element_size);
  return bits;
}

class OneHot : public OpKernelBase<OneHot> {
 public:
  OneHot(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", -1);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = -1;
};

OrtStatus* OneHot::Compute(Ort::KernelContext& ctx) const {
  musaStream_t stream = GetComputeStream(ctx);
  Ort::ConstValue indices = ctx.GetInput(0);
  Ort::ConstValue depth = ctx.GetInput(1);
  Ort::ConstValue values = ctx.GetInput(2);

  auto indices_info = indices.GetTensorTypeAndShapeInfo();
  auto indices_shape = indices_info.GetShape();
  auto indices_type = indices_info.GetElementType();
  if (indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "OneHot unsupported indices dtype");
  }
  if (!IsGpuMemory(indices.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "OneHot requires MUSA indices");
  }

  auto depth_info = depth.GetTensorTypeAndShapeInfo();
  if (depth_info.GetElementCount() != 1) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "OneHot depth must be a scalar tensor");
  }
  std::vector<int64_t> depth_values = ReadIntTensor(ctx, 1);
  const int64_t depth_value = depth_values.empty() ? 0 : depth_values[0];
  if (depth_value <= 0) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "OneHot depth must be positive");
  }

  auto values_info = values.GetTensorTypeAndShapeInfo();
  auto values_type = values_info.GetElementType();
  if (values_info.GetElementCount() != 2) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "OneHot values must contain two values");
  }
  const size_t element_size = ElementSize(values_type);
  if (element_size == 0 || element_size > sizeof(uint64_t)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "OneHot unsupported values dtype");
  }

  const int64_t output_rank = static_cast<int64_t>(indices_shape.size()) + 1;
  int64_t axis = axis_ < 0 ? axis_ + output_rank : axis_;
  if (axis < 0 || axis >= output_rank) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "OneHot axis is out of range");
  }

  std::vector<int64_t> output_shape;
  output_shape.reserve(indices_shape.size() + 1);
  int64_t suffix = 1;
  for (int64_t i = 0; i < output_rank; ++i) {
    if (i == axis) {
      output_shape.push_back(depth_value);
    } else {
      const size_t indices_dim = static_cast<size_t>(i < axis ? i : i - 1);
      output_shape.push_back(indices_shape[indices_dim]);
      if (i > axis) {
        suffix *= indices_shape[indices_dim];
      }
    }
  }

  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "OneHot requires MUSA output");
  }
  auto output_type = output.GetTensorTypeAndShapeInfo().GetElementType();
  if (output_type != values_type) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "OneHot output dtype must match values dtype");
  }

  std::vector<uint8_t> values_bytes;
  RETURN_IF_ERROR(CopyToHost(values, values_bytes, stream));
  if (values_bytes.size() != 2 * element_size) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "OneHot values byte size does not match dtype");
  }
  const uint64_t off_value = ReadScalarBits(values_bytes, element_size, 0);
  const uint64_t on_value = ReadScalarBits(values_bytes, element_size, 1);

  return LaunchStatus(LaunchMusaOneHotKernel(
      indices.GetTensorRawData(), output.GetTensorMutableRawData(),
      static_cast<int32_t>(indices_type), static_cast<int32_t>(element_size),
      off_value, on_value, depth_value, suffix, NumElements(output_shape),
      stream));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    OneHot, kOnnxDomain, 11, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T1", IntTensorTypes())
         .AddTypeConstraint("T2", IntTensorTypes())
         .AddTypeConstraint("T3", AllFixedSizeTensorTypesNoBFloat16())
         .SetInputMemType(1, OrtMemTypeCPUInput)
         .SetInputMemType(2, OrtMemTypeCPUInput)),
    OneHot)
