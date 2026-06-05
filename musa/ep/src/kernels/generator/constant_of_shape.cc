// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/constant_of_shape_impl.h"
#include "shared_inc/op_kernel_common.h"

namespace {

class ConstantOfShape : public OpKernelBase<ConstantOfShape> {
 public:
  ConstantOfShape(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    InitValue(kernel_info);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  void InitValue(Ort::ConstKernelInfo& kernel_info) {
    float default_value = 0.0f;
    value_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    value_bits_ = 0;
    value_size_ = sizeof(default_value);
    std::memcpy(&value_bits_, &default_value, sizeof(default_value));

    Ort::AllocatorWithDefaultOptions allocator;
    try {
      Ort::Value value_tensor =
          kernel_info.GetTensorAttribute("value", allocator);
      auto info = value_tensor.GetTensorTypeAndShapeInfo();
      const size_t elem_count = info.GetElementCount();
      if (elem_count == 0) {
        return;
      }
      if (elem_count != 1) {
        throw std::runtime_error(
            "ConstantOfShape value attribute must be a scalar tensor");
      }
      value_type_ = info.GetElementType();
      value_size_ = ElementSize(value_type_);
      if (value_size_ == 0 || value_size_ > sizeof(uint64_t)) {
        throw std::runtime_error("ConstantOfShape unsupported value dtype");
      }
      value_bits_ = 0;
      std::memcpy(&value_bits_, value_tensor.GetTensorRawData(), value_size_);
    } catch (const Ort::Exception&) {
      return;
    }
  }

  ONNXTensorElementDataType value_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  uint64_t value_bits_ = 0;
  size_t value_size_ = sizeof(float);
};

OrtStatus* ConstantOfShape::Compute(Ort::KernelContext& ctx) const {
  std::vector<int64_t> output_shape = ReadIntTensor(ctx, 0);
  for (int64_t dim : output_shape) {
    if (dim < 0) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "ConstantOfShape requires non-negative output dimensions");
    }
  }

  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "ConstantOfShape requires MUSA output");
  }
  return LaunchStatus(LaunchMusaConstantOfShapeKernel(
      output.GetTensorMutableRawData(), value_bits_,
      static_cast<int32_t>(value_size_), NumElements(output_shape), nullptr));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ConstantOfShape, kOnnxDomain, 9, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T1",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("T2", AllFixedSizeTensorTypesNoBFloat16())),
    ConstantOfShape)
