// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class ConstantOfShape : public OpKernelBase<ConstantOfShape> {
 public:
  ConstantOfShape(const OrtKernelInfo* info, void* /*state*/);
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  ONNXTensorElementDataType dtype_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  std::vector<uint8_t> value_bytes_;  // bytes of the 1-element fill value
};

ConstantOfShape::ConstantOfShape(const OrtKernelInfo* info, void* /*state*/) {
  // Default: float 0.0
  dtype_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  value_bytes_.resize(sizeof(float), 0);

  try {
    Ort::ConstKernelInfo ki(info);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Value val = ki.GetTensorAttribute("value", allocator);
    auto tinfo = val.GetTensorTypeAndShapeInfo();
    dtype_ = tinfo.GetElementType();
    size_t nbytes = val.GetTensorSizeInBytes();
    value_bytes_.resize(nbytes);
    std::memcpy(value_bytes_.data(), val.GetTensorRawData(), nbytes);
  } catch (...) {
    // attribute not present – keep defaults
  }
}

OrtStatus* ConstantOfShape::Compute(Ort::KernelContext& ctx) const {
  std::vector<int64_t> shape = ReadIntTensor(ctx, 0);
  int64_t total = NumElements(shape);
  size_t elem_size = value_bytes_.size();
  std::vector<uint8_t> out(static_cast<size_t>(total) * elem_size);
  for (int64_t i = 0; i < total; ++i) {
    std::memcpy(out.data() + static_cast<size_t>(i) * elem_size,
                value_bytes_.data(), elem_size);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  return CopyFromHost(y, out.data(), out.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ConstantOfShape, kOnnxDomain, 9, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T2", AllTensorTypes())),
    ConstantOfShape)
