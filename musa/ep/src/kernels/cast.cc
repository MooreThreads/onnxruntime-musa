// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
constexpr int64_t kFloat = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr int64_t kInt32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
constexpr int64_t kInt64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

class Cast : public OpKernelBase<Cast> {
 public:
  Cast(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    to_ = AttrOrDefault<int64_t>(kernel_info, "to", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t to_ = 0;
};

OrtStatus* Cast::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();

  std::vector<uint8_t> in;
  RETURN_IF_ERROR(CopyToHost(input0, in));
  int64_t n = NumElements(shape0);
  Ort::UnownedValue y = ctx.GetOutput(0, shape0);
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && to_ == kInt64) {
    auto x = Span<float>(in);
    std::vector<int64_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      out[static_cast<size_t>(i)] =
          static_cast<int64_t>(x[static_cast<size_t>(i)]);
    return WriteTyped<int64_t>(y, out);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && to_ == kInt32) {
    auto x = Span<float>(in);
    std::vector<int32_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      out[static_cast<size_t>(i)] =
          static_cast<int32_t>(x[static_cast<size_t>(i)]);
    return WriteTyped<int32_t>(y, out);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && to_ == kFloat) {
    auto x = Span<int64_t>(in);
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      out[static_cast<size_t>(i)] =
          static_cast<float>(x[static_cast<size_t>(i)]);
    return WriteTyped<float>(y, out);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && to_ == kFloat) {
    auto x = Span<int32_t>(in);
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      out[static_cast<size_t>(i)] =
          static_cast<float>(x[static_cast<size_t>(i)]);
    return WriteTyped<float>(y, out);
  }
  if ((elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && to_ == kInt32) ||
      (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && to_ == kInt64)) {
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      auto x = Span<int64_t>(in);
      std::vector<int32_t> out(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] =
            static_cast<int32_t>(x[static_cast<size_t>(i)]);
      return WriteTyped<int32_t>(y, out);
    }
    auto x = Span<int32_t>(in);
    std::vector<int64_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      out[static_cast<size_t>(i)] =
          static_cast<int64_t>(x[static_cast<size_t>(i)]);
    return WriteTyped<int64_t>(y, out);
  }
  return CopyFromHost(y, in.data(), in.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T1", AllTensorTypes())
         .AddTypeConstraint("T2", AllTensorTypes())),
    Cast)
