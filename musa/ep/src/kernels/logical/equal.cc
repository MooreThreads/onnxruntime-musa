// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
constexpr size_t kMudnnMaxElementwiseRank = 5;

bool TryMudnnEqual(Ort::KernelContext& ctx, const std::vector<int64_t>& shape0,
                   const std::vector<int64_t>& shape1,
                   ONNXTensorElementDataType elem_type) {
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  if (out_shape.size() > kMudnnMaxElementwiseRank ||
      shape0.size() > kMudnnMaxElementwiseRank ||
      shape1.size() > kMudnnMaxElementwiseRank ||
      !IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
    return false;
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, GetComputeStream(ctx));
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor lhs_tensor;
  ::musa::dnn::Tensor rhs_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnTensor(lhs_tensor, lhs.GetTensorRawData(), shape0, elem_type) ||
      !SetMudnnTensor(rhs_tensor, rhs.GetTensorRawData(), shape1, elem_type) ||
      !SetMudnnTensor(output_tensor, y.GetTensorMutableRawData(), out_shape,
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)) {
    return false;
  }

  ::musa::dnn::Binary op;
  if (op.SetMode(::musa::dnn::Binary::Mode::EQ) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, lhs_tensor, rhs_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

class Equal : public OpKernelBase<Equal> {
 public:
  Equal(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

class EqualString : public OpKernelBase<EqualString> {
 public:
  EqualString(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Equal::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape0 = info.GetShape();
  auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  if (TryMudnnEqual(ctx, shape0, shape1, elem_type)) {
    return nullptr;
  }
  return CompareDeviceCompute(ctx, shape0, shape1, elem_type,
                              MusaCompareOp::Equal, "Equal");
}

OrtStatus* EqualString::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  auto lhs_info = lhs.GetTensorTypeAndShapeInfo();
  auto rhs_info = rhs.GetTensorTypeAndShapeInfo();
  if (lhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING ||
      rhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "EqualString requires string inputs");
  }

  const std::vector<int64_t> lhs_shape = lhs_info.GetShape();
  const std::vector<int64_t> rhs_shape = rhs_info.GetShape();
  const std::vector<int64_t> out_shape = BroadcastShape(lhs_shape, rhs_shape);
  Ort::UnownedValue output = ctx.GetOutput(0, out_shape);
  if (IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      IsGpuMemory(rhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "EqualString requires CPU string inputs and MUSA bool output");
  }

  const int64_t total = NumElements(out_shape);
  const std::vector<int64_t> lhs_strides = Strides(lhs_shape);
  const std::vector<int64_t> rhs_strides = Strides(rhs_shape);
  std::vector<uint8_t> out(static_cast<size_t>(total));
  for (int64_t i = 0; i < total; ++i) {
    const std::vector<int64_t> coord = Coordinates(i, out_shape);
    const int64_t lhs_offset = BroadcastOffset(coord, lhs_shape, lhs_strides);
    const int64_t rhs_offset = BroadcastOffset(coord, rhs_shape, rhs_strides);
    out[static_cast<size_t>(i)] =
        lhs.GetStringTensorElement(static_cast<size_t>(lhs_offset)) ==
        rhs.GetStringTensorElement(static_cast<size_t>(rhs_offset));
  }
  return CopyFromHost(output, out.data(), out.size(), GetComputeStream(ctx));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Equal, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", EqualTensorTypes())), Equal)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Equal, kOnnxDomain, 19, 19,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T",
                                                          StringTensorTypes())
                                       .SetInputMemType(0, OrtMemTypeCPUInput)
                                       .SetInputMemType(1, OrtMemTypeCPUInput)),
                                  EqualString)
