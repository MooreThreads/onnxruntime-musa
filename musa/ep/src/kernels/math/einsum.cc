// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "math/einsum_impl.h"
#include "shared_inc/op_kernel_common.h"

namespace {

class Einsum : public OpKernelBase<Einsum> {
 public:
  Einsum(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    equation_ = AttrOrDefault<std::string>(kernel_info, "equation", "");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  OrtStatus* ComputeDiagonal(Ort::KernelContext& ctx) const;
  OrtStatus* ComputeBhijHk(Ort::KernelContext& ctx) const;

  std::string equation_;
};

OrtStatus* Einsum::ComputeDiagonal(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() != 2 || shape[0] != shape[1]) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Einsum aa->a requires square rank-2 input");
  }
  const size_t elem_size = ElementSize(info.GetElementType());
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum unsupported dtype");
  }
  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA input");
  }
  Ort::UnownedValue output = ctx.GetOutput(0, {shape[0]});
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA output");
  }
  return LaunchStatus(LaunchMusaEinsumDiagonalKernel(
      input.GetTensorRawData(), output.GetTensorMutableRawData(), shape[0],
      static_cast<int32_t>(elem_size), nullptr));
}

OrtStatus* Einsum::ComputeBhijHk(Ort::KernelContext& ctx) const {
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  auto lhs_info = lhs.GetTensorTypeAndShapeInfo();
  auto rhs_info = rhs.GetTensorTypeAndShapeInfo();
  if (lhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      rhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Einsum bhij,hk->bkij only supports float");
  }
  auto lhs_shape = lhs_info.GetShape();
  auto rhs_shape = rhs_info.GetShape();
  if (lhs_shape.size() != 4 || rhs_shape.size() != 2 ||
      lhs_shape[1] != rhs_shape[0]) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Einsum bhij,hk->bkij shape mismatch");
  }
  if (!IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA inputs");
  }

  std::vector<int64_t> output_shape = {
      lhs_shape[0], rhs_shape[1], lhs_shape[2], lhs_shape[3]};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA output");
  }
  return LaunchStatus(LaunchMusaEinsumBhijHkKernel(
      lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
      output.GetTensorMutableData<float>(), lhs_shape[0], lhs_shape[1],
      rhs_shape[1], lhs_shape[2], lhs_shape[3], nullptr));
}

OrtStatus* Einsum::Compute(Ort::KernelContext& ctx) const {
  if (equation_ == "aa->a") {
    return ComputeDiagonal(ctx);
  }
  if (equation_ == "bhij,hk->bkij") {
    return ComputeBhijHk(ctx);
  }
  const std::string message =
      "unsupported Einsum equation for MUSA device path: " + equation_;
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message.c_str());
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Einsum, kOnnxDomain, 12, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    Einsum)
