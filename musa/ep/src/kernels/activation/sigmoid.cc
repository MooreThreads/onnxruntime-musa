// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
constexpr size_t kMudnnMaxElementwiseRank = 5;

bool TryMudnnSigmoid(Ort::KernelContext& ctx,
                     const std::vector<int64_t>& shape,
                     ONNXTensorElementDataType elem_type) {
  Ort::ConstValue input = ctx.GetInput(0);
  if (shape.size() > kMudnnMaxElementwiseRank ||
      !IsGpuMemory(input.GetTensorMemoryInfo())) {
    return false;
  }

  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnTensor(input_tensor, input.GetTensorRawData(), shape,
                      elem_type) ||
      !SetMudnnTensor(output_tensor, y.GetTensorMutableRawData(), shape,
                      elem_type)) {
    return false;
  }

  ::musa::dnn::Unary op;
  if (op.SetMode(::musa::dnn::Unary::Mode::SIGMOID) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, input_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

class Sigmoid : public OpKernelBase<Sigmoid> {
 public:
  Sigmoid(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Sigmoid::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  if (TryMudnnSigmoid(ctx, info.GetShape(), elem_type)) {
    return nullptr;
  }
  return UnaryDeviceCompute(ctx, info.GetShape(), elem_type,
                            MusaUnaryOp::Sigmoid, "Sigmoid");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sigmoid, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    Sigmoid)
