// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
constexpr size_t kMudnnMaxElementwiseRank = 5;

bool TryMudnnLeakyRelu(Ort::KernelContext& ctx,
                       const std::vector<int64_t>& shape, float alpha,
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
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, GetComputeStream(ctx));
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
  if (op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU) !=
          ::musa::dnn::Status::SUCCESS ||
      op.SetAlpha(static_cast<double>(alpha)) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, input_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

class LeakyRelu : public OpKernelBase<LeakyRelu> {
 public:
  LeakyRelu(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    alpha_ = AttrOrDefault<float>(kernel_info, "alpha", 0.01f);
    alpha_ = AttrOrDefault<float>(kernel_info, "activation_alpha", alpha_);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  float alpha_ = 0.01f;
};

OrtStatus* LeakyRelu::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape = info.GetShape();
  float alpha = alpha_;
  if (OutputEmptyTensorIfNeeded(ctx, shape)) {
    return nullptr;
  }
  if (TryMudnnLeakyRelu(ctx, shape, alpha, elem_type)) {
    return nullptr;
  }
  return UnaryDeviceCompute(ctx, shape, elem_type, MusaUnaryOp::LeakyRelu,
                            "LeakyRelu", alpha);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LeakyRelu, kOnnxDomain, 6, 15,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())),
    LeakyRelu)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LeakyRelu, kOnnxDomain, 16, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    LeakyRelu)
