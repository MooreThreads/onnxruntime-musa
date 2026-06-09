// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
constexpr size_t kMudnnMaxElementwiseRank = 5;

bool TryMudnnGreater(Ort::KernelContext& ctx,
                     const std::vector<int64_t>& shape0,
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
  if (op.SetMode(::musa::dnn::Binary::Mode::GT) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, lhs_tensor, rhs_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

class Greater : public OpKernelBase<Greater> {
 public:
  Greater(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Greater::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape0 = info.GetShape();
  auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  if (TryMudnnGreater(ctx, shape0, shape1, elem_type)) {
    return nullptr;
  }
  return CompareDeviceCompute(ctx, shape0, shape1, elem_type,
                              MusaCompareOp::Greater, "Greater");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Greater, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", CompareTensorTypes())),
    Greater)
