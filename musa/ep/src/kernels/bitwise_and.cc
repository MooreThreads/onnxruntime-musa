// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class BitwiseAnd : public OpKernelBase<BitwiseAnd> {
 public:
  BitwiseAnd(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* BitwiseAnd::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape0 = info.GetShape();
  auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    return BinaryCompute<int32_t>(ctx, shape0, shape1,
                                  [](int32_t a, int32_t b) { return a & b; });
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    return BinaryCompute<int64_t>(ctx, shape0, shape1,
                                  [](int64_t a, int64_t b) { return a & b; });
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                    "BitwiseAnd: unsupported dtype");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    BitwiseAnd, kOnnxDomain, 18, 18,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", IntTensorTypes())),
    BitwiseAnd)
