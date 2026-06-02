// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "common/op_kernel_common.h"

namespace {

// Shared host-side ReduceMax used for float, int32, int64.
template <typename T>
OrtStatus* ReduceMaxCompute(Ort::KernelContext& ctx, std::vector<int64_t> axes,
                            bool keepdims) {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto shape0 = input0.GetTensorTypeAndShapeInfo().GetShape();
  if (ctx.GetInputCount() > 1) axes = ReadIntTensor(ctx, 1);
  auto axes_set = AxesSet(axes, shape0.size());

  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < shape0.size(); ++i) {
    if (axes_set.count(static_cast<int64_t>(i))) {
      if (keepdims) out_shape.push_back(1);
    } else {
      out_shape.push_back(shape0[i]);
    }
  }
  if (out_shape.empty()) out_shape.push_back(1);
  auto out_strides = Strides(out_shape);

  std::vector<T> x = ReadTyped<T>(input0);
  constexpr T kMin = std::numeric_limits<T>::lowest();
  std::vector<T> out(static_cast<size_t>(NumElements(out_shape)), kMin);

  for (int64_t i = 0; i < NumElements(shape0); ++i) {
    auto ic = Coordinates(i, shape0);
    std::vector<int64_t> oc;
    for (size_t d = 0; d < shape0.size(); ++d) {
      if (axes_set.count(static_cast<int64_t>(d))) {
        if (keepdims) oc.push_back(0);
      } else {
        oc.push_back(ic[d]);
      }
    }
    if (oc.empty()) oc.push_back(0);
    int64_t oo = Offset(oc, out_strides);
    T val = x[static_cast<size_t>(i)];
    if (val > out[static_cast<size_t>(oo)]) out[static_cast<size_t>(oo)] = val;
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<T>(y, out);
}

class ReduceMax : public OpKernelBase<ReduceMax> {
 public:
  ReduceMax(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo ki(info);
    keepdims_ = AttrOrDefault<int64_t>(ki, "keepdims", 1);
    axes_attr_ = AttrsOrEmpty(ki, "axes");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t keepdims_ = 1;
  std::vector<int64_t> axes_attr_;
};

OrtStatus* ReduceMax::Compute(Ort::KernelContext& ctx) const {
  auto elem_type = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetElementType();
  bool kd = keepdims_ != 0;
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return ReduceMaxCompute<float>(ctx, axes_attr_, kd);
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    return ReduceMaxCompute<int64_t>(ctx, axes_attr_, kd);
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    return ReduceMaxCompute<int32_t>(ctx, axes_attr_, kd);
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                    "ReduceMax: unsupported dtype");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ReduceMax, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())),
    ReduceMax)
