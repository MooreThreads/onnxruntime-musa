// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Softmax : public OpKernelBase<Softmax> {
 public:
  Softmax(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
};

OrtStatus* Softmax::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto info = input0.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Softmax only supports float");
  std::vector<int64_t> shape0 = info.GetShape();
  int64_t axis = NormalizeAxis(axis_, shape0.size());
  int64_t outer = 1, dim = shape0[static_cast<size_t>(axis)], inner = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= shape0[static_cast<size_t>(i)];
  for (size_t i = static_cast<size_t>(axis) + 1; i < shape0.size(); ++i)
    inner *= shape0[i];
  std::vector<float> x = ReadTyped<float>(input0);
  std::vector<float> out(x.size());
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t in = 0; in < inner; ++in) {
      float max_v = -std::numeric_limits<float>::infinity();
      for (int64_t d = 0; d < dim; ++d)
        max_v =
            std::max(max_v, x[static_cast<size_t>((o * dim + d) * inner + in)]);
      float sum = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        float e = std::exp(x[static_cast<size_t>((o * dim + d) * inner + in)] -
                           max_v);
        out[static_cast<size_t>((o * dim + d) * inner + in)] = e;
        sum += e;
      }
      for (int64_t d = 0; d < dim; ++d)
        out[static_cast<size_t>((o * dim + d) * inner + in)] /= sum;
    }
  }
  Ort::UnownedValue y = ctx.GetOutput(0, shape0);
  return WriteTyped<float>(y, out);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Softmax, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    Softmax)
