// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>

#include "common/op_kernel_common.h"

namespace {
class RandomUniformLike : public OpKernelBase<RandomUniformLike> {
 public:
  RandomUniformLike(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo ki(info);
    // dtype=-1 means use input's dtype
    dtype_ = AttrOrDefault<int64_t>(ki, "dtype", int64_t{-1});
    high_ = AttrOrDefault<float>(ki, "high", 1.0f);
    low_ = AttrOrDefault<float>(ki, "low", 0.0f);
    seed_ = AttrOrDefault<float>(ki, "seed", -1.0f);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t dtype_ = -1;
  float high_ = 1.0f;
  float low_ = 0.0f;
  float seed_ = -1.0f;
};

OrtStatus* RandomUniformLike::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  auto elem_type = dtype_ >= 0 ? static_cast<ONNXTensorElementDataType>(dtype_)
                               : info.GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "RandomUniformLike: only float32 supported");
  int64_t total = NumElements(shape);
  std::mt19937 rng(seed_ >= 0.0f ? static_cast<uint32_t>(seed_)
                                 : std::random_device{}());
  std::uniform_real_distribution<float> dist(low_, high_);
  std::vector<float> out(static_cast<size_t>(total));
  for (auto& v : out) v = dist(rng);
  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  return WriteTyped<float>(y, out);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    RandomUniformLike, kOnnxDomain, 1, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T1", AllTensorTypes())),
    RandomUniformLike)
