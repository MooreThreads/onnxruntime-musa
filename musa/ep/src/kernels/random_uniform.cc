// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>

#include "common/op_kernel_common.h"

namespace {
class RandomUniform : public OpKernelBase<RandomUniform> {
 public:
  RandomUniform(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo ki(info);
    dtype_ = AttrOrDefault<int64_t>(ki, "dtype", int64_t{1});  // 1 = float32
    high_ = AttrOrDefault<float>(ki, "high", 1.0f);
    low_ = AttrOrDefault<float>(ki, "low", 0.0f);
    seed_ = AttrOrDefault<float>(ki, "seed", -1.0f);
    shape_ = AttrsOrEmpty(ki, "shape");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t dtype_ = 1;
  float high_ = 1.0f;
  float low_ = 0.0f;
  float seed_ = -1.0f;
  std::vector<int64_t> shape_;
};

OrtStatus* RandomUniform::Compute(Ort::KernelContext& ctx) const {
  if (dtype_ != 1)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "RandomUniform: only float32 supported");
  int64_t total = NumElements(shape_);
  std::mt19937 rng(seed_ >= 0.0f ? static_cast<uint32_t>(seed_)
                                 : std::random_device{}());
  std::uniform_real_distribution<float> dist(low_, high_);
  std::vector<float> out(static_cast<size_t>(total));
  for (auto& v : out) v = dist(rng);
  Ort::UnownedValue y = ctx.GetOutput(0, shape_);
  return WriteTyped<float>(y, out);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    RandomUniform, kOnnxDomain, 1, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    RandomUniform)
