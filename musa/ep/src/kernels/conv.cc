// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class Conv : public OpKernelBase<Conv> {
 public:
  Conv(const OrtKernelInfo* info, void* /*state*/);
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::string auto_pad_;
  std::vector<int64_t> dilations_;
  int64_t group_ = 1;
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
};

Conv::Conv(const OrtKernelInfo* info, void* /*state*/) {
  Ort::ConstKernelInfo ki(info);
  auto_pad_ = AttrOrDefault<std::string>(ki, "auto_pad", std::string("NOTSET"));
  group_ = AttrOrDefault<int64_t>(ki, "group", int64_t{1});
  dilations_ = AttrsOrEmpty(ki, "dilations");
  kernel_shape_ = AttrsOrEmpty(ki, "kernel_shape");
  pads_ = AttrsOrEmpty(ki, "pads");
  strides_ = AttrsOrEmpty(ki, "strides");
}

OrtStatus* Conv::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue X_val = ctx.GetInput(0);
  Ort::ConstValue W_val = ctx.GetInput(1);
  auto x_info = X_val.GetTensorTypeAndShapeInfo();
  if (x_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Conv: only float32 supported");

  auto X_shape = x_info.GetShape();
  auto W_shape = W_val.GetTensorTypeAndShapeInfo().GetShape();
  // Only support 2D spatial (NCHW / MCKK)
  if (X_shape.size() != 4 || W_shape.size() != 4)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Conv: only 2D supported");
  if (auto_pad_ != "NOTSET")
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Conv: only auto_pad=NOTSET supported");

  const int64_t N = X_shape[0], C = X_shape[1], H = X_shape[2],
                W_dim = X_shape[3];
  const int64_t M = W_shape[0];
  const int64_t kH = W_shape[2], kW = W_shape[3];
  const int64_t C_per_group = C / group_;
  const int64_t M_per_group = M / group_;

  // Fill defaults
  int64_t dH = dilations_.size() >= 1 ? dilations_[0] : 1;
  int64_t dW = dilations_.size() >= 2 ? dilations_[1] : 1;
  int64_t sH = strides_.size() >= 1 ? strides_[0] : 1;
  int64_t sW = strides_.size() >= 2 ? strides_[1] : 1;
  int64_t pH = pads_.size() >= 1 ? pads_[0] : 0;
  int64_t pW = pads_.size() >= 2 ? pads_[1] : 0;
  // pads: [top, left, bottom, right]
  int64_t pH_end = pads_.size() >= 3 ? pads_[2] : 0;
  int64_t pW_end = pads_.size() >= 4 ? pads_[3] : 0;

  const int64_t oH = (H + pH + pH_end - dH * (kH - 1) - 1) / sH + 1;
  const int64_t oW = (W_dim + pW + pW_end - dW * (kW - 1) - 1) / sW + 1;

  std::vector<float> X = ReadTyped<float>(X_val);
  std::vector<float> Wt = ReadTyped<float>(W_val);
  bool has_bias = ctx.GetInputCount() >= 3;
  std::vector<float> B;
  if (has_bias) B = ReadTyped<float>(ctx.GetInput(2));

  std::vector<int64_t> out_shape = {N, M, oH, oW};
  std::vector<float> Y(static_cast<size_t>(N * M * oH * oW), 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t m = 0; m < M; ++m) {
      int64_t g = m / M_per_group;
      float bias_val = has_bias ? B[static_cast<size_t>(m)] : 0.0f;
      for (int64_t oh = 0; oh < oH; ++oh) {
        for (int64_t ow = 0; ow < oW; ++ow) {
          float sum = bias_val;
          for (int64_t c = 0; c < C_per_group; ++c) {
            int64_t ic = g * C_per_group + c;
            for (int64_t kh = 0; kh < kH; ++kh) {
              for (int64_t kw = 0; kw < kW; ++kw) {
                int64_t ih = oh * sH + kh * dH - pH;
                int64_t iw = ow * sW + kw * dW - pW;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W_dim) {
                  size_t x_idx = static_cast<size_t>((n * C + ic) * H * W_dim +
                                                     ih * W_dim + iw);
                  // W[m, c, kh, kw]
                  size_t w_idx = static_cast<size_t>(
                      (m * C_per_group + c) * kH * kW + kh * kW + kw);
                  sum += X[x_idx] * Wt[w_idx];
                }
              }
            }
          }
          Y[static_cast<size_t>((n * M + m) * oH * oW + oh * oW + ow)] = sum;
        }
      }
    }
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<float>(y, Y);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv, kOnnxDomain, 11, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())), Conv)
