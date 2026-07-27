// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nn/conv_impl.h"
#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {

std::vector<int64_t> NormalizePair(std::vector<int64_t> values,
                                   int64_t default_value) {
  if (values.empty()) {
    return {default_value, default_value};
  }
  if (values.size() == 1) {
    return {values[0], values[0]};
  }
  return {values[0], values[1]};
}

class Conv : public OpKernelBase<Conv> {
 public:
  Conv(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    pads_ = AttrsOrEmpty(kernel_info, "pads");
    strides_ = NormalizePair(AttrsOrEmpty(kernel_info, "strides"), 1);
    dilations_ = NormalizePair(AttrsOrEmpty(kernel_info, "dilations"), 1);
    group_ = AttrOrDefault<int64_t>(kernel_info, "group", 1);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  bool TryMudnnConv(Ort::KernelContext& ctx,
                    const std::vector<int64_t>& x_shape,
                    const std::vector<int64_t>& w_shape,
                    const std::vector<int64_t>& y_shape,
                    Ort::UnownedValue output) const;

  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilations_;
  int64_t group_ = 1;
};

bool Conv::TryMudnnConv(Ort::KernelContext& ctx,
                        const std::vector<int64_t>& x_shape,
                        const std::vector<int64_t>& w_shape,
                        const std::vector<int64_t>& y_shape,
                        Ort::UnownedValue output) const {
  // Conv1D exported as NCHW Conv with H=1 and kH=1 is covered by the custom
  // device fallback; avoid a muDNN path that is unstable for this layout.
  if (x_shape[2] == 1 && w_shape[2] == 1) {
    return false;
  }

  Ort::ConstValue x = ctx.GetInput(0);
  Ort::ConstValue w = ctx.GetInput(1);
  if (!IsGpuMemory(x.GetTensorMemoryInfo()) ||
      !IsGpuMemory(w.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, GetComputeStream(ctx));
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor x_tensor;
  ::musa::dnn::Tensor w_tensor;
  ::musa::dnn::Tensor y_tensor;
  if (!SetMudnnTensor(x_tensor, x.GetTensorRawData(), x_shape,
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !SetMudnnTensor(w_tensor, w.GetTensorRawData(), w_shape,
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !SetMudnnTensor(y_tensor, output.GetTensorMutableRawData(), y_shape,
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }

  const int pad[2] = {static_cast<int>(pads_[0]), static_cast<int>(pads_[1])};
  const int stride[2] = {static_cast<int>(strides_[0]),
                         static_cast<int>(strides_[1])};
  const int dilation[2] = {static_cast<int>(dilations_[0]),
                           static_cast<int>(dilations_[1])};
  ::musa::dnn::Convolution conv;
  if (conv.SetGroups(static_cast<int>(group_)) !=
          ::musa::dnn::Status::SUCCESS ||
      conv.SetNdInfo(2, pad, stride, dilation) !=
          ::musa::dnn::Status::SUCCESS ||
      conv.SetComputeMode(::musa::dnn::Convolution::ComputeMode::TENSOR) !=
          ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  return conv.Run(*handle, y_tensor, x_tensor, w_tensor,
                  ::musa::dnn::Convolution::Algorithm::IMPLICIT_GEMM,
                  nullptr) == ::musa::dnn::Status::SUCCESS;
}

OrtStatus* Conv::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue x = ctx.GetInput(0);
  Ort::ConstValue w = ctx.GetInput(1);
  auto x_info = x.GetTensorTypeAndShapeInfo();
  auto w_info = w.GetTensorTypeAndShapeInfo();
  if (x_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      w_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Conv only supports float");
  }
  auto x_shape = x_info.GetShape();
  auto w_shape = w_info.GetShape();
  if (x_shape.size() != 4 || w_shape.size() != 4) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Conv only supports 2D NCHW tensors");
  }
  if (group_ != 1 || w_shape[1] != x_shape[1]) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Conv only supports group=1");
  }

  std::vector<int64_t> pads = pads_;
  if (pads.empty()) {
    pads = {0, 0, 0, 0};
  }
  if (pads.size() != 4 || pads[0] != pads[2] || pads[1] != pads[3]) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Conv only supports symmetric 2D pads");
  }

  const int64_t out_h =
      (x_shape[2] + pads[0] + pads[2] - dilations_[0] * (w_shape[2] - 1) - 1) /
          strides_[0] +
      1;
  const int64_t out_w =
      (x_shape[3] + pads[1] + pads[3] - dilations_[1] * (w_shape[3] - 1) - 1) /
          strides_[1] +
      1;
  if (out_h < 0 || out_w < 0) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Conv invalid output shape");
  }

  std::vector<int64_t> y_shape = {x_shape[0], w_shape[0], out_h, out_w};
  Ort::UnownedValue output = ctx.GetOutput(0, y_shape);
  if (TryMudnnConv(ctx, x_shape, w_shape, y_shape, output)) {
    return nullptr;
  }

  if (!IsGpuMemory(x.GetTensorMemoryInfo()) ||
      !IsGpuMemory(w.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Conv requires MUSA tensors");
  }

  const float* bias = nullptr;
  if (ctx.GetInputCount() > 2 && ctx.GetInput(2) != nullptr) {
    Ort::ConstValue b = ctx.GetInput(2);
    auto b_info = b.GetTensorTypeAndShapeInfo();
    if (b_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        b_info.GetShape().size() != 1 || b_info.GetShape()[0] != w_shape[0] ||
        !IsGpuMemory(b.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "Conv unsupported bias");
    }
    bias = b.GetTensorData<float>();
  }

  MusaConv2DParams params{};
  params.n = x_shape[0];
  params.c = x_shape[1];
  params.h = x_shape[2];
  params.w = x_shape[3];
  params.m = w_shape[0];
  params.kernel_h = w_shape[2];
  params.kernel_w = w_shape[3];
  params.out_h = out_h;
  params.out_w = out_w;
  params.pad_h = pads[0];
  params.pad_w = pads[1];
  params.stride_h = strides_[0];
  params.stride_w = strides_[1];
  params.dilation_h = dilations_[0];
  params.dilation_w = dilations_[1];
  params.total_elements = NumElements(y_shape);
  return LaunchStatus(LaunchMusaConv2DFloatKernel(
      x.GetTensorData<float>(), w.GetTensorData<float>(), bias,
      output.GetTensorMutableData<float>(), params, GetComputeStream(ctx)));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv, kOnnxDomain, 11, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())), Conv)
