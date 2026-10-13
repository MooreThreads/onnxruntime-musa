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

#include <cstdlib>

#include "nn/conv_impl.h"
#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {

bool SetMudnnConvTensor(::musa::dnn::Tensor& tensor, const void* data,
                        const std::vector<int64_t>& shape) {
  if (shape.size() != 4 && shape.size() != 5) {
    return false;
  }
  if (tensor.SetAddr(data) != ::musa::dnn::Status::SUCCESS ||
      tensor.SetType(::musa::dnn::Tensor::Type::FLOAT) !=
          ::musa::dnn::Status::SUCCESS ||
      tensor.SetFormat(shape.size() == 5 ? ::musa::dnn::Tensor::Format::NCDHW
                                         : ::musa::dnn::Tensor::Format::NCHW) !=
          ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return tensor.SetNdInfo(static_cast<int>(shape.size()), shape.data()) ==
         ::musa::dnn::Status::SUCCESS;
}

class Conv : public OpKernelBase<Conv> {
 public:
  Conv(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    pads_ = AttrsOrEmpty(kernel_info, "pads");
    strides_ = AttrsOrEmpty(kernel_info, "strides");
    dilations_ = AttrsOrEmpty(kernel_info, "dilations");
    group_ = AttrOrDefault<int64_t>(kernel_info, "group", 1);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  bool TryMudnnConv(
      Ort::KernelContext& ctx, const std::vector<int64_t>& x_shape,
      const std::vector<int64_t>& w_shape, const std::vector<int64_t>& y_shape,
      const std::vector<int64_t>& pads, const std::vector<int64_t>& strides,
      const std::vector<int64_t>& dilations, Ort::UnownedValue output) const;

  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilations_;
  int64_t group_ = 1;
};

bool Conv::TryMudnnConv(
    Ort::KernelContext& ctx, const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& w_shape, const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& pads, const std::vector<int64_t>& strides,
    const std::vector<int64_t>& dilations, Ort::UnownedValue output) const {
  // Conv1D exported as NCHW Conv with H=1 and kH=1 is covered by the custom
  // device fallback; avoid a muDNN path that is unstable for this layout.
  if (x_shape.size() == 4 && x_shape[2] == 1 && w_shape[2] == 1) {
    return false;
  }
  if (x_shape.size() == 5 &&
      std::getenv("ORT_MUSA_CONV3D_FORCE_MUDNN_FAILURE_FOR_TEST") != nullptr) {
    return false;
  }

  Ort::ConstValue x = ctx.GetInput(0);
  Ort::ConstValue w = ctx.GetInput(1);
  const bool has_bias = ctx.GetInputCount() > 2 && ctx.GetInput(2) != nullptr;
  if (!IsGpuMemory(x.GetTensorMemoryInfo()) ||
      !IsGpuMemory(w.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return false;
  }
  if (has_bias) {
    Ort::ConstValue b = ctx.GetInput(2);
    auto b_info = b.GetTensorTypeAndShapeInfo();
    if (b_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        b_info.GetShape().size() != 1 || b_info.GetShape()[0] != w_shape[0] ||
        !IsGpuMemory(b.GetTensorMemoryInfo())) {
      return false;
    }
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
  if (!SetMudnnConvTensor(x_tensor, x.GetTensorRawData(), x_shape) ||
      !SetMudnnConvTensor(w_tensor, w.GetTensorRawData(), w_shape) ||
      !SetMudnnConvTensor(y_tensor, output.GetTensorMutableRawData(),
                          y_shape)) {
    return false;
  }

  const size_t spatial_rank = x_shape.size() - 2;
  std::vector<int> pad(spatial_rank);
  std::vector<int> stride(spatial_rank);
  std::vector<int> dilation(spatial_rank);
  for (size_t i = 0; i < spatial_rank; ++i) {
    pad[i] = static_cast<int>(pads[i]);
    stride[i] = static_cast<int>(strides[i]);
    dilation[i] = static_cast<int>(dilations[i]);
  }
  ::musa::dnn::Convolution conv;
  if (conv.SetGroups(static_cast<int>(group_)) !=
          ::musa::dnn::Status::SUCCESS ||
      conv.SetNdInfo(static_cast<int>(spatial_rank), pad.data(), stride.data(),
                     dilation.data()) != ::musa::dnn::Status::SUCCESS ||
      conv.SetComputeMode(::musa::dnn::Convolution::ComputeMode::TENSOR) !=
          ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  if (conv.Run(*handle, y_tensor, x_tensor, w_tensor,
               ::musa::dnn::Convolution::Algorithm::IMPLICIT_GEMM,
               nullptr) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  if (!has_bias) {
    return true;
  }

  Ort::ConstValue b = ctx.GetInput(2);
  ::musa::dnn::Tensor bias_tensor;
  std::vector<int64_t> bias_shape(x_shape.size(), 1);
  bias_shape[1] = w_shape[0];
  if (!SetMudnnConvTensor(bias_tensor, b.GetTensorRawData(), bias_shape)) {
    return false;
  }

  ::musa::dnn::Binary bias_add;
  if (bias_add.SetMode(::musa::dnn::Binary::Mode::ADD) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return bias_add.Run(*handle, y_tensor, y_tensor, bias_tensor) ==
         ::musa::dnn::Status::SUCCESS;
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
  const size_t rank = x_shape.size();
  if ((rank != 3 && rank != 4 && rank != 5) || w_shape.size() != rank) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Conv only supports 1D NCL, 2D NCHW or 3D NCDHW tensors");
  }
  if (group_ <= 0 || x_shape[1] != w_shape[1] * group_ ||
      w_shape[0] % group_ != 0) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "Conv group must divide output channels and match input channels");
  }

  // A Conv1D tensor is NCL. Executing it as a unit-height 2D NCHW tensor
  // ([N, C, 1, L] with a [M, C / group, 1, K] kernel) keeps the memory layout
  // identical, so the existing 2D muDNN and custom device paths are reused.
  const bool is_1d = rank == 3;
  const size_t spatial_rank = is_1d ? 1 : rank - 2;
  std::vector<int64_t> pads = pads_;
  if (pads.empty()) {
    pads.resize(spatial_rank * 2, 0);
  }
  std::vector<int64_t> strides = strides_;
  if (strides.empty()) {
    strides.resize(spatial_rank, 1);
  }
  std::vector<int64_t> dilations = dilations_;
  if (dilations.empty()) {
    dilations.resize(spatial_rank, 1);
  }
  if (pads.size() != spatial_rank * 2 || strides.size() != spatial_rank ||
      dilations.size() != spatial_rank) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Conv spatial attributes do not match tensor rank");
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    if (pads[i] != pads[i + spatial_rank]) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "Conv only supports symmetric spatial pads");
    }
    if (strides[i] <= 0 || dilations[i] <= 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "Conv strides and dilations must be positive");
    }
  }

  std::vector<int64_t> out_spatial;
  out_spatial.reserve(spatial_rank);
  for (size_t i = 0; i < spatial_rank; ++i) {
    const int64_t output_dim =
        (x_shape[i + 2] + pads[i] + pads[i + spatial_rank] -
         dilations[i] * (w_shape[i + 2] - 1) - 1) /
            strides[i] +
        1;
    if (output_dim < 0) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Conv invalid output shape");
    }
    out_spatial.push_back(output_dim);
  }

  std::vector<int64_t> y_shape = {x_shape[0], w_shape[0]};
  y_shape.insert(y_shape.end(), out_spatial.begin(), out_spatial.end());
  Ort::UnownedValue output = ctx.GetOutput(0, y_shape);

  std::vector<int64_t> x_exec = x_shape;
  std::vector<int64_t> w_exec = w_shape;
  std::vector<int64_t> y_exec = y_shape;
  std::vector<int64_t> pads_exec = pads;
  std::vector<int64_t> strides_exec = strides;
  std::vector<int64_t> dilations_exec = dilations;
  if (is_1d) {
    x_exec = {x_shape[0], x_shape[1], 1, x_shape[2]};
    w_exec = {w_shape[0], w_shape[1], 1, w_shape[2]};
    y_exec = {y_shape[0], y_shape[1], 1, out_spatial[0]};
    pads_exec = {0, pads[0], 0, pads[1]};
    strides_exec = {1, strides[0]};
    dilations_exec = {1, dilations[0]};
  }

  if (TryMudnnConv(ctx, x_exec, w_exec, y_exec, pads_exec, strides_exec,
                   dilations_exec, output)) {
    return nullptr;
  }
  if (rank == 5) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Conv3D requires the muDNN path; custom fallback is 2D-only");
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
  params.n = x_exec[0];
  params.c = x_exec[1];
  params.group = group_;
  params.h = x_exec[2];
  params.w = x_exec[3];
  params.m = w_exec[0];
  params.kernel_h = w_exec[2];
  params.kernel_w = w_exec[3];
  params.out_h = y_exec[2];
  params.out_w = y_exec[3];
  params.pad_h = pads_exec[0];
  params.pad_w = pads_exec[1];
  params.stride_h = strides_exec[0];
  params.stride_w = strides_exec[1];
  params.dilation_h = dilations_exec[0];
  params.dilation_w = dilations_exec[1];
  params.total_elements = NumElements(y_exec);
  return LaunchStatus(LaunchMusaConv2DFloatKernel(
      x.GetTensorData<float>(), w.GetTensorData<float>(), bias,
      output.GetTensorMutableData<float>(), params, GetComputeStream(ctx)));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv, kOnnxDomain, 1, 10,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())), Conv)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv, kOnnxDomain, 11, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())), Conv)
