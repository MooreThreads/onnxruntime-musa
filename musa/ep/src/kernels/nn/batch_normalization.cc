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

#include "nn/batch_norm_impl.h"
#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
bool TryMudnnBatchNormalization(Ort::KernelContext& ctx,
                                const std::vector<int64_t>& shape,
                                ONNXTensorElementDataType elem_type,
                                float epsilon, Ort::UnownedValue y) {
  if (!AllGpuInputs(ctx) || !IsGpuMemory(y.GetTensorMemoryInfo())) {
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
  ::musa::dnn::Tensor scale_tensor;
  ::musa::dnn::Tensor bias_tensor;
  ::musa::dnn::Tensor mean_tensor;
  ::musa::dnn::Tensor variance_tensor;
  if (!SetMudnnTensor(input_tensor, ctx.GetInput(0).GetTensorRawData(), shape,
                      elem_type) ||
      !SetMudnnTensor(output_tensor, y.GetTensorMutableRawData(), shape,
                      elem_type) ||
      !SetMudnnTensor(scale_tensor, ctx.GetInput(1).GetTensorRawData(),
                      ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape(),
                      elem_type) ||
      !SetMudnnTensor(bias_tensor, ctx.GetInput(2).GetTensorRawData(),
                      ctx.GetInput(2).GetTensorTypeAndShapeInfo().GetShape(),
                      elem_type) ||
      !SetMudnnTensor(mean_tensor, ctx.GetInput(3).GetTensorRawData(),
                      ctx.GetInput(3).GetTensorTypeAndShapeInfo().GetShape(),
                      elem_type) ||
      !SetMudnnTensor(variance_tensor, ctx.GetInput(4).GetTensorRawData(),
                      ctx.GetInput(4).GetTensorTypeAndShapeInfo().GetShape(),
                      elem_type)) {
    return false;
  }

  ::musa::dnn::BatchNorm op;
  if (op.SetEpsilon(static_cast<double>(epsilon)) !=
          ::musa::dnn::Status::SUCCESS ||
      op.SetTraining(false) != ::musa::dnn::Status::SUCCESS ||
      op.SetMode(::musa::dnn::BatchNorm::Mode::PER_CHANNEL) !=
          ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  return op.RunPure(*handle, output_tensor, input_tensor, mean_tensor,
                    variance_tensor, scale_tensor,
                    bias_tensor) == ::musa::dnn::Status::SUCCESS;
}

class BatchNormalization : public OpKernelBase<BatchNormalization> {
 public:
  BatchNormalization(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    epsilon_ = AttrOrDefault<float>(kernel_info, "epsilon", 1e-5f);
    training_mode_ = AttrOrDefault<int64_t>(kernel_info, "training_mode", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  float epsilon_ = 1e-5f;
  int64_t training_mode_ = 0;
};

OrtStatus* BatchNormalization::Compute(Ort::KernelContext& ctx) const {
  if (ctx.GetInputCount() != 5) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "BatchNormalization requires 5 inputs");
  }
  if (ctx.GetOutputCount() != 1) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "BatchNormalization only supports inference output Y");
  }
  if (training_mode_ != 0) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "BatchNormalization only supports inference mode");
  }
  auto input_info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  const auto elem_type = input_info.GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported BatchNormalization dtype");
  }
  std::vector<int64_t> shape = input_info.GetShape();
  if (shape.size() < 2) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "BatchNormalization input rank must be at least 2");
  }
  const int64_t channels = shape[1];
  int64_t spatial_size = 1;
  for (size_t i = 2; i < shape.size(); ++i) spatial_size *= shape[i];

  auto validate_param = [&](size_t index, const char* name) -> OrtStatus* {
    auto info = ctx.GetInput(index).GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != elem_type) {
      const std::string msg =
          std::string("BatchNormalization ") + name + " dtype must match input";
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, msg.c_str());
    }
    auto param_shape = info.GetShape();
    if (param_shape.size() != 1 || param_shape[0] != channels) {
      const std::string msg =
          std::string("BatchNormalization invalid ") + name + " shape";
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, msg.c_str());
    }
    return nullptr;
  };

  RETURN_IF_ERROR(validate_param(1, "scale"));
  RETURN_IF_ERROR(validate_param(2, "bias"));
  RETURN_IF_ERROR(validate_param(3, "mean"));
  RETURN_IF_ERROR(validate_param(4, "variance"));

  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  if (IsGpuMemory(ctx.GetInput(0).GetTensorMemoryInfo()) &&
      IsGpuMemory(ctx.GetInput(1).GetTensorMemoryInfo()) &&
      IsGpuMemory(ctx.GetInput(2).GetTensorMemoryInfo()) &&
      IsGpuMemory(ctx.GetInput(3).GetTensorMemoryInfo()) &&
      IsGpuMemory(ctx.GetInput(4).GetTensorMemoryInfo()) &&
      IsGpuMemory(y.GetTensorMemoryInfo())) {
    if (TryMudnnBatchNormalization(ctx, shape, elem_type, epsilon_, y)) {
      return nullptr;
    }
    MusaBatchNormParams params{NumElements(shape), channels, spatial_size,
                               epsilon_};
    MusaElementType musa_elem_type;
    if (!ToMusaElementType(elem_type, musa_elem_type)) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "unsupported BatchNormalization dtype");
    }
    return LaunchStatus(LaunchMusaBatchNormalizationKernel(
        ctx.GetInput(0).GetTensorRawData(), ctx.GetInput(1).GetTensorRawData(),
        ctx.GetInput(2).GetTensorRawData(), ctx.GetInput(3).GetTensorRawData(),
        ctx.GetInput(4).GetTensorRawData(), y.GetTensorMutableRawData(), params,
        musa_elem_type, GetComputeStream(ctx)));
  }

  return Ort::GetApi().CreateStatus(
      ORT_NOT_IMPLEMENTED, "BatchNormalization requires MUSA device tensors");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    BatchNormalization, kOnnxDomain, 9, 13,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())),
    BatchNormalization)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(BatchNormalization, kOnnxDomain, 14, 14,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T", HfdTensorTypes())
                                       .AddTypeConstraint("U",
                                                          HfdTensorTypes())),
                                  BatchNormalization)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    BatchNormalization, kOnnxDomain, 15, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", HfdTensorTypes())
         .AddTypeConstraint("T1", HfdTensorTypes())
         .AddTypeConstraint("T2", HfdTensorTypes())),
    BatchNormalization)
