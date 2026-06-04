// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "nn/batch_norm_impl.h"

namespace {
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
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "BatchNormalization requires 5 inputs");
  }
  if (training_mode_ != 0) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "BatchNormalization only supports inference mode");
  }
  auto input_info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "BatchNormalization only supports float");
  }
  std::vector<int64_t> shape = input_info.GetShape();
  if (shape.size() < 2) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "BatchNormalization input rank must be at least 2");
  }
  const int64_t channels = shape[1];
  int64_t spatial_size = 1;
  for (size_t i = 2; i < shape.size(); ++i) spatial_size *= shape[i];

  auto read_param = [&](size_t index, const char* name,
                        std::vector<float>& out) -> OrtStatus* {
    auto info = ctx.GetInput(index).GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const std::string msg =
          std::string("BatchNormalization ") + name + " only supports float";
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, msg.c_str());
    }
    auto param_shape = info.GetShape();
    if (param_shape.size() != 1 || param_shape[0] != channels) {
      const std::string msg =
          std::string("BatchNormalization invalid ") + name + " shape";
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, msg.c_str());
    }
    out = ReadTyped<float>(ctx.GetInput(index));
    return nullptr;
  };

  std::vector<float> scale;
  std::vector<float> bias;
  std::vector<float> mean;
  std::vector<float> variance;
  RETURN_IF_ERROR(read_param(1, "scale", scale));
  RETURN_IF_ERROR(read_param(2, "bias", bias));
  RETURN_IF_ERROR(read_param(3, "mean", mean));
  RETURN_IF_ERROR(read_param(4, "variance", variance));

  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  if (IsGpuMemory(ctx.GetInput(0).GetTensorMemoryInfo()) &&
      IsGpuMemory(ctx.GetInput(1).GetTensorMemoryInfo()) &&
      IsGpuMemory(ctx.GetInput(2).GetTensorMemoryInfo()) &&
      IsGpuMemory(ctx.GetInput(3).GetTensorMemoryInfo()) &&
      IsGpuMemory(ctx.GetInput(4).GetTensorMemoryInfo()) &&
      IsGpuMemory(y.GetTensorMemoryInfo())) {
    MusaBatchNormParams params{NumElements(shape), channels, spatial_size,
                               epsilon_};
    return LaunchStatus(LaunchMusaBatchNormalizationFloatKernel(
        ctx.GetInput(0).GetTensorData<float>(),
        ctx.GetInput(1).GetTensorData<float>(),
        ctx.GetInput(2).GetTensorData<float>(),
        ctx.GetInput(3).GetTensorData<float>(),
        ctx.GetInput(4).GetTensorData<float>(), y.GetTensorMutableData<float>(),
        params, nullptr));
  }

  std::vector<float> input = ReadTyped<float>(ctx.GetInput(0));
  std::vector<float> output(input.size());
  for (int64_t i = 0; i < NumElements(shape); ++i) {
    const int64_t c = (i / spatial_size) % channels;
    output[static_cast<size_t>(i)] =
        (input[static_cast<size_t>(i)] - mean[static_cast<size_t>(c)]) *
            scale[static_cast<size_t>(c)] /
            std::sqrt(variance[static_cast<size_t>(c)] + epsilon_) +
        bias[static_cast<size_t>(c)];
  }
  return WriteTyped<float>(y, output);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    BatchNormalization, kOnnxDomain, 15, 17,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", FloatTensorTypes())
         .AddTypeConstraint("T1", FloatTensorTypes())
         .AddTypeConstraint("T2", FloatTensorTypes())),
    BatchNormalization)
