// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>

#include "nn/layer_norm_impl.h"
#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {

class DeviceTempBuffer {
 public:
  DeviceTempBuffer(size_t bytes, musaStream_t stream)
      : bytes_(bytes), stream_(stream) {
    if (bytes_ != 0) {
      status_ = musaMalloc(&ptr_, bytes_);
    }
  }
  ~DeviceTempBuffer() {
    if (ptr_ != nullptr) {
      FreeDeviceMemoryOnStream(ptr_, stream_);
    }
  }
  DeviceTempBuffer(const DeviceTempBuffer&) = delete;
  DeviceTempBuffer& operator=(const DeviceTempBuffer&) = delete;

  bool OK() const { return bytes_ == 0 || status_ == musaSuccess; }
  void* Get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t stream_ = nullptr;
  musaError_t status_ = musaSuccess;
};

std::vector<int64_t> MeanShape(const std::vector<int64_t>& input_shape,
                               int64_t axis) {
  std::vector<int64_t> out = input_shape;
  for (size_t i = static_cast<size_t>(axis); i < out.size(); ++i) {
    out[i] = 1;
  }
  return out;
}

bool ShapeEqualsSuffix(const std::vector<int64_t>& shape,
                       const std::vector<int64_t>& suffix) {
  return shape.size() == suffix.size() &&
         std::equal(shape.begin(), shape.end(), suffix.begin());
}

class LayerNormalization : public OpKernelBase<LayerNormalization> {
 public:
  LayerNormalization(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", -1);
    epsilon_ = AttrOrDefault<float>(kernel_info, "epsilon", 1e-5f);
    stash_type_ = AttrOrDefault<int64_t>(kernel_info, "stash_type", 1);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  bool TryMudnn(Ort::KernelContext& ctx,
                const std::vector<int64_t>& input_shape,
                const std::vector<int64_t>& scale_shape,
                const std::vector<int64_t>& mean_shape,
                ONNXTensorElementDataType elem_type, int64_t axis,
                Ort::UnownedValue output, Ort::UnownedValue mean,
                Ort::UnownedValue inv_std) const;

  int64_t axis_ = -1;
  float epsilon_ = 1e-5f;
  int64_t stash_type_ = 1;
};

bool LayerNormalization::TryMudnn(Ort::KernelContext& ctx,
                                  const std::vector<int64_t>& input_shape,
                                  const std::vector<int64_t>& scale_shape,
                                  const std::vector<int64_t>& mean_shape,
                                  ONNXTensorElementDataType elem_type,
                                  int64_t axis, Ort::UnownedValue output,
                                  Ort::UnownedValue mean,
                                  Ort::UnownedValue inv_std) const {
  Ort::ConstValue input = ctx.GetInput(0);
  Ort::ConstValue scale = ctx.GetInput(1);
  Ort::ConstValue bias =
      (ctx.GetInputCount() > 2) ? ctx.GetInput(2) : Ort::ConstValue{nullptr};
  if (bias == nullptr || !IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(scale.GetTensorMemoryInfo()) ||
      !IsGpuMemory(bias.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  musaStream_t stream = GetComputeStream(ctx);
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, stream);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  const int64_t rows =
      axis == 0
          ? 1
          : std::accumulate(input_shape.begin(), input_shape.begin() + axis,
                            int64_t{1}, std::multiplies<int64_t>());
  const size_t mean_bytes = static_cast<size_t>(rows) * sizeof(float);
  DeviceTempBuffer mean_tmp(mean == nullptr ? mean_bytes : 0, stream);
  DeviceTempBuffer inv_tmp(inv_std == nullptr ? mean_bytes : 0, stream);
  if (!mean_tmp.OK() || !inv_tmp.OK()) {
    return false;
  }
  void* mean_data =
      mean == nullptr ? mean_tmp.Get() : mean.GetTensorMutableData<float>();
  void* inv_std_data = inv_std == nullptr
                           ? inv_tmp.Get()
                           : inv_std.GetTensorMutableData<float>();
  if (mean_data == nullptr || inv_std_data == nullptr) {
    return false;
  }

  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor output_tensor;
  ::musa::dnn::Tensor scale_tensor;
  ::musa::dnn::Tensor bias_tensor;
  ::musa::dnn::Tensor mean_tensor;
  ::musa::dnn::Tensor inv_std_tensor;
  if (!SetMudnnTensor(input_tensor, input.GetTensorRawData(), input_shape,
                      elem_type) ||
      !SetMudnnTensor(output_tensor, output.GetTensorMutableRawData(),
                      input_shape, elem_type) ||
      !SetMudnnFloatTensor(scale_tensor, scale.GetTensorRawData(),
                           scale_shape) ||
      !SetMudnnFloatTensor(bias_tensor, bias.GetTensorRawData(), scale_shape) ||
      !SetMudnnFloatTensor(mean_tensor, mean_data, mean_shape) ||
      !SetMudnnFloatTensor(inv_std_tensor, inv_std_data, mean_shape)) {
    return false;
  }

  std::vector<int> axes;
  for (int64_t dim = axis; dim < static_cast<int64_t>(input_shape.size());
       ++dim) {
    axes.push_back(static_cast<int>(dim));
  }

  ::musa::dnn::LayerNorm op;
  if (op.SetEpsilon(static_cast<double>(epsilon_)) !=
          ::musa::dnn::Status::SUCCESS ||
      op.SetVarMode(::musa::dnn::LayerNorm::VarMode::DIRECT) !=
          ::musa::dnn::Status::SUCCESS ||
      op.SetAxis(axes.size(), axes.data()) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  ::musa::dnn::MemoryMaintainer maintainer =
      [stream](size_t bytes) -> ::musa::dnn::MemoryHandler {
    void* ptr = nullptr;
    if (bytes != 0 && musaMalloc(&ptr, bytes) != musaSuccess) {
      ptr = nullptr;
    }
    return ::musa::dnn::MemoryHandler(ptr, [stream](void* p) {
      FreeDeviceMemoryOnStream(p, stream);
    });
  };

  auto status = op.Run(*handle, output_tensor, mean_tensor, inv_std_tensor,
                       input_tensor, scale_tensor, bias_tensor, maintainer);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return true;
}

OrtStatus* LayerNormalization::Compute(Ort::KernelContext& ctx) const {
  if (stash_type_ != 1) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "LayerNormalization only supports stash_type=1");
  }
  Ort::ConstValue input = ctx.GetInput(0);
  Ort::ConstValue scale = ctx.GetInput(1);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto scale_info = scale.GetTensorTypeAndShapeInfo();
  auto elem_type = input_info.GetElementType();
  auto input_shape = input_info.GetShape();
  auto scale_shape = scale_info.GetShape();
  if (input_shape.empty()) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "LayerNormalization requires rank >= 1");
  }
  int64_t axis = NormalizeAxis(axis_, input_shape.size());
  if (axis < 0 || axis >= static_cast<int64_t>(input_shape.size())) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "LayerNormalization axis out of range");
  }
  std::vector<int64_t> normalized_shape(input_shape.begin() + axis,
                                        input_shape.end());
  if (!ShapeEqualsSuffix(scale_shape, normalized_shape) ||
      scale_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "LayerNormalization requires float scale matching normalized shape");
  }
  Ort::ConstValue bias =
      (ctx.GetInputCount() > 2) ? ctx.GetInput(2) : Ort::ConstValue{nullptr};
  if (bias != nullptr) {
    auto bias_info = bias.GetTensorTypeAndShapeInfo();
    if (!ShapeEqualsSuffix(bias_info.GetShape(), normalized_shape) ||
        bias_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "LayerNormalization requires float bias matching normalized shape");
    }
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
       elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
       elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 &&
       elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "LayerNormalization unsupported dtype");
  }
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(scale.GetTensorMemoryInfo()) ||
      (bias != nullptr && !IsGpuMemory(bias.GetTensorMemoryInfo()))) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "LayerNormalization requires MUSA device inputs");
  }

  const int64_t rows =
      axis == 0
          ? 1
          : std::accumulate(input_shape.begin(), input_shape.begin() + axis,
                            int64_t{1}, std::multiplies<int64_t>());
  const int64_t norm_size =
      std::accumulate(input_shape.begin() + axis, input_shape.end(), int64_t{1},
                      std::multiplies<int64_t>());
  if (rows > INT32_MAX) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "LayerNormalization row count exceeds limit");
  }
  Ort::UnownedValue output = ctx.GetOutput(0, input_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "LayerNormalization requires MUSA device output");
  }

  std::vector<int64_t> mean_shape = MeanShape(input_shape, axis);
  Ort::UnownedValue mean = ctx.GetOutputCount() > 1
                               ? ctx.GetOutput(1, mean_shape)
                               : Ort::UnownedValue{nullptr};
  Ort::UnownedValue inv_std = ctx.GetOutputCount() > 2
                                  ? ctx.GetOutput(2, mean_shape)
                                  : Ort::UnownedValue{nullptr};
  if ((mean != nullptr && !IsGpuMemory(mean.GetTensorMemoryInfo())) ||
      (inv_std != nullptr && !IsGpuMemory(inv_std.GetTensorMemoryInfo()))) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "LayerNormalization requires MUSA device optional outputs");
  }

  if (TryMudnn(ctx, input_shape, scale_shape, mean_shape, elem_type, axis,
               output, mean, inv_std)) {
    return nullptr;
  }

  MusaLayerNormParams params{};
  params.rows = rows;
  params.norm_size = norm_size;
  params.has_bias = bias != nullptr ? 1 : 0;
  musaError_t status = LaunchMusaLayerNormKernel(
      input.GetTensorRawData(), scale.GetTensorData<float>(),
      bias == nullptr ? nullptr : bias.GetTensorData<float>(),
      output.GetTensorMutableRawData(),
      mean == nullptr ? nullptr : mean.GetTensorMutableData<float>(),
      inv_std == nullptr ? nullptr : inv_std.GetTensorMutableData<float>(),
      params, epsilon_, musa_elem_type, GetComputeStream(ctx));
  if (status == musaErrorNotSupported) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "LayerNormalization unsupported dtype, shape, or attribute");
  }
  return LaunchStatus(status);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LayerNormalization, kOnnxDomain, 17, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    LayerNormalization)
