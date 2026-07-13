// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "llm/attention_impl.h"
#include "math/matmul.h"
#include "shared_inc/op_kernel_common.h"

namespace {

class DeviceScratch {
 public:
  DeviceScratch() = default;
  DeviceScratch(const DeviceScratch&) = delete;
  DeviceScratch& operator=(const DeviceScratch&) = delete;

  ~DeviceScratch() {
    if (ptr_ != nullptr) {
      FreeDeviceMemoryOnStream(ptr_, stream_, bytes_);
    }
  }

  OrtStatus* Allocate(size_t bytes, musaStream_t stream) {
    bytes_ = bytes;
    stream_ = stream;
    if (bytes == 0) {
      return nullptr;
    }
    ptr_ = AllocateDeviceMemoryOnStream(bytes_, stream_);
    if (ptr_ == nullptr) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL, MusaErrorString(musaErrorMemoryAllocation));
    }
    return nullptr;
  }

  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t stream_ = nullptr;
};

class Attention : public OpKernelBase<Attention> {
 public:
  Attention(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    num_heads_ = AttrOrDefault<int64_t>(kernel_info, "num_heads", 0);
    scale_ = AttrOrDefault<float>(kernel_info, "scale",
                                  std::numeric_limits<float>::quiet_NaN());
    qkv_hidden_sizes_ = AttrsOrEmpty(kernel_info, "qkv_hidden_sizes");
    unidirectional_ = AttrOrDefault<int64_t>(kernel_info, "unidirectional", 0);
  }

  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t num_heads_ = 0;
  float scale_ = std::numeric_limits<float>::quiet_NaN();
  std::vector<int64_t> qkv_hidden_sizes_;
  int64_t unidirectional_ = 0;
};

OrtStatus* CheckShapeRank(const std::vector<int64_t>& shape, size_t rank,
                          const char* name) {
  if (shape.size() != rank) {
    std::string message = std::string("Attention requires ") + name + " rank " +
                          std::to_string(rank);
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message.c_str());
  }
  for (int64_t dim : shape) {
    if (dim < 0) {
      std::string message =
          std::string("Attention requires concrete ") + name + " shape";
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message.c_str());
    }
  }
  return nullptr;
}

OrtStatus* Attention::Compute(Ort::KernelContext& ctx) const {
  if (unidirectional_ != 0) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "MUSA Attention does not support unidirectional yet");
  }
  if (num_heads_ <= 0 || qkv_hidden_sizes_.size() != 3) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "Attention requires num_heads and qkv_hidden_sizes");
  }

  Ort::ConstValue input = ctx.GetInput(0);
  Ort::ConstValue weights = ctx.GetInput(1);
  Ort::ConstValue bias = ctx.GetInput(2);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto weights_info = weights.GetTensorTypeAndShapeInfo();
  auto bias_info = bias.GetTensorTypeAndShapeInfo();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      weights_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      bias_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "MUSA Attention currently supports float input/weights/bias only");
  }

  std::vector<int64_t> input_shape = input_info.GetShape();
  std::vector<int64_t> weights_shape = weights_info.GetShape();
  std::vector<int64_t> bias_shape = bias_info.GetShape();
  RETURN_IF_ERROR(CheckShapeRank(input_shape, 3, "input"));
  RETURN_IF_ERROR(CheckShapeRank(weights_shape, 2, "weights"));
  RETURN_IF_ERROR(CheckShapeRank(bias_shape, 1, "bias"));

  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  const int64_t input_hidden_size = input_shape[2];
  const int64_t q_hidden_size = qkv_hidden_sizes_[0];
  const int64_t k_hidden_size = qkv_hidden_sizes_[1];
  const int64_t v_hidden_size = qkv_hidden_sizes_[2];
  const int64_t qkv_hidden_size = q_hidden_size + k_hidden_size + v_hidden_size;
  if (weights_shape[0] != input_hidden_size ||
      weights_shape[1] != qkv_hidden_size || bias_shape[0] != qkv_hidden_size) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "Attention qkv weight or bias shape mismatch");
  }
  if (q_hidden_size % num_heads_ != 0 || k_hidden_size % num_heads_ != 0 ||
      v_hidden_size % num_heads_ != 0) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "Attention hidden sizes must be divisible by num_heads");
  }
  const int64_t q_head_size = q_hidden_size / num_heads_;
  const int64_t k_head_size = k_hidden_size / num_heads_;
  const int64_t v_head_size = v_hidden_size / num_heads_;
  if (q_head_size != k_head_size) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "MUSA Attention requires Q and K head sizes to match");
  }

  Ort::ConstValue mask{nullptr};
  bool has_mask = ctx.GetInputCount() > 3;
  std::vector<int64_t> mask_shape;
  if (has_mask) {
    mask = ctx.GetInput(3);
    if (mask == nullptr) {
      has_mask = false;
    } else {
      auto mask_info = mask.GetTensorTypeAndShapeInfo();
      if (mask_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "MUSA Attention currently supports int32 mask only");
      }
      mask_shape = mask_info.GetShape();
      RETURN_IF_ERROR(CheckShapeRank(mask_shape, 4, "mask"));
      if ((mask_shape[0] != 1 && mask_shape[0] != batch_size) ||
          (mask_shape[1] != 1 && mask_shape[1] != num_heads_) ||
          mask_shape[2] != sequence_length ||
          mask_shape[3] != sequence_length) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "MUSA Attention only supports broadcastable 4D masks "
            "[B|1,H|1,S,S]");
      }
    }
  }

  std::vector<int64_t> output_shape = {batch_size, sequence_length,
                                       v_hidden_size};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Attention requires MUSA output");
  }

  musaStream_t stream = GetComputeStream(ctx);
  DeviceInputBuffer input_buffer;
  DeviceInputBuffer weights_buffer;
  DeviceInputBuffer bias_buffer;
  DeviceInputBuffer mask_buffer;
  RETURN_IF_ERROR(input_buffer.Bind(input, stream));
  RETURN_IF_ERROR(weights_buffer.Bind(weights, stream));
  RETURN_IF_ERROR(bias_buffer.Bind(bias, stream));
  if (has_mask) {
    RETURN_IF_ERROR(mask_buffer.Bind(mask, stream));
  }

  const int64_t tokens = batch_size * sequence_length;
  DeviceScratch qkv_buffer;
  DeviceScratch scores_buffer;
  RETURN_IF_ERROR(qkv_buffer.Allocate(
      static_cast<size_t>(tokens * qkv_hidden_size) * sizeof(float), stream));
  RETURN_IF_ERROR(scores_buffer.Allocate(
      static_cast<size_t>(batch_size * num_heads_ * sequence_length *
                          sequence_length) *
          sizeof(float),
      stream));

  RETURN_IF_ERROR(ComputeMusaMatMulDevice(
      input_buffer.data(), weights_buffer.data(), qkv_buffer.get(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {tokens, input_hidden_size},
      weights_shape, {tokens, qkv_hidden_size}, false, false, false, false,
      1.0f, stream));
  RETURN_IF_ERROR(LaunchStatus(LaunchMusaAttentionAddBiasKernel(
      static_cast<float*>(qkv_buffer.get()),
      static_cast<const float*>(bias_buffer.data()), tokens * qkv_hidden_size,
      qkv_hidden_size, stream)));

  MusaAttentionParams params{};
  params.batch_size = batch_size;
  params.sequence_length = sequence_length;
  params.input_hidden_size = input_hidden_size;
  params.q_hidden_size = q_hidden_size;
  params.k_hidden_size = k_hidden_size;
  params.v_hidden_size = v_hidden_size;
  params.q_head_size = q_head_size;
  params.k_head_size = k_head_size;
  params.v_head_size = v_head_size;
  params.num_heads = num_heads_;
  params.qkv_hidden_size = qkv_hidden_size;
  params.mask_batch = has_mask ? mask_shape[0] : 1;
  params.mask_heads = has_mask ? mask_shape[1] : 1;
  params.scale = std::isnan(scale_)
                     ? 1.0f / std::sqrt(static_cast<float>(q_head_size))
                     : scale_;
  params.has_mask = has_mask ? 1 : 0;

  RETURN_IF_ERROR(LaunchStatus(LaunchMusaAttentionScoreKernel(
      static_cast<const float*>(qkv_buffer.get()),
      has_mask ? static_cast<const int32_t*>(mask_buffer.data()) : nullptr,
      static_cast<float*>(scores_buffer.get()), params, stream)));
  return LaunchStatus(LaunchMusaAttentionValueKernel(
      static_cast<const float*>(qkv_buffer.get()),
      static_cast<const float*>(scores_buffer.get()),
      static_cast<float*>(output.GetTensorMutableRawData()), params, stream));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Attention, kMSDomain, 1, 1,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint(
             "M", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32))),
    Attention)
