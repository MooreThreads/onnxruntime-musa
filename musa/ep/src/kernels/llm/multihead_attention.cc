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

#include <musa_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "llm/multihead_attention_impl.h"
#include "llm/reduced_mha_flash_impl.h"
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
    ptr_ = AllocateDeviceMemoryOnStream(bytes, stream);
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

OrtStatus* CheckConcreteRank3(const Ort::TensorTypeAndShapeInfo& info,
                              const char* name, std::vector<int64_t>* shape) {
  *shape = info.GetShape();
  if (shape->size() != 3) {
    const std::string message =
        std::string("MUSA MultiHeadAttention requires rank-3 ") + name;
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message.c_str());
  }
  for (int64_t dim : *shape) {
    if (dim < 0) {
      const std::string message =
          std::string("MUSA MultiHeadAttention requires concrete ") + name +
          " shape";
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message.c_str());
    }
  }
  return nullptr;
}

class MultiHeadAttention : public OpKernelBase<MultiHeadAttention> {
 public:
  MultiHeadAttention(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    num_heads_ = AttrOrDefault<int64_t>(kernel_info, "num_heads", 0);
    mask_filter_value_ =
        AttrOrDefault<float>(kernel_info, "mask_filter_value", -10000.0f);
    scale_ = AttrOrDefault<float>(kernel_info, "scale",
                                  std::numeric_limits<float>::quiet_NaN());
    unidirectional_ = AttrOrDefault<int64_t>(kernel_info, "unidirectional", 0);
  }

  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t num_heads_ = 0;
  float mask_filter_value_ = -10000.0f;
  float scale_ = std::numeric_limits<float>::quiet_NaN();
  int64_t unidirectional_ = 0;
};

OrtStatus* MultiHeadAttention::Compute(Ort::KernelContext& ctx) const {
  if (num_heads_ <= 0) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "MUSA MultiHeadAttention requires num_heads > 0");
  }
  if (unidirectional_ != 0) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "MUSA MultiHeadAttention does not support unidirectional attention");
  }
  if (ctx.GetInputCount() < 4) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "MUSA MultiHeadAttention requires separate Q/K/V and bias inputs");
  }
  for (size_t index = 5; index < ctx.GetInputCount(); ++index) {
    if (ctx.GetInput(index) != nullptr) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "MUSA MultiHeadAttention does not support attention_bias, KV cache, "
          "past_sequence_length, or cache_indirection");
    }
  }
  if (ctx.GetOutputCount() != 1) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "MUSA MultiHeadAttention currently supports output only, without "
        "present_key, present_value, or qk outputs");
  }

  Ort::ConstValue query = ctx.GetInput(0);
  Ort::ConstValue key = ctx.GetInput(1);
  Ort::ConstValue value = ctx.GetInput(2);
  Ort::ConstValue bias = ctx.GetInput(3);
  if (query == nullptr || key == nullptr || value == nullptr ||
      bias == nullptr) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "MUSA MultiHeadAttention requires non-packed Q/K/V and bias");
  }

  auto query_info = query.GetTensorTypeAndShapeInfo();
  auto key_info = key.GetTensorTypeAndShapeInfo();
  auto value_info = value.GetTensorTypeAndShapeInfo();
  auto bias_info = bias.GetTensorTypeAndShapeInfo();
  if (query_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      key_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      value_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      bias_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "MUSA MultiHeadAttention currently supports FP32 Q/K/V/bias only");
  }

  std::vector<int64_t> query_shape;
  std::vector<int64_t> key_shape;
  std::vector<int64_t> value_shape;
  RETURN_IF_ERROR(CheckConcreteRank3(query_info, "query", &query_shape));
  RETURN_IF_ERROR(CheckConcreteRank3(key_info, "key", &key_shape));
  RETURN_IF_ERROR(CheckConcreteRank3(value_info, "value", &value_shape));
  if (query_shape != key_shape || query_shape != value_shape) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "MUSA MultiHeadAttention currently requires self-attention with equal "
        "Q/K/V shapes [B,S,D]");
  }

  const int64_t batch = query_shape[0];
  const int64_t sequence = query_shape[1];
  const int64_t hidden = query_shape[2];
  if (hidden <= 0 || hidden % num_heads_ != 0) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "MUSA MultiHeadAttention requires hidden_size divisible by num_heads");
  }
  const std::vector<int64_t> bias_shape = bias_info.GetShape();
  if (bias_shape.size() != 1 || bias_shape[0] != 3 * hidden) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "MUSA MultiHeadAttention requires bias shape [3 * hidden_size]");
  }

  Ort::ConstValue mask{nullptr};
  bool has_mask = ctx.GetInputCount() > 4 && ctx.GetInput(4) != nullptr;
  std::vector<int64_t> mask_shape;
  if (has_mask) {
    mask = ctx.GetInput(4);
    auto mask_info = mask.GetTensorTypeAndShapeInfo();
    if (mask_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "MUSA MultiHeadAttention currently supports INT32 mask only");
    }
    mask_shape = mask_info.GetShape();
    if (mask_shape.size() != 3 || mask_shape[0] != batch ||
        mask_shape[1] != sequence || mask_shape[2] != sequence) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "MUSA MultiHeadAttention currently requires mask [B,S,S]");
    }
  }

  Ort::UnownedValue output = ctx.GetOutput(0, {batch, sequence, hidden});
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "MUSA MultiHeadAttention requires MUSA output");
  }
  if (batch == 0 || sequence == 0) {
    return nullptr;
  }

  musaStream_t stream = GetComputeStream(ctx);
  DeviceInputBuffer query_buffer;
  DeviceInputBuffer key_buffer;
  DeviceInputBuffer value_buffer;
  DeviceInputBuffer bias_buffer;
  DeviceInputBuffer mask_buffer;
  RETURN_IF_ERROR(query_buffer.Bind(query, stream));
  RETURN_IF_ERROR(key_buffer.Bind(key, stream));
  RETURN_IF_ERROR(value_buffer.Bind(value, stream));
  RETURN_IF_ERROR(bias_buffer.Bind(bias, stream));
  if (has_mask) {
    RETURN_IF_ERROR(mask_buffer.Bind(mask, stream));
  }

  DeviceScratch packed_qkv;
  RETURN_IF_ERROR(packed_qkv.Allocate(
      static_cast<size_t>(batch * sequence * 3 * hidden) * sizeof(float),
      stream));
  RETURN_IF_ERROR(LaunchStatus(LaunchMusaPackQkvBiasKernel(
      static_cast<const float*>(query_buffer.data()),
      static_cast<const float*>(key_buffer.data()),
      static_cast<const float*>(value_buffer.data()),
      static_cast<const float*>(bias_buffer.data()),
      static_cast<float*>(packed_qkv.get()), batch * sequence, hidden,
      stream)));

  const int64_t head_dim = hidden / num_heads_;
  MusaReducedMhaFlashParams params{
      batch,
      sequence,
      hidden,
      num_heads_,
      head_dim,
      has_mask ? mask_shape[0] : 1,
      1,
      std::isnan(scale_) ? 1.0f / std::sqrt(static_cast<float>(head_dim))
                         : scale_,
      mask_filter_value_,
      has_mask,
      true,
      true};
  return LaunchStatus(LaunchMusaReducedMhaFlashKernel(
      static_cast<const float*>(packed_qkv.get()),
      has_mask ? static_cast<const int32_t*>(mask_buffer.data()) : nullptr,
      output.GetTensorMutableData<float>(), params, stream));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MultiHeadAttention, kMSDomain, 1, 1,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("QK",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint(
             "M", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32))),
    MultiHeadAttention)
