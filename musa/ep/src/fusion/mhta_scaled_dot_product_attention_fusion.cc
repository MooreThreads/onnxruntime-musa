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

#include "fusion/mhta_scaled_dot_product_attention_fusion.h"

#include <mudnncxx/mudnn.h>
#include <musa_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fusion/mhta_scaled_dot_product_attention_utils.h"
#include "graph/graph_utils.h"
#include "kernels/llm/mhta_sdpa_fp32_impl.h"
#include "kernels/shared_inc/blas_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"

namespace {

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(Ort::ConstNode node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

std::vector<int64_t> Shape(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

float ReadScalarFloatAttributeInput(Ort::ConstValueInfo input) {
  auto value = musa_ep::ReadScalarFloatInitializer(input);
  if (!value.has_value()) {
    throw std::runtime_error(
        "MHTA SDPA scalar initializer must be a floating-point scalar");
  }
  return *value;
}

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> indices;
  std::vector<Ort::ConstValueInfo> inputs = fused_node.GetInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    indices.emplace(Name(inputs[i]), i);
  }
  return indices;
}

size_t InputIndex(const std::unordered_map<std::string, size_t>& indices,
                  Ort::ConstValueInfo input) {
  auto it = indices.find(Name(input));
  if (it == indices.end()) {
    throw std::runtime_error("unable to map MHTA SDPA fused input");
  }
  return it->second;
}

enum class MhtaSdpaLayout {
  kBhsd,
  kSimRank3,
};

class MhtaScaledDotProductAttentionFusionCompute final
    : public FusionNodeCompute {
 public:
  MhtaScaledDotProductAttentionFusionCompute(size_t q_index, size_t k_index,
                                             size_t v_index, size_t mask_index,
                                             float scale, float mask_scale,
                                             MhtaSdpaLayout layout)
      : q_index_(q_index),
        k_index_(k_index),
        v_index_(v_index),
        mask_index_(mask_index),
        scale_(scale),
        mask_scale_(mask_scale),
        layout_(layout) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

 private:
  size_t q_index_;
  size_t k_index_;
  size_t v_index_;
  size_t mask_index_;
  float scale_;
  float mask_scale_;
  MhtaSdpaLayout layout_;
};

bool IsMhtaSdpaTensorType(ONNXTensorElementDataType elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

void ValidateMhtaSdpaTensor(Ort::ConstValue value, const char* name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (!IsMhtaSdpaTensorType(info.GetElementType())) {
    throw std::runtime_error(std::string("MHTA SDPA only supports floating ") +
                             name);
  }
}

class DeviceBuffer {
 public:
  ~DeviceBuffer() {
    if (ptr_ != nullptr) (void)musaFree(ptr_);
  }
  void Resize(size_t bytes) {
    if (bytes <= bytes_) return;
    if (ptr_ != nullptr) (void)musaFree(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
    if (bytes != 0 && musaMalloc(&ptr_, bytes) != musaSuccess) {
      throw std::runtime_error(MusaErrorString(musaErrorMemoryAllocation));
    }
    bytes_ = bytes;
  }
  void* data() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

size_t MhtaSdpaElementSize(ONNXTensorElementDataType elem_type) {
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return sizeof(float);
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) return sizeof(double);
  return sizeof(uint16_t);
}

void CheckStatus(::musa::dnn::Status status, const char* message) {
  if (status != ::musa::dnn::Status::SUCCESS) {
    throw std::runtime_error(std::string(message) + ", status: " +
                             std::to_string(static_cast<int>(status)));
  }
}

void SetupTensor(::musa::dnn::Tensor& tensor, const void* data,
                 const std::vector<int64_t>& shape,
                 ONNXTensorElementDataType elem_type, const char* name) {
  if (!SetMudnnTensor(tensor, data, shape, elem_type)) {
    throw std::runtime_error(std::string("failed to set MHTA SDPA tensor ") +
                             name);
  }
}

uint16_t FloatToHalfBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  const uint32_t exponent = (bits >> 23) & 0xff;
  const uint32_t mantissa = bits & 0x7fffff;
  if (exponent >= 143) return static_cast<uint16_t>(sign | 0x7c00);
  if (exponent <= 112) return static_cast<uint16_t>(sign);
  const uint32_t rounded = mantissa + 0x1000;
  return static_cast<uint16_t>(sign | ((exponent - 112) << 10) |
                               (rounded >> 13));
}

void WriteMhtaSdpaScalar(void* dst, float value,
                         ONNXTensorElementDataType elem_type) {
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::memcpy(dst, &value, sizeof(value));
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    const double double_value = value;
    std::memcpy(dst, &double_value, sizeof(double_value));
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const uint16_t bits = FloatToHalfBits(value);
    std::memcpy(dst, &bits, sizeof(bits));
  } else {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t bf16 =
        static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
    std::memcpy(dst, &bf16, sizeof(bf16));
  }
}

bool SetupMhtaSdpaMaskParams(const std::vector<int64_t>& mask_shape,
                             int64_t batch, int64_t heads, int64_t seqlen_q,
                             int64_t seqlen_k, MusaMhtaSdpaFp32Params* params) {
  if (mask_shape.empty() || mask_shape.size() > 4) {
    return false;
  }
  int64_t dims[4] = {1, 1, 1, 1};
  const size_t padding = 4 - mask_shape.size();
  for (size_t i = 0; i < mask_shape.size(); ++i) {
    if (mask_shape[i] <= 0) {
      return false;
    }
    dims[padding + i] = mask_shape[i];
  }
  const int64_t target[4] = {batch, heads, seqlen_q, seqlen_k};
  for (size_t i = 0; i < 4; ++i) {
    if (dims[i] != 1 && dims[i] != target[i]) {
      return false;
    }
  }
  params->mask_b = dims[0];
  params->mask_h = dims[1];
  params->mask_q = dims[2];
  params->mask_k = dims[3];
  return true;
}

OrtStatus* MhtaScaledDotProductAttentionFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    musaStream_t stream = GetComputeStream(ctx);

    Ort::ConstValue q = ctx.GetInput(q_index_);
    Ort::ConstValue k = ctx.GetInput(k_index_);
    Ort::ConstValue v = ctx.GetInput(v_index_);
    Ort::ConstValue mask = ctx.GetInput(mask_index_);
    ValidateMhtaSdpaTensor(q, "Q");
    ValidateMhtaSdpaTensor(k, "K");
    ValidateMhtaSdpaTensor(v, "V");
    ValidateMhtaSdpaTensor(mask, "mask");
    const ONNXTensorElementDataType elem_type =
        q.GetTensorTypeAndShapeInfo().GetElementType();
    if (k.GetTensorTypeAndShapeInfo().GetElementType() != elem_type ||
        v.GetTensorTypeAndShapeInfo().GetElementType() != elem_type ||
        mask.GetTensorTypeAndShapeInfo().GetElementType() != elem_type) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "MHTA SDPA requires Q/K/V/mask to have the same element type");
    }

    std::vector<int64_t> q_shape = Shape(q);
    std::vector<int64_t> k_shape = Shape(k);
    std::vector<int64_t> v_shape = Shape(v);
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "MHTA SDPA requires 4D Q/K/V inputs");
    }

    const bool sim_rank3 = layout_ == MhtaSdpaLayout::kSimRank3;
    std::vector<int64_t> output_shape;
    if (sim_rank3) {
      if (q_shape[1] != 1 || k_shape[0] != q_shape[0] ||
          v_shape[0] != q_shape[0] || k_shape[2] != q_shape[2] ||
          v_shape[1] != q_shape[2] || k_shape[3] != q_shape[3] ||
          v_shape[3] != q_shape[3] || k_shape[1] != v_shape[2]) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "MHTA SDPA sim layout expects Q[B,1,H,D], K[B,S,H,D], V[B,H,S,D]");
      }
      output_shape = {q_shape[0], q_shape[1], q_shape[2] * q_shape[3]};
    } else {
      if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] ||
          q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1] ||
          q_shape[2] != k_shape[3] || q_shape[2] != v_shape[2] ||
          q_shape[3] != k_shape[2] || q_shape[3] != v_shape[3]) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "MHTA SDPA expects Q[B,H,S,D], K[B,H,D,S], V[B,H,S,D]");
      }
      output_shape = {q_shape[0], q_shape[1], q_shape[2], v_shape[3]};
    }
    Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
    if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "MHTA SDPA requires MUSA output");
    }

    const int64_t batch = q_shape[0];
    const int64_t heads = sim_rank3 ? q_shape[2] : q_shape[1];
    const int64_t seqlen_q = sim_rank3 ? q_shape[1] : q_shape[2];
    const int64_t seqlen_k = sim_rank3 ? k_shape[1] : k_shape[3];
    const int64_t head_dim = q_shape[3];
    if (batch <= 0 || heads <= 0 || seqlen_q <= 0 || seqlen_k <= 0 ||
        head_dim <= 0) {
      return nullptr;
    }

    DeviceInputBuffer q_buffer;
    DeviceInputBuffer k_buffer;
    DeviceInputBuffer v_buffer;
    DeviceInputBuffer mask_buffer;
    RETURN_IF_ERROR(q_buffer.Bind(q, stream));
    RETURN_IF_ERROR(k_buffer.Bind(k, stream));
    RETURN_IF_ERROR(v_buffer.Bind(v, stream));
    RETURN_IF_ERROR(mask_buffer.Bind(mask, stream));
    const std::vector<int64_t> mask_shape = Shape(mask);

    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      MusaMhtaSdpaFp32Params params{
          batch, heads, seqlen_q, seqlen_k, head_dim,   scale_,   mask_scale_,
          0,     0,     0,        0,        !sim_rank3, sim_rank3};
      if (!SetupMhtaSdpaMaskParams(mask_shape, batch, heads, seqlen_q, seqlen_k,
                                   &params)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "MHTA SDPA FP32 kernel requires a broadcastable rank-1 to rank-4 "
            "mask");
      }
      musaError_t launch_status = LaunchMusaMhtaSdpaFp32Kernel(
          static_cast<const float*>(q_buffer.data()),
          static_cast<const float*>(k_buffer.data()),
          static_cast<const float*>(v_buffer.data()),
          static_cast<const float*>(mask_buffer.data()),
          output.GetTensorMutableData<float>(), params, stream);
      if (launch_status != musaSuccess) {
        throw std::runtime_error(std::string("MHTA SDPA FP32 kernel failed: ") +
                                 MusaErrorString(launch_status));
      }
      return nullptr;
    }

    DeviceBuffer q_transposed;
    DeviceBuffer k_transposed;
    DeviceBuffer mask_scaled;
    DeviceBuffer scalar;
    const void* q_data = q_buffer.data();
    const void* k_data = k_buffer.data();
    std::vector<int64_t> sdpa_q_shape = q_shape;
    std::vector<int64_t> sdpa_k_shape = k_shape;
    if (sim_rank3) {
      sdpa_q_shape = {batch, heads, seqlen_q, head_dim};
      sdpa_k_shape = {batch, heads, seqlen_k, head_dim};
      q_transposed.Resize(
          static_cast<size_t>(batch * heads * seqlen_q * head_dim) *
          MhtaSdpaElementSize(elem_type));
      k_transposed.Resize(
          static_cast<size_t>(batch * heads * seqlen_k * head_dim) *
          MhtaSdpaElementSize(elem_type));
      ::musa::dnn::Handle* handle = nullptr;
      RETURN_IF_ERROR(EnsureMudnnHandle(&handle, stream));
      ::musa::dnn::Tensor input_tensor, output_tensor;
      SetupTensor(input_tensor, q_data, q_shape, elem_type, "Q");
      SetupTensor(output_tensor, q_transposed.data(), sdpa_q_shape, elem_type,
                  "Q transpose");
      const int64_t perm[] = {0, 2, 1, 3};
      ::musa::dnn::Permute permute;
      CheckStatus(permute.ConfigDimStride(output_tensor, input_tensor, 4, perm),
                  "failed to configure MHTA SDPA Q permute");
      CheckStatus(permute.Run(*handle, output_tensor, input_tensor),
                  "MHTA SDPA Q permute failed");
      SetupTensor(input_tensor, k_data, k_shape, elem_type, "K");
      SetupTensor(output_tensor, k_transposed.data(), sdpa_k_shape, elem_type,
                  "K transpose");
      CheckStatus(permute.ConfigDimStride(output_tensor, input_tensor, 4, perm),
                  "failed to configure MHTA SDPA K permute");
      CheckStatus(permute.Run(*handle, output_tensor, input_tensor),
                  "MHTA SDPA K permute failed");
      q_data = q_transposed.data();
      k_data = k_transposed.data();
    } else {
      // FlashAttention consumes K in BHSD.  The MatMul graph provides BHDS.
      sdpa_k_shape = {batch, heads, seqlen_k, head_dim};
      k_transposed.Resize(
          static_cast<size_t>(batch * heads * seqlen_k * head_dim) *
          MhtaSdpaElementSize(elem_type));
      ::musa::dnn::Handle* handle = nullptr;
      RETURN_IF_ERROR(EnsureMudnnHandle(&handle, stream));
      ::musa::dnn::Tensor input_tensor, output_tensor;
      SetupTensor(input_tensor, k_data, k_shape, elem_type, "K");
      SetupTensor(output_tensor, k_transposed.data(), sdpa_k_shape, elem_type,
                  "K transpose");
      const int64_t perm[] = {0, 1, 3, 2};
      ::musa::dnn::Permute permute;
      CheckStatus(permute.ConfigDimStride(output_tensor, input_tensor, 4, perm),
                  "failed to configure MHTA SDPA K permute");
      CheckStatus(permute.Run(*handle, output_tensor, input_tensor),
                  "MHTA SDPA K permute failed");
      k_data = k_transposed.data();
    }

    ::musa::dnn::Handle* handle = nullptr;
    RETURN_IF_ERROR(EnsureMudnnHandle(&handle, stream));
    const void* mask_data = mask_buffer.data();
    if (mask_scale_ != 1.0f) {
      const size_t mask_bytes = static_cast<size_t>(NumElements(mask_shape)) *
                                MhtaSdpaElementSize(elem_type);
      mask_scaled.Resize(mask_bytes);
      scalar.Resize(MhtaSdpaElementSize(elem_type));
      uint8_t scalar_host[sizeof(double)] = {};
      WriteMhtaSdpaScalar(scalar_host, mask_scale_, elem_type);
      musaError_t copy_status = musaMemcpyAsync(scalar.data(), scalar_host,
                                                MhtaSdpaElementSize(elem_type),
                                                musaMemcpyHostToDevice, stream);
      if (copy_status != musaSuccess)
        throw std::runtime_error(MusaErrorString(copy_status));
      ::musa::dnn::Tensor mask_tensor, scalar_tensor, scaled_tensor;
      SetupTensor(mask_tensor, mask_data, mask_shape, elem_type, "mask");
      SetupTensor(scalar_tensor, scalar.data(), {1}, elem_type, "mask scale");
      SetupTensor(scaled_tensor, mask_scaled.data(), mask_shape, elem_type,
                  "scaled mask");
      ::musa::dnn::Binary mul;
      CheckStatus(mul.SetMode(::musa::dnn::Binary::Mode::MUL),
                  "failed to set MHTA SDPA mask scale mode");
      CheckStatus(mul.Run(*handle, scaled_tensor, mask_tensor, scalar_tensor),
                  "MHTA SDPA mask scale failed");
      mask_data = mask_scaled.data();
    }

    ::musa::dnn::Tensor q_tensor, k_tensor, v_tensor, mask_tensor, out_tensor,
        lse_tensor, dropout_tensor;
    SetupTensor(q_tensor, q_data, sdpa_q_shape, elem_type, "Q");
    SetupTensor(k_tensor, k_data, sdpa_k_shape, elem_type, "K");
    SetupTensor(v_tensor, v_buffer.data(), v_shape, elem_type, "V");
    SetupTensor(mask_tensor, mask_data, mask_shape, elem_type, "mask");
    SetupTensor(out_tensor, output.GetTensorMutableData<void>(),
                sim_rank3 ? sdpa_q_shape : output_shape, elem_type, "output");
    DeviceBuffer lse;
    lse.Resize(static_cast<size_t>(batch * heads * seqlen_q) * sizeof(float));
    SetupTensor(lse_tensor, lse.data(), {batch, heads, seqlen_q},
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "logsumexp");
    // Inference has dropout disabled.  Passing a null tensor avoids Flash's
    // output-mask shape contract [B,H,Sq,Sk] and its needless allocation.
    SetupTensor(dropout_tensor, nullptr, {1}, elem_type, "dropout mask");
    ::musa::dnn::ScaledDotProductAttention sdpa;
    CheckStatus(
        sdpa.SetComputeMode(
            ::musa::dnn::ScaledDotProductAttention::ComputeMode::SCALAR),
        "failed to set MHTA SDPA compute mode");
    CheckStatus(sdpa.SetEmbedDim(static_cast<int>(heads * head_dim)),
                "failed to set MHTA SDPA embed dim");
    CheckStatus(sdpa.SetBatchSize(static_cast<int>(batch)),
                "failed to set MHTA SDPA batch size");
    CheckStatus(sdpa.SetHeadsNum(static_cast<int>(heads)),
                "failed to set MHTA SDPA heads");
    CheckStatus(sdpa.SetDropoutP(0.0), "failed to set MHTA SDPA dropout");
    CheckStatus(sdpa.SetScale(static_cast<double>(scale_)),
                "failed to set MHTA SDPA scale");
    CheckStatus(sdpa.SetTraining(false), "failed to set MHTA SDPA training");
    CheckStatus(sdpa.SetMaskMode(false), "failed to set MHTA SDPA mask mode");
    CheckStatus(sdpa.SetKeyFormat(false), "failed to set MHTA SDPA key format");
    CheckStatus(sdpa.SetCausal(false), "failed to set MHTA SDPA causal");
    CheckStatus(sdpa.SetIsDeterministic(true),
                "failed to set MHTA SDPA deterministic");
    CheckStatus(sdpa.SetMaxSeqlenQ(static_cast<int>(seqlen_q)),
                "failed to set MHTA SDPA max seqlen q");
    CheckStatus(sdpa.SetMaxSeqlenK(static_cast<int>(seqlen_k)),
                "failed to set MHTA SDPA max seqlen k");
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
        elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      CheckStatus(
          sdpa.RunFlash(*handle, out_tensor, lse_tensor, q_tensor, k_tensor,
                        v_tensor, mask_tensor, dropout_tensor),
          "MHTA SDPA RunFlash failed");
    } else {
      DeviceBuffer attn_probs;
      attn_probs.Resize(
          static_cast<size_t>(batch * heads * seqlen_q * seqlen_k) *
          MhtaSdpaElementSize(elem_type));
      ::musa::dnn::Tensor probs_tensor;
      SetupTensor(probs_tensor, attn_probs.data(),
                  {batch, heads, seqlen_q, seqlen_k}, elem_type,
                  "attention probabilities");
      auto allocator = [](size_t bytes) -> ::musa::dnn::MemoryHandler {
        void* workspace = nullptr;
        if (bytes != 0 && musaMalloc(&workspace, bytes) != musaSuccess) {
          throw std::runtime_error(MusaErrorString(musaErrorMemoryAllocation));
        }
        return ::musa::dnn::MemoryHandler(
            workspace, [](void* ptr) { (void)musaFree(ptr); });
      };
      CheckStatus(
          sdpa.RunMath(*handle, out_tensor, probs_tensor, q_tensor, k_tensor,
                       v_tensor, mask_tensor, dropout_tensor, allocator),
          "MHTA SDPA RunMath failed");
    }
    return nullptr;
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}

}  // namespace

bool IsMhtaScaledDotProductAttentionFusionGraph(Ort::ConstGraph graph) {
  size_t matmul_count = 0;
  size_t einsum_count = 0;
  size_t mul_count = 0;
  size_t add_count = 0;
  size_t div_count = 0;
  size_t softmax_count = 0;
  size_t unsqueeze_count = 0;
  size_t reshape_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "MatMul")) {
      ++matmul_count;
    } else if (IsOnnxOp(node, "Einsum")) {
      ++einsum_count;
    } else if (IsOnnxOp(node, "Mul")) {
      ++mul_count;
    } else if (IsOnnxOp(node, "Add")) {
      ++add_count;
    } else if (IsOnnxOp(node, "Div")) {
      ++div_count;
    } else if (IsOnnxOp(node, "Softmax")) {
      ++softmax_count;
    } else if (IsOnnxOp(node, "Unsqueeze")) {
      ++unsqueeze_count;
    } else if (IsOnnxOp(node, "Reshape")) {
      ++reshape_count;
    } else {
      return false;
    }
  }
  const bool simple_bhsd = matmul_count == 2 && einsum_count == 0 &&
                           mul_count == 1 && add_count == 1 && div_count == 1 &&
                           softmax_count == 1 && unsqueeze_count == 0 &&
                           reshape_count == 0;
  const bool sim_rank3 = matmul_count == 1 && einsum_count == 1 &&
                         add_count == 1 && softmax_count == 1 &&
                         unsqueeze_count == 1 && reshape_count == 1 &&
                         ((mul_count == 2 && div_count == 0) ||
                          (mul_count == 1 && div_count == 1));
  return simple_bhsd || sim_rank3;
}

std::unique_ptr<FusionNodeCompute> CreateMhtaScaledDotProductAttentionFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode value_matmul{nullptr};
  Ort::ConstNode score_matmul{nullptr};
  Ort::ConstNode mul_node{nullptr};
  Ort::ConstNode add_node{nullptr};
  Ort::ConstNode div_node{nullptr};
  Ort::ConstNode softmax_node{nullptr};
  Ort::ConstNode einsum_node{nullptr};
  Ort::ConstNode unsqueeze_node{nullptr};
  Ort::ConstNode reshape_node{nullptr};
  std::unordered_map<std::string, Ort::ConstNode> producers;
  std::vector<Ort::ConstNode> mul_nodes;
  for (Ort::ConstNode node : graph.GetNodes()) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      producers.emplace(Name(output), node);
    }
    if (IsOnnxOp(node, "Mul")) {
      mul_node = node;
      mul_nodes.push_back(node);
    } else if (IsOnnxOp(node, "Add")) {
      add_node = node;
    } else if (IsOnnxOp(node, "Div")) {
      div_node = node;
    } else if (IsOnnxOp(node, "Softmax")) {
      softmax_node = node;
    } else if (IsOnnxOp(node, "Einsum")) {
      einsum_node = node;
    } else if (IsOnnxOp(node, "Unsqueeze")) {
      unsqueeze_node = node;
    } else if (IsOnnxOp(node, "Reshape")) {
      reshape_node = node;
    }
  }
  if (!add_node || !softmax_node) {
    throw std::runtime_error("invalid MHTA SDPA fused graph");
  }

  auto fused_indices = FusedInputIndices(fused_node);

  if (einsum_node) {
    if (!unsqueeze_node || !reshape_node ||
        !((mul_nodes.size() == 2 && !div_node) ||
          (mul_nodes.size() == 1 && div_node))) {
      throw std::runtime_error("invalid MHTA SDPA sim fused graph");
    }

    std::vector<Ort::ConstValueInfo> softmax_inputs = softmax_node.GetInputs();
    Ort::ConstNode temperature_node = producers.at(Name(softmax_inputs[0]));
    if (!IsOnnxOp(temperature_node, "Mul") &&
        !IsOnnxOp(temperature_node, "Div")) {
      throw std::runtime_error("invalid MHTA SDPA temperature topology");
    }
    std::vector<Ort::ConstValueInfo> temperature_inputs =
        temperature_node.GetInputs();

    int64_t temp_data_index = -1;
    const int64_t temperature_data_end =
        IsOnnxOp(temperature_node, "Div") ? 1 : 2;
    for (int64_t i = 0; i < temperature_data_end; ++i) {
      auto it =
          producers.find(Name(temperature_inputs[static_cast<size_t>(i)]));
      if (it != producers.end() && IsOnnxOp(it->second, "Add")) {
        temp_data_index = i;
        add_node = it->second;
        break;
      }
    }
    if (temp_data_index < 0) {
      throw std::runtime_error("invalid MHTA SDPA temperature topology");
    }
    const float temperature_value = ReadScalarFloatAttributeInput(
        temperature_inputs[static_cast<size_t>(1 - temp_data_index)]);
    if (temperature_value == 0.0f) {
      throw std::runtime_error("MHTA SDPA temperature must not be zero");
    }
    const float temp_recip = IsOnnxOp(temperature_node, "Div")
                                 ? 1.0f / temperature_value
                                 : temperature_value;

    std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
    Ort::ConstNode scale_mul_node{nullptr};
    int64_t add_score_index = -1;
    int64_t scale_data_index = -1;
    for (int64_t i = 0; i < 2; ++i) {
      auto it = producers.find(Name(add_inputs[static_cast<size_t>(i)]));
      if (it == producers.end() || !IsOnnxOp(it->second, "Mul")) {
        continue;
      }
      std::vector<Ort::ConstValueInfo> candidate_inputs =
          it->second.GetInputs();
      if (candidate_inputs.size() != 2) {
        continue;
      }
      for (int64_t j = 0; j < 2; ++j) {
        auto score_it =
            producers.find(Name(candidate_inputs[static_cast<size_t>(j)]));
        if (score_it != producers.end() &&
            IsOnnxOp(score_it->second, "Einsum")) {
          scale_mul_node = it->second;
          einsum_node = score_it->second;
          add_score_index = i;
          scale_data_index = j;
          break;
        }
      }
      if (scale_mul_node) {
        break;
      }
    }
    if (!scale_mul_node) {
      throw std::runtime_error("invalid MHTA SDPA score scale topology");
    }
    std::vector<Ort::ConstValueInfo> scale_mul_inputs =
        scale_mul_node.GetInputs();
    if (scale_data_index < 0 ||
        !musa_ep::IsSupportedMhtaSimRank3Equation(
            musa_ep::GetStringAttribute(einsum_node, "equation")
                .value_or(""))) {
      throw std::runtime_error("invalid MHTA SDPA Einsum topology");
    }
    const float scale =
        ReadScalarFloatAttributeInput(
            scale_mul_inputs[static_cast<size_t>(1 - scale_data_index)]) *
        temp_recip;

    std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
        unsqueeze_node.GetInputs();
    std::vector<Ort::ConstValueInfo> value_inputs =
        producers.at(Name(reshape_node.GetInputs()[0])).GetInputs();
    std::vector<Ort::ConstValueInfo> score_inputs = einsum_node.GetInputs();
    if (unsqueeze_inputs.empty() ||
        Name(unsqueeze_inputs[0]) != Name(softmax_node.GetOutputs()[0]) ||
        score_inputs.size() != 2 || value_inputs.size() != 2) {
      throw std::runtime_error("invalid MHTA SDPA sim value topology");
    }

    const size_t mask_input_index = static_cast<size_t>(1 - add_score_index);
    return std::make_unique<MhtaScaledDotProductAttentionFusionCompute>(
        InputIndex(fused_indices, score_inputs[1]),
        InputIndex(fused_indices, score_inputs[0]),
        InputIndex(fused_indices, value_inputs[1]),
        InputIndex(fused_indices, add_inputs[mask_input_index]), scale,
        temp_recip, MhtaSdpaLayout::kSimRank3);
  }

  if (!mul_node || !div_node) {
    throw std::runtime_error("invalid MHTA SDPA fused graph");
  }

  std::vector<Ort::ConstValueInfo> softmax_inputs = softmax_node.GetInputs();
  std::vector<Ort::ConstValueInfo> div_inputs = div_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> mul_inputs = mul_node.GetInputs();
  score_matmul = producers.at(Name(mul_inputs[0]));
  Ort::ConstValueInfo softmax_output = softmax_node.GetOutputs()[0];
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "MatMul") &&
        Name(node.GetInputs()[0]) == Name(softmax_output)) {
      value_matmul = node;
      break;
    }
  }
  if (!score_matmul || !value_matmul || !IsOnnxOp(score_matmul, "MatMul")) {
    throw std::runtime_error("invalid MHTA SDPA MatMul topology");
  }

  int64_t add_score_index = -1;
  for (int64_t i = 0; i < 2; ++i) {
    if (Name(add_inputs[static_cast<size_t>(i)]) ==
        Name(mul_node.GetOutputs()[0])) {
      add_score_index = i;
    }
  }
  if (add_score_index < 0) {
    throw std::runtime_error("invalid MHTA SDPA Add topology");
  }

  const float scale = ReadScalarFloatAttributeInput(mul_inputs[1]);
  const float temperature = ReadScalarFloatAttributeInput(div_inputs[1]);
  if (temperature == 0.0f) {
    throw std::runtime_error("MHTA SDPA temperature must be non-zero");
  }

  std::vector<Ort::ConstValueInfo> score_inputs = score_matmul.GetInputs();
  std::vector<Ort::ConstValueInfo> value_inputs = value_matmul.GetInputs();
  const size_t mask_input_index = static_cast<size_t>(1 - add_score_index);
  return std::make_unique<MhtaScaledDotProductAttentionFusionCompute>(
      InputIndex(fused_indices, score_inputs[0]),
      InputIndex(fused_indices, score_inputs[1]),
      InputIndex(fused_indices, value_inputs[1]),
      InputIndex(fused_indices, add_inputs[mask_input_index]),
      scale / temperature, 1.0f / temperature, MhtaSdpaLayout::kBhsd);
}
