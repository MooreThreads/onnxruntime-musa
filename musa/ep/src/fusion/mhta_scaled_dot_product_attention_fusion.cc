// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "fusion/mhta_scaled_dot_product_attention_fusion.h"

#include <mudnncxx/mudnn.h>
#include <musa_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph/graph_utils.h"
#include "kernels/shared_inc/blas_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"

namespace {

void NoOpDelete(void*) {}

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

void CheckStatus(::musa::dnn::Status status, const char* message) {
  if (status != ::musa::dnn::Status::SUCCESS) {
    throw std::runtime_error(std::string(message) + ", status: " +
                             std::to_string(static_cast<int>(status)));
  }
}

float ReadScalarFloatAttributeInput(Ort::ConstValueInfo input) {
  if (!input.IsConstantInitializer()) {
    throw std::runtime_error("MHTA SDPA scalar input is not an initializer");
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = input.GetInitializer(value);
  if (!status.IsOK()) {
    throw std::runtime_error("failed to read MHTA SDPA scalar initializer");
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      info.GetElementCount() != 1) {
    throw std::runtime_error("MHTA SDPA scalar initializer must be float");
  }
  return value.GetTensorData<float>()[0];
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

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)musaFree(ptr_);
    }
  }

  void Resize(size_t bytes, musaStream_t stream) {
    if (bytes <= bytes_) {
      stream_ = stream;
      return;
    }
    if (ptr_ != nullptr) {
      FreeDeviceMemoryOnStream(ptr_, stream_, bytes_);
      ptr_ = nullptr;
      bytes_ = 0;
    }
    if (bytes == 0) {
      return;
    }
    ptr_ = AllocateDeviceMemoryOnStream(bytes, stream);
    if (ptr_ == nullptr) {
      throw std::runtime_error(MusaErrorString(musaErrorMemoryAllocation));
    }
    stream_ = stream;
    bytes_ = bytes;
  }

  void* get() const { return ptr_; }

  template <typename T>
  T* data() const {
    return reinterpret_cast<T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t stream_ = nullptr;
};

struct MhtaSdpaScratch {
  DeviceBuffer logsumexp;
  DeviceBuffer dropout_mask;
  DeviceBuffer attn_probs;
  DeviceBuffer q_transposed;
  DeviceBuffer k_transposed;
  DeviceBuffer mask_scaled;
  DeviceBuffer mask_scale_scalar;
  std::vector<std::unique_ptr<DeviceBuffer>> workspaces;
  size_t workspace_index = 0;

  void ResetWorkspace() { workspace_index = 0; }

  void* Workspace(size_t bytes, musaStream_t stream) {
    if (bytes == 0) {
      return nullptr;
    }
    if (workspace_index == workspaces.size()) {
      workspaces.push_back(std::make_unique<DeviceBuffer>());
    }
    DeviceBuffer& buffer = *workspaces[workspace_index++];
    buffer.Resize(bytes, stream);
    return buffer.get();
  }
};

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

void ValidateFloatTensor(Ort::ConstValue value, const char* name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(std::string("MHTA SDPA only supports float ") +
                             name);
  }
}

void SetupFloatTensor(::musa::dnn::Tensor& tensor, const void* data,
                      const std::vector<int64_t>& shape, const char* name) {
  if (!SetMudnnFloatTensor(tensor, data, shape)) {
    throw std::runtime_error(std::string("failed to set MHTA SDPA tensor ") +
                             name);
  }
}

void RunMudnnPermute(::musa::dnn::Handle& handle, const void* input_data,
                     DeviceBuffer& output,
                     const std::vector<int64_t>& input_shape,
                     const std::vector<int64_t>& output_shape,
                     const std::vector<int64_t>& perm, musaStream_t stream,
                     const char* name) {
  output.Resize(static_cast<size_t>(NumElements(output_shape)) * sizeof(float),
                stream);
  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor output_tensor;
  SetupFloatTensor(input_tensor, input_data, input_shape, name);
  SetupFloatTensor(output_tensor, output.data<float>(), output_shape, name);
  ::musa::dnn::Permute op;
  CheckStatus(op.ConfigDimStride(output_tensor, input_tensor,
                                 static_cast<int>(perm.size()), perm.data()),
              "failed to configure MHTA SDPA permute");
  CheckStatus(op.Run(handle, output_tensor, input_tensor),
              "MHTA SDPA permute failed");
}

const void* ScaleMaskIfNeeded(::musa::dnn::Handle& handle,
                              const void* mask_data,
                              const std::vector<int64_t>& mask_shape,
                              float mask_scale, MhtaSdpaScratch& scratch,
                              musaStream_t stream) {
  if (mask_scale == 1.0f) {
    return mask_data;
  }

  scratch.mask_scaled.Resize(
      static_cast<size_t>(NumElements(mask_shape)) * sizeof(float), stream);
  scratch.mask_scale_scalar.Resize(sizeof(float), stream);
  musaError_t copy_status =
      musaMemcpyAsync(scratch.mask_scale_scalar.data<float>(), &mask_scale,
                      sizeof(float), musaMemcpyHostToDevice, stream);
  if (copy_status != musaSuccess) {
    throw std::runtime_error(MusaErrorString(copy_status));
  }

  ::musa::dnn::Tensor mask_tensor;
  ::musa::dnn::Tensor scalar_tensor;
  ::musa::dnn::Tensor output_tensor;
  SetupFloatTensor(mask_tensor, mask_data, mask_shape, "mask");
  SetupFloatTensor(scalar_tensor, scratch.mask_scale_scalar.data<float>(), {1},
                   "mask scale");
  SetupFloatTensor(output_tensor, scratch.mask_scaled.data<float>(), mask_shape,
                   "scaled mask");

  ::musa::dnn::Binary mul;
  CheckStatus(mul.SetMode(::musa::dnn::Binary::Mode::MUL),
              "failed to set MHTA SDPA mask scale mode");
  CheckStatus(mul.Run(handle, output_tensor, mask_tensor, scalar_tensor),
              "MHTA SDPA mask scale failed");
  return scratch.mask_scaled.data<float>();
}

OrtStatus* MhtaScaledDotProductAttentionFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    musaStream_t stream = GetComputeStream(ctx);
    thread_local MhtaSdpaScratch scratch;

    Ort::ConstValue q = ctx.GetInput(q_index_);
    Ort::ConstValue k = ctx.GetInput(k_index_);
    Ort::ConstValue v = ctx.GetInput(v_index_);
    Ort::ConstValue mask = ctx.GetInput(mask_index_);
    ValidateFloatTensor(q, "Q");
    ValidateFloatTensor(k, "K");
    ValidateFloatTensor(v, "V");
    ValidateFloatTensor(mask, "mask");

    std::vector<int64_t> q_shape = Shape(q);
    std::vector<int64_t> k_shape = Shape(k);
    std::vector<int64_t> v_shape = Shape(v);
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "MHTA SDPA requires 4D Q/K/V inputs");
    }

    const bool sim_rank3 = layout_ == MhtaSdpaLayout::kSimRank3;
    std::vector<int64_t> sdpa_q_shape;
    std::vector<int64_t> sdpa_k_shape;
    std::vector<int64_t> sdpa_v_shape;
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
      sdpa_q_shape = {q_shape[0], q_shape[2], q_shape[1], q_shape[3]};
      sdpa_k_shape = {k_shape[0], k_shape[2], k_shape[1], k_shape[3]};
      sdpa_v_shape = v_shape;
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
      sdpa_q_shape = q_shape;
      sdpa_k_shape = k_shape;
      sdpa_v_shape = v_shape;
      output_shape = {q_shape[0], q_shape[1], q_shape[2], v_shape[3]};
    }
    Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
    if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "MHTA SDPA requires MUSA output");
    }

    const int64_t batch = sdpa_q_shape[0];
    const int64_t heads = sim_rank3 ? q_shape[2] : sdpa_q_shape[1];
    const int64_t seqlen_q = sim_rank3 ? q_shape[1] : sdpa_q_shape[2];
    const int64_t seqlen_k = sim_rank3 ? sdpa_k_shape[2] : sdpa_k_shape[3];
    const int64_t head_dim = sdpa_q_shape[3];
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

    scratch.logsumexp.Resize(
        static_cast<size_t>(batch * heads * seqlen_q) * sizeof(float), stream);
    scratch.dropout_mask.Resize(sizeof(float), stream);

    ::musa::dnn::Handle* handle = nullptr;
    RETURN_IF_ERROR(EnsureMudnnHandle(&handle, stream));
    CheckStatus(handle->SetAllowTF32(false),
                "failed to disable TF32 for MHTA SDPA");

    const void* q_data = q_buffer.data();
    const void* k_data = k_buffer.data();
    if (sim_rank3) {
      RunMudnnPermute(*handle, q_buffer.data(), scratch.q_transposed, q_shape,
                      sdpa_q_shape, {0, 2, 1, 3}, stream, "Q");
      RunMudnnPermute(*handle, k_buffer.data(), scratch.k_transposed, k_shape,
                      sdpa_k_shape, {0, 2, 1, 3}, stream, "K");
      q_data = scratch.q_transposed.data<float>();
      k_data = scratch.k_transposed.data<float>();
    }
    const void* mask_data = ScaleMaskIfNeeded(
        *handle, mask_buffer.data(), mask_shape, mask_scale_, scratch, stream);

    ::musa::dnn::Tensor q_tensor;
    ::musa::dnn::Tensor k_tensor;
    ::musa::dnn::Tensor v_tensor;
    ::musa::dnn::Tensor mask_tensor;
    ::musa::dnn::Tensor output_tensor;
    ::musa::dnn::Tensor lse_tensor;
    ::musa::dnn::Tensor dropout_tensor;
    SetupFloatTensor(q_tensor, q_data, sdpa_q_shape, "Q");
    SetupFloatTensor(k_tensor, k_data, sdpa_k_shape, "K");
    SetupFloatTensor(v_tensor, v_buffer.data(), sdpa_v_shape, "V");
    SetupFloatTensor(mask_tensor, mask_data, mask_shape, "mask");
    SetupFloatTensor(output_tensor, output.GetTensorMutableData<float>(),
                     sim_rank3 ? sdpa_q_shape : output_shape, "output");
    SetupFloatTensor(lse_tensor, scratch.logsumexp.data<float>(),
                     {batch, heads, seqlen_q}, "logsumexp");
    SetupFloatTensor(dropout_tensor, scratch.dropout_mask.data<float>(), {1},
                     "dropout mask");

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
    CheckStatus(sdpa.SetKeyFormat(!sim_rank3),
                "failed to set MHTA SDPA key format");
    CheckStatus(sdpa.SetCausal(false), "failed to set MHTA SDPA causal");
    CheckStatus(sdpa.SetIsDeterministic(true),
                "failed to set MHTA SDPA deterministic");
    CheckStatus(sdpa.SetMaxSeqlenQ(static_cast<int>(seqlen_q)),
                "failed to set MHTA SDPA max seqlen q");
    CheckStatus(sdpa.SetMaxSeqlenK(static_cast<int>(seqlen_k)),
                "failed to set MHTA SDPA max seqlen k");

    scratch.attn_probs.Resize(
        static_cast<size_t>(batch * heads * seqlen_q * seqlen_k) *
            sizeof(float),
        stream);
    ::musa::dnn::Tensor attn_probs_tensor;
    SetupFloatTensor(attn_probs_tensor, scratch.attn_probs.data<float>(),
                     {batch, heads, seqlen_q, seqlen_k}, "attention probs");
    scratch.ResetWorkspace();
    auto allocator = [stream](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) {
        return ::musa::dnn::MemoryHandler(nullptr, NoOpDelete);
      }
      return ::musa::dnn::MemoryHandler(scratch.Workspace(size, stream),
                                        NoOpDelete);
    };
    CheckStatus(sdpa.RunMath(*handle, output_tensor, attn_probs_tensor,
                             q_tensor, k_tensor, v_tensor, mask_tensor,
                             dropout_tensor, allocator),
                "MHTA SDPA RunMath failed");
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
                         mul_count == 2 && add_count == 1 && div_count == 0 &&
                         softmax_count == 1 && unsqueeze_count == 1 &&
                         reshape_count == 1;
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
    if (!unsqueeze_node || !reshape_node || mul_nodes.size() != 2) {
      throw std::runtime_error("invalid MHTA SDPA sim fused graph");
    }

    std::vector<Ort::ConstValueInfo> softmax_inputs = softmax_node.GetInputs();
    std::vector<Ort::ConstValueInfo> temp_mul_inputs =
        producers.at(Name(softmax_inputs[0])).GetInputs();
    Ort::ConstNode temp_mul_node = producers.at(Name(softmax_inputs[0]));
    if (!IsOnnxOp(temp_mul_node, "Mul")) {
      throw std::runtime_error("invalid MHTA SDPA temperature topology");
    }

    int64_t temp_data_index = -1;
    for (int64_t i = 0; i < 2; ++i) {
      auto it = producers.find(Name(temp_mul_inputs[static_cast<size_t>(i)]));
      if (it != producers.end() && IsOnnxOp(it->second, "Add")) {
        temp_data_index = i;
        add_node = it->second;
        break;
      }
    }
    if (temp_data_index < 0) {
      throw std::runtime_error("invalid MHTA SDPA temperature topology");
    }
    const float temp_recip = ReadScalarFloatAttributeInput(
        temp_mul_inputs[static_cast<size_t>(1 - temp_data_index)]);

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
        musa_ep::GetStringAttribute(einsum_node, "equation").value_or("") !=
            "ilhw,bjhw->bhl") {
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
