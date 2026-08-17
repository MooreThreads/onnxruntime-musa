#include "fusion/multi_kqv_mha_output_projection_fusion.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "kernels/llm/attention_impl.h"
#include "kernels/llm/reduced_mha_flash_impl.h"
#include "kernels/math/matmul.h"
#include "kernels/shared_inc/blas_utils.h"
#include "kernels/shared_inc/kernel_memory.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

using Pattern = MultiKqvMhaOutputProjectionPattern;
using Branch = MultiKqvMhaProjectionBranch;

Ort::ConstNode OnlyConsumer(Ort::ConstValueInfo value) {
  auto consumers = value.GetConsumers();
  return consumers.size() == 1 ? consumers[0].node : Ort::ConstNode{nullptr};
}

bool HasConsumerAt(Ort::ConstValueInfo value, Ort::ConstNode expected,
                   int64_t expected_index) {
  auto consumers = value.GetConsumers();
  return consumers.size() == 1 &&
         consumers[0].node.GetId() == expected.GetId() &&
         consumers[0].index == expected_index;
}

bool IsAxisZero(Ort::ConstNode node) {
  return GetIntAttribute(node, "axis").value_or(0) == 0;
}

float GetFloatAttribute(Ort::ConstNode node, const char* name, float fallback);

bool IsSimpleSlice(Ort::ConstNode node) {
  if (!IsOnnxOp(node, "Slice")) return false;
  auto inputs = node.GetInputs();
  auto outputs = node.GetOutputs();
  if (inputs.size() != 5 || outputs.size() != 1 ||
      !IsFloatTensorValueInfo(inputs[0]) ||
      !IsFloatTensorValueInfo(outputs[0]) || !IsIntTensorValueInfo(inputs[1]) ||
      !IsIntTensorValueInfo(inputs[2])) {
    return false;
  }
  // The source topology uses row slices with unit stride.  If axes/steps are
  // graph inputs rather than initializers, leave their value to ONNX Runtime;
  // the runtime implementation uses the same contiguous row partition.
  if (auto axes = ReadSmallIntInitializer(inputs[3]);
      axes.has_value() && (axes->size() != 1 || (*axes)[0] != 0)) {
    return false;
  }
  if (auto steps = ReadSmallIntInitializer(inputs[4]);
      steps.has_value() && (steps->size() != 1 || (*steps)[0] != 1)) {
    return false;
  }
  return true;
}

bool IsProjectionBranch(Ort::ConstNode slice, Branch& branch) {
  if (!IsSimpleSlice(slice)) return false;
  auto matmul = OnlyConsumer(slice.GetOutputs()[0]);
  if (!IsOnnxOp(matmul, "MatMul")) return false;
  auto mi = matmul.GetInputs();
  auto mo = matmul.GetOutputs();
  if (mi.size() != 2 || mo.size() != 1 || !IsFloatTensorValueInfo(mi[1]) ||
      !mi[1].IsConstantInitializer() || !IsFloatTensorValueInfo(mo[0])) {
    return false;
  }
  auto split = OnlyConsumer(mo[0]);
  const int64_t split_axis = GetIntAttribute(split, "axis").value_or(1);
  if (!IsOnnxOp(split, "Split") || (split_axis != 1 && split_axis != -1))
    return false;
  auto so = split.GetOutputs();
  if (so.size() != 3) return false;
  for (auto output : so) {
    auto consumer = OnlyConsumer(output);
    if (!IsOnnxOp(consumer, "Concat") || !IsAxisZero(consumer)) return false;
  }
  branch = {slice, matmul, split};
  return true;
}

bool SameNode(Ort::ConstNode lhs, Ort::ConstNode rhs) {
  return lhs && rhs && lhs.GetId() == rhs.GetId();
}

bool BuildBranches(Ort::ConstNode candidate, std::vector<Branch>& branches) {
  auto ci = candidate.GetInputs();
  if (ci.empty()) return false;
  Ort::ConstValueInfo source = ci[0];
  if (!IsFloatTensorValueInfo(source)) return false;

  for (const auto& consumer : source.GetConsumers()) {
    Branch branch;
    if (IsProjectionBranch(consumer.node, branch)) branches.push_back(branch);
  }
  if (branches.size() < 2) return false;
  for (const auto& branch : branches) {
    if (branch.slice.GetInputs()[0].GetName() != source.GetName()) return false;
  }
  return true;
}

bool ConcatContainsSplitOutputs(Ort::ConstNode concat,
                                const std::vector<Branch>& branches,
                                size_t split_output_index) {
  auto inputs = concat.GetInputs();
  if (inputs.size() != branches.size()) return false;
  std::unordered_set<std::string> expected;
  for (const auto& branch : branches) {
    expected.insert(branch.split.GetOutputs()[split_output_index].GetName());
  }
  if (expected.size() != branches.size()) return false;
  for (auto input : inputs) {
    if (expected.erase(input.GetName()) == 0) return false;
  }
  return expected.empty();
}

bool BuildQkvConcatNodes(const std::vector<Branch>& branches,
                         Ort::ConstNode& q_concat, Ort::ConstNode& k_concat,
                         Ort::ConstNode& v_concat) {
  std::array<Ort::ConstNode, 3> found{Ort::ConstNode{nullptr},
                                      Ort::ConstNode{nullptr},
                                      Ort::ConstNode{nullptr}};
  for (size_t output_index = 0; output_index < 3; ++output_index) {
    auto first_output = branches[0].split.GetOutputs()[output_index];
    auto concat = OnlyConsumer(first_output);
    if (!IsOnnxOp(concat, "Concat") || !IsAxisZero(concat) ||
        !ConcatContainsSplitOutputs(concat, branches, output_index)) {
      return false;
    }
    found[output_index] = concat;
  }
  if (SameNode(found[0], found[1]) || SameNode(found[0], found[2]) ||
      SameNode(found[1], found[2])) {
    return false;
  }
  q_concat = found[0];
  k_concat = found[1];
  v_concat = found[2];
  return true;
}

bool BuildUnsqueezeAndMha(const Pattern& partial, Ort::ConstNode& q_unsqueeze,
                          Ort::ConstNode& k_unsqueeze,
                          Ort::ConstNode& v_unsqueeze, Ort::ConstNode& mha) {
  Ort::ConstNode unsqueezes[3] = {
      OnlyConsumer(partial.q_concat.GetOutputs()[0]),
      OnlyConsumer(partial.k_concat.GetOutputs()[0]),
      OnlyConsumer(partial.v_concat.GetOutputs()[0])};
  for (auto node : unsqueezes) {
    if (!IsOnnxOp(node, "Unsqueeze") || !ReadUnsqueezeAxes(node).has_value() ||
        ReadUnsqueezeAxes(node)->size() != 1 ||
        ReadUnsqueezeAxes(node).value()[0] != 0) {
      return false;
    }
  }
  auto candidate_mha = OnlyConsumer(unsqueezes[0].GetOutputs()[0]);
  if (!candidate_mha ||
      candidate_mha.GetOperatorType() != "MultiHeadAttention" ||
      candidate_mha.GetDomain() != "com.microsoft") {
    return false;
  }
  for (size_t i = 0; i < 3; ++i) {
    if (!HasConsumerAt(unsqueezes[i].GetOutputs()[0], candidate_mha,
                       static_cast<int64_t>(i))) {
      return false;
    }
  }
  auto inputs = candidate_mha.GetInputs();
  if (inputs.size() != 4 && inputs.size() != 5) return false;
  if (!IsFloatTensorValueInfo(inputs[0]) ||
      !IsFloatTensorValueInfo(inputs[1]) ||
      !IsFloatTensorValueInfo(inputs[2]) ||
      !IsFloatTensorValueInfo(inputs[3]) ||
      !inputs[3].IsConstantInitializer() ||
      (inputs.size() == 5 &&
       (!IsInt32TensorValueInfo(inputs[4]) ||
        GetTensorShape(inputs[4]).value_or(std::vector<int64_t>{}).size() !=
            3))) {
    return false;
  }
  if (GetIntAttribute(candidate_mha, "num_heads").value_or(0) <= 0)
    return false;
  q_unsqueeze = unsqueezes[0];
  k_unsqueeze = unsqueezes[1];
  v_unsqueeze = unsqueezes[2];
  mha = candidate_mha;
  return true;
}

bool BuildOutputProjection(
    const Pattern& partial, Ort::ConstNode& gather,
    std::vector<Ort::ConstNode>& output_slices,
    std::vector<Ort::ConstNode>& output_gemms, Ort::ConstNode& output_concat,
    const std::unordered_set<std::string>& graph_outputs) {
  auto mha_output = partial.mha.GetOutputs();
  if (mha_output.size() != 1) return false;
  gather = OnlyConsumer(mha_output[0]);
  if (!IsOnnxOp(gather, "Gather") ||
      GetIntAttribute(gather, "axis").value_or(0) != 0 ||
      gather.GetInputs().size() != 2 ||
      ReadScalarIntInitializer(gather.GetInputs()[1]).value_or(-1) != 0) {
    return false;
  }
  auto gather_output = gather.GetOutputs();
  if (gather_output.size() != 1) return false;
  auto consumers = gather_output[0].GetConsumers();
  if (consumers.size() < 2) return false;

  std::vector<Ort::ConstNode> discovered_slices;
  for (const auto& consumer : consumers) {
    if (!IsSimpleSlice(consumer.node) || consumer.index != 0 ||
        consumer.node.GetInputs()[0].GetName() != gather_output[0].GetName()) {
      return false;
    }
    discovered_slices.push_back(consumer.node);
  }
  if (discovered_slices.size() < 2) return false;

  std::vector<Ort::ConstNode> discovered_gemms;
  for (auto slice : discovered_slices) {
    auto gemm = OnlyConsumer(slice.GetOutputs()[0]);
    if (!IsOnnxOp(gemm, "Gemm")) return false;
    auto inputs = gemm.GetInputs();
    if (inputs.size() != 3 || !IsFloatTensorValueInfo(inputs[1]) ||
        !IsFloatTensorValueInfo(inputs[2]) ||
        !inputs[1].IsConstantInitializer() ||
        !inputs[2].IsConstantInitializer() ||
        GetIntAttribute(gemm, "transA").value_or(0) != 0 ||
        GetIntAttribute(gemm, "transB").value_or(0) != 1 ||
        GetFloatAttribute(gemm, "alpha", 1.0f) != 1.0f ||
        GetFloatAttribute(gemm, "beta", 1.0f) != 1.0f) {
      return false;
    }
    discovered_gemms.push_back(gemm);
  }

  auto first_output = discovered_gemms[0].GetOutputs();
  if (first_output.size() != 1) return false;
  output_concat = OnlyConsumer(first_output[0]);
  if (!IsOnnxOp(output_concat, "Concat") || !IsAxisZero(output_concat) ||
      output_concat.GetInputs().size() != discovered_gemms.size()) {
    return false;
  }
  output_slices.clear();
  output_gemms.clear();
  for (auto input : output_concat.GetInputs()) {
    Ort::ConstNode gemm{nullptr};
    if (!GetProducer(input, gemm) || !IsOnnxOp(gemm, "Gemm")) return false;
    auto gi = gemm.GetInputs();
    if (gi.size() != 3) return false;
    Ort::ConstNode slice{nullptr};
    if (!GetProducer(gi[0], slice)) return false;
    if (!IsOnnxOp(slice, "Slice") ||
        slice.GetInputs()[0].GetName() != gather_output[0].GetName()) {
      return false;
    }
    output_gemms.push_back(gemm);
    output_slices.push_back(slice);
  }
  if (output_gemms.size() != discovered_gemms.size() ||
      output_slices.size() != discovered_slices.size()) {
    return false;
  }
  (void)graph_outputs;
  return true;
}

std::vector<Ort::ConstNode> PatternNodes(const Pattern& p) {
  std::vector<Ort::ConstNode> nodes;
  for (const auto& branch : p.branches) {
    nodes.push_back(branch.slice);
    nodes.push_back(branch.matmul);
    nodes.push_back(branch.split);
  }
  nodes.insert(nodes.end(),
               {p.q_concat, p.k_concat, p.v_concat, p.q_unsqueeze,
                p.k_unsqueeze, p.v_unsqueeze, p.mha, p.output_gather});
  nodes.insert(nodes.end(), p.output_slices.begin(), p.output_slices.end());
  nodes.insert(nodes.end(), p.output_gemms.begin(), p.output_gemms.end());
  nodes.push_back(p.output_concat);
  return nodes;
}

bool BuildPattern(Ort::ConstNode candidate,
                  const std::unordered_set<std::string>& graph_outputs,
                  Pattern& pattern) {
  if (!BuildBranches(candidate, pattern.branches)) return false;
  if (!BuildQkvConcatNodes(pattern.branches, pattern.q_concat, pattern.k_concat,
                           pattern.v_concat)) {
    return false;
  }
  if (!BuildUnsqueezeAndMha(pattern, pattern.q_unsqueeze, pattern.k_unsqueeze,
                            pattern.v_unsqueeze, pattern.mha)) {
    return false;
  }
  if (!BuildOutputProjection(pattern, pattern.output_gather,
                             pattern.output_slices, pattern.output_gemms,
                             pattern.output_concat, graph_outputs)) {
    return false;
  }

  // Reorder projection branches according to the Q concat.  This is the
  // sequence order consumed by MHA and avoids relying on node-name suffixes.
  std::unordered_map<size_t, Branch> by_split;
  for (const auto& branch : pattern.branches) {
    by_split.emplace(branch.split.GetId(), branch);
  }
  std::vector<Branch> ordered;
  for (auto input : pattern.q_concat.GetInputs()) {
    Ort::ConstNode split{nullptr};
    if (!GetProducer(input, split)) return false;
    auto it = by_split.find(split.GetId());
    if (it == by_split.end()) return false;
    ordered.push_back(it->second);
  }
  if (ordered.size() != pattern.branches.size()) return false;
  for (size_t i = 0; i < ordered.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      auto expected = ordered[i].split.GetOutputs()[j].GetName();
      auto actual = pattern.k_concat.GetInputs()[i + 0].GetName();
      (void)actual;
      if (j == 1 && expected != pattern.k_concat.GetInputs()[i].GetName()) {
        return false;
      }
      if (j == 2 && expected != pattern.v_concat.GetInputs()[i].GetName()) {
        return false;
      }
    }
  }
  pattern.branches = std::move(ordered);

  std::vector<Ort::ConstNode> nodes = PatternNodes(pattern);
  std::unordered_set<size_t> selected;
  for (auto node : nodes) {
    if (!selected.insert(node.GetId()).second) continue;
  }
  return FusionHasNoExternalPathBetweenSelectedNodes(nodes, selected);
}

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused) {
  std::unordered_map<std::string, size_t> result;
  auto inputs = fused.GetInputs();
  for (size_t i = 0; i < inputs.size(); ++i) result.emplace(Name(inputs[i]), i);
  return result;
}

size_t FusedInputIndex(const std::unordered_map<std::string, size_t>& indices,
                       Ort::ConstValueInfo value) {
  auto it = indices.find(Name(value));
  if (it == indices.end()) {
    throw std::runtime_error("MultiKqvMhaOutputProjection missing fused input");
  }
  return it->second;
}

float GetFloatAttribute(Ort::ConstNode node, const char* name, float fallback) {
  Ort::ConstOpAttr attr;
  float value = fallback;
  return node.GetAttributeByName(name, attr).IsOK() &&
                 attr.GetValue(value).IsOK()
             ? value
             : fallback;
}

class Scratch {
 public:
  ~Scratch() {
    if (ptr_ != nullptr) FreeDeviceMemoryOnStream(ptr_, stream_, bytes_);
  }
  void Allocate(size_t bytes, musaStream_t stream) {
    bytes_ = bytes;
    stream_ = stream;
    if (bytes != 0) {
      ptr_ = AllocateDeviceMemoryOnStream(bytes, stream);
      if (ptr_ == nullptr) {
        throw std::runtime_error(MusaErrorString(musaErrorMemoryAllocation));
      }
    }
  }
  void* data() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t stream_ = nullptr;
};

OrtStatus* RunProjectionWithBias(float* y, const void* x, const void* weight,
                                 const void* bias,
                                 const std::vector<int64_t>& x_shape,
                                 const std::vector<int64_t>& weight_shape,
                                 const std::vector<int64_t>& y_shape,
                                 musaStream_t stream) {
  const std::vector<int64_t> bias_shape{y_shape.back()};
  if (TryMudnnGemm(y, x, weight, bias, x_shape, weight_shape, bias_shape,
                   y_shape, false, true, 1.0f, 1.0f, true,
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, stream)) {
    return nullptr;
  }
  RETURN_IF_ERROR(ComputeMusaMatMulDevice(
      x, weight, y, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, x_shape, weight_shape,
      y_shape, false, true, false, false, 1.0f, stream));
  return LaunchStatus(LaunchMusaAttentionAddBiasKernel(
      y, static_cast<const float*>(bias), y_shape[0] * y_shape[1], y_shape[1],
      stream));
}

OrtStatus* ReadRuntimeScalarInteger(Ort::ConstValue value, musaStream_t stream,
                                    int64_t& result) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1 ||
      (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
       info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL,
        "MultiKqvMhaOutputProjection requires scalar INT32/INT64 Slice bounds");
  }
  std::vector<uint8_t> bytes;
  RETURN_IF_ERROR(CopyToHost(value, bytes, stream));
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    int32_t v = 0;
    std::memcpy(&v, bytes.data(), sizeof(v));
    result = v;
  } else {
    std::memcpy(&result, bytes.data(), sizeof(result));
  }
  return nullptr;
}

class MultiKqvMhaOutputProjectionFusionCompute final
    : public FusionNodeCompute {
 public:
  MultiKqvMhaOutputProjectionFusionCompute(
      size_t source, size_t mha_bias, size_t mask,
      std::vector<size_t> qkv_weights, std::vector<size_t> output_weights,
      std::vector<size_t> output_biases, std::vector<size_t> input_starts,
      std::vector<size_t> input_ends, std::vector<size_t> output_starts,
      std::vector<size_t> output_ends, int64_t heads, float scale)
      : source_(source),
        mha_bias_(mha_bias),
        mask_(mask),
        qkv_weights_(std::move(qkv_weights)),
        output_weights_(std::move(output_weights)),
        output_biases_(std::move(output_biases)),
        input_starts_(std::move(input_starts)),
        input_ends_(std::move(input_ends)),
        output_starts_(std::move(output_starts)),
        output_ends_(std::move(output_ends)),
        heads_(heads),
        scale_(scale) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      musaStream_t stream = GetComputeStream(ctx);
      Ort::ConstValue source = ctx.GetInput(source_);
      Ort::ConstValue mha_bias = ctx.GetInput(mha_bias_);
      auto source_info = source.GetTensorTypeAndShapeInfo();
      auto source_shape = source_info.GetShape();
      auto bias_shape = mha_bias.GetTensorTypeAndShapeInfo().GetShape();
      if (source_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          source_shape.size() != 2 || bias_shape.size() != 1 ||
          bias_shape[0] <= 0 || bias_shape[0] % 3 != 0 || heads_ <= 0 ||
          qkv_weights_.size() < 2 ||
          qkv_weights_.size() != output_weights_.size() ||
          qkv_weights_.size() != output_biases_.size() ||
          qkv_weights_.size() != input_starts_.size() ||
          qkv_weights_.size() != input_ends_.size() ||
          qkv_weights_.size() != output_starts_.size() ||
          qkv_weights_.size() != output_ends_.size()) {
        throw std::runtime_error(
            "MultiKqvMhaOutputProjection requires static FP32 rank-2 inputs");
      }
      const int64_t rows = source_shape[0];
      const int64_t input_dim = source_shape[1];
      const int64_t attention_dim = bias_shape[0] / 3;
      if (rows < 0 || input_dim <= 0 || attention_dim <= 0 ||
          attention_dim % heads_ != 0) {
        throw std::runtime_error(
            "MultiKqvMhaOutputProjection requires valid sequence slices");
      }

      std::vector<int64_t> source_starts(qkv_weights_.size());
      std::vector<int64_t> source_ends(qkv_weights_.size());
      std::vector<int64_t> output_starts(qkv_weights_.size());
      std::vector<int64_t> output_ends(qkv_weights_.size());
      for (size_t i = 0; i < qkv_weights_.size(); ++i) {
        RETURN_IF_ERROR(ReadRuntimeScalarInteger(ctx.GetInput(input_starts_[i]),
                                                 stream, source_starts[i]));
        RETURN_IF_ERROR(ReadRuntimeScalarInteger(ctx.GetInput(input_ends_[i]),
                                                 stream, source_ends[i]));
        RETURN_IF_ERROR(ReadRuntimeScalarInteger(
            ctx.GetInput(output_starts_[i]), stream, output_starts[i]));
        RETURN_IF_ERROR(ReadRuntimeScalarInteger(ctx.GetInput(output_ends_[i]),
                                                 stream, output_ends[i]));
        if (source_starts[i] < 0 || source_ends[i] < source_starts[i] ||
            source_ends[i] > rows || output_starts[i] < 0 ||
            output_ends[i] < output_starts[i] || output_ends[i] > rows ||
            source_ends[i] - source_starts[i] !=
                output_ends[i] - output_starts[i]) {
          throw std::runtime_error(
              "MultiKqvMhaOutputProjection Slice bounds are invalid");
        }
      }
      int64_t sequence = 0;
      for (size_t i = 0; i < qkv_weights_.size(); ++i) {
        if (source_starts[i] != sequence) {
          throw std::runtime_error(
              "MultiKqvMhaOutputProjection source slices are not contiguous");
        }
        sequence += source_ends[i] - source_starts[i];
      }
      if (sequence != rows) {
        throw std::runtime_error(
            "MultiKqvMhaOutputProjection source slices do not cover input");
      }

      std::vector<Ort::ConstValue> qkv_values;
      std::vector<Ort::ConstValue> output_values;
      std::vector<Ort::ConstValue> bias_values;
      qkv_values.reserve(qkv_weights_.size());
      output_values.reserve(output_weights_.size());
      bias_values.reserve(output_biases_.size());
      for (size_t i = 0; i < qkv_weights_.size(); ++i) {
        qkv_values.push_back(ctx.GetInput(qkv_weights_[i]));
        output_values.push_back(ctx.GetInput(output_weights_[i]));
        bias_values.push_back(ctx.GetInput(output_biases_[i]));
      }

      auto output_weight_shape =
          output_values[0].GetTensorTypeAndShapeInfo().GetShape();
      auto output_bias_shape =
          bias_values[0].GetTensorTypeAndShapeInfo().GetShape();
      if (output_weight_shape.size() != 2 || output_bias_shape.size() != 1 ||
          output_weight_shape[1] != attention_dim ||
          output_bias_shape[0] != output_weight_shape[0] ||
          output_weight_shape[0] <= 0) {
        throw std::runtime_error(
            "MultiKqvMhaOutputProjection output projection dimensions "
            "mismatch");
      }
      const int64_t output_dim = output_weight_shape[0];
      for (size_t i = 0; i < qkv_values.size(); ++i) {
        auto qkv_shape = qkv_values[i].GetTensorTypeAndShapeInfo().GetShape();
        auto weight_shape =
            output_values[i].GetTensorTypeAndShapeInfo().GetShape();
        auto out_bias = bias_values[i].GetTensorTypeAndShapeInfo().GetShape();
        if (qkv_values[i].GetTensorTypeAndShapeInfo().GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            qkv_shape.size() != 2 || qkv_shape[0] != input_dim ||
            qkv_shape[1] != 3 * attention_dim || weight_shape.size() != 2 ||
            weight_shape[0] != output_dim || weight_shape[1] != attention_dim ||
            out_bias.size() != 1 || out_bias[0] != output_dim) {
          throw std::runtime_error(
              "MultiKqvMhaOutputProjection projection dimensions mismatch");
        }
      }

      Ort::ConstValue mask{nullptr};
      bool has_mask = mask_ != SIZE_MAX;
      std::vector<int64_t> mask_shape;
      if (has_mask) {
        mask = ctx.GetInput(mask_);
        auto mask_info = mask.GetTensorTypeAndShapeInfo();
        mask_shape = mask_info.GetShape();
        if (mask_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
            mask_shape.size() != 3 || mask_shape[0] != 1 ||
            mask_shape[1] != rows || mask_shape[2] != rows) {
          throw std::runtime_error(
              "MultiKqvMhaOutputProjection requires INT32 mask [1,S,S]");
        }
      }

      Ort::UnownedValue output = ctx.GetOutput(0, {rows, output_dim});
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        throw std::runtime_error(
            "MultiKqvMhaOutputProjection requires MUSA output");
      }

      DeviceInputBuffer source_buffer;
      DeviceInputBuffer mha_bias_buffer;
      DeviceInputBuffer mask_buffer;
      RETURN_IF_ERROR(source_buffer.Bind(source, stream));
      RETURN_IF_ERROR(mha_bias_buffer.Bind(mha_bias, stream));
      if (has_mask) RETURN_IF_ERROR(mask_buffer.Bind(mask, stream));
      std::vector<DeviceInputBuffer> qkv_buffers(qkv_values.size());
      std::vector<DeviceInputBuffer> output_buffers(output_values.size());
      std::vector<DeviceInputBuffer> bias_buffers(bias_values.size());
      for (size_t i = 0; i < qkv_values.size(); ++i) {
        RETURN_IF_ERROR(qkv_buffers[i].Bind(qkv_values[i], stream));
        RETURN_IF_ERROR(output_buffers[i].Bind(output_values[i], stream));
        RETURN_IF_ERROR(bias_buffers[i].Bind(bias_values[i], stream));
      }

      Scratch packed_qkv;
      Scratch attention_output;
      packed_qkv.Allocate(
          static_cast<size_t>(rows * 3 * attention_dim) * sizeof(float),
          stream);
      attention_output.Allocate(
          static_cast<size_t>(rows * attention_dim) * sizeof(float), stream);

      const float* source_data =
          static_cast<const float*>(source_buffer.data());
      float* packed_data = static_cast<float*>(packed_qkv.data());
      const int64_t head_dim = attention_dim / heads_;
      int64_t packed_offset = 0;
      for (size_t i = 0; i < qkv_buffers.size(); ++i) {
        const int64_t segment = source_ends[i] - source_starts[i];
        RETURN_IF_ERROR(ComputeMusaMatMulDevice(
            source_data + source_starts[i] * input_dim, qkv_buffers[i].data(),
            packed_data + packed_offset * 3 * attention_dim,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {segment, input_dim},
            {input_dim, 3 * attention_dim}, {segment, 3 * attention_dim}, false,
            false, false, false, 1.0f, stream));
        packed_offset += segment;
      }
      RETURN_IF_ERROR(LaunchStatus(LaunchMusaAttentionAddBiasKernel(
          packed_data, static_cast<const float*>(mha_bias_buffer.data()),
          rows * 3 * attention_dim, 3 * attention_dim, stream)));

      MusaReducedMhaFlashParams params{
          1,
          rows,
          attention_dim,
          heads_,
          head_dim,
          1,
          1,
          std::isnan(scale_) ? 1.0f / std::sqrt(static_cast<float>(head_dim))
                             : scale_,
          -10000.0f,
          has_mask,
          true,
          true};
      RETURN_IF_ERROR(LaunchStatus(LaunchMusaReducedMhaFlashKernel(
          packed_data,
          has_mask ? static_cast<const int32_t*>(mask_buffer.data()) : nullptr,
          static_cast<float*>(attention_output.data()), params, stream)));

      float* output_data = output.GetTensorMutableData<float>();
      packed_offset = 0;
      for (size_t i = 0; i < output_buffers.size(); ++i) {
        const int64_t segment = output_ends[i] - output_starts[i];
        RETURN_IF_ERROR(RunProjectionWithBias(
            output_data + output_starts[i] * output_dim,
            static_cast<const float*>(attention_output.data()) +
                packed_offset * attention_dim,
            output_buffers[i].data(), bias_buffers[i].data(),
            {segment, attention_dim}, {output_dim, attention_dim},
            {segment, output_dim}, stream));
        packed_offset += segment;
      }
      return nullptr;
    } catch (const std::exception& e) {
      return Ort::Status(e.what(), ORT_EP_FAIL).release();
    }
  }

 private:
  size_t source_;
  size_t mha_bias_;
  size_t mask_;
  std::vector<size_t> qkv_weights_;
  std::vector<size_t> output_weights_;
  std::vector<size_t> output_biases_;
  std::vector<size_t> input_starts_;
  std::vector<size_t> input_ends_;
  std::vector<size_t> output_starts_;
  std::vector<size_t> output_ends_;
  int64_t heads_;
  float scale_;
};

}  // namespace

bool TryBuildMultiKqvMhaOutputProjectionPattern(
    Ort::ConstNode candidate,
    const std::unordered_set<std::string>& graph_output_names,
    MultiKqvMhaOutputProjectionPattern& pattern) {
  return BuildPattern(candidate, graph_output_names, pattern);
}

bool IsMultiKqvMhaOutputProjectionFusionGraph(Ort::ConstGraph graph) {
  std::unordered_set<std::string> graph_outputs;
  for (auto output : graph.GetOutputs()) graph_outputs.insert(Name(output));
  for (auto node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Slice")) continue;
    Pattern pattern;
    if (TryBuildMultiKqvMhaOutputProjectionPattern(node, graph_outputs,
                                                   pattern)) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<FusionNodeCompute> CreateMultiKqvMhaOutputProjectionFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  std::unordered_set<std::string> graph_outputs;
  for (auto output : graph.GetOutputs()) graph_outputs.insert(Name(output));
  Pattern pattern;
  bool found = false;
  for (auto node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Slice") && TryBuildMultiKqvMhaOutputProjectionPattern(
                                       node, graph_outputs, pattern)) {
      found = true;
      break;
    }
  }
  if (!found)
    throw std::runtime_error("invalid MultiKqvMhaOutputProjection graph");

  auto indices = FusedInputIndices(fused_node);
  auto mha_inputs = pattern.mha.GetInputs();
  std::vector<size_t> qkv_weights;
  std::vector<size_t> output_weights;
  std::vector<size_t> output_biases;
  std::vector<size_t> input_starts;
  std::vector<size_t> input_ends;
  std::vector<size_t> output_starts;
  std::vector<size_t> output_ends;
  qkv_weights.reserve(pattern.branches.size());
  input_starts.reserve(pattern.branches.size());
  input_ends.reserve(pattern.branches.size());
  output_weights.reserve(pattern.output_gemms.size());
  output_biases.reserve(pattern.output_gemms.size());
  output_starts.reserve(pattern.output_slices.size());
  output_ends.reserve(pattern.output_slices.size());
  for (const auto& branch : pattern.branches) {
    qkv_weights.push_back(
        FusedInputIndex(indices, branch.matmul.GetInputs()[1]));
    input_starts.push_back(
        FusedInputIndex(indices, branch.slice.GetInputs()[1]));
    input_ends.push_back(FusedInputIndex(indices, branch.slice.GetInputs()[2]));
  }
  for (size_t i = 0; i < pattern.output_gemms.size(); ++i) {
    auto gemm = pattern.output_gemms[i];
    auto inputs = gemm.GetInputs();
    output_weights.push_back(FusedInputIndex(indices, inputs[1]));
    output_biases.push_back(FusedInputIndex(indices, inputs[2]));
    auto slice_inputs = pattern.output_slices[i].GetInputs();
    output_starts.push_back(FusedInputIndex(indices, slice_inputs[1]));
    output_ends.push_back(FusedInputIndex(indices, slice_inputs[2]));
  }
  const size_t mask = mha_inputs.size() == 5
                          ? FusedInputIndex(indices, mha_inputs[4])
                          : SIZE_MAX;
  const int64_t heads = GetIntAttribute(pattern.mha, "num_heads").value_or(0);
  const int64_t attention_dim =
      GetTensorShape(mha_inputs[3]).value_or(std::vector<int64_t>{0})[0] / 3;
  const float default_scale =
      attention_dim > 0 && heads > 0
          ? 1.0f / std::sqrt(static_cast<float>(attention_dim / heads))
          : std::numeric_limits<float>::quiet_NaN();
  return std::make_unique<MultiKqvMhaOutputProjectionFusionCompute>(
      FusedInputIndex(indices, pattern.branches[0].slice.GetInputs()[0]),
      FusedInputIndex(indices, mha_inputs[3]), mask, std::move(qkv_weights),
      std::move(output_weights), std::move(output_biases),
      std::move(input_starts), std::move(input_ends), std::move(output_starts),
      std::move(output_ends), heads,
      GetFloatAttribute(pattern.mha, "scale", default_scale));
}

}  // namespace musa_ep
