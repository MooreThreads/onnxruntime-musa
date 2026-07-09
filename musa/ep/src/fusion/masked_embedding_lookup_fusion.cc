// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "fusion/masked_embedding_lookup_fusion.h"

#include <musa_runtime.h>

#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/masked_embedding_lookup_impl.h"
#include "plugin_ep_utils.h"

namespace {

std::unordered_map<std::string, Ort::ConstNode> ProducersInGraph(
    Ort::ConstGraph graph) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : graph.GetNodes()) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      producers.emplace(musa_ep::Name(output), node);
    }
  }
  return producers;
}

Ort::ConstNode ProducerInGraph(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info) {
  auto it = producers.find(musa_ep::Name(value_info));
  return it == producers.end() ? Ort::ConstNode{nullptr} : it->second;
}

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_input_indices;
  std::vector<Ort::ConstValueInfo> fused_inputs = fused_node.GetInputs();
  for (size_t i = 0; i < fused_inputs.size(); ++i) {
    fused_input_indices.emplace(musa_ep::Name(fused_inputs[i]), i);
  }
  return fused_input_indices;
}

std::unordered_map<std::string, size_t> FusedOutputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_output_indices;
  std::vector<Ort::ConstValueInfo> fused_outputs = fused_node.GetOutputs();
  for (size_t i = 0; i < fused_outputs.size(); ++i) {
    fused_output_indices.emplace(musa_ep::Name(fused_outputs[i]), i);
  }
  return fused_output_indices;
}

size_t GetMappedIndex(const std::unordered_map<std::string, size_t>& indices,
                      const std::string& name, const char* kind) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(
        std::string("unable to map MaskedEmbeddingLookup ") + kind + " " +
        name);
  }
  return it->second;
}

int64_t ReadRequiredScalarIntInitializer(Ort::ConstValueInfo value_info,
                                         const char* kind) {
  std::optional<int64_t> value = musa_ep::ReadScalarIntInitializer(value_info);
  if (!value.has_value()) {
    throw std::runtime_error(
        std::string("MaskedEmbeddingLookup requires scalar ") + kind +
        " initializer");
  }
  return *value;
}

int64_t NumElementsFromShape(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 1;
  }
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

std::vector<int64_t> RuntimeOutputShape(
    const std::vector<int64_t>& static_output_shape,
    const std::vector<int64_t>& input_shape, int64_t embedding_dim) {
  if (!static_output_shape.empty()) {
    return static_output_shape;
  }
  return {1, NumElements(input_shape), embedding_dim};
}

struct MaskedEmbeddingLookupFusionCompute : FusionNodeCompute {
  MaskedEmbeddingLookupFusionCompute(size_t input_index, size_t table_index,
                                     size_t output_index, int64_t threshold,
                                     std::vector<int64_t> output_shape)
      : input_index(input_index),
        table_index(table_index),
        output_index(output_index),
        threshold(threshold),
        output_shape(std::move(output_shape)) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue ids = ctx.GetInput(input_index);
      Ort::ConstValue table = ctx.GetInput(table_index);

      auto ids_info = ids.GetTensorTypeAndShapeInfo();
      auto table_info = table.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> ids_shape = ids_info.GetShape();
      std::vector<int64_t> table_shape = table_info.GetShape();
      if (table_shape.size() != 2 || table_shape[0] <= 0 ||
          table_shape[1] <= 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "MaskedEmbeddingLookup requires a rank-2 embedding table");
      }

      const size_t elem_size = ElementSize(table_info.GetElementType());
      if (elem_size == 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "MaskedEmbeddingLookup unsupported table dtype");
      }
      const size_t index_elem_size = ElementSize(ids_info.GetElementType());
      if (index_elem_size != sizeof(int32_t) &&
          index_elem_size != sizeof(int64_t)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "MaskedEmbeddingLookup requires int32/int64 ids");
      }

      const int64_t sequence_count = NumElements(ids_shape);
      const int64_t embedding_dim = table_shape[1];
      std::vector<int64_t> out_shape =
          RuntimeOutputShape(output_shape, ids_shape, embedding_dim);
      if (NumElementsFromShape(out_shape) != sequence_count * embedding_dim) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "MaskedEmbeddingLookup output shape does not match ids/table");
      }

      Ort::UnownedValue output = ctx.GetOutput(output_index, out_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "MaskedEmbeddingLookup requires MUSA output");
      }

      musaStream_t stream = GetComputeStream(ctx);
      DeviceInputBuffer table_buffer;
      DeviceInputBuffer ids_buffer;
      RETURN_IF_ERROR(table_buffer.Bind(table, stream));
      RETURN_IF_ERROR(ids_buffer.Bind(ids, stream));

      const size_t output_bytes = output.GetTensorSizeInBytes();
      RETURN_IF_ERROR(LaunchStatus(musaMemsetAsync(
          output.GetTensorMutableRawData(), 0, output_bytes, stream)));
      return LaunchStatus(LaunchMusaMaskedEmbeddingLookupKernel(
          table_buffer.data(), ids_buffer.data(),
          output.GetTensorMutableRawData(), static_cast<int32_t>(elem_size),
          static_cast<int32_t>(index_elem_size), sequence_count, table_shape[0],
          embedding_dim, threshold, stream));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t input_index;
  size_t table_index;
  size_t output_index;
  int64_t threshold;
  std::vector<int64_t> output_shape;
};

}  // namespace

bool IsMaskedEmbeddingLookupFusionGraph(Ort::ConstGraph graph) {
  int reshape_count = 0;
  int greater_equal_count = 0;
  int nonzero_count = 0;
  int transpose_count = 0;
  int squeeze_count = 0;
  int gather_count = 0;
  int scatter_nd_count = 0;
  int unsqueeze_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (musa_ep::IsOnnxOp(node, "Reshape")) {
      ++reshape_count;
    } else if (musa_ep::IsOnnxOp(node, "GreaterOrEqual")) {
      ++greater_equal_count;
    } else if (musa_ep::IsOnnxOp(node, "NonZero")) {
      ++nonzero_count;
    } else if (musa_ep::IsOnnxOp(node, "Transpose")) {
      ++transpose_count;
    } else if (musa_ep::IsOnnxOp(node, "Squeeze")) {
      ++squeeze_count;
    } else if (musa_ep::IsOnnxOp(node, "Gather")) {
      ++gather_count;
    } else if (musa_ep::IsOnnxOp(node, "ScatterND")) {
      ++scatter_nd_count;
    } else if (musa_ep::IsOnnxOp(node, "Unsqueeze")) {
      ++unsqueeze_count;
    } else {
      return false;
    }
  }
  return reshape_count == 1 && greater_equal_count == 1 && nonzero_count == 1 &&
         transpose_count == 1 && squeeze_count == 1 && gather_count == 2 &&
         scatter_nd_count == 1 && unsqueeze_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateMaskedEmbeddingLookupFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto producers = ProducersInGraph(graph);
  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);

  Ort::ConstNode unsqueeze_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!musa_ep::IsOnnxOp(node, "Unsqueeze")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
    if (outputs.size() == 1 &&
        fused_output_indices.count(musa_ep::Name(outputs[0])) != 0) {
      unsqueeze_node = node;
      break;
    }
  }
  if (!unsqueeze_node) {
    throw std::runtime_error("MaskedEmbeddingLookup requires final Unsqueeze");
  }

  std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
      unsqueeze_node.GetInputs();
  std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
      unsqueeze_node.GetOutputs();
  if (unsqueeze_inputs.size() != 2 || unsqueeze_outputs.size() != 1) {
    throw std::runtime_error("MaskedEmbeddingLookup invalid Unsqueeze");
  }

  Ort::ConstNode scatter_node = ProducerInGraph(producers, unsqueeze_inputs[0]);
  if (!musa_ep::IsOnnxOp(scatter_node, "ScatterND")) {
    throw std::runtime_error("MaskedEmbeddingLookup requires ScatterND");
  }
  std::vector<Ort::ConstValueInfo> scatter_inputs = scatter_node.GetInputs();
  if (scatter_inputs.size() != 3) {
    throw std::runtime_error("MaskedEmbeddingLookup invalid ScatterND");
  }

  Ort::ConstNode embedding_gather =
      ProducerInGraph(producers, scatter_inputs[2]);
  if (!musa_ep::IsOnnxOp(embedding_gather, "Gather")) {
    throw std::runtime_error(
        "MaskedEmbeddingLookup requires embedding Gather before ScatterND");
  }
  std::vector<Ort::ConstValueInfo> embedding_gather_inputs =
      embedding_gather.GetInputs();
  if (embedding_gather_inputs.size() != 2) {
    throw std::runtime_error("MaskedEmbeddingLookup invalid embedding Gather");
  }

  Ort::ConstNode id_gather =
      ProducerInGraph(producers, embedding_gather_inputs[1]);
  if (!musa_ep::IsOnnxOp(id_gather, "Gather")) {
    throw std::runtime_error("MaskedEmbeddingLookup requires id Gather");
  }
  std::vector<Ort::ConstValueInfo> id_gather_inputs = id_gather.GetInputs();
  if (id_gather_inputs.size() != 2) {
    throw std::runtime_error("MaskedEmbeddingLookup invalid id Gather");
  }

  Ort::ConstNode reshape_node = ProducerInGraph(producers, id_gather_inputs[0]);
  Ort::ConstNode squeeze_node = ProducerInGraph(producers, id_gather_inputs[1]);
  if (!musa_ep::IsOnnxOp(reshape_node, "Reshape") ||
      !musa_ep::IsOnnxOp(squeeze_node, "Squeeze")) {
    throw std::runtime_error("MaskedEmbeddingLookup requires Reshape/Squeeze");
  }
  std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
  if (reshape_inputs.size() != 2) {
    throw std::runtime_error("MaskedEmbeddingLookup invalid Reshape");
  }

  Ort::ConstNode greater_equal_node{nullptr};
  for (const auto& consumer : reshape_node.GetOutputs()[0].GetConsumers()) {
    if (musa_ep::IsOnnxOp(consumer.node, "GreaterOrEqual")) {
      greater_equal_node = consumer.node;
      break;
    }
  }
  if (!greater_equal_node) {
    throw std::runtime_error("MaskedEmbeddingLookup requires GreaterOrEqual");
  }
  std::vector<Ort::ConstValueInfo> ge_inputs = greater_equal_node.GetInputs();
  if (ge_inputs.size() != 2) {
    throw std::runtime_error("MaskedEmbeddingLookup invalid GreaterOrEqual");
  }
  Ort::ConstValueInfo threshold_input =
      musa_ep::Name(ge_inputs[0]) == musa_ep::Name(reshape_node.GetOutputs()[0])
          ? ge_inputs[1]
          : ge_inputs[0];

  size_t input_index = GetMappedIndex(
      fused_input_indices, musa_ep::Name(reshape_inputs[0]), "input");
  size_t table_index = GetMappedIndex(fused_input_indices,
                                      musa_ep::Name(embedding_gather_inputs[0]),
                                      "embedding table");
  size_t output_index = GetMappedIndex(
      fused_output_indices, musa_ep::Name(unsqueeze_outputs[0]), "output");
  std::vector<int64_t> out_shape;
  auto static_shape = GetTensorShape(unsqueeze_outputs[0]);
  if (static_shape.has_value()) {
    out_shape = *static_shape;
  }
  return std::make_unique<MaskedEmbeddingLookupFusionCompute>(
      input_index, table_index, output_index,
      ReadRequiredScalarIntInitializer(threshold_input, "threshold"),
      std::move(out_shape));
}
