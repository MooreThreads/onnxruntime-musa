// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

bool IsZeroFloatInitializer(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer() ||
      !IsFloatTensorValueInfo(value_info)) {
    return false;
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  const float* data = value.GetTensorData<float>();
  for (int64_t i = 0; i < info.GetElementCount(); ++i) {
    if (data[i] != 0.0f) {
      return false;
    }
  }
  return true;
}

bool HasOnlyConsumers(Ort::ConstValueInfo output,
                      std::initializer_list<Ort::ConstNode> expected_nodes) {
  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
      output.GetConsumers();
  if (consumers.size() != expected_nodes.size()) {
    return false;
  }
  std::unordered_set<size_t> expected_node_ids;
  for (Ort::ConstNode node : expected_nodes) {
    if (!node) {
      return false;
    }
    expected_node_ids.insert(node.GetId());
  }
  for (const auto& consumer : consumers) {
    if (expected_node_ids.count(consumer.node.GetId()) == 0) {
      return false;
    }
  }
  return true;
}

bool CanFuseMaskedEmbeddingLookup(
    Ort::ConstNode unsqueeze_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(unsqueeze_node, "Unsqueeze") ||
      accepted_node_ids.count(unsqueeze_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
      unsqueeze_node.GetInputs();
  std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
      unsqueeze_node.GetOutputs();
  if (unsqueeze_inputs.size() != 2 || unsqueeze_outputs.size() != 1 ||
      !IsFloatTensorValueInfo(unsqueeze_outputs[0]) ||
      !ReadUnsqueezeAxes(unsqueeze_node).has_value() ||
      ReadUnsqueezeAxes(unsqueeze_node)->size() != 1 ||
      (*ReadUnsqueezeAxes(unsqueeze_node))[0] != 0) {
    return false;
  }

  Ort::ConstNode scatter_node = FindProducer(producers, unsqueeze_inputs[0]);
  if (!IsOnnxOp(scatter_node, "ScatterND") ||
      accepted_node_ids.count(scatter_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> scatter_inputs = scatter_node.GetInputs();
  std::vector<Ort::ConstValueInfo> scatter_outputs = scatter_node.GetOutputs();
  if (scatter_inputs.size() != 3 || scatter_outputs.size() != 1 ||
      graph_output_names.count(Name(scatter_outputs[0])) != 0 ||
      !HasOnlyConsumer(scatter_outputs[0], unsqueeze_node, 0) ||
      !IsZeroFloatInitializer(scatter_inputs[0])) {
    return false;
  }

  Ort::ConstNode transpose_node = FindProducer(producers, scatter_inputs[1]);
  Ort::ConstNode embedding_gather = FindProducer(producers, scatter_inputs[2]);
  if (!IsOnnxOp(transpose_node, "Transpose") ||
      !IsOnnxOp(embedding_gather, "Gather")) {
    return false;
  }
  if (accepted_node_ids.count(transpose_node.GetId()) != 0 ||
      accepted_node_ids.count(embedding_gather.GetId()) != 0) {
    return false;
  }
  if (GetIntAttribute(embedding_gather, "axis").value_or(0) != 0) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> transpose_inputs =
      transpose_node.GetInputs();
  std::vector<Ort::ConstValueInfo> transpose_outputs =
      transpose_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> embedding_inputs =
      embedding_gather.GetInputs();
  std::vector<Ort::ConstValueInfo> embedding_outputs =
      embedding_gather.GetOutputs();
  if (transpose_inputs.size() != 1 || transpose_outputs.size() != 1 ||
      embedding_inputs.size() != 2 || embedding_outputs.size() != 1 ||
      graph_output_names.count(Name(transpose_outputs[0])) != 0 ||
      graph_output_names.count(Name(embedding_outputs[0])) != 0 ||
      !HasOnlyConsumer(embedding_outputs[0], scatter_node, 2) ||
      !IsFloatTensorValueInfo(embedding_inputs[0]) ||
      !IsFloatTensorValueInfo(embedding_outputs[0])) {
    return false;
  }

  Ort::ConstNode id_gather = FindProducer(producers, embedding_inputs[1]);
  if (!IsOnnxOp(id_gather, "Gather") ||
      accepted_node_ids.count(id_gather.GetId()) != 0 ||
      GetIntAttribute(id_gather, "axis").value_or(0) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> id_gather_inputs = id_gather.GetInputs();
  std::vector<Ort::ConstValueInfo> id_gather_outputs = id_gather.GetOutputs();
  if (id_gather_inputs.size() != 2 || id_gather_outputs.size() != 1 ||
      graph_output_names.count(Name(id_gather_outputs[0])) != 0 ||
      !HasOnlyConsumer(id_gather_outputs[0], embedding_gather, 1) ||
      !IsIntTensorValueInfo(id_gather_outputs[0])) {
    return false;
  }

  Ort::ConstNode reshape_node = FindProducer(producers, id_gather_inputs[0]);
  Ort::ConstNode squeeze_node = FindProducer(producers, id_gather_inputs[1]);
  if (!IsOnnxOp(reshape_node, "Reshape") ||
      !IsOnnxOp(squeeze_node, "Squeeze") ||
      accepted_node_ids.count(reshape_node.GetId()) != 0 ||
      accepted_node_ids.count(squeeze_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> reshape_inputs = reshape_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reshape_outputs = reshape_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> squeeze_inputs = squeeze_node.GetInputs();
  std::vector<Ort::ConstValueInfo> squeeze_outputs = squeeze_node.GetOutputs();
  if (reshape_inputs.size() != 2 || reshape_outputs.size() != 1 ||
      squeeze_inputs.size() != 2 || squeeze_outputs.size() != 1 ||
      graph_output_names.count(Name(reshape_outputs[0])) != 0 ||
      graph_output_names.count(Name(squeeze_outputs[0])) != 0 ||
      !HasOnlyConsumers(transpose_outputs[0], {squeeze_node, scatter_node}) ||
      !HasOnlyConsumer(squeeze_outputs[0], id_gather, 1) ||
      !IsIntTensorValueInfo(reshape_inputs[0]) ||
      !IsIntTensorValueInfo(reshape_outputs[0]) ||
      !IsIntTensorValueInfo(squeeze_outputs[0])) {
    return false;
  }
  auto reshape_shape = ReadSmallIntInitializer(reshape_inputs[1]);
  auto squeeze_axes = ReadSmallIntInitializer(squeeze_inputs[1]);
  if (!reshape_shape.has_value() || reshape_shape->size() != 1 ||
      (*reshape_shape)[0] != -1 || !squeeze_axes.has_value() ||
      squeeze_axes->size() != 1 || (*squeeze_axes)[0] != 1 ||
      Name(squeeze_inputs[0]) != Name(transpose_outputs[0])) {
    return false;
  }

  Ort::ConstNode nonzero_node = FindProducer(producers, transpose_inputs[0]);
  if (!IsOnnxOp(nonzero_node, "NonZero") ||
      accepted_node_ids.count(nonzero_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> nonzero_inputs = nonzero_node.GetInputs();
  std::vector<Ort::ConstValueInfo> nonzero_outputs = nonzero_node.GetOutputs();
  if (nonzero_inputs.size() != 1 || nonzero_outputs.size() != 1 ||
      graph_output_names.count(Name(nonzero_outputs[0])) != 0 ||
      !HasOnlyConsumer(nonzero_outputs[0], transpose_node, 0)) {
    return false;
  }

  Ort::ConstNode greater_equal_node =
      FindProducer(producers, nonzero_inputs[0]);
  if (!IsOnnxOp(greater_equal_node, "GreaterOrEqual") ||
      accepted_node_ids.count(greater_equal_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> ge_inputs = greater_equal_node.GetInputs();
  std::vector<Ort::ConstValueInfo> ge_outputs = greater_equal_node.GetOutputs();
  if (ge_inputs.size() != 2 || ge_outputs.size() != 1 ||
      graph_output_names.count(Name(ge_outputs[0])) != 0 ||
      !HasOnlyConsumer(ge_outputs[0], nonzero_node, 0) ||
      !HasOnlyConsumers(reshape_outputs[0], {id_gather, greater_equal_node})) {
    return false;
  }

  Ort::ConstValueInfo threshold_input{nullptr};
  if (Name(ge_inputs[0]) == Name(reshape_outputs[0])) {
    threshold_input = ge_inputs[1];
  } else if (Name(ge_inputs[1]) == Name(reshape_outputs[0])) {
    threshold_input = ge_inputs[0];
  } else {
    return false;
  }
  std::optional<int64_t> threshold = ReadScalarIntInitializer(threshold_input);
  if (!threshold.has_value() || *threshold != 0) {
    return false;
  }

  auto source_shape = GetTensorShape(reshape_inputs[0]);
  auto reshape_output_shape = GetTensorShape(reshape_outputs[0]);
  auto nonzero_shape = GetTensorShape(nonzero_outputs[0]);
  auto transpose_shape = GetTensorShape(transpose_outputs[0]);
  auto table_shape = GetTensorShape(embedding_inputs[0]);
  auto scatter_shape = GetTensorShape(scatter_outputs[0]);
  auto output_shape = GetTensorShape(unsqueeze_outputs[0]);
  if (!reshape_output_shape.has_value() || reshape_output_shape->size() != 1) {
    return false;
  }
  if (!nonzero_shape.has_value() || nonzero_shape->size() != 2 ||
      (*nonzero_shape)[0] != 1 || !transpose_shape.has_value() ||
      transpose_shape->size() != 2 || (*transpose_shape)[1] != 1 ||
      !KnownDimsEqual((*nonzero_shape)[1], (*transpose_shape)[0])) {
    return false;
  }
  if (!table_shape.has_value() || table_shape->size() != 2 ||
      (*table_shape)[0] <= 0 || (*table_shape)[1] <= 0 ||
      !scatter_shape.has_value() || scatter_shape->size() != 2 ||
      !KnownDimsEqual((*scatter_shape)[0], (*reshape_output_shape)[0]) ||
      !KnownDimsEqual((*scatter_shape)[1], (*table_shape)[1]) ||
      !output_shape.has_value() || output_shape->size() != 3 ||
      (*output_shape)[0] != 1 ||
      !KnownDimsEqual((*output_shape)[1], (*scatter_shape)[0]) ||
      !KnownDimsEqual((*output_shape)[2], (*scatter_shape)[1])) {
    return false;
  }
  if (source_shape.has_value() && reshape_output_shape.has_value()) {
    int64_t source_count = 1;
    bool source_static = true;
    for (int64_t dim : *source_shape) {
      if (dim <= 0) {
        source_static = false;
        break;
      }
      source_count *= dim;
    }
    if (source_static &&
        !KnownDimsEqual(source_count, (*reshape_output_shape)[0])) {
      return false;
    }
  }

  std::unordered_set<size_t> selected_node_ids;
  fusion_nodes.clear();
  for (Ort::ConstNode node : {reshape_node, greater_equal_node, nonzero_node,
                              transpose_node, squeeze_node, id_gather,
                              embedding_gather, scatter_node, unsqueeze_node}) {
    if (!AddFusionNode(node, accepted_node_ids, selected_node_ids,
                       fusion_nodes)) {
      return false;
    }
  }
  return FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                     selected_node_ids);
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindMaskedEmbeddingLookupFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::unordered_map<std::string, Ort::ConstNode> producers =
      BuildProducerMap(all_nodes);
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (CanFuseMaskedEmbeddingLookup(node, producers, graph_output_names,
                                     accepted_node_ids, fusion_nodes)) {
      fusions.push_back(std::move(fusion_nodes));
    }
  }
  return fusions;
}

}  // namespace musa_ep
