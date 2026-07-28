#include <algorithm>
#include <array>
#include <optional>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

bool IsCastTo(Ort::ConstNode node, ONNXTensorElementDataType element_type) {
  return IsOnnxOp(node, "Cast") && GetIntAttribute(node, "to").value_or(-1) ==
                                       static_cast<int64_t>(element_type);
}

bool IsCastToInt(Ort::ConstNode node) {
  if (!IsOnnxOp(node, "Cast")) {
    return false;
  }
  const int64_t to = GetIntAttribute(node, "to").value_or(-1);
  return to == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         to == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool IsScalarIntInitializerValue(Ort::ConstValueInfo value_info,
                                 int64_t expected) {
  std::optional<int64_t> value = ReadScalarIntInitializer(value_info);
  return value.has_value() && *value == expected;
}

bool IsTensorOfType(Ort::ConstValueInfo value_info,
                    ONNXTensorElementDataType element_type) {
  return value_info != nullptr &&
         value_info.TypeInfo().GetONNXType() == ONNX_TYPE_TENSOR &&
         value_info.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType() ==
             element_type;
}

bool IsIntTensor(Ort::ConstValueInfo value_info) {
  return value_info != nullptr && IsIntTensorValueInfo(value_info);
}

bool IsExternalOrSelectedConsumer(
    Ort::ConstValueInfo output, const std::unordered_set<size_t>& selected_ids,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<std::string>& allowed_external_outputs) {
  if (allowed_external_outputs.count(Name(output)) != 0) {
    return true;
  }
  if (graph_output_names.count(Name(output)) != 0) {
    return false;
  }
  for (const auto& consumer : output.GetConsumers()) {
    if (selected_ids.count(consumer.node.GetId()) == 0) {
      return false;
    }
  }
  return true;
}

bool MatchedNodesHaveNoUnexpectedExternalConsumers(
    const std::vector<Ort::ConstNode>& fusion_nodes,
    const std::unordered_set<size_t>& selected_ids,
    const std::unordered_set<std::string>& graph_output_names,
    Ort::ConstValueInfo embedding_output,
    Ort::ConstValueInfo count_feature_output) {
  const std::unordered_set<std::string> allowed_external_outputs = {
      Name(embedding_output), Name(count_feature_output)};
  for (Ort::ConstNode node : fusion_nodes) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (!IsExternalOrSelectedConsumer(output, selected_ids,
                                        graph_output_names,
                                        allowed_external_outputs)) {
        return false;
      }
    }
  }
  return true;
}

bool FindCountFeatureCast(Ort::ConstValueInfo reduce_output,
                          Ort::ConstNode min_node,
                          Ort::ConstNode& count_unsqueeze,
                          Ort::ConstNode& count_cast) {
  for (const auto& consumer : reduce_output.GetConsumers()) {
    if (consumer.node.GetId() == min_node.GetId()) {
      continue;
    }
    if (!IsOnnxOp(consumer.node, "Unsqueeze")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
        consumer.node.GetOutputs();
    if (unsqueeze_outputs.size() != 1) {
      continue;
    }
    for (const auto& cast_consumer : unsqueeze_outputs[0].GetConsumers()) {
      if (IsCastTo(cast_consumer.node, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
        count_unsqueeze = consumer.node;
        count_cast = cast_consumer.node;
        return true;
      }
    }
  }
  return false;
}

Ort::ConstNode ProducerOf(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info) {
  return FindProducer(producers, value_info);
}

bool ResolveHitPath(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstNode hit_equal, Ort::ConstValueInfo source_ids,
    Ort::ConstNode& source_unsqueeze, Ort::ConstValueInfo& target_input) {
  if (!IsOnnxOp(hit_equal, "Equal")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> equal_inputs = hit_equal.GetInputs();
  if (equal_inputs.size() != 2) {
    return false;
  }

  Ort::ConstNode sub_node{nullptr};
  Ort::ConstValueInfo zero_input{nullptr};
  for (Ort::ConstValueInfo input : equal_inputs) {
    Ort::ConstNode producer = ProducerOf(producers, input);
    if (IsOnnxOp(producer, "Sub")) {
      sub_node = producer;
    } else {
      zero_input = input;
    }
  }
  if (!sub_node || !IsScalarIntInitializerValue(zero_input, 0)) {
    return false;
  }

  std::vector<Ort::ConstValueInfo> sub_inputs = sub_node.GetInputs();
  if (sub_inputs.size() != 2) {
    return false;
  }
  for (Ort::ConstValueInfo input : sub_inputs) {
    Ort::ConstNode producer = ProducerOf(producers, input);
    if (IsOnnxOp(producer, "Unsqueeze")) {
      std::vector<Ort::ConstValueInfo> unsqueeze_inputs = producer.GetInputs();
      if (!unsqueeze_inputs.empty() &&
          Name(unsqueeze_inputs[0]) == Name(source_ids)) {
        source_unsqueeze = producer;
      } else {
        target_input = input;
      }
    } else {
      target_input = input;
    }
  }
  return source_unsqueeze && target_input != nullptr;
}

bool ResolveValidPath(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstNode valid_unsqueeze, Ort::ConstValueInfo source_ids) {
  if (!IsOnnxOp(valid_unsqueeze, "Unsqueeze")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> valid_inputs = valid_unsqueeze.GetInputs();
  if (valid_inputs.size() != 2) {
    return false;
  }
  Ort::ConstNode not_node = ProducerOf(producers, valid_inputs[0]);
  if (!IsOnnxOp(not_node, "Not")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> not_inputs = not_node.GetInputs();
  if (not_inputs.size() != 1) {
    return false;
  }
  Ort::ConstNode pad_equal = ProducerOf(producers, not_inputs[0]);
  if (!IsOnnxOp(pad_equal, "Equal")) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> equal_inputs = pad_equal.GetInputs();
  if (equal_inputs.size() != 2) {
    return false;
  }
  bool has_source = false;
  bool has_pad = false;
  for (Ort::ConstValueInfo input : equal_inputs) {
    if (Name(input) == Name(source_ids)) {
      has_source = true;
    } else {
      has_pad = ReadScalarIntInitializer(input).has_value();
    }
  }
  return has_source && has_pad;
}

bool MatchTargetIdCountEmbedding(
    Ort::ConstNode gather_node,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(gather_node, "Gather") ||
      accepted_node_ids.count(gather_node.GetId()) != 0 ||
      GetIntAttribute(gather_node, "axis").value_or(0) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gather_outputs = gather_node.GetOutputs();
  if (gather_inputs.size() != 2 || gather_outputs.size() != 1 ||
      !IsTensorOfType(gather_inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !IsTensorOfType(gather_outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }

  Ort::ConstNode bucket_unsqueeze = ProducerOf(producers, gather_inputs[1]);
  if (!IsOnnxOp(bucket_unsqueeze, "Unsqueeze") ||
      accepted_node_ids.count(bucket_unsqueeze.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> bucket_unsqueeze_inputs =
      bucket_unsqueeze.GetInputs();
  if (bucket_unsqueeze_inputs.size() != 2) {
    return false;
  }
  Ort::ConstNode min_node = ProducerOf(producers, bucket_unsqueeze_inputs[0]);
  if (!IsOnnxOp(min_node, "Min") ||
      accepted_node_ids.count(min_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> min_inputs = min_node.GetInputs();
  if (min_inputs.size() != 2) {
    return false;
  }

  Ort::ConstNode reduce_node{nullptr};
  Ort::ConstValueInfo cap_input{nullptr};
  for (Ort::ConstValueInfo input : min_inputs) {
    Ort::ConstNode producer = ProducerOf(producers, input);
    if (IsOnnxOp(producer, "ReduceSum")) {
      reduce_node = producer;
    } else {
      cap_input = input;
    }
  }
  if (!reduce_node || accepted_node_ids.count(reduce_node.GetId()) != 0 ||
      !ReadScalarIntInitializer(cap_input).has_value()) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> reduce_inputs = reduce_node.GetInputs();
  std::vector<Ort::ConstValueInfo> reduce_outputs = reduce_node.GetOutputs();
  if (reduce_inputs.size() != 2 || reduce_outputs.size() != 1 ||
      !ReadSmallIntInitializer(reduce_inputs[1]).has_value()) {
    return false;
  }

  Ort::ConstNode count_unsqueeze{nullptr};
  Ort::ConstNode count_cast{nullptr};
  if (!FindCountFeatureCast(reduce_outputs[0], min_node, count_unsqueeze,
                            count_cast) ||
      accepted_node_ids.count(count_unsqueeze.GetId()) != 0 ||
      accepted_node_ids.count(count_cast.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> count_cast_outputs = count_cast.GetOutputs();
  if (count_cast_outputs.size() != 1 ||
      !IsTensorOfType(count_cast_outputs[0],
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }

  Ort::ConstNode mask_cast = ProducerOf(producers, reduce_inputs[0]);
  if (!IsCastToInt(mask_cast) ||
      accepted_node_ids.count(mask_cast.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> mask_cast_inputs = mask_cast.GetInputs();
  if (mask_cast_inputs.size() != 1) {
    return false;
  }
  Ort::ConstNode and_node = ProducerOf(producers, mask_cast_inputs[0]);
  if (!IsOnnxOp(and_node, "And") ||
      accepted_node_ids.count(and_node.GetId()) != 0) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> and_inputs = and_node.GetInputs();
  if (and_inputs.size() != 2) {
    return false;
  }

  Ort::ConstNode hit_equal{nullptr};
  Ort::ConstNode valid_unsqueeze{nullptr};
  for (Ort::ConstValueInfo input : and_inputs) {
    Ort::ConstNode producer = ProducerOf(producers, input);
    if (IsOnnxOp(producer, "Equal")) {
      hit_equal = producer;
    } else if (IsOnnxOp(producer, "Unsqueeze")) {
      valid_unsqueeze = producer;
    }
  }
  if (!hit_equal || !valid_unsqueeze ||
      accepted_node_ids.count(hit_equal.GetId()) != 0 ||
      accepted_node_ids.count(valid_unsqueeze.GetId()) != 0) {
    return false;
  }

  Ort::ConstNode source_unsqueeze{nullptr};
  Ort::ConstValueInfo target_input{nullptr};
  Ort::ConstValueInfo source_ids{nullptr};
  for (Ort::ConstValueInfo input : hit_equal.GetInputs()) {
    Ort::ConstNode producer = ProducerOf(producers, input);
    if (!IsOnnxOp(producer, "Sub")) {
      continue;
    }
    for (Ort::ConstValueInfo sub_input : producer.GetInputs()) {
      Ort::ConstNode sub_producer = ProducerOf(producers, sub_input);
      if (IsOnnxOp(sub_producer, "Unsqueeze")) {
        std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
            sub_producer.GetInputs();
        if (!unsqueeze_inputs.empty() && IsIntTensor(unsqueeze_inputs[0])) {
          source_ids = unsqueeze_inputs[0];
          break;
        }
      }
    }
  }
  if (!source_ids ||
      !ResolveHitPath(producers, hit_equal, source_ids, source_unsqueeze,
                      target_input) ||
      accepted_node_ids.count(source_unsqueeze.GetId()) != 0 ||
      !ResolveValidPath(producers, valid_unsqueeze, source_ids)) {
    return false;
  }

  Ort::ConstNode not_node =
      ProducerOf(producers, valid_unsqueeze.GetInputs()[0]);
  Ort::ConstNode pad_equal = ProducerOf(producers, not_node.GetInputs()[0]);
  Ort::ConstNode sub_node{nullptr};
  for (Ort::ConstValueInfo input : hit_equal.GetInputs()) {
    Ort::ConstNode producer = ProducerOf(producers, input);
    if (IsOnnxOp(producer, "Sub")) {
      sub_node = producer;
      break;
    }
  }
  if (!sub_node || accepted_node_ids.count(sub_node.GetId()) != 0 ||
      accepted_node_ids.count(not_node.GetId()) != 0 ||
      accepted_node_ids.count(pad_equal.GetId()) != 0) {
    return false;
  }

  std::array<Ort::ConstNode, 13> matched = {
      source_unsqueeze, sub_node,    hit_equal,      pad_equal,   not_node,
      valid_unsqueeze,  and_node,    mask_cast,      reduce_node, min_node,
      bucket_unsqueeze, gather_node, count_unsqueeze};

  std::unordered_set<size_t> selected_ids;
  fusion_nodes.clear();
  fusion_nodes.reserve(matched.size() + 1);
  for (Ort::ConstNode node : matched) {
    if (!AddFusionNode(node, accepted_node_ids, selected_ids, fusion_nodes)) {
      fusion_nodes.clear();
      return false;
    }
  }
  if (!AddFusionNode(count_cast, accepted_node_ids, selected_ids,
                     fusion_nodes)) {
    fusion_nodes.clear();
    return false;
  }
  if (!MatchedNodesHaveNoUnexpectedExternalConsumers(
          fusion_nodes, selected_ids, graph_output_names, gather_outputs[0],
          count_cast_outputs[0])) {
    fusion_nodes.clear();
    return false;
  }

  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindTargetIdCountEmbeddingFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  const auto producers = BuildProducerMap(all_nodes);
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (MatchTargetIdCountEmbedding(node, producers, graph_output_names,
                                    accepted_node_ids, fusion_nodes)) {
      fusions.push_back(std::move(fusion_nodes));
    }
  }
  return fusions;
}

}  // namespace musa_ep
