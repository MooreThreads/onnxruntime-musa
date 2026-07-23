#include <algorithm>
#include <array>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

bool IsIntValues(Ort::ConstValueInfo value_info,
                 std::initializer_list<int64_t> expected) {
  std::optional<std::vector<int64_t>> values =
      ReadSmallIntInitializer(value_info);
  return values.has_value() && values->size() == expected.size() &&
         std::equal(values->begin(), values->end(), expected.begin());
}

bool IsCastTo(Ort::ConstNode node, ONNXTensorElementDataType element_type) {
  return IsOnnxOp(node, "Cast") && GetIntAttribute(node, "to").value_or(-1) ==
                                       static_cast<int64_t>(element_type);
}

bool IsTensorOfType(Ort::ConstValueInfo value_info,
                    ONNXTensorElementDataType element_type) {
  return value_info != nullptr &&
         value_info.TypeInfo().GetTensorTypeAndShapeInfo().GetElementType() ==
             element_type;
}

bool IsInputFrom(Ort::ConstNode consumer, size_t input_index,
                 Ort::ConstNode producer, int64_t output_index) {
  std::vector<Ort::ConstValueInfo> inputs = consumer.GetInputs();
  if (input_index >= inputs.size()) {
    return false;
  }
  Ort::ValueInfoConsumerProducerInfo info =
      inputs[input_index].GetProducerNode();
  return info.node && info.node.GetId() == producer.GetId() &&
         info.index == output_index;
}

bool GetInputProducer(Ort::ConstNode consumer, size_t input_index,
                      const char* op_type, int64_t output_index,
                      Ort::ConstNode& producer) {
  std::vector<Ort::ConstValueInfo> inputs = consumer.GetInputs();
  if (input_index >= inputs.size()) {
    return false;
  }
  Ort::ValueInfoConsumerProducerInfo info =
      inputs[input_index].GetProducerNode();
  if (!info.node || info.index != output_index ||
      !IsOnnxOp(info.node, op_type)) {
    return false;
  }
  producer = info.node;
  return true;
}

bool HasInputCount(Ort::ConstNode node, size_t count) {
  return node.GetInputs().size() == count;
}

bool HasNoEscapingInternalOutputs(
    const std::vector<Ort::ConstNode>& nodes, Ort::ConstNode final_node,
    const std::unordered_set<std::string>& graph_output_names) {
  std::unordered_set<size_t> node_ids;
  for (Ort::ConstNode node : nodes) {
    node_ids.insert(node.GetId());
  }

  for (Ort::ConstNode node : nodes) {
    const bool is_final_node = node.GetId() == final_node.GetId();
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (output == nullptr) {
        continue;
      }
      if (!is_final_node && graph_output_names.count(Name(output)) != 0) {
        return false;
      }
      for (const auto& consumer : output.GetConsumers()) {
        if (!is_final_node && node_ids.count(consumer.node.GetId()) == 0) {
          return false;
        }
      }
    }
  }
  return true;
}

bool MatchSegmentMaxBroadcast(
    Ort::ConstNode final_gather,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(final_gather, "Gather") || !HasInputCount(final_gather, 2) ||
      GetIntAttribute(final_gather, "axis").value_or(0) != 0) {
    return false;
  }

  Ort::ConstNode final_reduce{nullptr};
  Ort::ConstNode index_cast32{nullptr};
  Ort::ConstNode value_gather{nullptr};
  Ort::ConstNode scatter{nullptr};
  Ort::ConstNode constant_of_shape{nullptr};
  Ort::ConstNode index_concat{nullptr};
  Ort::ConstNode topk{nullptr};
  Ort::ConstNode shape_concat{nullptr};
  Ort::ConstNode topk_unsqueeze{nullptr};
  Ort::ConstNode mod_unsqueeze{nullptr};
  Ort::ConstNode size_unsqueeze{nullptr};
  Ort::ConstNode count_reduce{nullptr};
  Ort::ConstNode mod{nullptr};
  Ort::ConstNode size_cast64{nullptr};
  Ort::ConstNode range{nullptr};
  Ort::ConstNode count_gather{nullptr};
  Ort::ConstNode size_squeeze{nullptr};
  Ort::ConstNode size_slice{nullptr};
  Ort::ConstNode size_shape_cast32{nullptr};
  Ort::ConstNode first_unique_shape{nullptr};
  Ort::ConstNode id_shape_squeeze{nullptr};
  Ort::ConstNode second_unique{nullptr};
  Ort::ConstNode id_shape{nullptr};
  Ort::ConstNode index_cast64{nullptr};
  Ort::ConstNode first_unique{nullptr};

  if (!GetInputProducer(final_gather, 0, "ReduceMax", 0, final_reduce) ||
      !GetInputProducer(final_gather, 1, "Cast", 0, index_cast32) ||
      !HasInputCount(final_reduce, 2) ||
      GetIntAttribute(final_reduce, "keepdims").value_or(1) != 0 ||
      !IsIntValues(final_reduce.GetInputs()[1], {1}) ||
      !GetInputProducer(final_reduce, 0, "Gather", 0, value_gather) ||
      !HasInputCount(value_gather, 2) ||
      GetIntAttribute(value_gather, "axis").value_or(0) != 0 ||
      !GetInputProducer(value_gather, 1, "ScatterND", 0, scatter) ||
      !HasInputCount(scatter, 3) ||
      !GetInputProducer(scatter, 0, "ConstantOfShape", 0, constant_of_shape) ||
      !GetInputProducer(scatter, 1, "Concat", 0, index_concat) ||
      !GetInputProducer(scatter, 2, "TopK", 1, topk) ||
      !HasInputCount(constant_of_shape, 1) ||
      !GetInputProducer(constant_of_shape, 0, "Concat", 0, shape_concat) ||
      !HasInputCount(index_concat, 2) ||
      GetIntAttribute(index_concat, "axis").value_or(0) != 1 ||
      !GetInputProducer(index_concat, 0, "Unsqueeze", 0, topk_unsqueeze) ||
      !GetInputProducer(index_concat, 1, "Unsqueeze", 0, mod_unsqueeze) ||
      !HasInputCount(shape_concat, 2) ||
      GetIntAttribute(shape_concat, "axis").value_or(0) != 0 ||
      !GetInputProducer(shape_concat, 0, "Unsqueeze", 0, size_unsqueeze) ||
      !GetInputProducer(shape_concat, 1, "ReduceMax", 0, count_reduce) ||
      !HasInputCount(topk_unsqueeze, 2) ||
      !IsIntValues(topk_unsqueeze.GetInputs()[1], {-1}) ||
      !IsInputFrom(topk_unsqueeze, 0, topk, 0) ||
      !HasInputCount(mod_unsqueeze, 2) ||
      !IsIntValues(mod_unsqueeze.GetInputs()[1], {-1}) ||
      !GetInputProducer(mod_unsqueeze, 0, "Mod", 0, mod) ||
      !HasInputCount(size_unsqueeze, 2) ||
      !IsIntValues(size_unsqueeze.GetInputs()[1], {0}) ||
      !GetInputProducer(size_unsqueeze, 0, "Cast", 0, size_cast64) ||
      !HasInputCount(count_reduce, 2) ||
      GetIntAttribute(count_reduce, "keepdims").value_or(1) != 1 ||
      !IsIntValues(count_reduce.GetInputs()[1], {0}) ||
      !HasInputCount(mod, 2) || !GetInputProducer(mod, 0, "Range", 0, range) ||
      !GetInputProducer(mod, 1, "Gather", 0, count_gather) ||
      !HasInputCount(size_cast64, 1) ||
      !IsCastTo(size_cast64, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      !GetInputProducer(size_cast64, 0, "Squeeze", 0, size_squeeze) ||
      !HasInputCount(range, 3) || !IsIntValues(range.GetInputs()[0], {0}) ||
      !IsIntValues(range.GetInputs()[2], {1}) ||
      !GetInputProducer(range, 1, "Squeeze", 0, id_shape_squeeze) ||
      !HasInputCount(count_gather, 2) ||
      GetIntAttribute(count_gather, "axis").value_or(0) != 0 ||
      !GetInputProducer(count_gather, 0, "Unique", 3, second_unique) ||
      !IsInputFrom(count_gather, 1, second_unique, 2) ||
      !HasInputCount(size_squeeze, 2) ||
      !IsIntValues(size_squeeze.GetInputs()[1], {0}) ||
      !GetInputProducer(size_squeeze, 0, "Slice", 0, size_slice) ||
      !HasInputCount(size_slice, 4) ||
      !IsIntValues(size_slice.GetInputs()[1], {0}) ||
      !IsIntValues(size_slice.GetInputs()[2], {1}) ||
      !IsIntValues(size_slice.GetInputs()[3], {0}) ||
      !GetInputProducer(size_slice, 0, "Cast", 0, size_shape_cast32) ||
      !HasInputCount(size_shape_cast32, 1) ||
      !IsCastTo(size_shape_cast32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) ||
      !GetInputProducer(size_shape_cast32, 0, "Shape", 0, first_unique_shape) ||
      !HasInputCount(first_unique_shape, 1) ||
      !HasInputCount(id_shape_squeeze, 2) ||
      !IsIntValues(id_shape_squeeze.GetInputs()[1], {0}) ||
      !GetInputProducer(id_shape_squeeze, 0, "Shape", 0, id_shape) ||
      !HasInputCount(second_unique, 1) ||
      GetIntAttribute(second_unique, "axis").value_or(-1) != 0 ||
      GetIntAttribute(second_unique, "sorted").value_or(1) != 1 ||
      !IsInputFrom(second_unique, 0, topk, 0) ||
      !IsInputFrom(count_reduce, 0, second_unique, 3) ||
      !GetInputProducer(topk, 0, "Cast", 0, index_cast64) ||
      !HasInputCount(id_shape, 1) ||
      !IsInputFrom(id_shape, 0, index_cast64, 0) || !HasInputCount(topk, 2) ||
      GetIntAttribute(topk, "axis").value_or(-1) != 0 ||
      GetIntAttribute(topk, "largest").value_or(1) != 0 ||
      GetIntAttribute(topk, "sorted").value_or(1) != 1 ||
      !IsInputFrom(topk, 0, index_cast64, 0) ||
      !IsInputFrom(topk, 1, id_shape, 0) || !HasInputCount(index_cast64, 1) ||
      !IsCastTo(index_cast64, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      !GetInputProducer(index_cast64, 0, "Cast", 0, index_cast32) ||
      !HasInputCount(index_cast32, 1) ||
      !IsCastTo(index_cast32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) ||
      !GetInputProducer(index_cast32, 0, "Unique", 2, first_unique) ||
      !HasInputCount(first_unique, 1) ||
      GetIntAttribute(first_unique, "sorted").value_or(1) != 0 ||
      !IsInputFrom(first_unique_shape, 0, first_unique, 0) ||
      !IsTensorOfType(first_unique.GetInputs()[0],
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      !IsTensorOfType(value_gather.GetInputs()[0],
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      !IsTensorOfType(final_gather.GetOutputs()[0],
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    return false;
  }

  std::array<Ort::ConstNode, 26> matched = {
      first_unique,      index_cast32,   index_cast64,
      id_shape,          topk,           topk_unsqueeze,
      second_unique,     count_reduce,   count_gather,
      id_shape_squeeze,  range,          mod,
      mod_unsqueeze,     index_concat,   first_unique_shape,
      size_shape_cast32, size_slice,     size_squeeze,
      size_cast64,       size_unsqueeze, shape_concat,
      constant_of_shape, scatter,        value_gather,
      final_reduce,      final_gather};

  std::unordered_set<size_t> selected_ids;
  fusion_nodes.clear();
  fusion_nodes.reserve(matched.size());
  for (Ort::ConstNode node : matched) {
    if (!AddFusionNode(node, accepted_node_ids, selected_ids, fusion_nodes)) {
      return false;
    }
  }
  if (!HasNoEscapingInternalOutputs(fusion_nodes, final_gather,
                                    graph_output_names)) {
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

std::vector<std::vector<Ort::ConstNode>> FindSegmentMaxBroadcastFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  if (std::getenv("ORT_MUSA_DISABLE_SEGMENT_MAX_BROADCAST_FUSION") != nullptr) {
    return {};
  }

  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion_nodes;
    if (MatchSegmentMaxBroadcast(node, graph_output_names, accepted_node_ids,
                                 fusion_nodes)) {
      fusions.push_back(std::move(fusion_nodes));
    }
  }
  return fusions;
}

}  // namespace musa_ep
