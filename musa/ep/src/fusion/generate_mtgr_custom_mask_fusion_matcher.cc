// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

using ProducerMap = std::unordered_map<std::string, Ort::ConstNode>;

struct ShapeGatherMatch {
  Ort::ConstNode shape{nullptr};
  Ort::ConstNode gather{nullptr};
  Ort::ConstValueInfo source{nullptr};
};

struct PatternMatch {
  std::vector<Ort::ConstNode> nodes;
  Ort::ConstNode target_gather{nullptr};
  Ort::ConstNode total_length_add{nullptr};
  Ort::ConstNode mask_unsqueeze{nullptr};
  Ort::ConstNode final_mask{nullptr};
  bool cropped = false;
};

Ort::ConstNode Producer(const ProducerMap& producers,
                        Ort::ConstValueInfo value) {
  return FindProducer(producers, value);
}

bool IsTensorType(Ort::ConstValueInfo value,
                  ONNXTensorElementDataType expected) {
  auto type = GetTensorElementType(value);
  return type.has_value() && *type == expected;
}

bool HasShape(Ort::ConstValueInfo value, const std::vector<int64_t>& expected) {
  auto shape = GetTensorShape(value);
  return shape.has_value() && *shape == expected;
}

bool IsScalarInt64(Ort::ConstValueInfo value) {
  return IsTensorType(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) &&
         HasShape(value, {});
}

bool HasIotaShape(Ort::ConstValueInfo value, bool row_iota,
                  std::optional<int64_t> capacity = std::nullopt) {
  auto shape = GetTensorShape(value);
  if (!shape.has_value() || shape->size() != 3 || (*shape)[0] != 1 ||
      (row_iota ? (*shape)[2] != 1 : (*shape)[1] != 1)) {
    return false;
  }
  const int64_t inferred_capacity = row_iota ? (*shape)[1] : (*shape)[2];
  return inferred_capacity > 0 &&
         (!capacity.has_value() || inferred_capacity == *capacity);
}

std::optional<int64_t> ReadIotaCapacity(Ort::ConstValueInfo value,
                                        bool row_iota) {
  if (!IsTensorType(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      !HasIotaShape(value, row_iota)) {
    return std::nullopt;
  }
  auto shape = GetTensorShape(value);
  const int64_t capacity = row_iota ? (*shape)[1] : (*shape)[2];
  auto values = ReadIntInitializerNoLimit(value);
  if (!values.has_value() || values->size() != static_cast<size_t>(capacity)) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < capacity; ++i) {
    if ((*values)[static_cast<size_t>(i)] != i) {
      return std::nullopt;
    }
  }
  return capacity;
}

bool IsIotaInitializer(Ort::ConstValueInfo value, bool row_iota,
                       int64_t capacity) {
  return ReadIotaCapacity(value, row_iota).value_or(-1) == capacity;
}

bool NodeHasSingleOutput(Ort::ConstNode node) {
  return node && node.GetOutputs().size() == 1;
}

bool MatchShapeGather(Ort::ConstNode gather, const ProducerMap& producers,
                      size_t expected_rank, ShapeGatherMatch& match) {
  if (!IsOnnxOp(gather, "Gather") ||
      GetIntAttribute(gather, "axis").value_or(0) != 0) {
    return false;
  }
  auto inputs = gather.GetInputs();
  auto outputs = gather.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 ||
      ReadScalarIntInitializer(inputs[1]).value_or(-1) != 1 ||
      !IsScalarInt64(outputs[0])) {
    return false;
  }

  Ort::ConstNode shape = Producer(producers, inputs[0]);
  if (!IsOnnxOp(shape, "Shape") || shape.GetInputs().size() != 1 ||
      !NodeHasSingleOutput(shape)) {
    return false;
  }
  Ort::ConstValueInfo source = shape.GetInputs()[0];
  auto source_shape = GetTensorShape(source);
  if (!source_shape.has_value() || source_shape->size() != expected_rank) {
    return false;
  }
  match = {shape, gather, source};
  return true;
}

bool SplitBinaryProducers(Ort::ConstNode node, const char* op_type,
                          const ProducerMap& producers, Ort::ConstNode& first,
                          Ort::ConstNode& second) {
  if (!IsOnnxOp(node, op_type)) {
    return false;
  }
  auto inputs = node.GetInputs();
  if (inputs.size() != 2 || !NodeHasSingleOutput(node)) {
    return false;
  }
  first = Producer(producers, inputs[0]);
  second = Producer(producers, inputs[1]);
  return first && second;
}

bool SplitByOpTypes(Ort::ConstNode node, const char* node_type,
                    const char* first_type, const char* second_type,
                    const ProducerMap& producers, Ort::ConstNode& first,
                    Ort::ConstNode& second) {
  Ort::ConstNode lhs{nullptr};
  Ort::ConstNode rhs{nullptr};
  if (!SplitBinaryProducers(node, node_type, producers, lhs, rhs)) {
    return false;
  }
  if (IsOnnxOp(lhs, first_type) && IsOnnxOp(rhs, second_type)) {
    first = lhs;
    second = rhs;
    return true;
  }
  if (IsOnnxOp(rhs, first_type) && IsOnnxOp(lhs, second_type)) {
    first = rhs;
    second = lhs;
    return true;
  }
  return false;
}

bool InputsAreProducedBy(Ort::ConstNode node, Ort::ConstNode first,
                         Ort::ConstNode second, const ProducerMap& producers) {
  Ort::ConstNode lhs{nullptr};
  Ort::ConstNode rhs{nullptr};
  if (!SplitBinaryProducers(node, node.GetOperatorType().c_str(), producers,
                            lhs, rhs)) {
    return false;
  }
  return (lhs.GetId() == first.GetId() && rhs.GetId() == second.GetId()) ||
         (lhs.GetId() == second.GetId() && rhs.GetId() == first.GetId());
}

bool MatchComparison(Ort::ConstNode node, const char* op_type, bool row_iota,
                     int64_t capacity, Ort::ConstNode bound,
                     const ProducerMap& producers) {
  if (!IsOnnxOp(node, op_type) || node.GetInputs().size() != 2 ||
      !NodeHasSingleOutput(node)) {
    return false;
  }
  auto inputs = node.GetInputs();
  return IsIotaInitializer(inputs[0], row_iota, capacity) &&
         Producer(producers, inputs[1]).GetId() == bound.GetId();
}

bool MatchCastSub(Ort::ConstNode cast, bool row_iota, int64_t capacity,
                  Ort::ConstNode bound, const ProducerMap& producers,
                  Ort::ConstNode& sub) {
  if (!IsOnnxOp(cast, "Cast") || cast.GetInputs().size() != 1 ||
      !NodeHasSingleOutput(cast) ||
      GetIntAttribute(cast, "to").value_or(-1) !=
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return false;
  }
  sub = Producer(producers, cast.GetInputs()[0]);
  if (!IsOnnxOp(sub, "Sub") || sub.GetInputs().size() != 2 ||
      !NodeHasSingleOutput(sub)) {
    return false;
  }
  auto inputs = sub.GetInputs();
  return IsIotaInitializer(inputs[0], row_iota, capacity) &&
         Producer(producers, inputs[1]).GetId() == bound.GetId();
}

bool MatchReshape(Ort::ConstNode reshape, const ProducerMap& producers,
                  Ort::ConstNode& data_producer) {
  if (!IsOnnxOp(reshape, "Reshape") || reshape.GetInputs().size() != 2 ||
      !NodeHasSingleOutput(reshape) ||
      GetIntAttribute(reshape, "allowzero").value_or(0) != 0) {
    return false;
  }
  auto inputs = reshape.GetInputs();
  auto shape = ReadSmallIntInitializer(inputs[1]);
  if (!shape.has_value() || *shape != std::vector<int64_t>({1, 1, 1})) {
    return false;
  }
  data_producer = Producer(producers, inputs[0]);
  return data_producer != nullptr;
}

bool MatchAddChain(Ort::ConstNode final_add, const ProducerMap& producers,
                   std::array<ShapeGatherMatch, 4>& upstream,
                   std::vector<Ort::ConstNode>& add_nodes,
                   int64_t& prefix_length) {
  Ort::ConstNode current = final_add;
  for (int index = 3; index >= 1; --index) {
    Ort::ConstNode previous_add{nullptr};
    Ort::ConstNode gather{nullptr};
    if (!SplitByOpTypes(current, "Add", "Add", "Gather", producers,
                        previous_add, gather) ||
        !MatchShapeGather(gather, producers, 3,
                          upstream[static_cast<size_t>(index)])) {
      return false;
    }
    add_nodes.push_back(current);
    current = previous_add;
  }

  if (!IsOnnxOp(current, "Add") || current.GetInputs().size() != 2 ||
      !NodeHasSingleOutput(current)) {
    return false;
  }
  auto initial_inputs = current.GetInputs();
  Ort::ConstNode gather = Producer(producers, initial_inputs[0]);
  auto prefix = ReadScalarIntInitializer(initial_inputs[1]);
  if (!IsOnnxOp(gather, "Gather") || !prefix.has_value() || *prefix < 0 ||
      !MatchShapeGather(gather, producers, 3, upstream[0])) {
    return false;
  }
  prefix_length = *prefix;
  add_nodes.push_back(current);

  std::unordered_set<std::string> source_names;
  for (const auto& item : upstream) {
    if (!source_names.insert(Name(item.source)).second) {
      return false;
    }
  }
  return true;
}

bool IsGraphOutput(Ort::ConstValueInfo value,
                   const std::unordered_set<std::string>& graph_output_names) {
  return graph_output_names.count(Name(value)) != 0;
}

bool MatchSlice(Ort::ConstNode slice, Ort::ConstNode data_producer,
                Ort::ConstValueInfo ends,
                const std::vector<int64_t>& expected_starts,
                const std::vector<int64_t>& expected_axes,
                const std::vector<int64_t>& expected_steps,
                const ProducerMap& producers) {
  if (!IsOnnxOp(slice, "Slice") || slice.GetInputs().size() != 5 ||
      !NodeHasSingleOutput(slice)) {
    return false;
  }
  auto inputs = slice.GetInputs();
  return Producer(producers, inputs[0]).GetId() == data_producer.GetId() &&
         Name(inputs[2]) == Name(ends) &&
         ReadSmallIntInitializer(inputs[1]).value_or(std::vector<int64_t>{}) ==
             expected_starts &&
         ReadSmallIntInitializer(inputs[3]).value_or(std::vector<int64_t>{}) ==
             expected_axes &&
         ReadSmallIntInitializer(inputs[4]).value_or(std::vector<int64_t>{}) ==
             expected_steps;
}

bool MatchFinalInt32Cast(Ort::ConstNode cast, Ort::ConstNode input_producer,
                         const ProducerMap& producers) {
  return IsOnnxOp(cast, "Cast") && cast.GetInputs().size() == 1 &&
         NodeHasSingleOutput(cast) &&
         Producer(producers, cast.GetInputs()[0]).GetId() ==
             input_producer.GetId() &&
         GetIntAttribute(cast, "to").value_or(-1) ==
             ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
         IsTensorType(cast.GetOutputs()[0],
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
}

bool MatchCroppedTail(
    PatternMatch& match, const ProducerMap& producers,
    const std::unordered_set<std::string>& graph_output_names) {
  auto full_mask_output = match.mask_unsqueeze.GetOutputs()[0];
  auto total_length_output = match.total_length_add.GetOutputs()[0];
  if (IsGraphOutput(full_mask_output, graph_output_names) ||
      IsGraphOutput(total_length_output, graph_output_names)) {
    return false;
  }

  auto mask_consumers = full_mask_output.GetConsumers();
  auto length_consumers = total_length_output.GetConsumers();
  if (mask_consumers.size() != 1 || length_consumers.size() != 1 ||
      mask_consumers[0].index != 0 || length_consumers[0].index != 0) {
    return false;
  }
  Ort::ConstNode first_slice = mask_consumers[0].node;
  Ort::ConstNode length_unsqueeze = length_consumers[0].node;
  if (!IsOnnxOp(length_unsqueeze, "Unsqueeze") ||
      length_unsqueeze.GetInputs().size() != 2 ||
      !NodeHasSingleOutput(length_unsqueeze) ||
      ReadUnsqueezeAxes(length_unsqueeze).value_or(std::vector<int64_t>{}) !=
          std::vector<int64_t>({0}) ||
      !IsTensorType(length_unsqueeze.GetOutputs()[0],
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }
  Ort::ConstValueInfo length_vector = length_unsqueeze.GetOutputs()[0];

  std::vector<Ort::ConstNode> tail_nodes = {length_unsqueeze};
  Ort::ConstNode final_slice{nullptr};
  auto first_inputs = first_slice.GetInputs();
  if (first_inputs.size() != 5) {
    return false;
  }
  Ort::ConstNode ends_producer = Producer(producers, first_inputs[2]);
  if (Name(first_inputs[2]) == Name(length_vector)) {
    if (!MatchSlice(first_slice, match.mask_unsqueeze, length_vector, {0}, {2},
                    {1}, producers)) {
      return false;
    }
    auto first_consumers = first_slice.GetOutputs()[0].GetConsumers();
    if (first_consumers.size() != 1 || first_consumers[0].index != 0) {
      return false;
    }
    Ort::ConstNode second_slice = first_consumers[0].node;
    if (!MatchSlice(second_slice, first_slice, length_vector, {0}, {3}, {1},
                    producers)) {
      return false;
    }
    auto length_vector_consumers = length_vector.GetConsumers();
    if (length_vector_consumers.size() != 2) {
      return false;
    }
    std::unordered_set<size_t> expected_slice_ids = {first_slice.GetId(),
                                                     second_slice.GetId()};
    for (const auto& consumer : length_vector_consumers) {
      if (consumer.index != 2 ||
          expected_slice_ids.count(consumer.node.GetId()) == 0) {
        return false;
      }
    }
    tail_nodes.insert(tail_nodes.end(), {first_slice, second_slice});
    final_slice = second_slice;
  } else {
    Ort::ConstNode concat = ends_producer;
    if (!IsOnnxOp(concat, "Concat") || concat.GetInputs().size() != 2 ||
        !NodeHasSingleOutput(concat) ||
        GetIntAttribute(concat, "axis").value_or(-1) != 0) {
      return false;
    }
    for (Ort::ConstValueInfo input : concat.GetInputs()) {
      if (Name(input) != Name(length_vector)) {
        return false;
      }
    }
    if (!HasOnlyConsumer(concat.GetOutputs()[0], first_slice, 2) ||
        !MatchSlice(first_slice, match.mask_unsqueeze, concat.GetOutputs()[0],
                    {0, 0}, {2, 3}, {1, 1}, producers)) {
      return false;
    }
    auto length_vector_consumers = length_vector.GetConsumers();
    if (length_vector_consumers.size() != 2) {
      return false;
    }
    for (const auto& consumer : length_vector_consumers) {
      if (consumer.node.GetId() != concat.GetId() ||
          (consumer.index != 0 && consumer.index != 1)) {
        return false;
      }
    }
    tail_nodes.insert(tail_nodes.end(), {concat, first_slice});
    final_slice = first_slice;
  }

  if (IsGraphOutput(final_slice.GetOutputs()[0], graph_output_names)) {
    return false;
  }
  auto final_consumers = final_slice.GetOutputs()[0].GetConsumers();
  if (final_consumers.size() != 1 || final_consumers[0].index != 0) {
    return false;
  }
  Ort::ConstNode final_cast = final_consumers[0].node;
  if (!MatchFinalInt32Cast(final_cast, final_slice, producers)) {
    return false;
  }
  tail_nodes.push_back(final_cast);
  match.nodes.insert(match.nodes.end(), tail_nodes.begin(), tail_nodes.end());
  match.final_mask = final_cast;
  match.cropped = true;
  return true;
}

bool ValidateInternalConsumers(
    const PatternMatch& match,
    const std::unordered_set<std::string>& graph_output_names) {
  std::unordered_set<size_t> selected;
  for (Ort::ConstNode node : match.nodes) {
    selected.insert(node.GetId());
  }
  std::unordered_set<size_t> boundary_nodes = {match.target_gather.GetId()};
  if (match.cropped) {
    boundary_nodes.insert(match.final_mask.GetId());
  } else {
    boundary_nodes.insert(match.total_length_add.GetId());
    boundary_nodes.insert(match.mask_unsqueeze.GetId());
  }

  for (Ort::ConstNode node : match.nodes) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      const bool boundary = boundary_nodes.count(node.GetId()) != 0;
      if (!boundary && graph_output_names.count(Name(output)) != 0) {
        return false;
      }
      for (const auto& consumer : output.GetConsumers()) {
        if (!boundary && selected.count(consumer.node.GetId()) == 0) {
          return false;
        }
      }
    }
  }
  return true;
}

bool CanFuseGenerateMTGRCustomMask(
    Ort::ConstNode unsqueeze, const ProducerMap& producers,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids,
    std::vector<Ort::ConstNode>& fusion_nodes) {
  if (!IsOnnxOp(unsqueeze, "Unsqueeze") ||
      accepted_node_ids.count(unsqueeze.GetId()) != 0 ||
      unsqueeze.GetInputs().size() != 2 || !NodeHasSingleOutput(unsqueeze)) {
    return false;
  }
  auto axes = ReadUnsqueezeAxes(unsqueeze);
  if (!axes.has_value() || *axes != std::vector<int64_t>({1}) ||
      !IsTensorType(unsqueeze.GetOutputs()[0],
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)) {
    return false;
  }

  Ort::ConstNode and7 = Producer(producers, unsqueeze.GetInputs()[0]);
  Ort::ConstNode or1{nullptr};
  Ort::ConstNode and6{nullptr};
  if (!SplitByOpTypes(and7, "And", "Or", "And", producers, or1, and6)) {
    return false;
  }
  Ort::ConstNode or0{nullptr};
  Ort::ConstNode and5{nullptr};
  if (!SplitByOpTypes(or1, "Or", "Or", "And", producers, or0, and5)) {
    return false;
  }
  Ort::ConstNode and4{nullptr};
  Ort::ConstNode equal{nullptr};
  if (!SplitByOpTypes(and5, "And", "And", "Equal", producers, and4, equal)) {
    return false;
  }

  Ort::ConstNode upstream_and{nullptr};
  Ort::ConstNode target_to_upstream_and{nullptr};
  if (!SplitByOpTypes(or0, "Or", "And", "And", producers, upstream_and,
                      target_to_upstream_and)) {
    return false;
  }
  auto upstream_inputs = upstream_and.GetInputs();
  if (upstream_inputs.size() != 2) {
    return false;
  }
  Ort::ConstNode upstream_lhs = Producer(producers, upstream_inputs[0]);
  Ort::ConstNode upstream_rhs = Producer(producers, upstream_inputs[1]);
  if (!IsOnnxOp(upstream_lhs, "Less") || !IsOnnxOp(upstream_rhs, "Less")) {
    return false;
  }

  Ort::ConstNode reshape_u{nullptr};
  Ort::ConstNode reshape_t{nullptr};
  Ort::ConstNode add_l{nullptr};
  Ort::ConstNode add9{nullptr};
  Ort::ConstNode row_less_u{nullptr};
  Ort::ConstNode col_less_u{nullptr};

  // Both upstream Less nodes use the same U reshape. Identify row/column by
  // their iota initializer shapes, exactly as seen in unirank_onnx.mmd.
  for (Ort::ConstNode candidate : {upstream_lhs, upstream_rhs}) {
    auto inputs = candidate.GetInputs();
    if (inputs.size() != 2) {
      return false;
    }
    Ort::ConstNode bound = Producer(producers, inputs[1]);
    if (!IsOnnxOp(bound, "Reshape")) {
      return false;
    }
    if (!reshape_u) {
      reshape_u = bound;
    } else if (reshape_u.GetId() != bound.GetId()) {
      return false;
    }
    if (HasIotaShape(inputs[0], true)) {
      row_less_u = candidate;
    } else if (HasIotaShape(inputs[0], false)) {
      col_less_u = candidate;
    } else {
      return false;
    }
  }
  if (!row_less_u || !col_less_u) {
    return false;
  }

  Ort::ConstNode u_add{nullptr};
  if (!MatchReshape(reshape_u, producers, u_add) || !IsOnnxOp(u_add, "Add")) {
    return false;
  }

  Ort::ConstNode row_ge_u{nullptr};
  Ort::ConstNode shared_col_less{nullptr};
  if (!SplitByOpTypes(target_to_upstream_and, "And", "GreaterOrEqual", "Less",
                      producers, row_ge_u, shared_col_less) ||
      shared_col_less.GetId() != col_less_u.GetId() ||
      row_ge_u.GetInputs().size() != 2) {
    return false;
  }
  // The row GreaterOrEqual A input defines the mask capacity. All other iota
  // initializers must have the same inferred extent and sequence values.
  auto capacity = ReadIotaCapacity(row_ge_u.GetInputs()[0], true);
  if (!capacity.has_value() ||
      !MatchComparison(row_ge_u, "GreaterOrEqual", true, *capacity, reshape_u,
                       producers) ||
      !IsIotaInitializer(row_less_u.GetInputs()[0], true, *capacity) ||
      !IsIotaInitializer(col_less_u.GetInputs()[0], false, *capacity)) {
    return false;
  }

  Ort::ConstNode row_less_l{nullptr};
  Ort::ConstNode col_less_l{nullptr};
  Ort::ConstNode and6_lhs{nullptr};
  Ort::ConstNode and6_rhs{nullptr};
  if (!SplitBinaryProducers(and6, "And", producers, and6_lhs, and6_rhs)) {
    return false;
  }
  for (Ort::ConstNode candidate : {and6_lhs, and6_rhs}) {
    if (!IsOnnxOp(candidate, "Less") || candidate.GetInputs().size() != 2) {
      return false;
    }
    auto inputs = candidate.GetInputs();
    Ort::ConstNode bound = Producer(producers, inputs[1]);
    if (!add9) {
      add9 = bound;
    } else if (!bound || add9.GetId() != bound.GetId()) {
      return false;
    }
    if (IsIotaInitializer(inputs[0], true, *capacity)) {
      row_less_l = candidate;
    } else if (IsIotaInitializer(inputs[0], false, *capacity)) {
      col_less_l = candidate;
    } else {
      return false;
    }
  }
  if (!row_less_l || !col_less_l || !IsOnnxOp(add9, "Add")) {
    return false;
  }

  Ort::ConstNode add9_lhs{nullptr};
  Ort::ConstNode add9_rhs{nullptr};
  if (!SplitBinaryProducers(add9, "Add", producers, add9_lhs, add9_rhs) ||
      !IsOnnxOp(add9_lhs, "Reshape") || !IsOnnxOp(add9_rhs, "Reshape")) {
    return false;
  }
  if (add9_lhs.GetId() == reshape_u.GetId()) {
    reshape_t = add9_rhs;
  } else if (add9_rhs.GetId() == reshape_u.GetId()) {
    reshape_t = add9_lhs;
  } else {
    return false;
  }
  Ort::ConstNode target_gather{nullptr};
  if (!MatchReshape(reshape_t, producers, target_gather)) {
    return false;
  }
  ShapeGatherMatch target;
  if (!MatchShapeGather(target_gather, producers, 2, target)) {
    return false;
  }

  std::array<ShapeGatherMatch, 4> upstream;
  std::vector<Ort::ConstNode> upstream_adds;
  int64_t prefix_length = 0;
  if (!MatchAddChain(u_add, producers, upstream, upstream_adds,
                     prefix_length)) {
    return false;
  }

  // L is independently present in the optimized graph as U + T.
  for (const auto& consumer : u_add.GetOutputs()[0].GetConsumers()) {
    Ort::ConstNode candidate = consumer.node;
    if (!IsOnnxOp(candidate, "Add") || candidate.GetId() == add9.GetId()) {
      continue;
    }
    auto inputs = candidate.GetInputs();
    if (inputs.size() != 2) {
      continue;
    }
    bool has_u = false;
    bool has_t = false;
    for (Ort::ConstValueInfo input : inputs) {
      Ort::ConstNode producer = Producer(producers, input);
      has_u |= producer && producer.GetId() == u_add.GetId();
      has_t |= producer && producer.GetId() == target_gather.GetId();
    }
    if (has_u && has_t) {
      add_l = candidate;
      break;
    }
  }
  if (!add_l) {
    return false;
  }

  Ort::ConstNode and2{nullptr};
  Ort::ConstNode and3{nullptr};
  if (!SplitByOpTypes(and4, "And", "And", "And", producers, and2, and3)) {
    return false;
  }
  auto is_row_range = [&](Ort::ConstNode node) {
    return InputsAreProducedBy(node, row_ge_u, row_less_l, producers);
  };
  Ort::ConstNode col_ge_u{nullptr};
  Ort::ConstNode row_range{nullptr};
  Ort::ConstNode col_range{nullptr};
  if (is_row_range(and2)) {
    row_range = and2;
    col_range = and3;
  } else if (is_row_range(and3)) {
    row_range = and3;
    col_range = and2;
  } else {
    return false;
  }
  Ort::ConstNode col_range_lhs{nullptr};
  Ort::ConstNode col_range_rhs{nullptr};
  if (!SplitBinaryProducers(col_range, "And", producers, col_range_lhs,
                            col_range_rhs)) {
    return false;
  }
  if (IsOnnxOp(col_range_lhs, "GreaterOrEqual") &&
      col_range_rhs.GetId() == col_less_l.GetId()) {
    col_ge_u = col_range_lhs;
  } else if (IsOnnxOp(col_range_rhs, "GreaterOrEqual") &&
             col_range_lhs.GetId() == col_less_l.GetId()) {
    col_ge_u = col_range_rhs;
  } else {
    return false;
  }
  if (!MatchComparison(col_ge_u, "GreaterOrEqual", false, *capacity, reshape_u,
                       producers)) {
    return false;
  }

  Ort::ConstNode equal_lhs{nullptr};
  Ort::ConstNode equal_rhs{nullptr};
  if (!SplitBinaryProducers(equal, "Equal", producers, equal_lhs, equal_rhs) ||
      !IsOnnxOp(equal_lhs, "Cast") || !IsOnnxOp(equal_rhs, "Cast")) {
    return false;
  }
  Ort::ConstNode row_cast{nullptr};
  Ort::ConstNode col_cast{nullptr};
  Ort::ConstNode row_sub{nullptr};
  Ort::ConstNode col_sub{nullptr};
  if (MatchCastSub(equal_lhs, true, *capacity, reshape_u, producers, row_sub) &&
      MatchCastSub(equal_rhs, false, *capacity, reshape_u, producers,
                   col_sub)) {
    row_cast = equal_lhs;
    col_cast = equal_rhs;
  } else if (MatchCastSub(equal_rhs, true, *capacity, reshape_u, producers,
                          row_sub) &&
             MatchCastSub(equal_lhs, false, *capacity, reshape_u, producers,
                          col_sub)) {
    row_cast = equal_rhs;
    col_cast = equal_lhs;
  } else {
    return false;
  }

  PatternMatch match;
  match.target_gather = target_gather;
  match.total_length_add = add_l;
  match.mask_unsqueeze = unsqueeze;
  match.final_mask = unsqueeze;
  for (const auto& item : upstream) {
    match.nodes.push_back(item.shape);
    match.nodes.push_back(item.gather);
  }
  match.nodes.push_back(target.shape);
  match.nodes.push_back(target.gather);
  match.nodes.insert(match.nodes.end(), upstream_adds.begin(),
                     upstream_adds.end());
  match.nodes.insert(match.nodes.end(),
                     {add_l,        reshape_u,  reshape_t,
                      add9,         row_less_u, col_less_u,
                      upstream_and, row_ge_u,   target_to_upstream_and,
                      or0,          row_less_l, col_less_l,
                      and6,         row_range,  col_ge_u,
                      col_range,    and4,       row_sub,
                      row_cast,     col_sub,    col_cast,
                      equal,        and5,       or1,
                      and7,         unsqueeze});

  // Prefer the phase-2 CroppedInt32 form. If the strict Slice/Cast tail is not
  // present, retain the phase-1 FullBool candidate for other compatible graphs.
  MatchCroppedTail(match, producers, graph_output_names);

  std::unordered_set<size_t> selected_ids;
  fusion_nodes.clear();
  for (Ort::ConstNode node : match.nodes) {
    if (!AddFusionNode(node, accepted_node_ids, selected_ids, fusion_nodes)) {
      return false;
    }
  }
  if (fusion_nodes.size() != (match.cropped ? 44u : 40u) ||
      !ValidateInternalConsumers(match, graph_output_names) ||
      !FusionHasNoExternalPathBetweenSelectedNodes(fusion_nodes,
                                                   selected_ids)) {
    return false;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return true;
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindGenerateMTGRCustomMaskFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  ProducerMap producers = BuildProducerMap(all_nodes);
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion;
    if (CanFuseGenerateMTGRCustomMask(node, producers, graph_output_names,
                                      accepted_node_ids, fusion)) {
      fusions.push_back(std::move(fusion));
    }
  }
  return fusions;
}

}  // namespace musa_ep
