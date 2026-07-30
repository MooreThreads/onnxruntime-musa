// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"

namespace musa_ep {
namespace {

bool IsScalarIntInitializer(Ort::ConstValueInfo value, int64_t expected) {
  auto values = ReadIntInitializerNoLimit(value);
  return values.has_value() && values->size() == 1 && (*values)[0] == expected;
}

bool CanFuseStridedView(Ort::ConstNode concat,
                        const std::unordered_set<std::string>& outputs,
                        const std::unordered_set<size_t>& accepted,
                        std::vector<Ort::ConstNode>& fusion) {
  if (!IsOnnxOp(concat, "Concat") || accepted.count(concat.GetId()))
    return false;
  auto concat_inputs = concat.GetInputs();
  auto concat_outputs = concat.GetOutputs();
  auto axis = GetIntAttribute(concat, "axis");
  if (!axis.has_value() || *axis != 2 || concat_inputs.size() < 2 ||
      concat_outputs.size() != 1 || outputs.count(Name(concat_outputs[0])))
    return false;

  Ort::ConstNode source{nullptr};
  std::vector<Ort::ConstNode> selected{concat};
  std::unordered_set<size_t> ids{concat.GetId()};
  for (size_t i = 0; i < concat_inputs.size(); ++i) {
    Ort::ConstNode slice = concat_inputs[i].GetProducerNode().node;
    if (!IsOnnxOp(slice, "Slice") || accepted.count(slice.GetId()))
      return false;
    auto in = slice.GetInputs();
    auto out = slice.GetOutputs();
    if (in.size() != 4 || out.size() != 1 ||
        !HasOnlyConsumer(out[0], concat, static_cast<int>(i)) ||
        !IsScalarIntInitializer(in[3], 0))
      return false;
    Ort::ConstNode this_source = in[0].GetProducerNode().node;
    if (!IsOnnxOp(this_source, "MatMul")) return false;
    if (source && source.GetId() != this_source.GetId()) return false;
    source = this_source;
    if (!ids.insert(slice.GetId()).second) return false;
    selected.push_back(slice);
    if (i == 0 && !IsScalarIntInitializer(in[1], 0)) return false;
  }
  if (!source) return false;
  std::sort(
      selected.begin(), selected.end(),
      [](Ort::ConstNode a, Ort::ConstNode b) { return a.GetId() < b.GetId(); });
  fusion = std::move(selected);
  return true;
}
}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindStridedViewFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;
  for (Ort::ConstNode node : all_nodes) {
    std::vector<Ort::ConstNode> fusion;
    if (CanFuseStridedView(node, graph_output_names, accepted_node_ids, fusion))
      fusions.push_back(std::move(fusion));
  }
  return fusions;
}
}  // namespace musa_ep
