#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "fusion/multi_kqv_mha_output_projection_fusion.h"
#include "graph/graph_utils.h"

namespace musa_ep {

std::vector<std::vector<Ort::ConstNode>> FindMultiKqvMhaOutputProjectionFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> result;
  for (Ort::ConstNode candidate : all_nodes) {
    if (!IsOnnxOp(candidate, "Slice") ||
        accepted_node_ids.count(candidate.GetId()) != 0) {
      continue;
    }
    MultiKqvMhaOutputProjectionPattern pattern;
    if (!TryBuildMultiKqvMhaOutputProjectionPattern(
            candidate, graph_output_names, pattern)) {
      continue;
    }
    std::vector<Ort::ConstNode> nodes;
    std::unordered_set<size_t> selected;
    auto add = [&](Ort::ConstNode node) {
      if (!node || accepted_node_ids.count(node.GetId()) != 0) return false;
      if (selected.insert(node.GetId()).second) nodes.push_back(node);
      return true;
    };
    for (const auto& branch : pattern.branches) {
      if (!add(branch.slice) || !add(branch.matmul) || !add(branch.split)) {
        nodes.clear();
        break;
      }
    }
    if (nodes.empty()) continue;
    for (Ort::ConstNode node :
         {pattern.q_concat, pattern.k_concat, pattern.v_concat,
          pattern.q_unsqueeze, pattern.k_unsqueeze, pattern.v_unsqueeze,
          pattern.mha, pattern.output_gather}) {
      if (!add(node)) {
        nodes.clear();
        break;
      }
    }
    if (nodes.empty()) continue;
    for (Ort::ConstNode node : pattern.output_slices) {
      if (!add(node)) {
        nodes.clear();
        break;
      }
    }
    if (nodes.empty()) continue;
    for (Ort::ConstNode node : pattern.output_gemms) {
      if (!add(node)) {
        nodes.clear();
        break;
      }
    }
    if (nodes.empty() || !add(pattern.output_concat)) continue;
    if (!FusionHasNoExternalPathBetweenSelectedNodes(nodes, selected)) continue;
    result.push_back(std::move(nodes));
  }
  return result;
}

}  // namespace musa_ep
