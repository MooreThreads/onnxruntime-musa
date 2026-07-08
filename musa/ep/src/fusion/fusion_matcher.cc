// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "fusion/fusion_matcher.h"

#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <utility>

namespace musa_ep {
namespace {

bool NormalizeCandidate(const std::vector<Ort::ConstNode>& candidate,
                        std::vector<size_t>& node_ids) {
  if (candidate.empty()) {
    return false;
  }

  std::unordered_set<size_t> candidate_node_ids;
  node_ids.clear();
  node_ids.reserve(candidate.size());
  for (Ort::ConstNode node : candidate) {
    if (!node) {
      return false;
    }
    const size_t node_id = node.GetId();
    if (!candidate_node_ids.insert(node_id).second) {
      return false;
    }
    node_ids.push_back(node_id);
  }
  return true;
}

bool CandidateOverlapsAccepted(
    const std::vector<size_t>& node_ids,
    const std::unordered_set<size_t>& accepted_node_ids) {
  return std::any_of(node_ids.begin(), node_ids.end(), [&](size_t node_id) {
    return accepted_node_ids.count(node_id) != 0;
  });
}

std::vector<std::vector<Ort::ConstNode>> AcceptFusionCandidates(
    std::vector<std::vector<Ort::ConstNode>> candidates,
    std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> accepted;
  accepted.reserve(candidates.size());

  for (auto& candidate : candidates) {
    std::vector<size_t> node_ids;
    if (!NormalizeCandidate(candidate, node_ids) ||
        CandidateOverlapsAccepted(node_ids, accepted_node_ids)) {
      continue;
    }

    accepted_node_ids.insert(node_ids.begin(), node_ids.end());
    accepted.push_back(std::move(candidate));
  }
  return accepted;
}

bool FusionMatchesHaveNoOverlap(const std::vector<FusionMatch>& matches) {
  std::unordered_set<size_t> seen_node_ids;
  for (const FusionMatch& match : matches) {
    for (const auto& fusion : match.fusions) {
      std::vector<size_t> node_ids;
      if (!NormalizeCandidate(fusion, node_ids)) {
        return false;
      }
      for (size_t node_id : node_ids) {
        if (!seen_node_ids.insert(node_id).second) {
          return false;
        }
      }
    }
  }
  return true;
}

void AddFusionMatch(std::vector<FusionMatch>& matches, const char* finder,
                    bool drop_constant_initializers,
                    std::vector<std::vector<Ort::ConstNode>> candidates,
                    std::unordered_set<size_t>& accepted_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> accepted_fusions =
      AcceptFusionCandidates(std::move(candidates), accepted_node_ids);
  matches.push_back(
      {finder, drop_constant_initializers, std::move(accepted_fusions)});
}

}  // namespace

std::vector<FusionMatch> FindFusionMatches(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names) {
  std::unordered_set<size_t> accepted_node_ids;
  std::vector<FusionMatch> matches;

  auto split_unsqueeze_concat_fusions = FindSplitUnsqueezeConcatFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSplitUnsqueezeConcatFusions", false,
                 std::move(split_unsqueeze_concat_fusions), accepted_node_ids);

  auto split_concat_reorder_fusions = FindSplitConcatReorderFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSplitConcatReorderFusions", false,
                 std::move(split_concat_reorder_fusions), accepted_node_ids);

  auto concat_matmul_fusions =
      FindConcatMatMulFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindConcatMatMulFusions", false,
                 std::move(concat_matmul_fusions), accepted_node_ids);

  auto concat_split_fusions =
      FindConcatSplitFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindConcatSplitFusions", false,
                 std::move(concat_split_fusions), accepted_node_ids);

  auto slice_concat_fusions =
      FindSliceConcatFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSliceConcatFusions", false,
                 std::move(slice_concat_fusions), accepted_node_ids);

  auto tile_concat_fusions =
      FindTileConcatFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindTileConcatFusions", false,
                 std::move(tile_concat_fusions), accepted_node_ids);

  auto gemm_activation_fusions = FindGemmActivationFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindGemmActivationFusions", false,
                 std::move(gemm_activation_fusions), accepted_node_ids);

  auto fused_gemm_fusions =
      FindFusedGemmFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindFusedGemmFusions", false,
                 std::move(fused_gemm_fusions), accepted_node_ids);

  auto shape_reshape_fusions =
      FindShapeReshapeFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindShapeReshapeFusions", true,
                 std::move(shape_reshape_fusions), accepted_node_ids);

  auto centered_reduce_fusions = FindCenteredReduceFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindCenteredReduceFusions", false,
                 std::move(centered_reduce_fusions), accepted_node_ids);

  auto split_reduce_fusions =
      FindSplitReduceFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSplitReduceFusions", false,
                 std::move(split_reduce_fusions), accepted_node_ids);

  auto rms_norm_fusions =
      FindRmsNormFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindRmsNormFusions", false,
                 std::move(rms_norm_fusions), accepted_node_ids);

  auto modulo_gather_fusions =
      FindModuloGatherFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindModuloGatherFusions", false,
                 std::move(modulo_gather_fusions), accepted_node_ids);

  auto parallel_matmul_concat_fusions = FindParallelMatMulConcatFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindParallelMatMulConcatFusions", false,
                 std::move(parallel_matmul_concat_fusions), accepted_node_ids);

  auto parallel_einsum_activation_fusions = FindParallelEinsumActivationFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindParallelEinsumActivationFusions", false,
                 std::move(parallel_einsum_activation_fusions),
                 accepted_node_ids);

  auto math_concat_log_fusions = FindMathConcatLogFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindMathConcatLogFusions", false,
                 std::move(math_concat_log_fusions), accepted_node_ids);

  const bool no_overlap = FusionMatchesHaveNoOverlap(matches);
  assert(no_overlap);
  (void)no_overlap;
  return matches;
}

}  // namespace musa_ep
