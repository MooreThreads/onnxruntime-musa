// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fusion/fusion_matcher.h"

#include <algorithm>
#include <cassert>
#include <string>
#include <unordered_set>
#include <utility>

namespace musa_ep {
namespace {

bool CandidateSatisfiesDTypeContract(
    const std::vector<Ort::ConstNode>& candidate,
    const FusionDTypeContract& contract, std::string& reason) {
  if (std::string(contract.storage_types) == "none") {
    reason = "missing dtype contract";
    return false;
  }

  for (Ort::ConstNode node : candidate) {
    for (Ort::ConstValueInfo value_info : node.GetInputs()) {
      auto elem_type = GetTensorElementType(value_info);
      if (elem_type.has_value() && IsFloatingStorageType(*elem_type) &&
          !IsSupportedStorageType(*elem_type, contract)) {
        reason = "dtype contract saw unsupported floating input " +
                 std::string(FusionDTypeName(*elem_type));
        return false;
      }
    }
    for (Ort::ConstValueInfo value_info : node.GetOutputs()) {
      auto elem_type = GetTensorElementType(value_info);
      if (elem_type.has_value() && IsFloatingStorageType(*elem_type) &&
          !IsSupportedStorageType(*elem_type, contract)) {
        reason = "dtype contract saw unsupported floating output " +
                 std::string(FusionDTypeName(*elem_type));
        return false;
      }
    }
  }
  return true;
}

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
    const FusionDTypeContract& dtype_contract,
    std::unordered_set<size_t>& accepted_node_ids,
    std::vector<FusionDTypeRejection>& dtype_rejections) {
  std::vector<std::vector<Ort::ConstNode>> accepted;
  accepted.reserve(candidates.size());

  for (auto& candidate : candidates) {
    std::vector<size_t> node_ids;
    if (!NormalizeCandidate(candidate, node_ids) ||
        CandidateOverlapsAccepted(node_ids, accepted_node_ids)) {
      continue;
    }
    std::string rejection_reason;
    if (!CandidateSatisfiesDTypeContract(candidate, dtype_contract,
                                         rejection_reason)) {
      dtype_rejections.push_back(
          {candidate, rejection_reason.empty() ? "dtype contract rejected"
                                               : std::move(rejection_reason)});
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
  const FusionDTypeContract& dtype_contract =
      FusionDTypeContractForFinder(finder);
  std::vector<FusionDTypeRejection> dtype_rejections;
  std::vector<std::vector<Ort::ConstNode>> accepted_fusions =
      AcceptFusionCandidates(std::move(candidates), dtype_contract,
                             accepted_node_ids, dtype_rejections);
  matches.push_back({finder, drop_constant_initializers, dtype_contract,
                     std::move(accepted_fusions), std::move(dtype_rejections)});
}

}  // namespace

std::vector<FusionMatch> FindFusionMatches(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names) {
  std::unordered_set<size_t> accepted_node_ids;
  std::vector<FusionMatch> matches;

  auto multi_kqv_mha_output_projection_fusions =
      FindMultiKqvMhaOutputProjectionFusions(all_nodes, graph_output_names,
                                             accepted_node_ids);
  AddFusionMatch(matches, "FindMultiKqvMhaOutputProjectionFusions", false,
                 std::move(multi_kqv_mha_output_projection_fusions),
                 accepted_node_ids);

  auto qkv_attention_output_projection_fusions =
      FindQkvAttentionOutputProjectionFusions(all_nodes, graph_output_names,
                                              accepted_node_ids);
  AddFusionMatch(matches, "FindQkvAttentionOutputProjectionFusions", false,
                 std::move(qkv_attention_output_projection_fusions),
                 accepted_node_ids);

  // Keep the broader QKV attention/output-projection pattern ahead of the
  // local SDPA matcher so future ONNX-node-based expansions do not get
  // preempted by the narrower attention chain.
  auto mhta_scaled_dot_product_attention_fusions =
      FindMhtaScaledDotProductAttentionFusions(all_nodes, graph_output_names,
                                               accepted_node_ids);
  AddFusionMatch(matches, "FindMhtaScaledDotProductAttentionFusions", false,
                 std::move(mhta_scaled_dot_product_attention_fusions),
                 accepted_node_ids);

  auto generate_mtgr_custom_mask_fusions = FindGenerateMTGRCustomMaskFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindGenerateMTGRCustomMaskFusions", true,
                 std::move(generate_mtgr_custom_mask_fusions),
                 accepted_node_ids);

  auto moe_fusions =
      FindMoEFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindMoEFusions", true, std::move(moe_fusions),
                 accepted_node_ids);

  auto split_sequence_moe_fusions = FindSplitSequenceMoEFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSplitSequenceMoEFusions", true,
                 std::move(split_sequence_moe_fusions), accepted_node_ids);

  auto parallel_einsum_activation_fusions = FindParallelEinsumActivationFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindParallelEinsumActivationFusions", false,
                 std::move(parallel_einsum_activation_fusions),
                 accepted_node_ids);

  auto cast_rms_norm_fusions =
      FindCastRmsNormFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindCastRmsNormFusions", false,
                 std::move(cast_rms_norm_fusions), accepted_node_ids);

  auto rms_norm_fusions =
      FindRmsNormFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindRmsNormFusions", false,
                 std::move(rms_norm_fusions), accepted_node_ids);

  auto centered_reduce_fusions = FindCenteredReduceFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindCenteredReduceFusions", false,
                 std::move(centered_reduce_fusions), accepted_node_ids);

  auto segment_max_broadcast_fusions = FindSegmentMaxBroadcastFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSegmentMaxBroadcastFusions", true,
                 std::move(segment_max_broadcast_fusions), accepted_node_ids);

  auto rec_rank_calibration_fusions = FindRecRankCalibrationFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindRecRankCalibrationFusions", true,
                 std::move(rec_rank_calibration_fusions), accepted_node_ids);

  auto target_id_count_embedding_fusions = FindTargetIdCountEmbeddingFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindTargetIdCountEmbeddingFusions", false,
                 std::move(target_id_count_embedding_fusions),
                 accepted_node_ids);

  auto masked_embedding_lookup_fusions = FindMaskedEmbeddingLookupFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindMaskedEmbeddingLookupFusions", false,
                 std::move(masked_embedding_lookup_fusions), accepted_node_ids);

  auto sparse_id_to_mask_fusions = FindSparseIdToMaskFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSparseIdToMaskFusions", false,
                 std::move(sparse_id_to_mask_fusions), accepted_node_ids);

  auto bucketize_gather_fusions = FindBucketizeGatherFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindBucketizeGatherFusions", false,
                 std::move(bucketize_gather_fusions), accepted_node_ids);

  auto log_bucketize_gather_fusions = FindLogBucketizeGatherFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindLogBucketizeGatherFusions", false,
                 std::move(log_bucketize_gather_fusions), accepted_node_ids);

  auto modulo_gather_fusions =
      FindModuloGatherFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindModuloGatherFusions", false,
                 std::move(modulo_gather_fusions), accepted_node_ids);

  auto replace_invalid_id_fusions = FindReplaceInvalidIdFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindReplaceInvalidIdFusions", true,
                 std::move(replace_invalid_id_fusions), accepted_node_ids);

  auto math_concat_log_fusions = FindMathConcatLogFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindMathConcatLogFusions", false,
                 std::move(math_concat_log_fusions), accepted_node_ids);

  auto split_unsqueeze_concat_fusions = FindSplitUnsqueezeConcatFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSplitUnsqueezeConcatFusions", false,
                 std::move(split_unsqueeze_concat_fusions), accepted_node_ids);

  auto split_reduce_fusions =
      FindSplitReduceFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSplitReduceFusions", false,
                 std::move(split_reduce_fusions), accepted_node_ids);

  // ConcatSplit can absorb a downstream Concat. Let the more specific
  // Concat -> MatMul consumer claim that downstream pair first, then
  // ConcatSplit can safely shrink to the remaining Concat/Split/Sum nodes.
  auto concat_matmul_fusions =
      FindConcatMatMulFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindConcatMatMulFusions", false,
                 std::move(concat_matmul_fusions), accepted_node_ids);

  auto concat_split_fusions =
      FindConcatSplitFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindConcatSplitFusions", false,
                 std::move(concat_split_fusions), accepted_node_ids);

  auto split_concat_fusions =
      FindSplitConcatFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSplitConcatFusions", false,
                 std::move(split_concat_fusions), accepted_node_ids);

  // The MatMul -> Unsqueeze -> Concat pattern is more specific than a bare
  // parallel MatMul and must claim its nodes first.
  auto parallel_matmul_concat_fusions = FindParallelMatMulConcatFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindParallelMatMulConcatFusions", false,
                 std::move(parallel_matmul_concat_fusions), accepted_node_ids);

  // This larger, more specific pattern must run before the generic per-branch
  // linear matchers below; otherwise those matchers consume its nodes first.
  auto parallel_linear_fusions = FindParallelLinearFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindParallelLinearFusions", false,
                 std::move(parallel_linear_fusions), accepted_node_ids);

  auto strided_view_fusions =
      FindStridedViewFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindStridedViewFusions", true,
                 std::move(strided_view_fusions), accepted_node_ids);

  auto shape_reshape_fusions =
      FindShapeReshapeFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindShapeReshapeFusions", true,
                 std::move(shape_reshape_fusions), accepted_node_ids);

  auto tile_concat_fusions =
      FindTileConcatFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindTileConcatFusions", false,
                 std::move(tile_concat_fusions), accepted_node_ids);

  auto slice_concat_fusions =
      FindSliceConcatFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSliceConcatFusions", false,
                 std::move(slice_concat_fusions), accepted_node_ids);

  auto concat_reshape_fusions = FindConcatReshapeFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindConcatReshapeFusions", true,
                 std::move(concat_reshape_fusions), accepted_node_ids);

  auto gemm_activation_fusions = FindGemmActivationFusions(
      all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindGemmActivationFusions", false,
                 std::move(gemm_activation_fusions), accepted_node_ids);

  auto fused_gemm_fusions =
      FindFusedGemmFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindFusedGemmFusions", false,
                 std::move(fused_gemm_fusions), accepted_node_ids);

  auto silu_fusions =
      FindSiluFusions(all_nodes, graph_output_names, accepted_node_ids);
  AddFusionMatch(matches, "FindSiluFusions", false, std::move(silu_fusions),
                 accepted_node_ids);

  const bool no_overlap = FusionMatchesHaveNoOverlap(matches);
  assert(no_overlap);
  (void)no_overlap;
  return matches;
}

}  // namespace musa_ep
