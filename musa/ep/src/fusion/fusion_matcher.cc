// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/fusion_matcher.h"

#include <unordered_set>
#include <utility>

namespace musa_ep {
namespace {

void AddFusionMatch(std::vector<FusionMatch>& matches, const char* finder,
                    bool drop_constant_initializers,
                    std::vector<std::vector<Ort::ConstNode>> fusions) {
  matches.push_back({finder, drop_constant_initializers, std::move(fusions)});
}

}  // namespace

std::vector<FusionMatch> FindFusionMatches(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names) {
  std::unordered_set<size_t> fused_node_ids;
  std::vector<FusionMatch> matches;

  auto split_unsqueeze_concat_fusions = FindSplitUnsqueezeConcatFusions(
      all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindSplitUnsqueezeConcatFusions", false,
                 std::move(split_unsqueeze_concat_fusions));

  auto split_concat_reorder_fusions = FindSplitConcatReorderFusions(
      all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindSplitConcatReorderFusions", false,
                 std::move(split_concat_reorder_fusions));

  auto concat_matmul_fusions =
      FindConcatMatMulFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindConcatMatMulFusions", false,
                 std::move(concat_matmul_fusions));

  auto concat_split_fusions =
      FindConcatSplitFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindConcatSplitFusions", false,
                 std::move(concat_split_fusions));

  auto slice_concat_fusions =
      FindSliceConcatFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindSliceConcatFusions", false,
                 std::move(slice_concat_fusions));

  auto tile_concat_fusions =
      FindTileConcatFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindTileConcatFusions", false,
                 std::move(tile_concat_fusions));

  auto gemm_activation_fusions =
      FindGemmActivationFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindGemmActivationFusions", false,
                 std::move(gemm_activation_fusions));

  auto fused_gemm_fusions =
      FindFusedGemmFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindFusedGemmFusions", false,
                 std::move(fused_gemm_fusions));

  auto shape_reshape_fusions =
      FindShapeReshapeFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindShapeReshapeFusions", true,
                 std::move(shape_reshape_fusions));

  auto centered_reduce_fusions =
      FindCenteredReduceFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindCenteredReduceFusions", false,
                 std::move(centered_reduce_fusions));

  auto split_reduce_fusions =
      FindSplitReduceFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindSplitReduceFusions", false,
                 std::move(split_reduce_fusions));

  auto rms_norm_fusions =
      FindRmsNormFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindRmsNormFusions", false,
                 std::move(rms_norm_fusions));

  auto modulo_gather_fusions =
      FindModuloGatherFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindModuloGatherFusions", false,
                 std::move(modulo_gather_fusions));

  auto parallel_matmul_concat_fusions = FindParallelMatMulConcatFusions(
      all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindParallelMatMulConcatFusions", false,
                 std::move(parallel_matmul_concat_fusions));

  auto parallel_einsum_activation_fusions = FindParallelEinsumActivationFusions(
      all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindParallelEinsumActivationFusions", false,
                 std::move(parallel_einsum_activation_fusions));

  auto math_concat_log_fusions =
      FindMathConcatLogFusions(all_nodes, graph_output_names, fused_node_ids);
  AddFusionMatch(matches, "FindMathConcatLogFusions", false,
                 std::move(math_concat_log_fusions));

  return matches;
}

}  // namespace musa_ep
