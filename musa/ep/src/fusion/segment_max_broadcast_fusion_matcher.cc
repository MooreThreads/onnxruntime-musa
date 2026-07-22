#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
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

struct ExpectedNode {
  const char* name;
  const char* op_type;
};

constexpr ExpectedNode kSimSegmentMaxNodes[] = {
    {"Unique", "Unique"},
    {"Unique__10183_cast", "Cast"},
    {"Cast__11406", "Cast"},
    {"Shape__11411", "Shape"},
    {"UnsortedSegmentMax_TopK__11415", "TopK"},
    {"Unsqueeze__11434", "Unsqueeze"},
    {"UnsortedSegmentMax_Unique__11417", "Unique"},
    {"ReduceMax__11420", "ReduceMax"},
    {"Gather__11425", "Gather"},
    {"Squeeze__11414", "Squeeze"},
    {"Range__11424", "Range"},
    {"Mod__11428", "Mod"},
    {"Unsqueeze__11437", "Unsqueeze"},
    {"Concat__11439", "Concat"},
    {"Shape_1", "Shape"},
    {"Shape_1__10185", "Cast"},
    {"strided_slice_1", "Slice"},
    {"strided_slice_1__10189", "Squeeze"},
    {"Cast__11408", "Cast"},
    {"Unsqueeze__11410", "Unsqueeze"},
    {"Concat__11431", "Concat"},
    {"ConstantOfShape__11432", "ConstantOfShape"},
    {"ScatterND__11442", "ScatterND"},
    {"Gather__11458", "Gather"},
    {"UnsortedSegmentMax_ReduceMax__11463", "ReduceMax"},
    {"GatherV2_11", "Gather"},
};

bool HasExactInputs(
    const std::unordered_map<std::string, Ort::ConstNode>& by_name,
    const char* node_name, std::initializer_list<const char*> expected) {
  auto it = by_name.find(node_name);
  if (it == by_name.end()) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> inputs = it->second.GetInputs();
  if (inputs.size() != expected.size()) {
    return false;
  }
  size_t index = 0;
  for (const char* name : expected) {
    if (Name(inputs[index++]) != name) {
      return false;
    }
  }
  return true;
}

bool HasExpectedTopology(
    const std::unordered_map<std::string, Ort::ConstNode>& by_name) {
  return HasExactInputs(by_name, "Unique", {"Reshape:0"}) &&
         HasExactInputs(by_name, "Unique__10183_cast", {"idx__10180"}) &&
         HasExactInputs(by_name, "Cast__11406", {"Unique__10183_cast:0"}) &&
         HasExactInputs(by_name, "Shape__11411", {"Cast__11406:0"}) &&
         HasExactInputs(by_name, "UnsortedSegmentMax_TopK__11415",
                        {"Cast__11406:0", "Shape__11411:0"}) &&
         HasExactInputs(
             by_name, "Unsqueeze__11434",
             {"UnsortedSegmentMax_TopK__11415:0", "const_fold_opt__11684"}) &&
         HasExactInputs(by_name, "UnsortedSegmentMax_Unique__11417",
                        {"UnsortedSegmentMax_TopK__11415:0"}) &&
         HasExactInputs(
             by_name, "ReduceMax__11420",
             {"UnsortedSegmentMax_Unique__11417:3", "const_axes__11245"}) &&
         HasExactInputs(by_name, "Gather__11425",
                        {"UnsortedSegmentMax_Unique__11417:3",
                         "UnsortedSegmentMax_Unique__11417:2"}) &&
         HasExactInputs(by_name, "Squeeze__11414",
                        {"Shape__11411:0", "const_axes__11245"}) &&
         HasExactInputs(
             by_name, "Range__11424",
             {"VocabFileEmbeddingLookup/brow_300_time_list/GreaterEqual/y:0",
              "Squeeze__11414:0", "add_9/y:0"}) &&
         HasExactInputs(by_name, "Mod__11428",
                        {"Range__11424:0", "Gather__11425:0"}) &&
         HasExactInputs(by_name, "Unsqueeze__11437",
                        {"Mod__11428:0", "const_fold_opt__11684"}) &&
         HasExactInputs(by_name, "Concat__11439",
                        {"Unsqueeze__11434:0", "Unsqueeze__11437:0"}) &&
         HasExactInputs(by_name, "Shape_1", {"y__10178"}) &&
         HasExactInputs(by_name, "Shape_1__10185", {"Shape_1:0"}) &&
         HasExactInputs(by_name, "strided_slice_1",
                        {"Shape_1__10185:0", "const_axes__11245",
                         "axes_const__9894", "const_axes__11245"}) &&
         HasExactInputs(by_name, "strided_slice_1__10189",
                        {"strided_slice_1:0", "const_axes__11245"}) &&
         HasExactInputs(by_name, "Cast__11408", {"strided_slice_1__10189:0"}) &&
         HasExactInputs(by_name, "Unsqueeze__11410",
                        {"Cast__11408:0", "const_axes__11245"}) &&
         HasExactInputs(by_name, "Concat__11431",
                        {"Unsqueeze__11410:0", "ReduceMax__11420:0"}) &&
         HasExactInputs(by_name, "ConstantOfShape__11432",
                        {"Concat__11431:0"}) &&
         HasExactInputs(by_name, "ScatterND__11442",
                        {"ConstantOfShape__11432:0", "Concat__11439:0",
                         "UnsortedSegmentMax_TopK__11415:1"}) &&
         HasExactInputs(by_name, "Gather__11458",
                        {"Concat__11456:0", "ScatterND__11442:0"}) &&
         HasExactInputs(by_name, "UnsortedSegmentMax_ReduceMax__11463",
                        {"Gather__11458:0", "axes_const__9894"}) &&
         HasExactInputs(
             by_name, "GatherV2_11",
             {"UnsortedSegmentMax_ReduceMax__11463:0", "Unique__10183_cast:0"});
}

}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindSegmentMaxBroadcastFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids) {
  (void)graph_output_names;
  if (std::getenv("ORT_MUSA_DISABLE_SEGMENT_MAX_BROADCAST_FUSION") != nullptr) {
    return {};
  }
  std::unordered_map<std::string, Ort::ConstNode> by_name;
  for (Ort::ConstNode node : all_nodes) {
    by_name.emplace(node.GetName(), node);
  }
  if (!HasExpectedTopology(by_name)) {
    return {};
  }

  std::unordered_set<size_t> selected_ids;
  std::vector<Ort::ConstNode> fusion_nodes;
  fusion_nodes.reserve(std::size(kSimSegmentMaxNodes));
  for (const ExpectedNode& expected : kSimSegmentMaxNodes) {
    auto it = by_name.find(expected.name);
    if (it == by_name.end() || !IsOnnxOp(it->second, expected.op_type) ||
        !AddFusionNode(it->second, accepted_node_ids, selected_ids,
                       fusion_nodes)) {
      return {};
    }
  }

  std::sort(fusion_nodes.begin(), fusion_nodes.end(),
            [](Ort::ConstNode lhs, Ort::ConstNode rhs) {
              return lhs.GetId() < rhs.GetId();
            });
  return {std::move(fusion_nodes)};
}

}  // namespace musa_ep
