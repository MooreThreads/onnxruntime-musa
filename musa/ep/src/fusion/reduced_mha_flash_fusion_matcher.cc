#include <unordered_set>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"

namespace musa_ep {
namespace {

float ReadFloatAttribute(Ort::ConstNode node, const char* name,
                         float default_value) {
  Ort::ConstOpAttr attribute;
  float value = default_value;
  return node.GetAttributeByName(name, attribute).IsOK() &&
                 attribute.GetValue(value).IsOK()
             ? value
             : default_value;
}

bool CanFuse(Ort::ConstNode gemm,
             const std::unordered_set<std::string>& graph_outputs,
             const std::unordered_set<size_t>& accepted,
             std::vector<Ort::ConstNode>& nodes) {
  if (!IsOnnxOp(gemm, "Gemm") || accepted.count(gemm.GetId()) != 0 ||
      GetIntAttribute(gemm, "transA").value_or(0) != 0 ||
      GetIntAttribute(gemm, "transB").value_or(0) != 1 ||
      ReadFloatAttribute(gemm, "alpha", 1.0f) != 1.0f ||
      ReadFloatAttribute(gemm, "beta", 1.0f) != 1.0f)
    return false;
  auto gi = gemm.GetInputs();
  auto go = gemm.GetOutputs();
  if (gi.size() != 3 || go.size() != 1 || !IsFloatTensorValueInfo(gi[0]) ||
      !IsFloatTensorValueInfo(gi[1]) || !IsFloatTensorValueInfo(gi[2]) ||
      !gi[1].IsConstantInitializer() || !gi[2].IsConstantInitializer())
    return false;
  Ort::ConstNode reshape;
  if (!GetProducer(gi[0], reshape) || !IsOnnxOp(reshape, "Reshape"))
    return false;
  auto ri = reshape.GetInputs();
  auto ro = reshape.GetOutputs();
  if (ri.size() != 2 || ro.size() != 1 ||
      !HasSingleConsumerAt(ro[0], gemm, 0, graph_outputs))
    return false;
  Ort::ConstNode attention;
  if (!GetProducer(ri[0], attention) ||
      attention.GetOperatorType() != "Attention" ||
      attention.GetDomain() != "com.microsoft")
    return false;
  auto ai = attention.GetInputs();
  auto ao = attention.GetOutputs();
  if ((ai.size() != 3 && ai.size() != 4) || ao.size() != 1 ||
      !HasSingleConsumerAt(ao[0], reshape, 0, graph_outputs) ||
      !IsFloatTensorValueInfo(ai[0]) || !IsFloatTensorValueInfo(ai[1]) ||
      !IsFloatTensorValueInfo(ai[2]) || !ai[1].IsConstantInitializer() ||
      !ai[2].IsConstantInitializer())
    return false;
  if (GetIntAttribute(attention, "unidirectional").value_or(0) != 0)
    return false;
  if (ai.size() == 4 && !IsIntTensorValueInfo(ai[3])) return false;
  auto qkv = GetIntsAttribute(attention, "qkv_hidden_sizes");
  if (!qkv.has_value() || qkv->size() != 3 || (*qkv)[0] != (*qkv)[1] ||
      (*qkv)[1] != (*qkv)[2] ||
      GetIntAttribute(attention, "num_heads").value_or(0) <= 0)
    return false;
  Ort::ConstNode unsqueeze;
  if (!GetProducer(ai[0], unsqueeze) || !IsOnnxOp(unsqueeze, "Unsqueeze"))
    return false;
  auto ui = unsqueeze.GetInputs();
  auto uo = unsqueeze.GetOutputs();
  auto axes = ReadUnsqueezeAxes(unsqueeze);
  if (ui.size() != 2 || uo.size() != 1 || !axes.has_value() ||
      axes->size() != 1 || (*axes)[0] != 0 ||
      !HasSingleConsumerAt(uo[0], attention, 0, graph_outputs))
    return false;
  std::unordered_set<size_t> selected;
  for (Ort::ConstNode n : {unsqueeze, attention, reshape, gemm})
    if (!AddFusionNode(n, accepted, selected, nodes)) return false;
  return FusionHasNoExternalPathBetweenSelectedNodes(nodes, selected);
}
}  // namespace

std::vector<std::vector<Ort::ConstNode>> FindReducedMhaFlashFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_outputs,
    const std::unordered_set<size_t>& accepted) {
  std::vector<std::vector<Ort::ConstNode>> result;
  for (Ort::ConstNode n : all_nodes) {
    std::vector<Ort::ConstNode> nodes;
    if (CanFuse(n, graph_outputs, accepted, nodes))
      result.push_back(std::move(nodes));
  }
  return result;
}
}  // namespace musa_ep
