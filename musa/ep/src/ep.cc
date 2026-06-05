// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <array>
#include <cassert>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "ep_factory.h"
#include "ep_profiling.h"
#include "fusion/concat_matmul_fusion.h"
#include "fusion/fusion_node_compute.h"
#include "fusion/linear_fusion.h"
#include "plugin_ep_utils.h"

namespace {

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(const Ort::ConstNode& node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

bool IsFloatTensorValueInfo(Ort::ConstValueInfo value_info) {
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

std::optional<std::vector<int64_t>> GetStaticShape(
    Ort::ConstValueInfo value_info) {
  auto shape = GetTensorShape(value_info);
  if (!shape.has_value()) {
    return std::nullopt;
  }

  for (int64_t dim : *shape) {
    if (dim <= 0) {
      return std::nullopt;
    }
  }

  return shape;
}

std::optional<int64_t> GetIntAttribute(Ort::ConstNode node,
                                       const std::string& name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return std::nullopt;
  }

  int64_t value = 0;
  status = attr.GetValue(value);
  if (!status.IsOK()) {
    return std::nullopt;
  }

  return value;
}

bool NormalizeAxis(int64_t axis, size_t rank, int64_t& normalized_axis) {
  const int64_t signed_rank = static_cast<int64_t>(rank);
  if (rank == 0 || axis < -signed_rank || axis >= signed_rank) {
    return false;
  }

  normalized_axis = axis < 0 ? axis + signed_rank : axis;
  return true;
}

bool CanFuseConcatMatMul(Ort::ConstNode concat_node, Ort::ConstNode matmul_node,
                         int64_t concat_input_idx) {
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> matmul_inputs = matmul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
  if (concat_inputs.size() < 2 || concat_outputs.size() != 1 ||
      matmul_inputs.size() != 2 || matmul_outputs.size() != 1 ||
      (concat_input_idx != 0 && concat_input_idx != 1)) {
    return false;
  }

  const std::string concat_output_name = concat_outputs[0].GetName();
  if (matmul_inputs[static_cast<size_t>(concat_input_idx)].GetName() !=
      concat_output_name) {
    return false;
  }

  if (!IsFloatTensorValueInfo(matmul_outputs[0]) ||
      !IsFloatTensorValueInfo(
          matmul_inputs[static_cast<size_t>(1 - concat_input_idx)])) {
    return false;
  }

  for (Ort::ConstValueInfo input : concat_inputs) {
    if (!IsFloatTensorValueInfo(input)) {
      return false;
    }
  }

  auto axis_attr = GetIntAttribute(concat_node, "axis");
  if (!axis_attr.has_value()) {
    return false;
  }

  // Dynamic dimensions are common for model batch inputs, but rank information
  // is still enough to reject MatMul shapes this fusion cannot execute. The
  // unfused Concat + MatMul kernels can handle ONNX MatMul rank broadcasting
  // such as [B, M, K] x [K, N], while ConcatMatMul currently requires equal
  // ranks, so do not claim those nodes as a fused region.
  auto first_concat_type_shape = GetTensorShape(concat_inputs[0]);
  if (first_concat_type_shape.has_value()) {
    if (first_concat_type_shape->size() < 2) {
      return false;
    }

    int64_t axis = 0;
    if (!NormalizeAxis(*axis_attr, first_concat_type_shape->size(), axis)) {
      return false;
    }

    for (Ort::ConstValueInfo input : concat_inputs) {
      auto shape = GetTensorShape(input);
      if (shape.has_value() &&
          shape->size() != first_concat_type_shape->size()) {
        return false;
      }
    }

    auto other_type_shape =
        GetTensorShape(matmul_inputs[static_cast<size_t>(1 - concat_input_idx)]);
    if (other_type_shape.has_value() &&
        other_type_shape->size() != first_concat_type_shape->size()) {
      return false;
    }
  }

  // Plugin EP ValueInfo frequently has type but no static shape for internal
  // tensors. Defer the detailed shape validation to Compile/Compute when ORT
  // provides the runtime tensors, and only reject here if available static
  // shapes prove the pattern is invalid.
  auto first_concat_shape = GetStaticShape(concat_inputs[0]);
  if (!first_concat_shape.has_value()) {
    return true;
  }
  if (first_concat_shape->size() < 2) {
    return false;
  }

  int64_t axis = 0;
  if (!NormalizeAxis(*axis_attr, first_concat_shape->size(), axis)) {
    return false;
  }

  std::vector<int64_t> concat_shape = *first_concat_shape;
  concat_shape[static_cast<size_t>(axis)] = 0;
  bool all_concat_shapes_known = true;
  for (Ort::ConstValueInfo input : concat_inputs) {
    auto shape = GetStaticShape(input);
    if (!shape.has_value()) {
      all_concat_shapes_known = false;
      continue;
    }
    if (shape->size() != concat_shape.size()) {
      return false;
    }

    for (size_t dim = 0; dim < shape->size(); ++dim) {
      if (dim == static_cast<size_t>(axis)) {
        continue;
      }
      if ((*shape)[dim] != concat_shape[dim]) {
        return false;
      }
    }
    concat_shape[static_cast<size_t>(axis)] +=
        (*shape)[static_cast<size_t>(axis)];
  }

  auto other_shape =
      GetStaticShape(matmul_inputs[static_cast<size_t>(1 - concat_input_idx)]);
  if (!other_shape.has_value() || !all_concat_shapes_known) {
    return true;
  }
  if (other_shape->size() != concat_shape.size()) {
    return false;
  }

  const std::vector<int64_t>& lhs_shape =
      concat_input_idx == 0 ? concat_shape : *other_shape;
  const std::vector<int64_t>& rhs_shape =
      concat_input_idx == 0 ? *other_shape : concat_shape;
  const size_t rank = lhs_shape.size();
  for (size_t dim = 0; dim + 2 < rank; ++dim) {
    if (lhs_shape[dim] != rhs_shape[dim]) {
      return false;
    }
  }
  return lhs_shape[rank - 1] == rhs_shape[rank - 2];
}

std::vector<std::vector<Ort::ConstNode>> FindConcatMatMulFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode concat_node : all_nodes) {
    if (!IsOnnxOp(concat_node, "Concat") ||
        fused_node_ids.count(concat_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
    if (concat_outputs.size() != 1 ||
        graph_output_names.count(concat_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers =
        concat_outputs[0].GetConsumers();
    if (consumers.size() != 1 || consumers[0].index < 0 ||
        consumers[0].index > 1) {
      continue;
    }

    Ort::ConstNode matmul_node = consumers[0].node;
    if (!IsOnnxOp(matmul_node, "MatMul") ||
        fused_node_ids.count(matmul_node.GetId()) != 0) {
      continue;
    }

    if (!CanFuseConcatMatMul(concat_node, matmul_node, consumers[0].index)) {
      continue;
    }

    fusions.push_back({concat_node, matmul_node});
    fused_node_ids.insert(concat_node.GetId());
    fused_node_ids.insert(matmul_node.GetId());
  }

  return fusions;
}

bool IsLinearActivationNode(Ort::ConstNode node) {
  return IsOnnxOp(node, "Relu") || IsOnnxOp(node, "LeakyRelu") ||
         IsOnnxOp(node, "Tanh");
}

bool IsBiasShapeForMatMulN(const std::vector<int64_t>& bias_shape, int64_t n) {
  if (n <= 0) {
    return true;
  }
  return (bias_shape.size() == 1 && bias_shape[0] == n) ||
         (bias_shape.size() == 2 && bias_shape[0] == 1 && bias_shape[1] == n);
}

bool CanFuseMatMulAddActivation(Ort::ConstNode matmul_node,
                                Ort::ConstNode add_node,
                                Ort::ConstNode activation_node,
                                int64_t add_matmul_input_idx) {
  std::vector<Ort::ConstValueInfo> matmul_inputs = matmul_node.GetInputs();
  std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> activation_inputs =
      activation_node.GetInputs();
  std::vector<Ort::ConstValueInfo> activation_outputs =
      activation_node.GetOutputs();
  if (matmul_inputs.size() != 2 || matmul_outputs.size() != 1 ||
      add_inputs.size() != 2 || add_outputs.size() != 1 ||
      activation_inputs.size() != 1 || activation_outputs.size() != 1 ||
      (add_matmul_input_idx != 0 && add_matmul_input_idx != 1)) {
    return false;
  }

  if (Name(matmul_outputs[0]) !=
          Name(add_inputs[static_cast<size_t>(add_matmul_input_idx)]) ||
      Name(add_outputs[0]) != Name(activation_inputs[0])) {
    return false;
  }

  const size_t bias_idx = static_cast<size_t>(1 - add_matmul_input_idx);
  if (!IsFloatTensorValueInfo(matmul_inputs[0]) ||
      !IsFloatTensorValueInfo(matmul_inputs[1]) ||
      !IsFloatTensorValueInfo(add_inputs[bias_idx]) ||
      !IsFloatTensorValueInfo(activation_outputs[0])) {
    return false;
  }

  auto b_shape = GetStaticShape(matmul_inputs[1]);
  if (b_shape.has_value() && b_shape->size() != 2) {
    return false;
  }

  auto a_shape = GetStaticShape(matmul_inputs[0]);
  if (a_shape.has_value() && a_shape->size() < 2) {
    return false;
  }

  if (a_shape.has_value() && b_shape.has_value() &&
      a_shape->back() != (*b_shape)[0]) {
    return false;
  }

  auto bias_shape = GetStaticShape(add_inputs[bias_idx]);
  if (bias_shape.has_value() && b_shape.has_value() &&
      !IsBiasShapeForMatMulN(*bias_shape, (*b_shape)[1])) {
    return false;
  }

  return true;
}

bool CanFuseGemmActivation(Ort::ConstNode gemm_node,
                           Ort::ConstNode activation_node) {
  std::vector<Ort::ConstValueInfo> gemm_inputs = gemm_node.GetInputs();
  std::vector<Ort::ConstValueInfo> gemm_outputs = gemm_node.GetOutputs();
  std::vector<Ort::ConstValueInfo> activation_inputs =
      activation_node.GetInputs();
  std::vector<Ort::ConstValueInfo> activation_outputs =
      activation_node.GetOutputs();
  if ((gemm_inputs.size() != 2 && gemm_inputs.size() != 3) ||
      gemm_outputs.size() != 1 || activation_inputs.size() != 1 ||
      activation_outputs.size() != 1 ||
      Name(gemm_outputs[0]) != Name(activation_inputs[0])) {
    return false;
  }

  if (!IsFloatTensorValueInfo(gemm_inputs[0]) ||
      !IsFloatTensorValueInfo(gemm_inputs[1]) ||
      !IsFloatTensorValueInfo(activation_outputs[0])) {
    return false;
  }
  if (gemm_inputs.size() == 3 && !IsFloatTensorValueInfo(gemm_inputs[2])) {
    return false;
  }

  auto a_shape = GetStaticShape(gemm_inputs[0]);
  auto b_shape = GetStaticShape(gemm_inputs[1]);
  if (a_shape.has_value() && a_shape->size() != 2) {
    return false;
  }
  if (b_shape.has_value() && b_shape->size() != 2) {
    return false;
  }
  return true;
}

std::vector<std::vector<Ort::ConstNode>> FindGemmActivationFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode gemm_node : all_nodes) {
    if (!IsOnnxOp(gemm_node, "Gemm") ||
        fused_node_ids.count(gemm_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> gemm_outputs = gemm_node.GetOutputs();
    if (gemm_outputs.size() != 1 ||
        graph_output_names.count(gemm_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> gemm_consumers =
        gemm_outputs[0].GetConsumers();
    if (gemm_consumers.size() != 1 || gemm_consumers[0].index != 0) {
      continue;
    }

    Ort::ConstNode activation_node = gemm_consumers[0].node;
    if (!IsLinearActivationNode(activation_node) ||
        fused_node_ids.count(activation_node.GetId()) != 0) {
      continue;
    }

    if (!CanFuseGemmActivation(gemm_node, activation_node)) {
      continue;
    }

    fusions.push_back({gemm_node, activation_node});
    fused_node_ids.insert(gemm_node.GetId());
    fused_node_ids.insert(activation_node.GetId());
  }

  return fusions;
}

std::vector<std::vector<Ort::ConstNode>> FindFusedGemmFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    std::unordered_set<size_t>& fused_node_ids) {
  std::vector<std::vector<Ort::ConstNode>> fusions;

  for (Ort::ConstNode matmul_node : all_nodes) {
    if (!IsOnnxOp(matmul_node, "MatMul") ||
        fused_node_ids.count(matmul_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> matmul_outputs = matmul_node.GetOutputs();
    if (matmul_outputs.size() != 1 ||
        graph_output_names.count(matmul_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> matmul_consumers =
        matmul_outputs[0].GetConsumers();
    if (matmul_consumers.size() != 1 || matmul_consumers[0].index < 0 ||
        matmul_consumers[0].index > 1) {
      continue;
    }

    Ort::ConstNode add_node = matmul_consumers[0].node;
    if (!IsOnnxOp(add_node, "Add") ||
        fused_node_ids.count(add_node.GetId()) != 0) {
      continue;
    }

    std::vector<Ort::ConstValueInfo> add_outputs = add_node.GetOutputs();
    if (add_outputs.size() != 1 ||
        graph_output_names.count(add_outputs[0].GetName()) != 0) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> add_consumers =
        add_outputs[0].GetConsumers();
    if (add_consumers.size() != 1 || add_consumers[0].index != 0) {
      continue;
    }

    Ort::ConstNode activation_node = add_consumers[0].node;
    if (!IsLinearActivationNode(activation_node) ||
        fused_node_ids.count(activation_node.GetId()) != 0) {
      continue;
    }

    if (!CanFuseMatMulAddActivation(matmul_node, add_node, activation_node,
                                    matmul_consumers[0].index)) {
      continue;
    }

    fusions.push_back({matmul_node, add_node, activation_node});
    fused_node_ids.insert(matmul_node.GetId());
    fused_node_ids.insert(add_node.GetId());
    fused_node_ids.insert(activation_node.GetId());
  }

  return fusions;
}

}  // namespace

MusaEp::MusaEp(MusaEpFactory& factory, const Config& config,
               const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values
                // are default initialized
      factory_{factory},
      ort_api_{factory.GetOrtApi()},
      ep_api_{factory.GetEpApi()},
      name_{factory.GetEpName()},
      config_{config},
      logger_{logger} {
  ort_version_supported =
      ORT_API_VERSION;  // set to the ORT version we were compiled with.

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  CreateProfiler = CreateProfilerImpl;

  IGNORE_ORTSTATUS(ort_api_.Logger_LogMessage(
      &logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
      ("MUSAExecutionProvider has been created with name " + name_).c_str(),
      ORT_FILE, __LINE__, __FUNCTION__));
}

MusaEp::~MusaEp() = default;

/*static*/
const char* ORT_API_CALL MusaEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const MusaEp*>(this_ptr);
  return ep->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL
MusaEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                          OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    MusaEp* ep = static_cast<MusaEp*>(this_ptr);

    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> all_nodes = graph.GetNodes();

    if (all_nodes.empty()) {
      return nullptr;  // No nodes to process
    }

    std::unordered_set<std::string> graph_output_names;
    for (Ort::ConstValueInfo output : graph.GetOutputs()) {
      graph_output_names.insert(output.GetName());
    }

    std::unordered_set<size_t> fused_node_ids;
    std::vector<std::vector<Ort::ConstNode>> concat_matmul_fusions =
        FindConcatMatMulFusions(all_nodes, graph_output_names, fused_node_ids);

    for (const auto& fusion_nodes : concat_matmul_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    std::vector<std::vector<Ort::ConstNode>> gemm_activation_fusions =
        FindGemmActivationFusions(all_nodes, graph_output_names,
                                  fused_node_ids);
    std::vector<std::vector<Ort::ConstNode>> fused_gemm_fusions =
        FindFusedGemmFusions(all_nodes, graph_output_names, fused_node_ids);

    for (const auto& fusion_nodes : gemm_activation_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    for (const auto& fusion_nodes : fused_gemm_fusions) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(fusion_nodes.data()),
          fusion_nodes.size(), &node_fusion_options));
    }

    // Mark non-fused nodes as supported if we have a registered kernel.
    for (const auto& node : all_nodes) {
      if (fused_node_ids.count(node.GetId()) != 0) {
        continue;
      }

      const OrtKernelDef* kernel_def = nullptr;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_LookUpKernel(
          graph_support_info, node, &kernel_def));

      if (kernel_def != nullptr) {
        RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddSingleNode(
            graph_support_info, node));
      }
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL MusaEp::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr, _Outptr_result_maybenull_ const OrtKernelRegistry**
                              kernel_registry) noexcept {
  MusaEp* ep = static_cast<MusaEp*>(this_ptr);

  *kernel_registry = nullptr;

  // Get the cached kernel registry from parent factory to avoid recreating the
  // kernel registry for every EP instance.
  RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(*ep, kernel_registry));
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL MusaEp::CreateProfilerImpl(
    OrtEp* this_ptr, OrtEpProfilerImpl** profiler) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  MusaEp* ep = static_cast<MusaEp*>(this_ptr);
  auto profiler_unique_ptr = std::make_unique<MusaEpProfiler>(ep->ep_api_);

  *profiler = profiler_unique_ptr.release();
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}
