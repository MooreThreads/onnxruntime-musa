#include "ep.h"

#include <optional>
#include <vector>

#include "ep_factory.h"
#include "kernel_utils.h"

MusaEp::MusaEp(MusaEpFactory& factory, const Config& config,
               const OrtLogger& logger)
    : OrtEp{},
      factory_(factory),
      ort_api_(factory.GetOrtApi()),
      ep_api_(factory.GetEpApi()),
      name_(factory.GetEpName()),
      config_(config),
      logger_(logger) {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  Compile = nullptr;
  ReleaseNodeComputeInfos = nullptr;

  IGNORE_ORT_STATUS(
      ort_api_.Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_INFO,
                                 "MusaExecutionProvider plugin EP created",
                                 ORT_MUSA_FILE, __LINE__, __FUNCTION__));
}

MusaEp::~MusaEp() = default;

const char* ORT_API_CALL MusaEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const MusaEp*>(this_ptr);
  return ep->name_.c_str();
}

OrtStatus* ORT_API_CALL
MusaEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                          OrtEpGraphSupportInfo* graph_support_info) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  auto* ep = static_cast<MusaEp*>(this_ptr);
  Ort::ConstGraph ort_graph{graph};
  for (const Ort::ConstNode& node : ort_graph.GetNodes()) {
    const std::string op_type = node.GetOperatorType();
    bool candidate = false;

    if (op_type == "Add") {
      const auto inputs = node.GetInputs();
      if (inputs.size() == 2) {
        const auto lhs =
            inputs[0].TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
        const auto rhs =
            inputs[1].TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> output_shape;
        candidate = BroadcastShapes(lhs, rhs, output_shape);
      }
    } else if (op_type == "MatMul") {
      const auto inputs = node.GetInputs();
      if (inputs.size() == 2) {
        const auto lhs =
            inputs[0].TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
        const auto rhs =
            inputs[1].TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
        candidate = lhs.size() == 2 && rhs.size() == 2 && HasStaticShape(lhs) &&
                    HasStaticShape(rhs) && lhs[1] == rhs[0];
      }
    }

    if (!candidate) {
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
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL MusaEp::GetKernelRegistryImpl(
    OrtEp* this_ptr, const OrtKernelRegistry** kernel_registry) noexcept {
  auto* ep = static_cast<MusaEp*>(this_ptr);
  *kernel_registry = nullptr;
  RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(*ep, kernel_registry));
  return nullptr;
}
