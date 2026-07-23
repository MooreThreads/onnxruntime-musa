#include "fusion/segment_max_broadcast_fusion.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/graph_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/segment_max_broadcast_impl.h"

namespace {

std::unordered_map<std::string, size_t> ValueIndices(
    const std::vector<Ort::ConstValueInfo>& values) {
  std::unordered_map<std::string, size_t> indices;
  for (size_t i = 0; i < values.size(); ++i) {
    indices.emplace(musa_ep::Name(values[i]), i);
  }
  return indices;
}

size_t MappedIndex(const std::unordered_map<std::string, size_t>& indices,
                   const char* name) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("SegmentMaxBroadcast missing value ") +
                             name);
  }
  return it->second;
}

struct SegmentMaxBroadcastCompute : FusionNodeCompute {
  SegmentMaxBroadcastCompute(size_t segment_ids_index, size_t values_index,
                             size_t output_index)
      : segment_ids_index(segment_ids_index),
        values_index(values_index),
        output_index(output_index) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue segment_ids = ctx.GetInput(segment_ids_index);
      Ort::ConstValue values = ctx.GetInput(values_index);
      auto ids_info = segment_ids.GetTensorTypeAndShapeInfo();
      auto values_info = values.GetTensorTypeAndShapeInfo();
      const int64_t count = ids_info.GetElementCount();
      if (!IsGpuMemory(segment_ids.GetTensorMemoryInfo()) ||
          !IsGpuMemory(values.GetTensorMemoryInfo()) ||
          ids_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
          values_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          values_info.GetElementCount() != count + 1) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "invalid SegmentMaxBroadcast inputs");
      }

      Ort::UnownedValue output =
          ctx.GetOutput(output_index, ids_info.GetShape());
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "SegmentMaxBroadcast requires MUSA output");
      }
      return LaunchStatus(LaunchMusaSegmentMaxBroadcastKernel(
          segment_ids.GetTensorData<int64_t>(), values.GetTensorData<float>(),
          output.GetTensorMutableData<float>(), count, GetComputeStream(ctx)));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t segment_ids_index;
  size_t values_index;
  size_t output_index;
};

}  // namespace

bool IsSegmentMaxBroadcastFusionGraph(Ort::ConstGraph graph) {
  const std::unordered_map<std::string, size_t> expected_counts = {
      {"Cast", 4},   {"Concat", 2},    {"ConstantOfShape", 1}, {"Gather", 3},
      {"Mod", 1},    {"Range", 1},     {"ReduceMax", 2},       {"ScatterND", 1},
      {"Shape", 2},  {"Slice", 1},     {"Squeeze", 2},         {"TopK", 1},
      {"Unique", 2}, {"Unsqueeze", 3},
  };
  std::unordered_map<std::string, size_t> actual_counts;
  size_t node_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    ++actual_counts[node.GetOperatorType()];
    ++node_count;
  }
  return node_count == 26 && actual_counts == expected_counts;
}

std::unique_ptr<FusionNodeCompute> CreateSegmentMaxBroadcastFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  (void)graph;
  auto inputs = ValueIndices(fused_node.GetInputs());
  auto outputs = ValueIndices(fused_node.GetOutputs());
  return std::make_unique<SegmentMaxBroadcastCompute>(
      MappedIndex(inputs, "Reshape:0"), MappedIndex(inputs, "Concat__11456:0"),
      MappedIndex(outputs, "GatherV2_11:0"));
}
