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
                   const std::string& name, const char* role) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("SegmentMaxBroadcast missing ") +
                             role);
  }
  return it->second;
}

std::unordered_map<std::string, Ort::ConstNode> ProducersInGraph(
    Ort::ConstGraph graph) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : graph.GetNodes()) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (output != nullptr) {
        producers.emplace(musa_ep::Name(output), node);
      }
    }
  }
  return producers;
}

Ort::ConstNode ProducerInGraph(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info, const char* op_type) {
  auto it = producers.find(musa_ep::Name(value_info));
  if (it == producers.end() || !musa_ep::IsOnnxOp(it->second, op_type)) {
    return Ort::ConstNode{nullptr};
  }
  return it->second;
}

bool IsProducedBy(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info, Ort::ConstNode producer,
    int64_t output_index) {
  auto it = producers.find(musa_ep::Name(value_info));
  if (it == producers.end() || it->second.GetId() != producer.GetId()) {
    return false;
  }
  std::vector<Ort::ConstValueInfo> outputs = producer.GetOutputs();
  return output_index >= 0 &&
         static_cast<size_t>(output_index) < outputs.size() &&
         musa_ep::Name(outputs[output_index]) == musa_ep::Name(value_info);
}

struct SegmentMaxBroadcastValues {
  Ort::ConstValueInfo segment_ids{nullptr};
  Ort::ConstValueInfo values{nullptr};
  Ort::ConstValueInfo output{nullptr};
};

bool ResolveSegmentMaxBroadcastValues(Ort::ConstGraph graph,
                                      SegmentMaxBroadcastValues& resolved) {
  auto producers = ProducersInGraph(graph);
  for (Ort::ConstNode final_gather : graph.GetNodes()) {
    if (!musa_ep::IsOnnxOp(final_gather, "Gather")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> final_inputs = final_gather.GetInputs();
    std::vector<Ort::ConstValueInfo> final_outputs = final_gather.GetOutputs();
    if (final_inputs.size() != 2 || final_outputs.size() != 1) {
      continue;
    }

    Ort::ConstNode final_reduce =
        ProducerInGraph(producers, final_inputs[0], "ReduceMax");
    Ort::ConstNode index_cast =
        ProducerInGraph(producers, final_inputs[1], "Cast");
    if (!final_reduce || !index_cast) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> reduce_inputs = final_reduce.GetInputs();
    std::vector<Ort::ConstValueInfo> cast_inputs = index_cast.GetInputs();
    if (reduce_inputs.size() != 2 || cast_inputs.size() != 1) {
      continue;
    }

    Ort::ConstNode value_gather =
        ProducerInGraph(producers, reduce_inputs[0], "Gather");
    Ort::ConstNode first_unique =
        ProducerInGraph(producers, cast_inputs[0], "Unique");
    if (!value_gather || !first_unique ||
        !IsProducedBy(producers, cast_inputs[0], first_unique, 2)) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> value_gather_inputs =
        value_gather.GetInputs();
    std::vector<Ort::ConstValueInfo> unique_inputs = first_unique.GetInputs();
    if (value_gather_inputs.size() != 2 || unique_inputs.size() != 1 ||
        !ProducerInGraph(producers, value_gather_inputs[1], "ScatterND")) {
      continue;
    }

    resolved = {unique_inputs[0], value_gather_inputs[0], final_outputs[0]};
    return true;
  }
  return false;
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
  SegmentMaxBroadcastValues resolved;
  return node_count == 26 && actual_counts == expected_counts &&
         ResolveSegmentMaxBroadcastValues(graph, resolved);
}

std::unique_ptr<FusionNodeCompute> CreateSegmentMaxBroadcastFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  SegmentMaxBroadcastValues resolved;
  if (!ResolveSegmentMaxBroadcastValues(graph, resolved)) {
    throw std::runtime_error("unable to resolve SegmentMaxBroadcast values");
  }
  auto inputs = ValueIndices(fused_node.GetInputs());
  auto outputs = ValueIndices(fused_node.GetOutputs());
  return std::make_unique<SegmentMaxBroadcastCompute>(
      MappedIndex(inputs, musa_ep::Name(resolved.segment_ids),
                  "segment-id input"),
      MappedIndex(inputs, musa_ep::Name(resolved.values), "value input"),
      MappedIndex(outputs, musa_ep::Name(resolved.output), "output"));
}
