// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/slice_sum_concat_fusion.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/slice_sum_concat_impl.h"

namespace {

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(Ort::ConstNode node, const char* op_type) {
  return node && node.GetOperatorType() == op_type &&
         IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

int64_t ReadIntAttribute(Ort::ConstNode node, const std::string& name,
                         int64_t default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK()) {
    return default_value;
  }
  int64_t value = default_value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

std::vector<int64_t> ReadIntInitializer(Ort::ConstValueInfo value_info) {
  if (!value_info || !value_info.IsConstantInitializer()) {
    throw std::runtime_error("SliceSumConcat requires constant int initializer");
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("SliceSumConcat failed to read initializer");
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  const auto elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ReadTyped<int64_t>(value);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> values = ReadTyped<int32_t>(value);
    return std::vector<int64_t>(values.begin(), values.end());
  }
  throw std::runtime_error("SliceSumConcat requires int32/int64 initializer");
}

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> indices;
  std::vector<Ort::ConstValueInfo> inputs = fused_node.GetInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    indices.emplace(Name(inputs[i]), i);
  }
  return indices;
}

std::unordered_map<std::string, size_t> FusedOutputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> indices;
  std::vector<Ort::ConstValueInfo> outputs = fused_node.GetOutputs();
  for (size_t i = 0; i < outputs.size(); ++i) {
    indices.emplace(Name(outputs[i]), i);
  }
  return indices;
}

size_t GetMappedIndex(const std::unordered_map<std::string, size_t>& indices,
                      const std::string& name, const char* kind) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("unable to map SliceSumConcat ") +
                             kind + " " + name);
  }
  return it->second;
}

struct SliceSpec {
  size_t input_index;
  int64_t start_col;
};

struct SegmentPlan {
  bool direct;
  size_t direct_input_index;
  int64_t width;
  std::vector<SliceSpec> slices;
};

struct ParsedSlice {
  Ort::ConstValueInfo input;
  int64_t start_col;
  int64_t width;
};

ParsedSlice ParseRank2ColumnSlice(Ort::ConstNode slice_node) {
  std::vector<Ort::ConstValueInfo> inputs = slice_node.GetInputs();
  if (inputs.size() < 3 || inputs.size() > 5) {
    throw std::runtime_error("SliceSumConcat requires Slice with 3-5 inputs");
  }
  std::vector<int64_t> starts = ReadIntInitializer(inputs[1]);
  std::vector<int64_t> ends = ReadIntInitializer(inputs[2]);
  if (starts.size() != ends.size()) {
    throw std::runtime_error("SliceSumConcat Slice starts/ends mismatch");
  }
  std::vector<int64_t> axes(starts.size());
  std::iota(axes.begin(), axes.end(), 0);
  if (inputs.size() > 3 && inputs[3]) {
    axes = ReadIntInitializer(inputs[3]);
  }
  std::vector<int64_t> steps(starts.size(), 1);
  if (inputs.size() > 4 && inputs[4]) {
    steps = ReadIntInitializer(inputs[4]);
  }
  if (axes.size() != starts.size() || steps.size() != starts.size()) {
    throw std::runtime_error("SliceSumConcat Slice axes/steps mismatch");
  }

  int64_t start_col = -1;
  int64_t end_col = -1;
  bool saw_col = false;
  for (size_t i = 0; i < axes.size(); ++i) {
    const int64_t axis = axes[i] < 0 ? axes[i] + 2 : axes[i];
    if (axis < 0 || axis > 1 || steps[i] != 1) {
      throw std::runtime_error("SliceSumConcat requires rank-2 step=1 Slice");
    }
    if (axis == 0) {
      if (starts[i] != 0 ||
          ends[i] < std::numeric_limits<int64_t>::max() / 4) {
        throw std::runtime_error("SliceSumConcat requires full row Slice");
      }
      continue;
    }
    saw_col = true;
    start_col = starts[i];
    end_col = ends[i];
  }
  if (!saw_col || start_col < 0 || end_col <= start_col) {
    throw std::runtime_error("SliceSumConcat requires positive column Slice");
  }
  return ParsedSlice{inputs[0], start_col, end_col - start_col};
}

Ort::ConstNode FindConcatNode(Ort::ConstGraph graph) {
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Concat")) {
      return node;
    }
  }
  return Ort::ConstNode{nullptr};
}

std::unordered_map<std::string, Ort::ConstNode> ProducersInGraph(
    Ort::ConstGraph graph) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : graph.GetNodes()) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      producers.emplace(Name(output), node);
    }
  }
  return producers;
}

Ort::ConstNode ProducerInGraph(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info) {
  auto it = producers.find(Name(value_info));
  return it == producers.end() ? Ort::ConstNode{nullptr} : it->second;
}

}  // namespace

struct SliceSumConcatFusionCompute : FusionNodeCompute {
  SliceSumConcatFusionCompute(size_t output_index,
                              std::vector<SegmentPlan> segments)
      : output_index(output_index), segments(std::move(segments)) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      if (segments.size() > kMusaSliceSumConcatMaxSegments) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "SliceSumConcat segment count exceeds limit");
      }

      MusaSliceSumConcatParams params{};
      params.segment_count = static_cast<int32_t>(segments.size());
      int64_t batch = -1;
      int64_t output_cols = 0;
      int32_t slice_count = 0;

      auto check_input = [&](Ort::ConstValue input,
                             std::vector<int64_t>& shape) -> OrtStatus* {
        auto info = input.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            !IsGpuMemory(input.GetTensorMemoryInfo())) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED,
              "SliceSumConcat requires MUSA float inputs");
        }
        shape = info.GetShape();
        if (shape.size() != 2 || shape[0] < 0 || shape[1] <= 0) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "SliceSumConcat requires rank-2 inputs");
        }
        if (batch < 0) {
          batch = shape[0];
        } else if (shape[0] != batch) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT, "SliceSumConcat batch mismatch");
        }
        return nullptr;
      };

      for (size_t i = 0; i < segments.size(); ++i) {
        const SegmentPlan& plan = segments[i];
        MusaSliceSumConcatSegment& segment = params.segments[i];
        segment.dst_col = output_cols;
        segment.mode = plan.direct
                           ? static_cast<int32_t>(
                                 MusaSliceSumConcatSegmentMode::Direct)
                           : static_cast<int32_t>(
                                 MusaSliceSumConcatSegmentMode::SumSlices);

        if (plan.direct) {
          Ort::ConstValue input = ctx.GetInput(plan.direct_input_index);
          std::vector<int64_t> shape;
          if (OrtStatus* status = check_input(input, shape)) {
            return status;
          }
          segment.width = shape[1];
          segment.direct_input = input.GetTensorData<float>();
          segment.direct_input_cols = shape[1];
        } else {
          if (plan.slices.empty() ||
              slice_count + static_cast<int32_t>(plan.slices.size()) >
                  kMusaSliceSumConcatMaxSlices) {
            return Ort::GetApi().CreateStatus(
                ORT_NOT_IMPLEMENTED, "SliceSumConcat slice count exceeds limit");
          }
          segment.width = plan.width;
          segment.slice_start = slice_count;
          segment.slice_count = static_cast<int32_t>(plan.slices.size());
          for (const SliceSpec& spec : plan.slices) {
            Ort::ConstValue input = ctx.GetInput(spec.input_index);
            std::vector<int64_t> shape;
            if (OrtStatus* status = check_input(input, shape)) {
              return status;
            }
            if (spec.start_col < 0 || spec.start_col + plan.width > shape[1]) {
              return Ort::GetApi().CreateStatus(
                  ORT_INVALID_ARGUMENT, "SliceSumConcat Slice range invalid");
            }
            params.slices[slice_count++] = MusaSliceSumConcatSlice{
                input.GetTensorData<float>(), shape[1], spec.start_col};
          }
        }
        output_cols += segment.width;
      }

      params.batch = batch;
      params.output_cols = output_cols;
      params.slice_count = slice_count;
      Ort::UnownedValue output = ctx.GetOutput(output_index, {batch, output_cols});
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "SliceSumConcat requires MUSA output");
      }
      return LaunchStatus(LaunchMusaSliceSumConcatFloat(
          params, output.GetTensorMutableData<float>(), nullptr));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t output_index;
  std::vector<SegmentPlan> segments;
};

bool IsSliceSumConcatFusionGraph(Ort::ConstGraph graph) {
  int concat_count = 0;
  int sum_count = 0;
  int slice_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Concat")) {
      ++concat_count;
    } else if (IsOnnxOp(node, "Sum")) {
      ++sum_count;
    } else if (IsOnnxOp(node, "Slice")) {
      ++slice_count;
    } else {
      return false;
    }
  }
  return concat_count == 1 && sum_count >= 1 && slice_count >= 2;
}

std::unique_ptr<FusionNodeCompute> CreateSliceSumConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode concat_node = FindConcatNode(graph);
  if (!concat_node || ReadIntAttribute(concat_node, "axis", 0) != 1) {
    throw std::runtime_error("SliceSumConcat requires Concat axis=1");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);
  auto producers = ProducersInGraph(graph);
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_outputs.size() != 1) {
    throw std::runtime_error("SliceSumConcat requires one Concat output");
  }

  std::vector<SegmentPlan> segments;
  segments.reserve(concat_inputs.size());
  for (Ort::ConstValueInfo concat_input : concat_inputs) {
    Ort::ConstNode producer = ProducerInGraph(producers, concat_input);
    if (!IsOnnxOp(producer, "Sum")) {
      segments.push_back(SegmentPlan{
          true, GetMappedIndex(fused_input_indices, Name(concat_input), "input"),
          0, {}});
      continue;
    }

    std::vector<Ort::ConstValueInfo> sum_inputs = producer.GetInputs();
    if (sum_inputs.size() < 2) {
      throw std::runtime_error("SliceSumConcat requires variadic Sum");
    }
    SegmentPlan segment{};
    segment.direct = false;
    segment.width = -1;
    segment.slices.reserve(sum_inputs.size());
    for (Ort::ConstValueInfo sum_input : sum_inputs) {
      Ort::ConstNode slice_node = ProducerInGraph(producers, sum_input);
      if (!IsOnnxOp(slice_node, "Slice")) {
        throw std::runtime_error(
            "SliceSumConcat requires Sum inputs from Slice");
      }
      ParsedSlice parsed = ParseRank2ColumnSlice(slice_node);
      if (segment.width < 0) {
        segment.width = parsed.width;
      } else if (segment.width != parsed.width) {
        throw std::runtime_error("SliceSumConcat Sum slice width mismatch");
      }
      segment.slices.push_back(SliceSpec{
          GetMappedIndex(fused_input_indices, Name(parsed.input), "input"),
          parsed.start_col});
    }
    segments.push_back(std::move(segment));
  }

  return std::make_unique<SliceSumConcatFusionCompute>(
      GetMappedIndex(fused_output_indices, Name(concat_outputs[0]), "output"),
      std::move(segments));
}
