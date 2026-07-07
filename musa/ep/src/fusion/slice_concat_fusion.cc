// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/slice_concat_fusion.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "kernels/shared_inc/op_kernel_common.h"

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

bool IsRank2ColumnAxis(int64_t axis) {
  if (axis < 0) {
    axis += 2;
  }
  return axis == 1;
}

std::vector<int64_t> ReadIntInitializer(Ort::ConstValueInfo value_info) {
  if (!value_info.IsConstantInitializer()) {
    throw std::runtime_error("SliceConcat requires constant Slice parameters");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("SliceConcat failed to read Slice initializer");
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ReadTyped<int64_t>(value);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> values = ReadTyped<int32_t>(value);
    return std::vector<int64_t>(values.begin(), values.end());
  }
  throw std::runtime_error("SliceConcat requires int32/int64 Slice params");
}

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_input_indices;
  std::vector<Ort::ConstValueInfo> fused_inputs = fused_node.GetInputs();
  for (size_t i = 0; i < fused_inputs.size(); ++i) {
    fused_input_indices.emplace(Name(fused_inputs[i]), i);
  }
  return fused_input_indices;
}

size_t GetFusedInputIndex(
    const std::unordered_map<std::string, size_t>& fused_input_indices,
    const std::string& input_name) {
  auto it = fused_input_indices.find(input_name);
  if (it == fused_input_indices.end()) {
    throw std::runtime_error("unable to map SliceConcat input " + input_name);
  }
  return it->second;
}

void RequireFloatTensor(Ort::ConstValueInfo value_info,
                        const std::string& name) {
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    throw std::runtime_error("SliceConcat requires tensor input " + name);
  }
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("SliceConcat requires float input " + name);
  }
}

Ort::ConstNode FindConcatNode(Ort::ConstGraph graph) {
  Ort::ConstNode concat_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Concat")) {
      if (concat_node) {
        throw std::runtime_error("SliceConcat expects one Concat node");
      }
      concat_node = node;
    }
  }
  if (!concat_node) {
    throw std::runtime_error("SliceConcat expects a Concat node");
  }
  return concat_node;
}

std::vector<int64_t> KnownRankShape(Ort::ConstValueInfo value_info,
                                    const std::string& name) {
  auto shape = GetTensorShape(value_info);
  if (!shape.has_value()) {
    throw std::runtime_error("SliceConcat requires tensor shape for " + name);
  }
  return *shape;
}

bool IsPositivePowerOfTwo(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int32_t PowerOfTwoShift(int64_t value) {
  int32_t shift = 0;
  while (value > 1) {
    value >>= 1;
    ++shift;
  }
  return shift;
}

int64_t SliceConcatBlockCount(int64_t rows, int64_t width) {
  return (rows * width + kSliceConcatThreadsPerBlock - 1) /
         kSliceConcatThreadsPerBlock;
}

SliceConcatInput MakeInput(
    Ort::ConstNode slice_node,
    const std::unordered_map<std::string, size_t>& fused_input_indices,
    int64_t dst_offset) {
  std::vector<Ort::ConstValueInfo> slice_inputs = slice_node.GetInputs();
  if (slice_inputs.size() < 3 || slice_inputs.size() > 5) {
    throw std::runtime_error("SliceConcat requires Slice with 3-5 inputs");
  }

  RequireFloatTensor(slice_inputs[0], Name(slice_inputs[0]));
  auto input_shape = GetTensorShape(slice_inputs[0]);
  if (input_shape.has_value() && input_shape->size() != 2) {
    throw std::runtime_error("SliceConcat requires rank-2 inputs");
  }

  std::vector<int64_t> starts = ReadIntInitializer(slice_inputs[1]);
  std::vector<int64_t> ends = ReadIntInitializer(slice_inputs[2]);
  std::vector<int64_t> axes = {0, 1};
  std::vector<int64_t> steps(starts.size(), 1);
  if (slice_inputs.size() > 3 && slice_inputs[3]) {
    axes = ReadIntInitializer(slice_inputs[3]);
  }
  if (slice_inputs.size() > 4 && slice_inputs[4]) {
    steps = ReadIntInitializer(slice_inputs[4]);
  }
  if (starts.size() != ends.size() || starts.size() != axes.size() ||
      starts.size() != steps.size()) {
    throw std::runtime_error("SliceConcat Slice param size mismatch");
  }

  int64_t row_start = 0;
  int64_t row_end = std::numeric_limits<int64_t>::max();
  int64_t col_start = 0;
  int64_t col_end = 0;
  bool has_col_slice = false;
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i] < 0 ? axes[i] + 2 : axes[i];
    if (axis < 0 || axis > 1 || steps[i] != 1) {
      throw std::runtime_error(
          "SliceConcat only supports rank-2 unit-step Slice");
    }
    if (axis == 0) {
      row_start = starts[i];
      row_end = ends[i];
    } else {
      if ((starts[i] < 0 || ends[i] < 0) &&
          (!input_shape.has_value() || (*input_shape)[1] <= 0)) {
        throw std::runtime_error(
            "SliceConcat requires known cols for negative Slice bounds");
      }
      int64_t start = starts[i] < 0 ? starts[i] + (*input_shape)[1] : starts[i];
      int64_t end = ends[i] < 0 ? ends[i] + (*input_shape)[1] : ends[i];
      if (input_shape.has_value() && (*input_shape)[1] > 0) {
        const int64_t dim = (*input_shape)[1];
        start = std::max<int64_t>(0, std::min(start, dim));
        end = std::max<int64_t>(0, std::min(end, dim));
      }
      col_start = start;
      col_end = end;
      has_col_slice = true;
    }
  }
  if (!has_col_slice || row_start != 0 ||
      row_end < std::numeric_limits<int64_t>::max() / 4 ||
      col_end <= col_start) {
    throw std::runtime_error(
        "SliceConcat requires full-batch positive-width column Slice");
  }

  return SliceConcatInput{
      GetFusedInputIndex(fused_input_indices, Name(slice_inputs[0])),
      input_shape.has_value() ? (*input_shape)[1] : -1,
      col_start,
      col_end - col_start,
      dst_offset,
      false,
  };
}

bool IsZeroFloatConstantOfShape(Ort::ConstNode node) {
  if (!IsOnnxOp(node, "ConstantOfShape")) {
    return false;
  }

  Ort::ConstOpAttr attr;
  Ort::Status attr_status = node.GetAttributeByName("value", attr);
  if (!attr_status.IsOK()) {
    return true;
  }

  Ort::Value value{nullptr};
  Ort::Status status = attr.GetTensorAttributeAsOrtValue(value);
  if (!status.IsOK() || !value) {
    return false;
  }

  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      info.GetElementCount() != 1) {
    return false;
  }

  float scalar = 0.0f;
  std::memcpy(&scalar, value.GetTensorRawData(), sizeof(scalar));
  return scalar == 0.0f;
}

std::optional<std::vector<int64_t>> ConstantOfShapeOutputShape(
    Ort::ConstNode node, Ort::ConstValueInfo output_info) {
  auto output_shape = GetTensorShape(output_info);
  if (output_shape.has_value() && output_shape->size() == 2 &&
      (*output_shape)[1] > 0) {
    return output_shape;
  }

  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  if (inputs.size() != 1) {
    return std::nullopt;
  }

  if (inputs[0].IsConstantInitializer()) {
    try {
      std::vector<int64_t> shape = ReadIntInitializer(inputs[0]);
      if (shape.size() == 2 && shape[1] > 0) {
        return shape;
      }
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }

  Ort::ValueInfoConsumerProducerInfo producer = inputs[0].GetProducerNode();
  if (!producer.node || !IsOnnxOp(producer.node, "Shape")) {
    return std::nullopt;
  }

  std::vector<Ort::ConstValueInfo> shape_inputs = producer.node.GetInputs();
  if (shape_inputs.size() != 1) {
    return std::nullopt;
  }
  auto shape = GetTensorShape(shape_inputs[0]);
  if (!shape.has_value() || shape->size() != 2 || (*shape)[1] <= 0) {
    return std::nullopt;
  }
  return shape;
}

SliceConcatInput MakeZeroInput(Ort::ConstValueInfo value_info,
                               Ort::ConstNode constant_of_shape_node,
                               int64_t dst_offset) {
  auto shape = ConstantOfShapeOutputShape(constant_of_shape_node, value_info);
  if (!shape.has_value()) {
    throw std::runtime_error(
        "SliceConcat zero ConstantOfShape requires rank-2 output shape");
  }
  return SliceConcatInput{
      0, 0, 0, (*shape)[1], dst_offset, true,
  };
}

SliceConcatInput MakeDirectInput(
    Ort::ConstValueInfo value_info,
    const std::unordered_map<std::string, size_t>& fused_input_indices,
    int64_t dst_offset) {
  auto shape = GetTensorShape(value_info);
  if (!shape.has_value() || shape->size() != 2 || (*shape)[1] <= 0) {
    throw std::runtime_error(
        "SliceConcat direct input requires rank-2 output shape");
  }
  RequireFloatTensor(value_info, Name(value_info));
  return SliceConcatInput{
      GetFusedInputIndex(fused_input_indices, Name(value_info)),
      (*shape)[1],
      0,
      (*shape)[1],
      dst_offset,
      false,
  };
}

}  // namespace

struct SliceConcatScratch {
  ~SliceConcatScratch() {
    if (device_segments != nullptr || pinned_segments != nullptr) {
      (void)musaDeviceSynchronize();
    }
    if (device_segments != nullptr) {
      (void)musaFree(device_segments);
    }
    if (pinned_segments != nullptr) {
      (void)musaFreeHost(pinned_segments);
    }
  }

  MusaSliceConcatSegment* device_segments = nullptr;
  MusaSliceConcatSegment* pinned_segments = nullptr;
  size_t capacity = 0;
  std::vector<MusaSliceConcatSegment> host_segments;
  struct RuntimeInput {
    const float* data = nullptr;
    int64_t cols = 0;
    bool initialized = false;
  };
  std::vector<RuntimeInput> runtime_inputs;
};

SliceConcatFusionCompute::SliceConcatFusionCompute(
    int64_t output_cols, std::vector<SliceConcatInput> inputs)
    : output_cols(output_cols), inputs(std::move(inputs)) {
  for (const SliceConcatInput& input : this->inputs) {
    if (!input.zero_fill) {
      input_slot_count = std::max(input_slot_count, input.input_index + 1);
    }
  }

  if (this->inputs.empty()) {
    return;
  }

  const int64_t width = this->inputs[0].width;
  if (!IsPositivePowerOfTwo(width) ||
      output_cols != width * static_cast<int64_t>(this->inputs.size())) {
    return;
  }

  for (size_t i = 0; i < this->inputs.size(); ++i) {
    const SliceConcatInput& input = this->inputs[i];
    if (input.width != width ||
        input.dst_offset != width * static_cast<int64_t>(i)) {
      return;
    }
  }

  equal_width = width;
  equal_width_shift = PowerOfTwoShift(width);
}

SliceConcatFusionCompute::~SliceConcatFusionCompute() = default;

OrtStatus* SliceConcatFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    musaStream_t stream = GetComputeStream(ctx);
    if (inputs.empty()) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "SliceConcat requires inputs");
    }
    const SliceConcatInput* first_tensor_input = nullptr;
    for (const SliceConcatInput& input : inputs) {
      if (!input.zero_fill) {
        first_tensor_input = &input;
        break;
      }
    }
    if (first_tensor_input == nullptr) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "SliceConcat requires at least one Slice input");
    }

    std::vector<int64_t> first_shape =
        ctx.GetInput(first_tensor_input->input_index)
            .GetTensorTypeAndShapeInfo()
            .GetShape();
    if (first_shape.size() != 2) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "SliceConcat requires rank-2 inputs");
    }
    const int64_t rows = first_shape[0];
    Ort::UnownedValue y = ctx.GetOutput(0, {rows, output_cols});
    if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "SliceConcat requires MUSA output");
    }

    std::lock_guard<std::mutex> lock(scratch_mutex);
    if (!scratch) {
      scratch = std::make_unique<SliceConcatScratch>();
    }
    if (scratch->capacity < inputs.size()) {
      if (scratch->device_segments != nullptr) {
        musaError_t free_status = musaFree(scratch->device_segments);
        if (free_status != musaSuccess) {
          return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                            MusaErrorString(free_status));
        }
      }
      if (scratch->pinned_segments != nullptr) {
        (void)musaDeviceSynchronize();
        musaError_t free_host_status = musaFreeHost(scratch->pinned_segments);
        if (free_host_status != musaSuccess) {
          return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                            MusaErrorString(free_host_status));
        }
        scratch->pinned_segments = nullptr;
      }
      const size_t bytes = inputs.size() * sizeof(MusaSliceConcatSegment);
      musaError_t alloc_status = musaMalloc(
          reinterpret_cast<void**>(&scratch->device_segments), bytes);
      if (alloc_status != musaSuccess) {
        scratch->device_segments = nullptr;
        scratch->capacity = 0;
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                          MusaErrorString(alloc_status));
      }
      musaError_t host_alloc_status =
          musaHostAlloc(reinterpret_cast<void**>(&scratch->pinned_segments),
                        bytes, musaHostAllocDefault);
      if (host_alloc_status != musaSuccess) {
        (void)musaFree(scratch->device_segments);
        scratch->device_segments = nullptr;
        scratch->capacity = 0;
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                          MusaErrorString(host_alloc_status));
      }
      scratch->capacity = inputs.size();
    }

    scratch->host_segments.resize(inputs.size());
    scratch->runtime_inputs.assign(input_slot_count, {});
    int64_t total_blocks = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const SliceConcatInput& spec = inputs[i];
      const int64_t block_count = SliceConcatBlockCount(rows, spec.width);
      const int64_t block_offset = total_blocks;
      total_blocks += block_count;
      if (spec.zero_fill) {
        scratch->host_segments[i] = MusaSliceConcatSegment{
            nullptr,      0,           0, spec.width, spec.dst_offset,
            block_offset, block_count, 1};
        continue;
      }

      if (spec.input_index >= scratch->runtime_inputs.size()) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "SliceConcat input index out of range");
      }
      SliceConcatScratch::RuntimeInput& runtime_input =
          scratch->runtime_inputs[spec.input_index];
      if (!runtime_input.initialized) {
        Ort::ConstValue input = ctx.GetInput(spec.input_index);
        if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
          return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                            "SliceConcat requires MUSA inputs");
        }
        int64_t input_cols = spec.input_cols;
        if (input_cols <= 0) {
          auto info = input.GetTensorTypeAndShapeInfo();
          if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return Ort::GetApi().CreateStatus(
                ORT_NOT_IMPLEMENTED, "SliceConcat requires float inputs");
          }
          std::vector<int64_t> shape = info.GetShape();
          if (shape.size() != 2 || shape[0] != rows) {
            return Ort::GetApi().CreateStatus(
                ORT_INVALID_ARGUMENT,
                "SliceConcat runtime input rank mismatch");
          }
          input_cols = shape[1];
        }
        runtime_input.data = input.GetTensorData<float>();
        runtime_input.cols = input_cols;
        runtime_input.initialized = true;
      }
      if (spec.start_col < 0 || spec.width <= 0 ||
          spec.start_col + spec.width > runtime_input.cols) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "SliceConcat runtime input cols mismatch");
      }
      scratch->host_segments[i] = MusaSliceConcatSegment{
          runtime_input.data, runtime_input.cols, spec.start_col, spec.width,
          spec.dst_offset,    block_offset,       block_count,    0,
      };
    }

    const size_t bytes = inputs.size() * sizeof(MusaSliceConcatSegment);
    std::memcpy(scratch->pinned_segments, scratch->host_segments.data(), bytes);
    musaError_t copy_status =
        musaMemcpyAsync(scratch->device_segments, scratch->pinned_segments,
                        bytes, musaMemcpyHostToDevice, stream);
    if (copy_status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        MusaErrorString(copy_status));
    }
    if (equal_width > 0) {
      return LaunchStatus(LaunchMusaSliceConcatEqualWidthKernel(
          scratch->device_segments, static_cast<int64_t>(inputs.size()),
          y.GetTensorMutableData<float>(), rows, output_cols, equal_width,
          equal_width_shift, stream));
    }
    return LaunchStatus(LaunchMusaSliceConcatSegmentedKernel(
        scratch->device_segments, static_cast<int64_t>(inputs.size()),
        total_blocks, y.GetTensorMutableData<float>(), rows, output_cols,
        stream));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsSliceConcatFusionGraph(Ort::ConstGraph graph) {
  Ort::ConstNode concat_node{nullptr};
  bool has_slice = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Concat")) {
      if (concat_node) {
        return false;
      }
      concat_node = node;
    } else if (IsOnnxOp(node, "Slice")) {
      has_slice = true;
    } else if (IsOnnxOp(node, "ConstantOfShape") || IsOnnxOp(node, "Shape")) {
      continue;
    } else {
      return false;
    }
  }
  return concat_node && has_slice &&
         IsRank2ColumnAxis(ReadIntAttribute(concat_node, "axis", 0));
}

std::unique_ptr<FusionNodeCompute> CreateSliceConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode concat_node = FindConcatNode(graph);
  if (!IsRank2ColumnAxis(ReadIntAttribute(concat_node, "axis", 0))) {
    throw std::runtime_error("SliceConcat requires Concat axis=1");
  }

  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> concat_outputs = concat_node.GetOutputs();
  if (concat_inputs.empty() || concat_outputs.size() != 1) {
    throw std::runtime_error("invalid SliceConcat fused graph");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  std::unordered_map<std::string, Ort::ConstNode> graph_producers;
  for (Ort::ConstNode node : graph.GetNodes()) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      graph_producers.emplace(Name(output), node);
    }
  }

  std::vector<SliceConcatInput> inputs;
  inputs.reserve(concat_inputs.size());
  int64_t dst_offset = 0;
  for (Ort::ConstValueInfo concat_input : concat_inputs) {
    auto producer_it = graph_producers.find(Name(concat_input));
    if (producer_it == graph_producers.end()) {
      SliceConcatInput input =
          MakeDirectInput(concat_input, fused_input_indices, dst_offset);
      dst_offset += input.width;
      inputs.push_back(input);
      continue;
    }
    Ort::ConstNode slice_node = producer_it->second;
    if (IsZeroFloatConstantOfShape(slice_node)) {
      SliceConcatInput input =
          MakeZeroInput(concat_input, slice_node, dst_offset);
      dst_offset += input.width;
      inputs.push_back(input);
      continue;
    }
    if (!IsOnnxOp(slice_node, "Slice")) {
      SliceConcatInput input =
          MakeDirectInput(concat_input, fused_input_indices, dst_offset);
      dst_offset += input.width;
      inputs.push_back(input);
      continue;
    }
    SliceConcatInput input =
        MakeInput(slice_node, fused_input_indices, dst_offset);
    dst_offset += input.width;
    inputs.push_back(input);
  }

  return std::make_unique<SliceConcatFusionCompute>(dst_offset,
                                                    std::move(inputs));
}
