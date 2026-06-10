// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion/concat_split_fusion.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "kernels/shared_inc/op_kernel_common.h"

namespace {

constexpr int64_t kConcatSplitCopyBlockSize = 256;

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

std::vector<int64_t> ReadIntInitializer(Ort::ConstValueInfo value_info) {
  if (!value_info.IsConstantInitializer()) {
    throw std::runtime_error("ConcatSplit requires constant Split sizes");
  }

  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("ConcatSplit failed to read Split initializer");
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
  throw std::runtime_error("ConcatSplit requires int32/int64 Split sizes");
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
    throw std::runtime_error("unable to map ConcatSplit input " + input_name);
  }
  return it->second;
}

Ort::ConstNode FindSingleNode(Ort::ConstGraph graph, const char* op_type) {
  Ort::ConstNode found{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, op_type)) {
      continue;
    }
    if (found) {
      throw std::runtime_error(std::string("ConcatSplit expects one ") +
                               op_type + " node");
    }
    found = node;
  }
  if (!found) {
    throw std::runtime_error(std::string("ConcatSplit expects a ") + op_type +
                             " node");
  }
  return found;
}

}  // namespace

struct ConcatSplitScratch {
  ~ConcatSplitScratch() {
    if (device_segments != nullptr) {
      (void)musaDeviceSynchronize();
      (void)musaFree(device_segments);
    }
    if (device_copy_blocks != nullptr) {
      (void)musaDeviceSynchronize();
      (void)musaFree(device_copy_blocks);
    }
    if (device_sum_outputs != nullptr) {
      (void)musaDeviceSynchronize();
      (void)musaFree(device_sum_outputs);
    }
    if (device_sum_terms != nullptr) {
      (void)musaDeviceSynchronize();
      (void)musaFree(device_sum_terms);
    }
  }

  MusaConcatSplitSegment* device_segments = nullptr;
  MusaConcatSplitCopyBlock* device_copy_blocks = nullptr;
  size_t capacity = 0;
  size_t copy_block_capacity = 0;
  std::vector<MusaConcatSplitSegment> host_segments;
  std::vector<MusaConcatSplitCopyBlock> host_copy_blocks;
  MusaConcatSplitSumOutput* device_sum_outputs = nullptr;
  MusaConcatSplitSumTerm* device_sum_terms = nullptr;
  size_t sum_output_capacity = 0;
  size_t sum_term_capacity = 0;
  std::vector<MusaConcatSplitSumOutput> host_sum_outputs;
  std::vector<MusaConcatSplitSumTerm> host_sum_terms;
};

ConcatSplitFusionCompute::ConcatSplitFusionCompute(
    std::vector<ConcatSplitOutput> outputs,
    std::vector<ConcatSplitSegmentSpec> segments,
    std::vector<ConcatSplitSumSpec> sums)
    : outputs(std::move(outputs)),
      segments(std::move(segments)),
      sums(std::move(sums)) {}

ConcatSplitFusionCompute::~ConcatSplitFusionCompute() = default;

OrtStatus* ConcatSplitFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    musaStream_t stream = GetComputeStream(ctx);
    if (outputs.empty() || (segments.empty() && sums.empty())) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "ConcatSplit requires outputs");
    }

    std::lock_guard<std::mutex> lock(scratch_mutex);
    if (!scratch) {
      scratch = std::make_unique<ConcatSplitScratch>();
    }

    const size_t input_count = ctx.GetInputCount();
    std::vector<const void*> input_data(input_count, nullptr);
    std::vector<int64_t> input_cols(input_count, 0);
    std::vector<uint8_t> input_seen(input_count, 0);

    int64_t rows = -1;
    int32_t element_size = 0;
    auto load_input = [&](size_t source_input_index) -> OrtStatus* {
      if (source_input_index >= input_count) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ConcatSplit source input index out of range");
      }
      if (input_seen[source_input_index]) {
        return nullptr;
      }

      Ort::ConstValue input = ctx.GetInput(source_input_index);
      if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "ConcatSplit requires MUSA inputs");
      }
      auto info = input.GetTensorTypeAndShapeInfo();
      const size_t size = ElementSize(info.GetElementType());
      if (size == 0) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "ConcatSplit unsupported dtype");
      }
      if (element_size == 0) {
        element_size = static_cast<int32_t>(size);
      } else if (element_size != static_cast<int32_t>(size)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "ConcatSplit requires matching dtypes");
      }
      std::vector<int64_t> shape = info.GetShape();
      if (shape.size() != 2 || shape[1] <= 0) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ConcatSplit requires rank-2 inputs");
      }
      if (rows < 0) {
        rows = shape[0];
      } else if (rows != shape[0]) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ConcatSplit input batch mismatch");
      }
      input_data[source_input_index] = input.GetTensorRawData();
      input_cols[source_input_index] = shape[1];
      input_seen[source_input_index] = 1;
      return nullptr;
    };

    for (const ConcatSplitSegmentSpec& spec : segments) {
      RETURN_IF_ERROR(load_input(spec.source_input_index));
      if (spec.source_offset < 0 || spec.width <= 0 ||
          spec.source_offset + spec.width > input_cols[spec.source_input_index]) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ConcatSplit output spec exceeds input");
      }
      if (spec.output_index >= outputs.size()) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ConcatSplit output index out of range");
      }
    }

    int64_t max_sum_width = 0;
    size_t sum_term_count = 0;
    for (const ConcatSplitSumSpec& sum : sums) {
      if (sum.output_index >= outputs.size() || sum.width <= 0 ||
          sum.terms.empty()) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ConcatSplit Sum output spec is invalid");
      }
      for (const ConcatSplitSumTermSpec& term : sum.terms) {
        RETURN_IF_ERROR(load_input(term.source_input_index));
        if (term.source_offset < 0 ||
            term.source_offset + sum.width > input_cols[term.source_input_index]) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT, "ConcatSplit Sum term exceeds input");
        }
      }
      max_sum_width = std::max(max_sum_width, sum.width);
      sum_term_count += sum.terms.size();
    }

    if (rows < 0) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "ConcatSplit requires input tensors");
    }
    if (!sums.empty() && element_size != static_cast<int32_t>(sizeof(float))) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "ConcatSplit Sum only supports float tensors");
    }

    std::vector<void*> output_data(outputs.size(), nullptr);
    for (size_t i = 0; i < outputs.size(); ++i) {
      Ort::UnownedValue output = ctx.GetOutput(i, {rows, outputs[i].width});
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                          "ConcatSplit requires MUSA outputs");
      }
      output_data[i] = output.GetTensorMutableRawData();
    }

    if (!segments.empty()) {
      if (scratch->capacity < segments.size()) {
        if (scratch->device_segments != nullptr) {
          musaError_t free_status = musaFree(scratch->device_segments);
          if (free_status != musaSuccess) {
            return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                              MusaErrorString(free_status));
          }
        }
        const size_t bytes = segments.size() * sizeof(MusaConcatSplitSegment);
        musaError_t alloc_status =
            musaMalloc(reinterpret_cast<void**>(&scratch->device_segments),
                       bytes);
        if (alloc_status != musaSuccess) {
          scratch->device_segments = nullptr;
          scratch->capacity = 0;
          return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                            MusaErrorString(alloc_status));
        }
        scratch->capacity = segments.size();
      }

      scratch->host_segments.resize(segments.size());
      scratch->host_copy_blocks.clear();
      for (size_t i = 0; i < segments.size(); ++i) {
        const ConcatSplitSegmentSpec& spec = segments[i];
        scratch->host_segments[i] = MusaConcatSplitSegment{
            input_data[spec.source_input_index],
            output_data[spec.output_index],
            input_cols[spec.source_input_index],
            outputs[spec.output_index].width,
            spec.source_offset,
            spec.width,
            spec.dst_offset,
        };
        const int64_t segment_elements = rows * spec.width;
        for (int64_t offset = 0; offset < segment_elements;
             offset += kConcatSplitCopyBlockSize) {
          scratch->host_copy_blocks.push_back(MusaConcatSplitCopyBlock{
              static_cast<int64_t>(i),
              offset,
          });
        }
      }

      if (scratch->copy_block_capacity < scratch->host_copy_blocks.size()) {
        if (scratch->device_copy_blocks != nullptr) {
          musaError_t free_status = musaFree(scratch->device_copy_blocks);
          if (free_status != musaSuccess) {
            return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                              MusaErrorString(free_status));
          }
        }
        const size_t block_bytes =
            scratch->host_copy_blocks.size() * sizeof(MusaConcatSplitCopyBlock);
        musaError_t alloc_status =
            musaMalloc(reinterpret_cast<void**>(&scratch->device_copy_blocks),
                       block_bytes);
        if (alloc_status != musaSuccess) {
          scratch->device_copy_blocks = nullptr;
          scratch->copy_block_capacity = 0;
          return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                            MusaErrorString(alloc_status));
        }
        scratch->copy_block_capacity = scratch->host_copy_blocks.size();
      }

      const size_t bytes = segments.size() * sizeof(MusaConcatSplitSegment);
      musaError_t copy_status = musaMemcpyAsync(
          scratch->device_segments, scratch->host_segments.data(), bytes,
          musaMemcpyHostToDevice, stream);
      if (copy_status == musaSuccess && !scratch->host_copy_blocks.empty()) {
        const size_t block_bytes =
            scratch->host_copy_blocks.size() * sizeof(MusaConcatSplitCopyBlock);
        copy_status = musaMemcpyAsync(
            scratch->device_copy_blocks, scratch->host_copy_blocks.data(),
            block_bytes, musaMemcpyHostToDevice, stream);
      }
      if (copy_status != musaSuccess) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                          MusaErrorString(copy_status));
      }
      RETURN_IF_ERROR(LaunchStatus(LaunchMusaConcatSplitBatchedCopy(
          scratch->device_segments, static_cast<int64_t>(segments.size()),
          scratch->device_copy_blocks,
          static_cast<int64_t>(scratch->host_copy_blocks.size()), rows,
          element_size, stream)));
    }

    if (!sums.empty()) {
      if (scratch->sum_output_capacity < sums.size()) {
        if (scratch->device_sum_outputs != nullptr) {
          musaError_t free_status = musaFree(scratch->device_sum_outputs);
          if (free_status != musaSuccess) {
            return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                              MusaErrorString(free_status));
          }
        }
        const size_t bytes = sums.size() * sizeof(MusaConcatSplitSumOutput);
        musaError_t alloc_status =
            musaMalloc(reinterpret_cast<void**>(&scratch->device_sum_outputs),
                       bytes);
        if (alloc_status != musaSuccess) {
          scratch->device_sum_outputs = nullptr;
          scratch->sum_output_capacity = 0;
          return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                            MusaErrorString(alloc_status));
        }
        scratch->sum_output_capacity = sums.size();
      }
      if (scratch->sum_term_capacity < sum_term_count) {
        if (scratch->device_sum_terms != nullptr) {
          musaError_t free_status = musaFree(scratch->device_sum_terms);
          if (free_status != musaSuccess) {
            return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                              MusaErrorString(free_status));
          }
        }
        const size_t bytes = sum_term_count * sizeof(MusaConcatSplitSumTerm);
        musaError_t alloc_status =
            musaMalloc(reinterpret_cast<void**>(&scratch->device_sum_terms),
                       bytes);
        if (alloc_status != musaSuccess) {
          scratch->device_sum_terms = nullptr;
          scratch->sum_term_capacity = 0;
          return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                            MusaErrorString(alloc_status));
        }
        scratch->sum_term_capacity = sum_term_count;
      }

      scratch->host_sum_outputs.resize(sums.size());
      scratch->host_sum_terms.resize(sum_term_count);
      size_t term_offset = 0;
      for (size_t sum_index = 0; sum_index < sums.size(); ++sum_index) {
        const ConcatSplitSumSpec& sum = sums[sum_index];
        scratch->host_sum_outputs[sum_index] = MusaConcatSplitSumOutput{
            static_cast<float*>(output_data[sum.output_index]),
            outputs[sum.output_index].width,
            static_cast<int64_t>(term_offset),
            static_cast<int64_t>(sum.terms.size()),
        };
        for (const ConcatSplitSumTermSpec& term : sum.terms) {
          scratch->host_sum_terms[term_offset++] = MusaConcatSplitSumTerm{
              static_cast<const float*>(input_data[term.source_input_index]),
              input_cols[term.source_input_index],
              term.source_offset,
          };
        }
      }

      const size_t output_bytes = sums.size() * sizeof(MusaConcatSplitSumOutput);
      musaError_t copy_status = musaMemcpyAsync(
          scratch->device_sum_outputs, scratch->host_sum_outputs.data(),
          output_bytes, musaMemcpyHostToDevice, stream);
      if (copy_status == musaSuccess) {
        const size_t term_bytes = sum_term_count * sizeof(MusaConcatSplitSumTerm);
        copy_status = musaMemcpyAsync(
            scratch->device_sum_terms, scratch->host_sum_terms.data(),
            term_bytes, musaMemcpyHostToDevice, stream);
      }
      if (copy_status != musaSuccess) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                          MusaErrorString(copy_status));
      }
      RETURN_IF_ERROR(LaunchStatus(LaunchMusaConcatSplitSums(
          scratch->device_sum_outputs, scratch->device_sum_terms,
          static_cast<int64_t>(sums.size()), rows, max_sum_width, stream)));
    }

    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsConcatSplitFusionGraph(Ort::ConstGraph graph) {
  int split_count = 0;
  int concat_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Concat")) {
      ++concat_count;
    } else if (IsOnnxOp(node, "Split")) {
      ++split_count;
    } else if (IsOnnxOp(node, "Sum")) {
      continue;
    } else {
      return false;
    }
  }
  return concat_count >= 1 && split_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateConcatSplitFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode split_node = FindSingleNode(graph, "Split");
  std::vector<Ort::ConstValueInfo> split_inputs = split_node.GetInputs();
  if (split_inputs.empty()) {
    throw std::runtime_error("ConcatSplit requires Split input");
  }

  Ort::ConstNode concat_node{nullptr};
  std::vector<Ort::ConstNode> downstream_concat_nodes;
  std::vector<Ort::ConstNode> downstream_sum_nodes;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Concat")) {
      std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
      if (outputs.size() == 1 && Name(outputs[0]) == Name(split_inputs[0])) {
        concat_node = node;
      } else {
        downstream_concat_nodes.push_back(node);
      }
    } else if (IsOnnxOp(node, "Sum")) {
      downstream_sum_nodes.push_back(node);
    }
  }
  if (!concat_node) {
    throw std::runtime_error("ConcatSplit requires upstream Concat");
  }

  if (ReadIntAttribute(concat_node, "axis", 0) != 1 ||
      ReadIntAttribute(split_node, "axis", 0) != 1) {
    throw std::runtime_error("ConcatSplit requires axis=1");
  }

  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  std::vector<Ort::ConstValueInfo> split_outputs = split_node.GetOutputs();
  if (split_inputs.size() != 2 || split_outputs.empty()) {
    throw std::runtime_error("ConcatSplit requires Split sizes input");
  }
  std::vector<int64_t> split_sizes = ReadIntInitializer(split_inputs[1]);
  if (split_sizes.size() != split_outputs.size()) {
    throw std::runtime_error("ConcatSplit Split size/output mismatch");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  std::vector<int64_t> concat_widths;
  concat_widths.reserve(concat_inputs.size());
  for (Ort::ConstValueInfo input : concat_inputs) {
    auto shape = GetTensorShape(input);
    if (!shape.has_value() || shape->size() != 2 || (*shape)[1] <= 0) {
      throw std::runtime_error("ConcatSplit requires static input widths");
    }
    concat_widths.push_back((*shape)[1]);
  }

  std::unordered_map<std::string, ConcatSplitSegmentSpec> split_specs;
  size_t concat_input_index = 0;
  int64_t concat_offset = 0;
  int64_t split_offset = 0;
  for (size_t i = 0; i < split_sizes.size(); ++i) {
    while (concat_input_index < concat_widths.size() &&
           split_offset >= concat_offset + concat_widths[concat_input_index]) {
      concat_offset += concat_widths[concat_input_index];
      ++concat_input_index;
    }
    if (concat_input_index >= concat_widths.size()) {
      throw std::runtime_error("ConcatSplit split offset exceeds inputs");
    }
    const int64_t source_offset = split_offset - concat_offset;
    const int64_t width = split_sizes[i];
    if (source_offset < 0 ||
        source_offset + width > concat_widths[concat_input_index]) {
      throw std::runtime_error(
          "ConcatSplit only supports splits within one Concat input");
    }
    split_specs.emplace(
        Name(split_outputs[i]),
        ConcatSplitSegmentSpec{
            GetFusedInputIndex(fused_input_indices,
                               Name(concat_inputs[concat_input_index])),
            0,
            source_offset,
            width,
            0,
        });
    split_offset += width;
  }

  std::vector<ConcatSplitOutput> outputs;
  std::vector<ConcatSplitSegmentSpec> segments;
  std::vector<ConcatSplitSumSpec> sums;
  std::unordered_map<std::string, std::vector<ConcatSplitSegmentSpec>>
      concat_specs;
  std::unordered_map<std::string, ConcatSplitSumSpec> sum_specs;
  for (Ort::ConstNode downstream_concat : downstream_concat_nodes) {
    if (ReadIntAttribute(downstream_concat, "axis", 0) != 1) {
      throw std::runtime_error("ConcatSplit downstream Concat requires axis=1");
    }
    std::vector<Ort::ConstValueInfo> downstream_outputs =
        downstream_concat.GetOutputs();
    if (downstream_outputs.size() != 1) {
      throw std::runtime_error("ConcatSplit downstream Concat output mismatch");
    }
    int64_t dst_offset = 0;
    std::vector<ConcatSplitSegmentSpec> concat_segments;
    for (Ort::ConstValueInfo input : downstream_concat.GetInputs()) {
      auto it = split_specs.find(Name(input));
      if (it == split_specs.end()) {
        throw std::runtime_error(
            "ConcatSplit downstream Concat input is not Split output");
      }
      ConcatSplitSegmentSpec segment = it->second;
      segment.dst_offset = dst_offset;
      concat_segments.push_back(segment);
      dst_offset += segment.width;
    }
    concat_specs.emplace(Name(downstream_outputs[0]),
                         std::move(concat_segments));
  }

  for (Ort::ConstNode downstream_sum : downstream_sum_nodes) {
    std::vector<Ort::ConstValueInfo> downstream_outputs =
        downstream_sum.GetOutputs();
    if (downstream_outputs.size() != 1) {
      throw std::runtime_error("ConcatSplit Sum output mismatch");
    }
    std::vector<Ort::ConstValueInfo> downstream_inputs =
        downstream_sum.GetInputs();
    if (downstream_inputs.size() < 2) {
      throw std::runtime_error("ConcatSplit Sum requires multiple inputs");
    }

    int64_t sum_width = -1;
    ConcatSplitSumSpec sum_spec;
    for (Ort::ConstValueInfo input : downstream_inputs) {
      auto it = split_specs.find(Name(input));
      if (it == split_specs.end()) {
        throw std::runtime_error(
            "ConcatSplit Sum input is not Split output");
      }
      if (sum_width < 0) {
        sum_width = it->second.width;
      } else if (sum_width != it->second.width) {
        throw std::runtime_error(
            "ConcatSplit Sum requires matching input widths");
      }
      sum_spec.terms.push_back(ConcatSplitSumTermSpec{
          it->second.source_input_index,
          it->second.source_offset,
      });
    }
    sum_spec.width = sum_width;
    sum_specs.emplace(Name(downstream_outputs[0]), std::move(sum_spec));
  }

  std::vector<Ort::ConstValueInfo> graph_outputs = graph.GetOutputs();
  outputs.reserve(graph_outputs.size());
  for (Ort::ConstValueInfo output : graph_outputs) {
    const std::string output_name = Name(output);
    const size_t output_index = outputs.size();
    auto split_it = split_specs.find(output_name);
    if (split_it != split_specs.end()) {
      ConcatSplitSegmentSpec segment = split_it->second;
      segment.output_index = output_index;
      segment.dst_offset = 0;
      outputs.push_back(ConcatSplitOutput{segment.width});
      segments.push_back(segment);
      continue;
    }

    auto concat_it = concat_specs.find(output_name);
    if (concat_it == concat_specs.end()) {
      auto sum_it = sum_specs.find(output_name);
      if (sum_it == sum_specs.end()) {
        throw std::runtime_error(
            "ConcatSplit graph output is not Split, downstream Concat, or Sum output");
      }
      ConcatSplitSumSpec sum = sum_it->second;
      sum.output_index = output_index;
      outputs.push_back(ConcatSplitOutput{sum.width});
      sums.push_back(std::move(sum));
      continue;
    }
    int64_t output_width = 0;
    for (const ConcatSplitSegmentSpec& segment : concat_it->second) {
      output_width += segment.width;
    }
    outputs.push_back(ConcatSplitOutput{output_width});
    for (ConcatSplitSegmentSpec segment : concat_it->second) {
      segment.output_index = output_index;
      segments.push_back(segment);
    }
  }
  if (outputs.empty() || (segments.empty() && sums.empty())) {
    throw std::runtime_error("ConcatSplit requires outputs");
  }

  return std::make_unique<ConcatSplitFusionCompute>(std::move(outputs),
                                                    std::move(segments),
                                                    std::move(sums));
}
