// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "fusion/log_bucketize_gather_fusion.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph/graph_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/log_bucketize_gather_impl.h"

namespace {

std::unordered_map<std::string, Ort::ConstNode> ProducersInGraph(
    Ort::ConstGraph graph) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : graph.GetNodes()) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      producers.emplace(musa_ep::Name(output), node);
    }
  }
  return producers;
}

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_input_indices;
  std::vector<Ort::ConstValueInfo> fused_inputs = fused_node.GetInputs();
  for (size_t i = 0; i < fused_inputs.size(); ++i) {
    fused_input_indices.emplace(musa_ep::Name(fused_inputs[i]), i);
  }
  return fused_input_indices;
}

std::unordered_map<std::string, size_t> FusedOutputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> fused_output_indices;
  std::vector<Ort::ConstValueInfo> fused_outputs = fused_node.GetOutputs();
  for (size_t i = 0; i < fused_outputs.size(); ++i) {
    fused_output_indices.emplace(musa_ep::Name(fused_outputs[i]), i);
  }
  return fused_output_indices;
}

size_t GetMappedIndex(const std::unordered_map<std::string, size_t>& indices,
                      const std::string& name, const char* kind) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("unable to map LogBucketizeGather ") +
                             kind + " " + name);
  }
  return it->second;
}

Ort::ConstNode ProducerOf(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info) {
  if (value_info == nullptr) {
    return Ort::ConstNode{nullptr};
  }
  auto it = producers.find(musa_ep::Name(value_info));
  if (it == producers.end()) {
    return Ort::ConstNode{nullptr};
  }
  return it->second;
}

float Float16ToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
  uint32_t exponent = (bits >> 10) & 0x1f;
  uint32_t mantissa = bits & 0x03ff;
  uint32_t out_bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      out_bits = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ff;
      out_bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1f) {
    out_bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    out_bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &out_bits, sizeof(result));
  return result;
}

template <typename ValueT>
float ReadTensorScalarAsFloat(const ValueT& value, const char* name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1) {
    throw std::runtime_error(std::string("LogBucketizeGather ") + name +
                             " must be scalar");
  }
  switch (info.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return value.template GetTensorData<float>()[0];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return static_cast<float>(value.template GetTensorData<double>()[0]);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return static_cast<float>(value.template GetTensorData<int64_t>()[0]);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return static_cast<float>(value.template GetTensorData<int32_t>()[0]);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: {
      const uint32_t bits =
          static_cast<uint32_t>(value.template GetTensorData<uint16_t>()[0])
          << 16;
      float result = 0.0f;
      std::memcpy(&result, &bits, sizeof(result));
      return result;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return Float16ToFloat(value.template GetTensorData<uint16_t>()[0]);
    default:
      throw std::runtime_error(std::string("LogBucketizeGather ") + name +
                               " has unsupported scalar dtype");
  }
}

template <typename ValueT>
int64_t ReadTensorScalarAsInt(const ValueT& value, const char* name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1) {
    throw std::runtime_error(std::string("LogBucketizeGather ") + name +
                             " must be scalar");
  }
  switch (info.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return value.template GetTensorData<int64_t>()[0];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return static_cast<int64_t>(value.template GetTensorData<int32_t>()[0]);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return static_cast<int64_t>(value.template GetTensorData<float>()[0]);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return static_cast<int64_t>(value.template GetTensorData<double>()[0]);
    default:
      throw std::runtime_error(std::string("LogBucketizeGather ") + name +
                               " has unsupported integer scalar dtype");
  }
}

Ort::ConstValue ReadInitializerValue(Ort::ConstValueInfo value_info,
                                     const char* name) {
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error(std::string("LogBucketizeGather failed to read ") +
                             name + " initializer");
  }
  return value;
}

Ort::Value ReadConstantNodeValue(Ort::ConstNode node, const char* name) {
  Ort::ConstOpAttr attr;
  Ort::Status attr_status = node.GetAttributeByName("value", attr);
  if (!attr_status.IsOK()) {
    throw std::runtime_error(std::string("LogBucketizeGather ") + name +
                             " requires Constant value");
  }
  Ort::Value value{nullptr};
  Ort::Status status = attr.GetTensorAttributeAsOrtValue(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error(std::string("LogBucketizeGather failed to read ") +
                             name + " Constant value");
  }
  return value;
}

float ReadScalarAsFloat(
    Ort::ConstValueInfo value_info,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const char* name);

int64_t ReadScalarAsInt(
    Ort::ConstValueInfo value_info,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const char* name);

float ReadScalarAsFloat(
    Ort::ConstValueInfo value_info,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const char* name) {
  if (value_info == nullptr) {
    throw std::runtime_error(std::string("LogBucketizeGather missing ") + name);
  }
  if (value_info.IsConstantInitializer()) {
    return ReadTensorScalarAsFloat(ReadInitializerValue(value_info, name),
                                   name);
  }
  Ort::ConstNode producer = ProducerOf(producers, value_info);
  if (musa_ep::IsOnnxOp(producer, "Constant")) {
    return ReadTensorScalarAsFloat(ReadConstantNodeValue(producer, name), name);
  }
  if (musa_ep::IsOnnxOp(producer, "Cast")) {
    std::vector<Ort::ConstValueInfo> inputs = producer.GetInputs();
    if (inputs.size() != 1) {
      throw std::runtime_error(std::string("invalid LogBucketizeGather Cast ") +
                               name);
    }
    return ReadScalarAsFloat(inputs[0], producers, name);
  }
  throw std::runtime_error(std::string("LogBucketizeGather ") + name +
                           " must be constant");
}

int64_t ReadScalarAsInt(
    Ort::ConstValueInfo value_info,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const char* name) {
  if (value_info == nullptr) {
    throw std::runtime_error(std::string("LogBucketizeGather missing ") + name);
  }
  if (value_info.IsConstantInitializer()) {
    return ReadTensorScalarAsInt(ReadInitializerValue(value_info, name), name);
  }
  Ort::ConstNode producer = ProducerOf(producers, value_info);
  if (musa_ep::IsOnnxOp(producer, "Constant")) {
    return ReadTensorScalarAsInt(ReadConstantNodeValue(producer, name), name);
  }
  if (musa_ep::IsOnnxOp(producer, "Cast")) {
    std::vector<Ort::ConstValueInfo> inputs = producer.GetInputs();
    if (inputs.size() != 1) {
      throw std::runtime_error(std::string("invalid LogBucketizeGather Cast ") +
                               name);
    }
    return ReadScalarAsInt(inputs[0], producers, name);
  }
  throw std::runtime_error(std::string("LogBucketizeGather ") + name +
                           " must be constant");
}

std::vector<int64_t> ReadShapeVector(
    Ort::ConstValueInfo value_info,
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    const char* name) {
  if (value_info == nullptr) {
    throw std::runtime_error(std::string("LogBucketizeGather missing ") + name);
  }
  if (value_info.IsConstantInitializer()) {
    Ort::ConstValue value = ReadInitializerValue(value_info, name);
    auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
        info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      throw std::runtime_error(std::string("LogBucketizeGather ") + name +
                               " must be int shape");
    }
    std::vector<int64_t> shape;
    const size_t count = static_cast<size_t>(info.GetElementCount());
    shape.reserve(count);
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const int64_t* data = value.GetTensorData<int64_t>();
      shape.assign(data, data + count);
    } else {
      const int32_t* data = value.GetTensorData<int32_t>();
      for (size_t i = 0; i < count; ++i) {
        shape.push_back(static_cast<int64_t>(data[i]));
      }
    }
    return shape;
  }
  Ort::ConstNode producer = ProducerOf(producers, value_info);
  if (musa_ep::IsOnnxOp(producer, "Constant")) {
    Ort::Value value = ReadConstantNodeValue(producer, name);
    auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
        info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      throw std::runtime_error(std::string("LogBucketizeGather ") + name +
                               " must be int shape");
    }
    std::vector<int64_t> shape;
    const size_t count = static_cast<size_t>(info.GetElementCount());
    shape.reserve(count);
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const int64_t* data = value.GetTensorData<int64_t>();
      shape.assign(data, data + count);
    } else {
      const int32_t* data = value.GetTensorData<int32_t>();
      for (size_t i = 0; i < count; ++i) {
        shape.push_back(static_cast<int64_t>(data[i]));
      }
    }
    return shape;
  }
  throw std::runtime_error(std::string("LogBucketizeGather ") + name +
                           " must be constant shape");
}

Ort::ConstNode RequiredProducer(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info, const char* expected_op,
    const char* message) {
  Ort::ConstNode node = ProducerOf(producers, value_info);
  if (!musa_ep::IsOnnxOp(node, expected_op)) {
    throw std::runtime_error(message);
  }
  return node;
}

std::vector<int64_t> UnsqueezedOutputShape(std::vector<int64_t> input_shape,
                                           int64_t axis) {
  const int64_t rank_after = static_cast<int64_t>(input_shape.size()) + 1;
  int64_t normalized_axis = 0;
  if (!musa_ep::NormalizeAxis(axis, static_cast<size_t>(rank_after),
                              normalized_axis)) {
    throw std::runtime_error("invalid LogBucketizeGather Unsqueeze axis");
  }
  input_shape.insert(input_shape.begin() + normalized_axis, 1);
  return input_shape;
}

struct LogBucketizeGatherParams {
  float diff_min;
  float time_scale;
  float value_min;
  float log_base;
  float bucket_start;
  float bucket_span;
  float bucket_scale;
  int64_t bucket_offset;
  int64_t bucket_min;
  int64_t bucket_max;
  float lower_threshold;
  float upper_threshold;
};

struct LogBucketizeGatherFusionCompute : FusionNodeCompute {
  LogBucketizeGatherFusionCompute(size_t table_index, size_t current_index,
                                  size_t sequence_index, int64_t unsqueeze_axis,
                                  LogBucketizeGatherParams params,
                                  size_t output_index)
      : table_index(table_index),
        current_index(current_index),
        sequence_index(sequence_index),
        unsqueeze_axis(unsqueeze_axis),
        params(params),
        output_index(output_index) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue table = ctx.GetInput(table_index);
      Ort::ConstValue current = ctx.GetInput(current_index);
      Ort::ConstValue sequence = ctx.GetInput(sequence_index);

      if (!IsGpuMemory(table.GetTensorMemoryInfo()) ||
          !IsGpuMemory(current.GetTensorMemoryInfo()) ||
          !IsGpuMemory(sequence.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "LogBucketizeGather requires MUSA input tensors");
      }

      auto table_info = table.GetTensorTypeAndShapeInfo();
      auto current_info = current.GetTensorTypeAndShapeInfo();
      auto sequence_info = sequence.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> table_shape = table_info.GetShape();
      std::vector<int64_t> current_shape = current_info.GetShape();
      std::vector<int64_t> sequence_shape = sequence_info.GetShape();
      if (table_shape.empty() || table_shape[0] <= params.bucket_max) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "LogBucketizeGather table does not cover bucket range");
      }

      const size_t elem_size = ElementSize(table_info.GetElementType());
      if (elem_size == 0) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "LogBucketizeGather unsupported table dtype");
      }
      const size_t current_elem_size =
          ElementSize(current_info.GetElementType());
      const size_t sequence_elem_size =
          ElementSize(sequence_info.GetElementType());
      if ((current_elem_size != sizeof(int32_t) &&
           current_elem_size != sizeof(int64_t)) ||
          (sequence_elem_size != sizeof(int32_t) &&
           sequence_elem_size != sizeof(int64_t))) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "LogBucketizeGather timestamp tensors must be int32/int64");
      }
      if (params.time_scale == 0.0f || params.log_base == 0.0f ||
          params.bucket_span == 0.0f || params.bucket_min < 0 ||
          params.bucket_max < params.bucket_min) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "LogBucketizeGather invalid bucket params");
      }

      int64_t block_size = 1;
      for (size_t i = 1; i < table_shape.size(); ++i) {
        block_size *= table_shape[i];
      }
      const int64_t current_count = NumElements(current_shape);
      const int64_t sequence_count = NumElements(sequence_shape);
      if (current_count != 1 && current_count != sequence_count) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "LogBucketizeGather only supports scalar or same-size current "
            "timestamps");
      }

      std::vector<int64_t> gather_shape = {sequence_count};
      gather_shape.insert(gather_shape.end(), table_shape.begin() + 1,
                          table_shape.end());
      std::vector<int64_t> output_shape =
          UnsqueezedOutputShape(std::move(gather_shape), unsqueeze_axis);
      Ort::UnownedValue output = ctx.GetOutput(output_index, output_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "LogBucketizeGather requires MUSA output");
      }
      if (output.GetTensorTypeAndShapeInfo().GetElementType() !=
          table_info.GetElementType()) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "LogBucketizeGather does not fuse payload dtype conversion");
      }

      DeviceInputBuffer table_buffer;
      DeviceInputBuffer current_buffer;
      DeviceInputBuffer sequence_buffer;
      musaStream_t stream = GetComputeStream(ctx);
      RETURN_IF_ERROR(table_buffer.Bind(table, stream));
      RETURN_IF_ERROR(current_buffer.Bind(current, stream));
      RETURN_IF_ERROR(sequence_buffer.Bind(sequence, stream));
      return LaunchStatus(LaunchMusaLogBucketizeGatherKernel(
          table_buffer.data(), current_buffer.data(), sequence_buffer.data(),
          output.GetTensorMutableRawData(), static_cast<int32_t>(elem_size),
          static_cast<int32_t>(current_elem_size),
          static_cast<int32_t>(sequence_elem_size), current_count,
          sequence_count, table_shape[0], block_size, params.diff_min,
          params.time_scale, params.value_min, params.log_base,
          params.bucket_start, params.bucket_span, params.bucket_scale,
          params.bucket_offset, params.bucket_min, params.bucket_max,
          params.lower_threshold, params.upper_threshold, stream));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t table_index;
  size_t current_index;
  size_t sequence_index;
  int64_t unsqueeze_axis;
  LogBucketizeGatherParams params;
  size_t output_index;
};

}  // namespace

bool IsLogBucketizeGatherFusionGraph(Ort::ConstGraph graph) {
  bool has_gather = false;
  bool has_log = false;
  bool has_floor = false;
  bool has_where = false;
  bool has_unsqueeze = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    has_gather = has_gather || musa_ep::IsOnnxOp(node, "Gather");
    has_log = has_log || musa_ep::IsOnnxOp(node, "Log");
    has_floor = has_floor || musa_ep::IsOnnxOp(node, "Floor");
    has_where = has_where || musa_ep::IsOnnxOp(node, "Where");
    has_unsqueeze = has_unsqueeze || musa_ep::IsOnnxOp(node, "Unsqueeze");
  }
  return has_gather && has_log && has_floor && has_where && has_unsqueeze;
}

std::unique_ptr<FusionNodeCompute> CreateLogBucketizeGatherFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto producers = ProducersInGraph(graph);
  Ort::ConstNode output_node{nullptr};
  Ort::ConstNode unsqueeze_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (musa_ep::IsOnnxOp(node, "Cast")) {
      std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
      if (inputs.size() == 1 &&
          musa_ep::IsOnnxOp(ProducerOf(producers, inputs[0]), "Unsqueeze")) {
        output_node = node;
        unsqueeze_node = ProducerOf(producers, inputs[0]);
        break;
      }
    }
  }
  if (!unsqueeze_node) {
    for (Ort::ConstNode node : graph.GetNodes()) {
      if (musa_ep::IsOnnxOp(node, "Unsqueeze")) {
        output_node = node;
        unsqueeze_node = node;
        break;
      }
    }
  }
  if (!unsqueeze_node) {
    throw std::runtime_error("LogBucketizeGather requires Unsqueeze");
  }

  std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
      unsqueeze_node.GetInputs();
  std::vector<Ort::ConstValueInfo> unsqueeze_outputs =
      unsqueeze_node.GetOutputs();
  if (unsqueeze_inputs.size() != 2 || unsqueeze_outputs.size() != 1) {
    throw std::runtime_error("invalid LogBucketizeGather Unsqueeze");
  }
  auto axes = musa_ep::ReadSmallIntInitializer(unsqueeze_inputs[1]);
  if (!axes.has_value() || axes->size() != 1) {
    throw std::runtime_error("LogBucketizeGather requires one Unsqueeze axis");
  }
  const int64_t unsqueeze_axis = (*axes)[0];

  Ort::ConstNode gather_node =
      RequiredProducer(producers, unsqueeze_inputs[0], "Gather",
                       "LogBucketizeGather requires Gather before Unsqueeze");
  std::vector<Ort::ConstValueInfo> gather_inputs = gather_node.GetInputs();
  if (gather_inputs.size() != 2) {
    throw std::runtime_error("invalid LogBucketizeGather Gather");
  }

  Ort::ConstNode low_where =
      RequiredProducer(producers, gather_inputs[1], "Where",
                       "LogBucketizeGather requires lower-bound Where");
  std::vector<Ort::ConstValueInfo> low_where_inputs = low_where.GetInputs();
  if (low_where_inputs.size() != 3) {
    throw std::runtime_error("invalid LogBucketizeGather lower Where");
  }
  Ort::ConstNode less_equal =
      RequiredProducer(producers, low_where_inputs[0], "LessOrEqual",
                       "LogBucketizeGather requires LessOrEqual");
  Ort::ConstNode high_where =
      RequiredProducer(producers, low_where_inputs[2], "Where",
                       "LogBucketizeGather requires upper-bound Where");
  std::vector<Ort::ConstValueInfo> less_inputs = less_equal.GetInputs();
  if (less_inputs.size() != 2) {
    throw std::runtime_error("invalid LogBucketizeGather LessOrEqual");
  }

  std::vector<Ort::ConstValueInfo> high_where_inputs = high_where.GetInputs();
  if (high_where_inputs.size() != 3) {
    throw std::runtime_error("invalid LogBucketizeGather upper Where");
  }
  Ort::ConstNode greater_equal =
      RequiredProducer(producers, high_where_inputs[0], "GreaterOrEqual",
                       "LogBucketizeGather requires GreaterOrEqual");
  Ort::ConstNode clip_index =
      RequiredProducer(producers, high_where_inputs[2], "Clip",
                       "LogBucketizeGather requires index Clip");
  std::vector<Ort::ConstValueInfo> greater_inputs = greater_equal.GetInputs();
  if (greater_inputs.size() != 2) {
    throw std::runtime_error("invalid LogBucketizeGather GreaterOrEqual");
  }

  Ort::ConstValueInfo log_bucket_value = less_inputs[0];
  if (musa_ep::Name(greater_inputs[0]) != musa_ep::Name(log_bucket_value)) {
    throw std::runtime_error("LogBucketizeGather comparisons must share input");
  }
  Ort::ConstNode log_div =
      RequiredProducer(producers, log_bucket_value, "Div",
                       "LogBucketizeGather requires log-base Div");
  std::vector<Ort::ConstValueInfo> log_div_inputs = log_div.GetInputs();
  if (log_div_inputs.size() != 2) {
    throw std::runtime_error("invalid LogBucketizeGather log Div");
  }
  Ort::ConstNode log_node = RequiredProducer(
      producers, log_div_inputs[0], "Log", "LogBucketizeGather requires Log");
  std::vector<Ort::ConstValueInfo> log_inputs = log_node.GetInputs();
  if (log_inputs.size() != 1) {
    throw std::runtime_error("invalid LogBucketizeGather Log");
  }
  Ort::ConstNode value_clip =
      RequiredProducer(producers, log_inputs[0], "Clip",
                       "LogBucketizeGather requires value Clip");
  std::vector<Ort::ConstValueInfo> value_clip_inputs = value_clip.GetInputs();
  if (value_clip_inputs.size() < 2) {
    throw std::runtime_error("invalid LogBucketizeGather value Clip");
  }
  Ort::ConstNode time_div =
      RequiredProducer(producers, value_clip_inputs[0], "Div",
                       "LogBucketizeGather requires time-scale Div");
  std::vector<Ort::ConstValueInfo> time_div_inputs = time_div.GetInputs();
  if (time_div_inputs.size() != 2) {
    throw std::runtime_error("invalid LogBucketizeGather time Div");
  }
  Ort::ConstNode diff_cast =
      RequiredProducer(producers, time_div_inputs[0], "Cast",
                       "LogBucketizeGather requires float Cast");
  std::vector<Ort::ConstValueInfo> diff_cast_inputs = diff_cast.GetInputs();
  if (diff_cast_inputs.size() != 1) {
    throw std::runtime_error("invalid LogBucketizeGather diff Cast");
  }
  Ort::ConstNode diff_clip =
      RequiredProducer(producers, diff_cast_inputs[0], "Clip",
                       "LogBucketizeGather requires diff Clip");
  std::vector<Ort::ConstValueInfo> diff_clip_inputs = diff_clip.GetInputs();
  if (diff_clip_inputs.size() < 2) {
    throw std::runtime_error("invalid LogBucketizeGather diff Clip");
  }
  Ort::ConstNode sub_node =
      RequiredProducer(producers, diff_clip_inputs[0], "Sub",
                       "LogBucketizeGather requires timestamp Sub");
  std::vector<Ort::ConstValueInfo> sub_inputs = sub_node.GetInputs();
  if (sub_inputs.size() != 2) {
    throw std::runtime_error("invalid LogBucketizeGather Sub");
  }
  Ort::ConstValueInfo current_input = sub_inputs[0];
  Ort::ConstValueInfo sequence_value = sub_inputs[1];
  Ort::ConstNode sequence_cast = ProducerOf(producers, sequence_value);
  if (musa_ep::IsOnnxOp(sequence_cast, "Cast")) {
    std::vector<Ort::ConstValueInfo> cast_inputs = sequence_cast.GetInputs();
    if (cast_inputs.size() != 1) {
      throw std::runtime_error("invalid LogBucketizeGather sequence Cast");
    }
    sequence_value = cast_inputs[0];
  }
  Ort::ConstNode sequence_reshape =
      RequiredProducer(producers, sequence_value, "Reshape",
                       "LogBucketizeGather requires sequence Reshape");
  std::vector<Ort::ConstValueInfo> reshape_inputs =
      sequence_reshape.GetInputs();
  if (reshape_inputs.size() != 2) {
    throw std::runtime_error("invalid LogBucketizeGather Reshape");
  }
  std::vector<int64_t> reshape_shape =
      ReadShapeVector(reshape_inputs[1], producers, "reshape shape");
  if (reshape_shape.size() != 1 || reshape_shape[0] != -1) {
    throw std::runtime_error("LogBucketizeGather requires Reshape[-1]");
  }

  std::vector<Ort::ConstValueInfo> clip_index_inputs = clip_index.GetInputs();
  if (clip_index_inputs.size() < 3) {
    throw std::runtime_error("invalid LogBucketizeGather index Clip");
  }
  Ort::ConstNode add_node =
      RequiredProducer(producers, clip_index_inputs[0], "Add",
                       "LogBucketizeGather requires index Add");
  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  if (add_inputs.size() != 2) {
    throw std::runtime_error("invalid LogBucketizeGather Add");
  }
  Ort::ConstNode floor_cast =
      RequiredProducer(producers, add_inputs[0], "Cast",
                       "LogBucketizeGather requires Floor Cast");
  std::vector<Ort::ConstValueInfo> floor_cast_inputs = floor_cast.GetInputs();
  if (floor_cast_inputs.size() != 1) {
    throw std::runtime_error("invalid LogBucketizeGather Floor Cast");
  }
  Ort::ConstNode floor_node =
      RequiredProducer(producers, floor_cast_inputs[0], "Floor",
                       "LogBucketizeGather requires Floor");
  std::vector<Ort::ConstValueInfo> floor_inputs = floor_node.GetInputs();
  if (floor_inputs.size() != 1) {
    throw std::runtime_error("invalid LogBucketizeGather Floor");
  }
  Ort::ConstNode scale_mul =
      RequiredProducer(producers, floor_inputs[0], "Mul",
                       "LogBucketizeGather requires scale Mul");
  std::vector<Ort::ConstValueInfo> scale_mul_inputs = scale_mul.GetInputs();
  if (scale_mul_inputs.size() != 2) {
    throw std::runtime_error("invalid LogBucketizeGather scale Mul");
  }
  Ort::ConstNode span_div =
      RequiredProducer(producers, scale_mul_inputs[0], "Div",
                       "LogBucketizeGather requires span Div");
  std::vector<Ort::ConstValueInfo> span_div_inputs = span_div.GetInputs();
  if (span_div_inputs.size() != 2) {
    throw std::runtime_error("invalid LogBucketizeGather span Div");
  }
  Ort::ConstNode start_sub =
      RequiredProducer(producers, span_div_inputs[0], "Sub",
                       "LogBucketizeGather requires start Sub");
  std::vector<Ort::ConstValueInfo> start_sub_inputs = start_sub.GetInputs();
  if (start_sub_inputs.size() != 2 ||
      musa_ep::Name(start_sub_inputs[0]) != musa_ep::Name(log_bucket_value)) {
    throw std::runtime_error("LogBucketizeGather start Sub must consume log");
  }

  LogBucketizeGatherParams params;
  params.diff_min =
      ReadScalarAsFloat(diff_clip_inputs[1], producers, "diff min");
  params.time_scale =
      ReadScalarAsFloat(time_div_inputs[1], producers, "time scale");
  params.value_min =
      ReadScalarAsFloat(value_clip_inputs[1], producers, "value min");
  params.log_base = ReadScalarAsFloat(log_div_inputs[1], producers, "log base");
  params.bucket_start =
      ReadScalarAsFloat(start_sub_inputs[1], producers, "bucket start");
  params.bucket_span =
      ReadScalarAsFloat(span_div_inputs[1], producers, "bucket span");
  params.bucket_scale =
      ReadScalarAsFloat(scale_mul_inputs[1], producers, "bucket scale");
  params.bucket_offset =
      ReadScalarAsInt(add_inputs[1], producers, "bucket offset");
  params.bucket_min =
      ReadScalarAsInt(clip_index_inputs[1], producers, "bucket min");
  params.bucket_max =
      ReadScalarAsInt(clip_index_inputs[2], producers, "bucket max");
  params.lower_threshold =
      ReadScalarAsFloat(less_inputs[1], producers, "lower threshold");
  params.upper_threshold =
      ReadScalarAsFloat(greater_inputs[1], producers, "upper threshold");

  auto fused_inputs = FusedInputIndices(fused_node);
  auto fused_outputs = FusedOutputIndices(fused_node);
  std::vector<Ort::ConstValueInfo> output_node_outputs =
      output_node.GetOutputs();
  if (output_node_outputs.size() != 1) {
    throw std::runtime_error("invalid LogBucketizeGather output node");
  }
  return std::make_unique<LogBucketizeGatherFusionCompute>(
      GetMappedIndex(fused_inputs, musa_ep::Name(gather_inputs[0]),
                     "table input"),
      GetMappedIndex(fused_inputs, musa_ep::Name(current_input),
                     "current timestamp input"),
      GetMappedIndex(fused_inputs, musa_ep::Name(reshape_inputs[0]),
                     "sequence timestamp input"),
      unsqueeze_axis, params,
      GetMappedIndex(fused_outputs, musa_ep::Name(output_node_outputs[0]),
                     "output"));
}
