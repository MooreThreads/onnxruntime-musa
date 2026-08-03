// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "fusion/rec_rank_calibration_fusion.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/graph_utils.h"
#include "kernels/reduction/rec_rank_calibration_impl.h"
#include "kernels/shared_inc/op_kernel_common.h"

namespace musa_ep {
namespace {

struct ParsedCalibrationGraph {
  Ort::ConstValueInfo task_relu{nullptr};
  Ort::ConstValueInfo task_scores{nullptr};
  Ort::ConstValueInfo output{nullptr};
  int64_t bucket_size = 0;
  float epsilon = 0.0f;
  bool complement_from_clipped_probability = false;
};

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

std::unordered_map<std::string, size_t> ValueIndices(
    const std::vector<Ort::ConstValueInfo>& value_infos) {
  std::unordered_map<std::string, size_t> indices;
  for (size_t i = 0; i < value_infos.size(); ++i) {
    indices.emplace(Name(value_infos[i]), i);
  }
  return indices;
}

size_t GetMappedIndex(const std::unordered_map<std::string, size_t>& indices,
                      Ort::ConstValueInfo value_info, const char* kind) {
  auto it = indices.find(Name(value_info));
  if (it == indices.end()) {
    throw std::runtime_error(std::string("unable to map RecRankCalibration ") +
                             kind + " " + Name(value_info));
  }
  return it->second;
}

double ReadScalarNumber(Ort::ConstValueInfo value_info, const char* kind) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    throw std::runtime_error(std::string("RecRankCalibration requires ") +
                             kind + " initializer");
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error(std::string("unable to read RecRankCalibration ") +
                             kind + " initializer");
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementCount() != 1) {
    throw std::runtime_error(std::string("RecRankCalibration ") + kind +
                             " must be scalar");
  }
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    return value.GetTensorData<double>()[0];
  }
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return static_cast<double>(value.GetTensorData<float>()[0]);
  }
  throw std::runtime_error(std::string("RecRankCalibration ") + kind +
                           " must be floating point");
}

ParsedCalibrationGraph ParseCalibrationGraph(Ort::ConstGraph graph) {
  ParsedCalibrationGraph parsed;
  auto producers = ProducersInGraph(graph);

  Ort::ConstNode mask_add{nullptr};
  Ort::ConstNode floor_node{nullptr};
  Ort::ConstNode probability_clip{nullptr};
  Ort::ConstNode complement_sub{nullptr};
  Ort::ConstNode log_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Add")) {
      mask_add = node;
    } else if (IsOnnxOp(node, "Floor")) {
      floor_node = node;
    } else if (IsOnnxOp(node, "Clip")) {
      std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
      if (!inputs.empty()) {
        Ort::ConstNode data_producer = ProducerInGraph(producers, inputs[0]);
        if (IsOnnxOp(data_producer, "Cast")) {
          probability_clip = node;
        } else if (IsOnnxOp(data_producer, "Sub")) {
          complement_sub = data_producer;
        }
      }
    } else if (IsOnnxOp(node, "Log")) {
      log_node = node;
    }
  }
  if (!mask_add || !floor_node || !probability_clip || !complement_sub ||
      !log_node) {
    throw std::runtime_error("invalid RecRankCalibration source graph");
  }

  std::vector<Ort::ConstValueInfo> mask_outputs = mask_add.GetOutputs();
  if (mask_outputs.size() != 1) {
    throw std::runtime_error("RecRankCalibration mask Add is invalid");
  }
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (!IsOnnxOp(node, "Mul")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    if (inputs.size() != 2) {
      continue;
    }
    if (Name(inputs[0]) == Name(mask_outputs[0])) {
      parsed.task_relu = inputs[1];
      break;
    }
    if (Name(inputs[1]) == Name(mask_outputs[0])) {
      parsed.task_relu = inputs[0];
      break;
    }
  }

  std::vector<Ort::ConstValueInfo> floor_inputs = floor_node.GetInputs();
  if (floor_inputs.size() != 1) {
    throw std::runtime_error("RecRankCalibration Floor is invalid");
  }
  Ort::ConstNode scaled_mul = ProducerInGraph(producers, floor_inputs[0]);
  if (!IsOnnxOp(scaled_mul, "Mul")) {
    throw std::runtime_error("RecRankCalibration scaled Mul is invalid");
  }
  std::vector<Ort::ConstValueInfo> scaled_inputs = scaled_mul.GetInputs();
  if (scaled_inputs.size() != 2) {
    throw std::runtime_error("RecRankCalibration scaled Mul is invalid");
  }
  Ort::ConstValueInfo bucket_initializer{nullptr};
  for (Ort::ConstValueInfo input : scaled_inputs) {
    if (input.IsConstantInitializer()) {
      bucket_initializer = input;
    } else {
      parsed.task_scores = input;
    }
  }
  const double bucket_value =
      ReadScalarNumber(bucket_initializer, "bucket_size");
  parsed.bucket_size = static_cast<int64_t>(std::llround(bucket_value));
  if (parsed.bucket_size <= 0 ||
      static_cast<double>(parsed.bucket_size) != bucket_value) {
    throw std::runtime_error(
        "RecRankCalibration bucket_size must be a positive integer");
  }

  std::vector<Ort::ConstValueInfo> clip_inputs = probability_clip.GetInputs();
  if (clip_inputs.size() != 3) {
    throw std::runtime_error("RecRankCalibration probability Clip is invalid");
  }
  parsed.epsilon =
      static_cast<float>(ReadScalarNumber(clip_inputs[1], "epsilon"));
  if (!(parsed.epsilon > 0.0f && parsed.epsilon < 1.0f)) {
    throw std::runtime_error(
        "RecRankCalibration epsilon must be between zero and one");
  }

  Ort::ConstNode output_cast = ProducerInGraph(producers, clip_inputs[0]);
  if (!IsOnnxOp(output_cast, "Cast")) {
    throw std::runtime_error("RecRankCalibration probability Cast is invalid");
  }
  std::vector<Ort::ConstValueInfo> output_cast_outputs =
      output_cast.GetOutputs();
  std::vector<Ort::ConstValueInfo> probability_clip_outputs =
      probability_clip.GetOutputs();
  std::vector<Ort::ConstValueInfo> complement_sub_inputs =
      complement_sub.GetInputs();
  if (output_cast_outputs.size() != 1 || probability_clip_outputs.size() != 1 ||
      complement_sub_inputs.size() != 2) {
    throw std::runtime_error("RecRankCalibration complement path is invalid");
  }
  if (Name(complement_sub_inputs[1]) == Name(probability_clip_outputs[0])) {
    parsed.complement_from_clipped_probability = true;
  } else if (Name(complement_sub_inputs[1]) != Name(output_cast_outputs[0])) {
    throw std::runtime_error("RecRankCalibration complement source is invalid");
  }

  std::vector<Ort::ConstValueInfo> log_outputs = log_node.GetOutputs();
  if (parsed.task_relu == nullptr || parsed.task_scores == nullptr ||
      log_outputs.size() != 1) {
    throw std::runtime_error("RecRankCalibration boundaries are invalid");
  }
  parsed.output = log_outputs[0];
  return parsed;
}

bool IsDoubleGpuTensor(Ort::ConstValue value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  return info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE &&
         IsGpuMemory(value.GetTensorMemoryInfo());
}

class RecRankCalibrationFusionCompute final : public FusionNodeCompute {
 public:
  RecRankCalibrationFusionCompute(size_t task_relu_index,
                                  size_t task_scores_index, size_t output_index,
                                  int64_t bucket_size, float epsilon,
                                  bool complement_from_clipped_probability)
      : task_relu_index_(task_relu_index),
        task_scores_index_(task_scores_index),
        output_index_(output_index),
        bucket_size_(bucket_size),
        epsilon_(epsilon),
        complement_from_clipped_probability_(
            complement_from_clipped_probability) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue task_relu = ctx.GetInput(task_relu_index_);
      Ort::ConstValue task_scores = ctx.GetInput(task_scores_index_);
      if (!IsDoubleGpuTensor(task_relu) || !IsDoubleGpuTensor(task_scores)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "RecRankCalibration requires MUSA double inputs");
      }

      auto relu_info = task_relu.GetTensorTypeAndShapeInfo();
      auto score_info = task_scores.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> relu_shape = relu_info.GetShape();
      std::vector<int64_t> score_shape = score_info.GetShape();
      if (relu_shape.size() < 2 || score_shape.size() != relu_shape.size() ||
          relu_shape.back() != bucket_size_ || score_shape.back() != 1) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "RecRankCalibration input shapes do not match bucket_size");
      }
      for (size_t i = 0; i + 1 < relu_shape.size(); ++i) {
        if (relu_shape[i] != score_shape[i] || relu_shape[i] < 0) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "RecRankCalibration input prefixes must match");
        }
      }

      const int64_t rows = static_cast<int64_t>(score_info.GetElementCount());
      if (relu_info.GetElementCount() !=
          static_cast<size_t>(rows * bucket_size_)) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "RecRankCalibration task_relu element count is invalid");
      }

      Ort::UnownedValue output = ctx.GetOutput(output_index_, score_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "RecRankCalibration requires a MUSA output");
      }
      return LaunchStatus(LaunchMusaRecRankCalibrationKernel(
          task_relu.GetTensorData<double>(),
          task_scores.GetTensorData<double>(),
          output.GetTensorMutableData<float>(), rows, bucket_size_, epsilon_,
          complement_from_clipped_probability_, GetComputeStream(ctx)));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

 private:
  size_t task_relu_index_;
  size_t task_scores_index_;
  size_t output_index_;
  int64_t bucket_size_;
  float epsilon_;
  bool complement_from_clipped_probability_;
};

}  // namespace

bool IsRecRankCalibrationFusionGraph(Ort::ConstGraph graph) {
  int mul_count = 0;
  int floor_count = 0;
  int less_count = 0;
  int cast_count = 0;
  int equal_count = 0;
  int sub_count = 0;
  int add_count = 0;
  int reduce_sum_count = 0;
  int clip_count = 0;
  int div_count = 0;
  int log_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Mul")) {
      ++mul_count;
    } else if (IsOnnxOp(node, "Floor")) {
      ++floor_count;
    } else if (IsOnnxOp(node, "Less")) {
      ++less_count;
    } else if (IsOnnxOp(node, "Cast")) {
      ++cast_count;
    } else if (IsOnnxOp(node, "Equal")) {
      ++equal_count;
    } else if (IsOnnxOp(node, "Sub")) {
      ++sub_count;
    } else if (IsOnnxOp(node, "Add")) {
      ++add_count;
    } else if (IsOnnxOp(node, "ReduceSum")) {
      ++reduce_sum_count;
    } else if (IsOnnxOp(node, "Clip")) {
      ++clip_count;
    } else if (IsOnnxOp(node, "Div")) {
      ++div_count;
    } else if (IsOnnxOp(node, "Log")) {
      ++log_count;
    } else {
      return false;
    }
  }
  return mul_count == 4 && floor_count == 1 && less_count == 1 &&
         cast_count == 3 && equal_count == 1 && sub_count == 2 &&
         add_count == 1 && reduce_sum_count == 1 && clip_count == 2 &&
         div_count == 1 && log_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateRecRankCalibrationFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  ParsedCalibrationGraph parsed = ParseCalibrationGraph(graph);
  auto input_indices = ValueIndices(fused_node.GetInputs());
  auto output_indices = ValueIndices(fused_node.GetOutputs());
  return std::make_unique<RecRankCalibrationFusionCompute>(
      GetMappedIndex(input_indices, parsed.task_relu, "task_relu"),
      GetMappedIndex(input_indices, parsed.task_scores, "task_scores"),
      GetMappedIndex(output_indices, parsed.output, "output"),
      parsed.bucket_size, parsed.epsilon,
      parsed.complement_from_clipped_probability);
}

}  // namespace musa_ep
