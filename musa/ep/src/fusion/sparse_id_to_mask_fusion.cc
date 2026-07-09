// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "fusion/sparse_id_to_mask_fusion.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/graph_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/sparse_id_to_mask_impl.h"

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

Ort::ConstNode ProducerInGraph(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info) {
  auto it = producers.find(musa_ep::Name(value_info));
  return it == producers.end() ? Ort::ConstNode{nullptr} : it->second;
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
    throw std::runtime_error(std::string("unable to map SparseIdToMask ") +
                             kind + " " + name);
  }
  return it->second;
}

std::vector<int64_t> ReadScalarIntInitializerOrThrow(
    Ort::ConstValueInfo value_info, const char* kind) {
  std::optional<std::vector<int64_t>> values =
      musa_ep::ReadSmallIntInitializer(value_info);
  if (!values.has_value() || values->size() != 1) {
    throw std::runtime_error(std::string("SparseIdToMask requires scalar ") +
                             kind + " initializer");
  }
  return *values;
}

bool IsGpuTensor(Ort::ConstValue value) {
  return IsGpuMemory(value.GetTensorMemoryInfo());
}

bool SameShape(const std::vector<int64_t>& lhs,
               const std::vector<int64_t>& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

}  // namespace

struct SparseIdToMaskFusionCompute : FusionNodeCompute {
  SparseIdToMaskFusionCompute(size_t dense_input_index,
                              size_t bound_input_index,
                              size_t sparse_input_index, size_t output_index,
                              int64_t default_id)
      : dense_input_index(dense_input_index),
        bound_input_index(bound_input_index),
        sparse_input_index(sparse_input_index),
        output_index(output_index),
        default_id(default_id) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue dense_ids = ctx.GetInput(dense_input_index);
      Ort::ConstValue bound_ids = ctx.GetInput(bound_input_index);
      Ort::ConstValue sparse_ids = ctx.GetInput(sparse_input_index);
      if (!IsGpuTensor(dense_ids) || !IsGpuTensor(bound_ids) ||
          !IsGpuTensor(sparse_ids)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "SparseIdToMask requires MUSA integer inputs");
      }

      auto dense_info = dense_ids.GetTensorTypeAndShapeInfo();
      auto bound_info = bound_ids.GetTensorTypeAndShapeInfo();
      auto sparse_info = sparse_ids.GetTensorTypeAndShapeInfo();
      const ONNXTensorElementDataType elem_type = dense_info.GetElementType();
      if (elem_type != bound_info.GetElementType() ||
          elem_type != sparse_info.GetElementType()) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "SparseIdToMask requires inputs with the same dtype");
      }
      std::vector<int64_t> output_shape = dense_info.GetShape();
      const int64_t output_count = dense_info.GetElementCount();
      const int64_t candidate_count = bound_info.GetElementCount();
      const int64_t sparse_count = sparse_info.GetElementCount();
      if (candidate_count <= 0 || (candidate_count != output_count &&
                                   output_count % candidate_count != 0)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "SparseIdToMask requires broadcastable candidate inputs");
      }
      if (sparse_count <= 0 ||
          (sparse_count != 1 && sparse_count != candidate_count)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "SparseIdToMask requires scalar or candidate-shaped sparse input");
      }

      Ort::UnownedValue output = ctx.GetOutput(output_index, output_shape);
      const ONNXTensorElementDataType output_type =
          output.GetTensorTypeAndShapeInfo().GetElementType();
      if (!IsGpuMemory(output.GetTensorMemoryInfo()) ||
          (output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
           output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "SparseIdToMask requires a MUSA float or int32 output");
      }

      if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          return LaunchStatus(LaunchMusaSparseIdToMaskInt32Kernel(
              dense_ids.GetTensorData<int32_t>(),
              bound_ids.GetTensorData<int32_t>(),
              sparse_ids.GetTensorData<int32_t>(),
              output.GetTensorMutableData<float>(), output_count,
              candidate_count, sparse_count, static_cast<int32_t>(default_id),
              GetComputeStream(ctx)));
        }
        return LaunchStatus(LaunchMusaSparseIdToMaskInt32Kernel(
            dense_ids.GetTensorData<int32_t>(),
            bound_ids.GetTensorData<int32_t>(),
            sparse_ids.GetTensorData<int32_t>(),
            output.GetTensorMutableData<int32_t>(), output_count,
            candidate_count, sparse_count, static_cast<int32_t>(default_id),
            GetComputeStream(ctx)));
      }
      if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          return LaunchStatus(LaunchMusaSparseIdToMaskInt64Kernel(
              dense_ids.GetTensorData<int64_t>(),
              bound_ids.GetTensorData<int64_t>(),
              sparse_ids.GetTensorData<int64_t>(),
              output.GetTensorMutableData<float>(), output_count,
              candidate_count, sparse_count, default_id,
              GetComputeStream(ctx)));
        }
        return LaunchStatus(LaunchMusaSparseIdToMaskInt64Kernel(
            dense_ids.GetTensorData<int64_t>(),
            bound_ids.GetTensorData<int64_t>(),
            sparse_ids.GetTensorData<int64_t>(),
            output.GetTensorMutableData<int32_t>(), output_count,
            candidate_count, sparse_count, default_id, GetComputeStream(ctx)));
      }
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "SparseIdToMask supports int32/int64 inputs");
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t dense_input_index;
  size_t bound_input_index;
  size_t sparse_input_index;
  size_t output_index;
  int64_t default_id;
};

bool IsSparseIdToMaskFusionGraph(Ort::ConstGraph graph) {
  int less_equal_count = 0;
  int not_count = 0;
  int cast_count = 0;
  int mul_count = 0;
  int add_count = 0;
  int equal_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (musa_ep::IsOnnxOp(node, "LessOrEqual")) {
      ++less_equal_count;
    } else if (musa_ep::IsOnnxOp(node, "Not")) {
      ++not_count;
    } else if (musa_ep::IsOnnxOp(node, "Cast")) {
      ++cast_count;
    } else if (musa_ep::IsOnnxOp(node, "Mul")) {
      ++mul_count;
    } else if (musa_ep::IsOnnxOp(node, "Add")) {
      ++add_count;
    } else if (musa_ep::IsOnnxOp(node, "Equal")) {
      ++equal_count;
    } else {
      return false;
    }
  }
  return less_equal_count == 1 && not_count == 1 && cast_count == 3 &&
         mul_count == 2 && add_count == 1 && equal_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateSparseIdToMaskFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto producers = ProducersInGraph(graph);
  auto fused_input_indices = FusedInputIndices(fused_node);
  auto fused_output_indices = FusedOutputIndices(fused_node);

  Ort::ConstNode output_cast_node{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (musa_ep::IsOnnxOp(node, "Cast")) {
      std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
      if (outputs.size() == 1 &&
          fused_output_indices.count(musa_ep::Name(outputs[0])) != 0) {
        output_cast_node = node;
        break;
      }
    }
  }
  if (!output_cast_node) {
    throw std::runtime_error("SparseIdToMask requires final Cast");
  }

  std::vector<Ort::ConstValueInfo> output_cast_inputs =
      output_cast_node.GetInputs();
  if (output_cast_inputs.size() != 1) {
    throw std::runtime_error("SparseIdToMask final Cast is invalid");
  }
  Ort::ConstNode equal_node = ProducerInGraph(producers, output_cast_inputs[0]);
  if (!musa_ep::IsOnnxOp(equal_node, "Equal")) {
    throw std::runtime_error("SparseIdToMask requires Equal before Cast");
  }

  std::vector<Ort::ConstValueInfo> equal_inputs = equal_node.GetInputs();
  if (equal_inputs.size() != 2) {
    throw std::runtime_error("SparseIdToMask Equal is invalid");
  }
  Ort::ConstValueInfo dense_input{nullptr};
  Ort::ConstNode add_node{nullptr};
  for (Ort::ConstValueInfo input : equal_inputs) {
    Ort::ConstNode producer = ProducerInGraph(producers, input);
    if (musa_ep::IsOnnxOp(producer, "Add")) {
      add_node = producer;
    } else {
      dense_input = input;
    }
  }
  if (!musa_ep::IsOnnxOp(add_node, "Add") || dense_input == nullptr) {
    throw std::runtime_error("SparseIdToMask requires Add and dense input");
  }

  std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
  if (add_inputs.size() != 2) {
    throw std::runtime_error("SparseIdToMask Add is invalid");
  }

  Ort::ConstNode less_equal_node{nullptr};
  Ort::ConstValueInfo sparse_input{nullptr};
  Ort::ConstValueInfo bound_input{nullptr};
  Ort::ConstValueInfo default_input{nullptr};
  for (Ort::ConstValueInfo add_input : add_inputs) {
    Ort::ConstNode mul_node = ProducerInGraph(producers, add_input);
    if (!musa_ep::IsOnnxOp(mul_node, "Mul")) {
      throw std::runtime_error("SparseIdToMask requires Mul before Add");
    }
    std::vector<Ort::ConstValueInfo> mul_inputs = mul_node.GetInputs();
    if (mul_inputs.size() != 2) {
      throw std::runtime_error("SparseIdToMask Mul is invalid");
    }

    Ort::ConstValueInfo constant_input{nullptr};
    Ort::ConstValueInfo dynamic_input{nullptr};
    Ort::ConstNode cast_node{nullptr};
    for (Ort::ConstValueInfo mul_input : mul_inputs) {
      if (musa_ep::IsConstantInitializerValueInfo(mul_input)) {
        constant_input = mul_input;
      } else {
        Ort::ConstNode producer = ProducerInGraph(producers, mul_input);
        if (musa_ep::IsOnnxOp(producer, "Cast")) {
          cast_node = producer;
        } else {
          dynamic_input = mul_input;
        }
      }
    }
    if (!musa_ep::IsOnnxOp(cast_node, "Cast")) {
      throw std::runtime_error("SparseIdToMask Mul requires Cast input");
    }

    std::vector<Ort::ConstValueInfo> cast_inputs = cast_node.GetInputs();
    if (cast_inputs.size() != 1) {
      throw std::runtime_error("SparseIdToMask Cast is invalid");
    }
    Ort::ConstNode cast_input_producer =
        ProducerInGraph(producers, cast_inputs[0]);
    if (musa_ep::IsOnnxOp(cast_input_producer, "LessOrEqual")) {
      if (constant_input == nullptr) {
        throw std::runtime_error(
            "SparseIdToMask default branch requires a constant");
      }
      less_equal_node = cast_input_producer;
      default_input = constant_input;
    } else if (musa_ep::IsOnnxOp(cast_input_producer, "Not")) {
      if (constant_input != nullptr || dynamic_input == nullptr) {
        throw std::runtime_error(
            "SparseIdToMask bound branch requires a dynamic input");
      }
      bound_input = dynamic_input;
      std::vector<Ort::ConstValueInfo> not_inputs =
          cast_input_producer.GetInputs();
      if (not_inputs.size() != 1) {
        throw std::runtime_error("SparseIdToMask Not is invalid");
      }
      Ort::ConstNode not_input_producer =
          ProducerInGraph(producers, not_inputs[0]);
      if (less_equal_node != nullptr &&
          not_input_producer.GetId() != less_equal_node.GetId()) {
        throw std::runtime_error("SparseIdToMask Not must consume LessOrEqual");
      }
      less_equal_node = not_input_producer;
    } else {
      throw std::runtime_error("SparseIdToMask Cast input is invalid");
    }
  }

  if (!musa_ep::IsOnnxOp(less_equal_node, "LessOrEqual") ||
      bound_input == nullptr || default_input == nullptr) {
    throw std::runtime_error("SparseIdToMask missing LessOrEqual constants");
  }

  std::vector<Ort::ConstValueInfo> less_equal_inputs =
      less_equal_node.GetInputs();
  if (less_equal_inputs.size() != 2 ||
      musa_ep::Name(less_equal_inputs[0]) != musa_ep::Name(bound_input)) {
    throw std::runtime_error(
        "SparseIdToMask expects LessOrEqual(bound, sparse_id)");
  }
  sparse_input = less_equal_inputs[1];
  bound_input = less_equal_inputs[0];

  int64_t default_id =
      ReadScalarIntInitializerOrThrow(default_input, "default")[0];

  return std::make_unique<SparseIdToMaskFusionCompute>(
      GetMappedIndex(fused_input_indices, musa_ep::Name(dense_input), "input"),
      GetMappedIndex(fused_input_indices, musa_ep::Name(bound_input), "input"),
      GetMappedIndex(fused_input_indices, musa_ep::Name(sparse_input), "input"),
      GetMappedIndex(fused_output_indices,
                     musa_ep::Name(output_cast_node.GetOutputs()[0]), "output"),
      default_id);
}
