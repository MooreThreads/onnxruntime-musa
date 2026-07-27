// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fusion/parallel_matmul_concat_fusion.h"

#include <mudnn.h>
#include <musa_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "kernels/math/matmul.h"
#include "kernels/shared_inc/blas_utils.h"

/*
 * ParallelMatMulConcat Fusion Pattern
 *
 *   X -> MatMul(X, W0) -> Unsqueeze(axis = rank(Y) - 1)
 *   X -> MatMul(X, W1) -> Unsqueeze(axis = rank(Y) - 1) -> Concat(axis)
 *   X -> MatMul(X, W2) -> Unsqueeze(axis = rank(Y) - 1)
 *   X -> MatMul(X, W3) -> Unsqueeze(axis = rank(Y) - 1)
 *
 * The supported layout is the DCN/MMoE gate pattern:
 *
 *   Concat(Unsqueeze(MatMul(X, Wi), axis = output_rank - 1), axis)
 *
 * With identical X and equal [K, N] weights, this is equivalent to:
 *
 *   MatMul(X, Concat(W0, W1, W2, W3, axis = 1))
 *
 * The MatMul writes directly into the original Concat output allocation. The
 * original output shape [..., 4, N] is a contiguous view of MatMul's
 * [..., 4 * N] result for the supported axis.
 */

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  void Resize(size_t bytes, musaStream_t stream) {
    if (bytes <= bytes_) {
      stream_ = stream;
      return;
    }

    if (ptr_ != nullptr) {
      FreeDeviceMemoryOnStream(ptr_, stream_, bytes_);
      ptr_ = nullptr;
      bytes_ = 0;
    }

    stream_ = stream;
    if (bytes == 0) {
      return;
    }

    ptr_ = AllocateDeviceMemoryOnStream(bytes, stream_);
    if (ptr_ == nullptr) {
      throw std::runtime_error(MusaErrorString(musaErrorMemoryAllocation));
    }
    bytes_ = bytes;
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)musaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  template <typename T>
  T* data() const {
    return reinterpret_cast<T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t stream_ = nullptr;
};

struct ParallelMatMulConcatScratch {
  DeviceBuffer weight_buffer;
  size_t merged_weight_bytes = 0;
  bool merged_initializer_weights_valid = false;
};

static ParallelMatMulConcatScratch& ThreadLocalScratchForStream(
    const ParallelMatMulConcatFusionCompute* owner, musaStream_t stream) {
  using StreamScratchMap =
      std::unordered_map<musaStream_t,
                         std::unique_ptr<ParallelMatMulConcatScratch>>;
  thread_local std::unordered_map<const ParallelMatMulConcatFusionCompute*,
                                  StreamScratchMap>
      scratch_by_owner;
  auto& scratch_by_stream = scratch_by_owner[owner];
  auto& scratch = scratch_by_stream[stream];
  if (!scratch) {
    scratch = std::make_unique<ParallelMatMulConcatScratch>();
  }
  return *scratch;
}

namespace {

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(Ort::ConstNode node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

std::vector<int64_t> TensorShape(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

void ValidateFloatTensor(Ort::ConstValue value, const char* name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(std::string("ParallelMatMulConcat only supports "
                                         "float tensors for ") +
                             name);
  }
}

bool IsGpuValue(Ort::ConstValue value) {
  return IsGpuMemory(value.GetTensorMemoryInfo());
}

std::vector<int64_t> ShapeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  if (shape.empty()) {
    return strides;
  }
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

void CheckMudnnStatus(::musa::dnn::Status status, const char* message) {
  if (status != ::musa::dnn::Status::SUCCESS) {
    throw std::runtime_error(std::string(message) + ", status: " +
                             std::to_string(static_cast<int>(status)));
  }
}

void SetupMudnn2DTensor(::musa::dnn::Tensor& tensor, const float* data,
                        const std::vector<int64_t>& shape) {
  CheckMudnnStatus(tensor.SetType(::musa::dnn::Tensor::Type::FLOAT),
                   "Failed to set ParallelMatMulConcat tensor type");
  if (data != nullptr) {
    CheckMudnnStatus(tensor.SetAddr(data),
                     "Failed to set ParallelMatMulConcat tensor address");
  } else if (NumElements(shape) > 0) {
    throw std::runtime_error(
        "ParallelMatMulConcat tensor data pointer is null");
  }
  CheckMudnnStatus(tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW),
                   "Failed to set ParallelMatMulConcat tensor format");
  std::vector<int64_t> strides = ShapeStrides(shape);
  CheckMudnnStatus(tensor.SetNdInfo(static_cast<int>(shape.size()),
                                    shape.data(), strides.data()),
                   "Failed to set ParallelMatMulConcat tensor shape");
}

void RunMudnnWeightConcat(const std::vector<Ort::ConstValue>& weights,
                          const std::vector<int64_t>& merged_shape,
                          float* merged_data, musaStream_t stream) {
  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* status = EnsureMudnnHandle(&handle, stream);
  if (status != nullptr) {
    Ort::GetApi().ReleaseStatus(status);
    throw std::runtime_error("ParallelMatMulConcat mudnn handle setup failed");
  }

  std::vector<::musa::dnn::Tensor> input_tensors(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    SetupMudnn2DTensor(input_tensors[i], weights[i].GetTensorData<float>(),
                       TensorShape(weights[i]));
  }

  ::musa::dnn::Tensor output_tensor;
  SetupMudnn2DTensor(output_tensor, merged_data, merged_shape);

  ::musa::dnn::Concat concat_op;
  CheckMudnnStatus(concat_op.SetAxis(1),
                   "Failed to set ParallelMatMulConcat concat axis");
  CheckMudnnStatus(concat_op.Run(*handle, output_tensor,
                                 static_cast<int>(input_tensors.size()),
                                 input_tensors.data()),
                   "ParallelMatMulConcat weight concat failed");
}

std::vector<int64_t> ComputeMatMulShape(const std::vector<int64_t>& input_shape,
                                        int64_t total_n) {
  if (input_shape.size() < 2) {
    throw std::runtime_error(
        "ParallelMatMulConcat input rank must be at least 2");
  }

  std::vector<int64_t> output_shape = input_shape;
  output_shape.back() = total_n;
  return output_shape;
}

std::vector<int64_t> ComputeConcatOutputShape(
    const std::vector<int64_t>& matmul_output_shape, int64_t part_count,
    int64_t part_width, int64_t concat_axis) {
  std::vector<int64_t> output_shape = matmul_output_shape;
  const size_t insert_pos = output_shape.size() - 1;
  if (concat_axis != static_cast<int64_t>(insert_pos)) {
    throw std::runtime_error(
        "ParallelMatMulConcat only supports Unsqueeze/Concat before the last "
        "MatMul output dimension");
  }
  output_shape.insert(output_shape.begin() + static_cast<int64_t>(insert_pos),
                      part_count);
  output_shape.back() = part_width;
  return output_shape;
}

OrtStatus* ComputeDeviceParallelMatMulConcat(
    Ort::UnownedValue y, Ort::ConstValue input,
    const std::vector<Ort::ConstValue>& weights, int64_t concat_axis,
    bool weights_are_initializers, ParallelMatMulConcatScratch& scratch,
    musaStream_t stream) {
  if (!IsGpuValue(input)) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ParallelMatMulConcat requires MUSA input");
  }
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ParallelMatMulConcat requires MUSA output");
  }
  for (Ort::ConstValue weight : weights) {
    if (!IsGpuValue(weight)) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "ParallelMatMulConcat requires MUSA weight inputs");
    }
  }

  std::vector<int64_t> input_shape = TensorShape(input);
  std::vector<int64_t> first_weight_shape = TensorShape(weights[0]);
  if (first_weight_shape.size() != 2 || first_weight_shape[0] <= 0 ||
      first_weight_shape[1] <= 0) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "ParallelMatMulConcat weights must have static 2D shapes");
  }
  for (Ort::ConstValue weight : weights) {
    std::vector<int64_t> shape = TensorShape(weight);
    if (shape != first_weight_shape) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "ParallelMatMulConcat weights must have identical shapes");
    }
  }

  const int64_t part_count = static_cast<int64_t>(weights.size());
  const int64_t part_width = first_weight_shape[1];
  const int64_t total_n = part_count * part_width;
  std::vector<int64_t> merged_weight_shape = {first_weight_shape[0], total_n};
  std::vector<int64_t> matmul_output_shape =
      ComputeMatMulShape(input_shape, total_n);
  std::vector<int64_t> concat_output_shape = ComputeConcatOutputShape(
      matmul_output_shape, part_count, part_width, concat_axis);

  const size_t merged_weight_bytes =
      static_cast<size_t>(NumElements(merged_weight_shape)) * sizeof(float);
  if (scratch.merged_weight_bytes != merged_weight_bytes) {
    scratch.merged_initializer_weights_valid = false;
    scratch.merged_weight_bytes = merged_weight_bytes;
  }
  scratch.weight_buffer.Resize(merged_weight_bytes, stream);
  if (!weights_are_initializers || !scratch.merged_initializer_weights_valid) {
    RunMudnnWeightConcat(weights, merged_weight_shape,
                         scratch.weight_buffer.data<float>(), stream);
    scratch.merged_initializer_weights_valid = weights_are_initializers;
  }

  float* y_data = y.GetTensorMutableData<float>();
  return ComputeMusaMatMulDevice(
      input.GetTensorData<float>(), scratch.weight_buffer.data<float>(), y_data,
      input_shape, merged_weight_shape, matmul_output_shape, stream);
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
    throw std::runtime_error("unable to map ParallelMatMulConcat fused input");
  }
  return it->second;
}

}  // namespace

bool IsParallelMatMulConcatFusionGraph(Ort::ConstGraph graph) {
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  size_t concat_count = 0;
  size_t matmul_count = 0;
  size_t unsqueeze_count = 0;
  for (Ort::ConstNode node : nodes) {
    if (IsOnnxOp(node, "Concat")) {
      ++concat_count;
    } else if (IsOnnxOp(node, "MatMul")) {
      ++matmul_count;
    } else if (IsOnnxOp(node, "Unsqueeze")) {
      ++unsqueeze_count;
    } else {
      return false;
    }
  }
  return concat_count == 1 && matmul_count >= 2 &&
         matmul_count == unsqueeze_count;
}

std::unique_ptr<FusionNodeCompute> CreateParallelMatMulConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  Ort::ConstNode concat_node{nullptr};
  std::unordered_map<std::string, Ort::ConstNode> producer_by_output;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Concat")) {
      concat_node = node;
    }
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      producer_by_output.emplace(Name(output), node);
    }
  }
  if (!concat_node) {
    throw std::runtime_error("ParallelMatMulConcat fusion expects Concat");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  std::vector<Ort::ConstValueInfo> concat_inputs = concat_node.GetInputs();
  if (concat_inputs.size() < 2) {
    throw std::runtime_error(
        "ParallelMatMulConcat expects at least two Concat inputs");
  }

  size_t input_index = 0;
  std::string common_input_name;
  std::vector<size_t> weight_indices;
  weight_indices.reserve(concat_inputs.size());
  bool weights_are_initializers = true;

  for (Ort::ConstValueInfo concat_input : concat_inputs) {
    auto unsqueeze_it = producer_by_output.find(Name(concat_input));
    if (unsqueeze_it == producer_by_output.end() ||
        !IsOnnxOp(unsqueeze_it->second, "Unsqueeze")) {
      throw std::runtime_error(
          "ParallelMatMulConcat expects Unsqueeze inputs to Concat");
    }

    std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
        unsqueeze_it->second.GetInputs();
    if (unsqueeze_inputs.empty()) {
      throw std::runtime_error("ParallelMatMulConcat invalid Unsqueeze");
    }

    auto matmul_it = producer_by_output.find(Name(unsqueeze_inputs[0]));
    if (matmul_it == producer_by_output.end() ||
        !IsOnnxOp(matmul_it->second, "MatMul")) {
      throw std::runtime_error(
          "ParallelMatMulConcat expects MatMul before Unsqueeze");
    }

    std::vector<Ort::ConstValueInfo> matmul_inputs =
        matmul_it->second.GetInputs();
    if (matmul_inputs.size() != 2) {
      throw std::runtime_error("ParallelMatMulConcat invalid MatMul");
    }
    const std::string input_name = Name(matmul_inputs[0]);
    if (common_input_name.empty()) {
      common_input_name = input_name;
      input_index = GetFusedInputIndex(fused_input_indices, input_name);
    } else if (common_input_name != input_name) {
      throw std::runtime_error(
          "ParallelMatMulConcat MatMul inputs are not shared");
    }
    weight_indices.push_back(
        GetFusedInputIndex(fused_input_indices, Name(matmul_inputs[1])));
    weights_are_initializers =
        weights_are_initializers && matmul_inputs[1].IsConstantInitializer();
  }

  int64_t concat_axis = 0;
  Ort::ConstOpAttr axis_attr;
  Ort::Status status = concat_node.GetAttributeByName("axis", axis_attr);
  if (!status.IsOK() || !axis_attr.GetValue(concat_axis).IsOK()) {
    throw std::runtime_error("ParallelMatMulConcat missing Concat axis");
  }

  return std::make_unique<ParallelMatMulConcatFusionCompute>(
      input_index, std::move(weight_indices), concat_axis,
      weights_are_initializers);
}

ParallelMatMulConcatFusionCompute::ParallelMatMulConcatFusionCompute(
    size_t input_index, std::vector<size_t> weight_indices, int64_t concat_axis,
    bool weights_are_initializers)
    : input_index(input_index),
      weight_indices(std::move(weight_indices)),
      concat_axis(concat_axis),
      weights_are_initializers(weights_are_initializers) {}

ParallelMatMulConcatFusionCompute::~ParallelMatMulConcatFusionCompute() =
    default;

OrtStatus* ParallelMatMulConcatFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    musaStream_t stream = GetComputeStream(ctx);

    Ort::ConstValue input = ctx.GetInput(input_index);
    ValidateFloatTensor(input, "input");
    std::vector<Ort::ConstValue> weights;
    weights.reserve(weight_indices.size());
    for (size_t index : weight_indices) {
      Ort::ConstValue weight = ctx.GetInput(index);
      ValidateFloatTensor(weight, "weight");
      weights.push_back(weight);
    }
    if (weights.empty()) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "ParallelMatMulConcat requires at least one weight");
    }

    std::vector<int64_t> input_shape = TensorShape(input);
    std::vector<int64_t> first_weight_shape = TensorShape(weights[0]);
    const int64_t part_count = static_cast<int64_t>(weights.size());
    const int64_t part_width = first_weight_shape.at(1);
    std::vector<int64_t> matmul_output_shape =
        ComputeMatMulShape(input_shape, part_count * part_width);
    std::vector<int64_t> concat_output_shape = ComputeConcatOutputShape(
        matmul_output_shape, part_count, part_width, concat_axis);

    Ort::UnownedValue y = ctx.GetOutput(0, concat_output_shape);
    ParallelMatMulConcatScratch& scratch =
        ThreadLocalScratchForStream(this, stream);
    return ComputeDeviceParallelMatMulConcat(y, input, weights, concat_axis,
                                             weights_are_initializers, scratch,
                                             stream);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}
