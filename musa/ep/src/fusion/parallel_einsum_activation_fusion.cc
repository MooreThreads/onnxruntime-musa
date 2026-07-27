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

#include "fusion/parallel_einsum_activation_fusion.h"

#include <musa_runtime.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/math/gemm_post_kernels.h"
#include "kernels/math/matmul.h"
#include "kernels/nn/parallel_einsum_activation_impl.h"
#include "kernels/shared_inc/op_kernel_common.h"

namespace {

constexpr size_t kParallelEinsumActivationHeads = 4;

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(Ort::ConstNode node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

std::vector<int64_t> TensorShape(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

void ValidateFloatTensor(Ort::ConstValue value, const char* name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(
        std::string("ParallelEinsumActivation only supports float ") + name);
  }
}

std::unordered_map<std::string, size_t> FusedInputIndices(
    Ort::ConstNode fused_node) {
  std::unordered_map<std::string, size_t> indices;
  std::vector<Ort::ConstValueInfo> fused_inputs = fused_node.GetInputs();
  for (size_t i = 0; i < fused_inputs.size(); ++i) {
    indices.emplace(Name(fused_inputs[i]), i);
  }
  return indices;
}

size_t GetFusedInputIndex(
    const std::unordered_map<std::string, size_t>& fused_input_indices,
    const std::string& input_name) {
  auto it = fused_input_indices.find(input_name);
  if (it == fused_input_indices.end()) {
    throw std::runtime_error(
        "unable to map ParallelEinsumActivation fused input");
  }
  return it->second;
}

struct ParsedBranch {
  Ort::ConstNode first_einsum{nullptr};
  Ort::ConstNode second_einsum{nullptr};
  Ort::ConstNode third_einsum{nullptr};
  Ort::ConstNode add{nullptr};
  Ort::ConstNode mul{nullptr};
};

}  // namespace

class CachedDeviceValue {
 public:
  CachedDeviceValue() = default;
  ~CachedDeviceValue() {
    if (ptr_ != nullptr) {
      (void)musaFree(ptr_);
    }
  }

  CachedDeviceValue(const CachedDeviceValue&) = delete;
  CachedDeviceValue& operator=(const CachedDeviceValue&) = delete;

  OrtStatus* BindConstant(Ort::ConstValue value, musaStream_t stream) {
    if (IsGpuMemory(value.GetTensorMemoryInfo())) {
      data_ = value.GetTensorRawData();
      return nullptr;
    }

    const size_t bytes = value.GetTensorSizeInBytes();
    const void* host_data = value.GetTensorRawData();
    if (bytes == 0) {
      data_ = host_data;
      return nullptr;
    }

    if (ptr_ == nullptr || bytes_ != bytes || host_data_ != host_data) {
      if (ptr_ != nullptr) {
        FreeDeviceMemoryOnStream(ptr_, stream_, bytes_);
        ptr_ = nullptr;
        bytes_ = 0;
        host_data_ = nullptr;
      }
      ptr_ = AllocateDeviceMemoryOnStream(bytes, stream);
      if (ptr_ == nullptr) {
        return Ort::GetApi().CreateStatus(
            ORT_EP_FAIL, MusaErrorString(musaErrorMemoryAllocation));
      }
      stream_ = stream;
      RETURN_IF_ERROR(
          CopyTemporaryHostToDevice(ptr_, host_data, bytes, stream));
      if (stream != nullptr) {
        musaError_t sync_status = musaStreamSynchronize(stream);
        if (sync_status != musaSuccess) {
          return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                            MusaErrorString(sync_status));
        }
      }
      bytes_ = bytes;
      host_data_ = host_data;
    }

    data_ = ptr_;
    return nullptr;
  }

  const void* data() const { return data_; }

 private:
  void* ptr_ = nullptr;
  const void* data_ = nullptr;
  const void* host_data_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t stream_ = nullptr;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)musaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  OrtStatus* Resize(size_t bytes, musaStream_t stream) {
    if (bytes <= bytes_) {
      stream_ = stream;
      return nullptr;
    }
    if (ptr_ != nullptr) {
      FreeDeviceMemoryOnStream(ptr_, stream_, bytes_);
      ptr_ = nullptr;
      bytes_ = 0;
    }
    if (bytes == 0) {
      return nullptr;
    }
    ptr_ = AllocateDeviceMemoryOnStream(bytes, stream);
    if (ptr_ == nullptr) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL, MusaErrorString(musaErrorMemoryAllocation));
    }
    stream_ = stream;
    bytes_ = bytes;
    return nullptr;
  }

  template <typename T>
  T* data() const {
    return reinterpret_cast<T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t stream_ = nullptr;
};

struct ParallelEinsumActivationDeviceConstants {
  std::array<CachedDeviceValue, 4> w1;
  std::array<CachedDeviceValue, 4> w2;
  std::array<CachedDeviceValue, 4> w3;
  CachedDeviceValue bias;
  DeviceBuffer packed_w1;
  bool packed_w1_valid = false;
  int64_t packed_input_dim = 0;
  int64_t packed_hidden_dim = 0;
};

struct ParallelEinsumActivationRuntimeBuffers {
  DeviceBuffer dynamic_packed_w1;
  DeviceBuffer stage1;
};

ParallelEinsumActivationRuntimeBuffers& ThreadLocalRuntimeBuffersForStream(
    const ParallelEinsumActivationFusionCompute* owner, musaStream_t stream) {
  using StreamBufferMap = std::unordered_map<
      musaStream_t, std::unique_ptr<ParallelEinsumActivationRuntimeBuffers>>;
  thread_local std::unordered_map<const ParallelEinsumActivationFusionCompute*,
                                  StreamBufferMap>
      buffers_by_owner;
  auto& buffers_by_stream = buffers_by_owner[owner];
  auto& buffers = buffers_by_stream[stream];
  if (!buffers) {
    buffers = std::make_unique<ParallelEinsumActivationRuntimeBuffers>();
  }
  return *buffers;
}

bool IsParallelEinsumActivationFusionGraph(Ort::ConstGraph graph) {
  size_t concat_count = 0;
  size_t einsum_count = 0;
  size_t tanh_count = 0;
  size_t add_count = 0;
  size_t mul_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Concat")) {
      ++concat_count;
    } else if (IsOnnxOp(node, "Einsum")) {
      ++einsum_count;
    } else if (IsOnnxOp(node, "Tanh")) {
      ++tanh_count;
    } else if (IsOnnxOp(node, "Add")) {
      ++add_count;
    } else if (IsOnnxOp(node, "Mul")) {
      ++mul_count;
    } else {
      return false;
    }
  }
  return concat_count == 1 && einsum_count == 12 && tanh_count == 8 &&
         add_count == 4 && mul_count == 4;
}

std::unique_ptr<FusionNodeCompute> CreateParallelEinsumActivationFusion(
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
    throw std::runtime_error("ParallelEinsumActivation fusion expects Concat");
  }

  auto fused_input_indices = FusedInputIndices(fused_node);
  std::string mlp_input_name;
  std::string gate_input_name;
  std::string bias_name;
  bool constants_are_initializers = true;
  std::vector<ParallelEinsumActivationBranchInputs> branches;
  branches.reserve(kParallelEinsumActivationHeads);

  for (Ort::ConstValueInfo concat_input : concat_node.GetInputs()) {
    Ort::ConstNode mul_node = producer_by_output.at(Name(concat_input));
    std::vector<Ort::ConstValueInfo> mul_inputs = mul_node.GetInputs();
    Ort::ConstNode add_node{nullptr};
    std::string branch_gate_input;
    for (Ort::ConstValueInfo input : mul_inputs) {
      auto it = producer_by_output.find(Name(input));
      if (it != producer_by_output.end() && IsOnnxOp(it->second, "Add")) {
        add_node = it->second;
      } else {
        branch_gate_input = Name(input);
      }
    }
    if (!add_node || branch_gate_input.empty()) {
      throw std::runtime_error("ParallelEinsumActivation invalid Mul branch");
    }
    if (gate_input_name.empty()) {
      gate_input_name = branch_gate_input;
    } else if (gate_input_name != branch_gate_input) {
      throw std::runtime_error(
          "ParallelEinsumActivation branches use different gates");
    }

    std::vector<Ort::ConstValueInfo> add_inputs = add_node.GetInputs();
    Ort::ConstNode third_einsum{nullptr};
    std::string branch_bias;
    bool branch_bias_is_initializer = false;
    for (Ort::ConstValueInfo input : add_inputs) {
      auto it = producer_by_output.find(Name(input));
      if (it != producer_by_output.end() && IsOnnxOp(it->second, "Einsum")) {
        third_einsum = it->second;
      } else {
        branch_bias = Name(input);
        branch_bias_is_initializer = input.IsConstantInitializer();
      }
    }
    if (!third_einsum || branch_bias.empty()) {
      throw std::runtime_error("ParallelEinsumActivation invalid Add branch");
    }
    if (bias_name.empty()) {
      bias_name = branch_bias;
    } else if (bias_name != branch_bias) {
      throw std::runtime_error(
          "ParallelEinsumActivation branches use different bias");
    }

    std::vector<Ort::ConstValueInfo> third_inputs = third_einsum.GetInputs();
    Ort::ConstNode second_tanh = producer_by_output.at(Name(third_inputs[1]));
    Ort::ConstNode second_einsum =
        producer_by_output.at(Name(second_tanh.GetInputs()[0]));
    Ort::ConstNode first_tanh =
        producer_by_output.at(Name(second_einsum.GetInputs()[1]));
    Ort::ConstNode first_einsum =
        producer_by_output.at(Name(first_tanh.GetInputs()[0]));

    std::vector<Ort::ConstValueInfo> first_inputs = first_einsum.GetInputs();
    const std::string branch_mlp_input = Name(first_inputs[1]);
    if (mlp_input_name.empty()) {
      mlp_input_name = branch_mlp_input;
    } else if (mlp_input_name != branch_mlp_input) {
      throw std::runtime_error(
          "ParallelEinsumActivation branches use different MLP inputs");
    }

    constants_are_initializers =
        constants_are_initializers && branch_bias_is_initializer &&
        first_inputs[0].IsConstantInitializer() &&
        second_einsum.GetInputs()[0].IsConstantInitializer() &&
        third_inputs[0].IsConstantInitializer();
    branches.push_back({
        GetFusedInputIndex(fused_input_indices, Name(first_inputs[0])),
        GetFusedInputIndex(fused_input_indices,
                           Name(second_einsum.GetInputs()[0])),
        GetFusedInputIndex(fused_input_indices, Name(third_inputs[0])),
    });
  }

  if (branches.size() != kParallelEinsumActivationHeads) {
    throw std::runtime_error("ParallelEinsumActivation expects four branches");
  }

  return std::make_unique<ParallelEinsumActivationFusionCompute>(
      GetFusedInputIndex(fused_input_indices, mlp_input_name),
      GetFusedInputIndex(fused_input_indices, gate_input_name),
      GetFusedInputIndex(fused_input_indices, bias_name), std::move(branches),
      constants_are_initializers);
}

ParallelEinsumActivationFusionCompute::ParallelEinsumActivationFusionCompute(
    size_t mlp_input_index, size_t gate_input_index, size_t bias_index,
    std::vector<ParallelEinsumActivationBranchInputs> branches,
    bool constants_are_initializers)
    : mlp_input_index(mlp_input_index),
      gate_input_index(gate_input_index),
      bias_index(bias_index),
      branches(std::move(branches)),
      constants_are_initializers(constants_are_initializers) {}

ParallelEinsumActivationFusionCompute::
    ~ParallelEinsumActivationFusionCompute() = default;

OrtStatus* ParallelEinsumActivationFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    musaStream_t stream = GetComputeStream(ctx);
    if (branches.size() != kParallelEinsumActivationHeads) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "ParallelEinsumActivation expects four heads");
    }

    Ort::ConstValue mlp_input = ctx.GetInput(mlp_input_index);
    Ort::ConstValue gate_input = ctx.GetInput(gate_input_index);
    Ort::ConstValue bias = ctx.GetInput(bias_index);
    ValidateFloatTensor(mlp_input, "MLP input");
    ValidateFloatTensor(gate_input, "gate input");
    ValidateFloatTensor(bias, "bias");

    std::vector<int64_t> mlp_shape = TensorShape(mlp_input);
    std::vector<int64_t> gate_shape = TensorShape(gate_input);
    std::vector<int64_t> bias_shape = TensorShape(bias);
    if (mlp_shape.size() != 3 || gate_shape != mlp_shape || mlp_shape[2] != 1 ||
        bias_shape.size() != 2 || bias_shape[0] != mlp_shape[1] ||
        bias_shape[1] != 1) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "ParallelEinsumActivation requires [B,D,1] inputs and [D,1] bias");
    }
    const int64_t batch = mlp_shape[0];
    const int64_t input_dim = mlp_shape[1];

    std::vector<Ort::ConstValue> w1;
    std::vector<Ort::ConstValue> w2;
    std::vector<Ort::ConstValue> w3;
    w1.reserve(kParallelEinsumActivationHeads);
    w2.reserve(kParallelEinsumActivationHeads);
    w3.reserve(kParallelEinsumActivationHeads);
    for (const auto& branch : branches) {
      w1.push_back(ctx.GetInput(branch.w1_index));
      w2.push_back(ctx.GetInput(branch.w2_index));
      w3.push_back(ctx.GetInput(branch.w3_index));
      ValidateFloatTensor(w1.back(), "w1");
      ValidateFloatTensor(w2.back(), "w2");
      ValidateFloatTensor(w3.back(), "w3");
    }

    std::vector<int64_t> w1_shape = TensorShape(w1[0]);
    std::vector<int64_t> w2_shape = TensorShape(w2[0]);
    std::vector<int64_t> w3_shape = TensorShape(w3[0]);
    if (w1_shape.size() != 2 || w2_shape.size() != 2 || w3_shape.size() != 2 ||
        w1_shape[1] != input_dim || w2_shape[0] != w1_shape[0] ||
        w2_shape[1] != w1_shape[0] || w3_shape[0] != input_dim ||
        w3_shape[1] != w1_shape[0]) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "ParallelEinsumActivation weight shape mismatch");
    }
    for (size_t i = 1; i < kParallelEinsumActivationHeads; ++i) {
      if (TensorShape(w1[i]) != w1_shape || TensorShape(w2[i]) != w2_shape ||
          TensorShape(w3[i]) != w3_shape) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "ParallelEinsumActivation branch weights must share shapes");
      }
    }
    const int64_t hidden_dim = w1_shape[0];

    DeviceInputBuffer mlp_input_buffer;
    DeviceInputBuffer gate_input_buffer;
    RETURN_IF_ERROR(mlp_input_buffer.Bind(mlp_input, stream));
    RETURN_IF_ERROR(gate_input_buffer.Bind(gate_input, stream));

    std::vector<int64_t> output_shape = {
        batch, input_dim, static_cast<int64_t>(kParallelEinsumActivationHeads)};
    Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
    if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "ParallelEinsumActivation requires MUSA output");
    }

    std::array<const float*, kParallelEinsumActivationHeads> w1_data{};
    std::array<const float*, kParallelEinsumActivationHeads> w2_data{};
    std::array<const float*, kParallelEinsumActivationHeads> w3_data{};
    const float* bias_data = nullptr;
    float* packed_w1_data = nullptr;
    std::vector<std::unique_ptr<DeviceInputBuffer>> temporary_buffers;
    if (constants_are_initializers) {
      std::lock_guard<std::mutex> lock(constants_mutex);
      if (!constants) {
        constants = std::make_unique<ParallelEinsumActivationDeviceConstants>();
      }
      for (size_t i = 0; i < kParallelEinsumActivationHeads; ++i) {
        RETURN_IF_ERROR(constants->w1[i].BindConstant(w1[i], stream));
        RETURN_IF_ERROR(constants->w2[i].BindConstant(w2[i], stream));
        RETURN_IF_ERROR(constants->w3[i].BindConstant(w3[i], stream));
        w1_data[i] = static_cast<const float*>(constants->w1[i].data());
        w2_data[i] = static_cast<const float*>(constants->w2[i].data());
        w3_data[i] = static_cast<const float*>(constants->w3[i].data());
      }
      RETURN_IF_ERROR(constants->bias.BindConstant(bias, stream));
      bias_data = static_cast<const float*>(constants->bias.data());
      const bool repack_w1 = !constants->packed_w1_valid ||
                             constants->packed_input_dim != input_dim ||
                             constants->packed_hidden_dim != hidden_dim;
      RETURN_IF_ERROR(constants->packed_w1.Resize(
          static_cast<size_t>(input_dim * kParallelEinsumActivationHeads *
                              hidden_dim) *
              sizeof(float),
          stream));
      packed_w1_data = constants->packed_w1.data<float>();
      if (repack_w1) {
        RETURN_IF_ERROR(
            LaunchStatus(LaunchMusaParallelEinsumActivationPackW1Kernel(
                w1_data[0], w1_data[1], w1_data[2], w1_data[3], packed_w1_data,
                input_dim, hidden_dim, stream)));
        constants->packed_w1_valid = true;
        constants->packed_input_dim = input_dim;
        constants->packed_hidden_dim = hidden_dim;
      }
    } else {
      temporary_buffers.reserve(kParallelEinsumActivationHeads * 3 + 1);
      for (size_t i = 0; i < kParallelEinsumActivationHeads; ++i) {
        temporary_buffers.push_back(std::make_unique<DeviceInputBuffer>());
        RETURN_IF_ERROR(temporary_buffers.back()->Bind(w1[i], stream));
        w1_data[i] =
            static_cast<const float*>(temporary_buffers.back()->data());
        temporary_buffers.push_back(std::make_unique<DeviceInputBuffer>());
        RETURN_IF_ERROR(temporary_buffers.back()->Bind(w2[i], stream));
        w2_data[i] =
            static_cast<const float*>(temporary_buffers.back()->data());
        temporary_buffers.push_back(std::make_unique<DeviceInputBuffer>());
        RETURN_IF_ERROR(temporary_buffers.back()->Bind(w3[i], stream));
        w3_data[i] =
            static_cast<const float*>(temporary_buffers.back()->data());
      }
      temporary_buffers.push_back(std::make_unique<DeviceInputBuffer>());
      RETURN_IF_ERROR(temporary_buffers.back()->Bind(bias, stream));
      bias_data = static_cast<const float*>(temporary_buffers.back()->data());
    }

    ParallelEinsumActivationRuntimeBuffers& runtime_buffers =
        ThreadLocalRuntimeBuffersForStream(this, stream);
    RETURN_IF_ERROR(runtime_buffers.stage1.Resize(
        static_cast<size_t>(batch * kParallelEinsumActivationHeads *
                            hidden_dim) *
            sizeof(float),
        stream));
    if (!constants_are_initializers) {
      RETURN_IF_ERROR(runtime_buffers.dynamic_packed_w1.Resize(
          static_cast<size_t>(input_dim * kParallelEinsumActivationHeads *
                              hidden_dim) *
              sizeof(float),
          stream));
      packed_w1_data = runtime_buffers.dynamic_packed_w1.data<float>();
    }

    if (!constants_are_initializers) {
      RETURN_IF_ERROR(
          LaunchStatus(LaunchMusaParallelEinsumActivationPackW1Kernel(
              w1_data[0], w1_data[1], w1_data[2], w1_data[3], packed_w1_data,
              input_dim, hidden_dim, stream)));
    }

    const std::vector<int64_t> mlp_matmul_shape = {batch, input_dim};
    const std::vector<int64_t> stage1_weight_shape = {
        input_dim,
        static_cast<int64_t>(kParallelEinsumActivationHeads) * hidden_dim};
    const std::vector<int64_t> stage1_shape = {
        batch,
        static_cast<int64_t>(kParallelEinsumActivationHeads) * hidden_dim};
    RETURN_IF_ERROR(ComputeMusaMatMulDevice(
        static_cast<const float*>(mlp_input_buffer.data()), packed_w1_data,
        runtime_buffers.stage1.data<float>(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, mlp_matmul_shape,
        stage1_weight_shape, stage1_shape, false, false, false, false, 1.0f,
        stream));
    MusaBroadcastParams stage1_params =
        MakeBroadcastParams(stage1_shape, stage1_shape, {1});
    RETURN_IF_ERROR(LaunchStatus(LaunchMusaGemmPostFloatKernel(
        runtime_buffers.stage1.data<float>(), nullptr, stage1_params, false,
        0.0f, MusaUnaryOp::Tanh, true, 0.0f, stream)));

    return LaunchStatus(LaunchMusaParallelEinsumActivationStage23Kernel(
        runtime_buffers.stage1.data<float>(),
        static_cast<const float*>(gate_input_buffer.data()), w2_data[0],
        w2_data[1], w2_data[2], w2_data[3], w3_data[0], w3_data[1], w3_data[2],
        w3_data[3], bias_data, output.GetTensorMutableData<float>(), batch,
        input_dim, hidden_dim, stream));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}
