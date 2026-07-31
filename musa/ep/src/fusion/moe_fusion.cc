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

#include "fusion/moe_fusion.h"

#include <musa_runtime.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "kernels/math/matmul.h"
#include "kernels/nn/moe_impl.h"
#include "kernels/shared_inc/op_kernel_common.h"

namespace {

std::string Name(Ort::ConstValueInfo value_info) {
  return value_info.GetName();
}

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(Ort::ConstNode node, const char* op_type) {
  return node && node.GetOperatorType() == op_type &&
         IsOnnxDomain(node.GetDomain());
}

std::vector<int64_t> TensorShape(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

void ValidateFloat(Ort::ConstValue value, const char* name) {
  if (value.GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(std::string("MoE requires float ") + name);
  }
}

std::unordered_map<std::string, size_t> ValueIndices(
    const std::vector<Ort::ConstValueInfo>& values) {
  std::unordered_map<std::string, size_t> indices;
  for (size_t i = 0; i < values.size(); ++i) {
    indices.emplace(Name(values[i]), i);
  }
  return indices;
}

size_t IndexOf(const std::unordered_map<std::string, size_t>& indices,
               const std::string& name) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error("MoE could not map fused value " + name);
  }
  return it->second;
}

std::vector<float> ReadFloatInitializer(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    throw std::runtime_error("MoE weights and biases must be initializers");
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("MoE failed to read initializer " +
                             Name(value_info));
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("MoE initializer is not float " +
                             Name(value_info));
  }
  const size_t count = static_cast<size_t>(info.GetElementCount());
  const float* data = value.GetTensorData<float>();
  return std::vector<float>(data, data + count);
}

void AppendBias(Ort::ConstValueInfo value_info, int64_t width,
                std::vector<float>& packed) {
  std::vector<float> values = ReadFloatInitializer(value_info);
  if (width < 0 || values.size() != static_cast<size_t>(width)) {
    throw std::runtime_error("MoE bias size mismatch for " + Name(value_info));
  }
  packed.insert(packed.end(), values.begin(), values.end());
}

void AppendWeight(Ort::ConstValueInfo value_info, bool transpose,
                  int64_t input_width, int64_t output_width,
                  std::vector<float>& packed) {
  std::vector<float> values = ReadFloatInitializer(value_info);
  if (input_width <= 0 || output_width <= 0 ||
      static_cast<uint64_t>(input_width) *
              static_cast<uint64_t>(output_width) !=
          values.size()) {
    throw std::runtime_error("MoE weight size mismatch for " +
                             Name(value_info));
  }
  if (!transpose) {
    packed.insert(packed.end(), values.begin(), values.end());
    return;
  }

  // MatMul stores [K, N], while the batched runtime matmul consumes a packed
  // expert-major [E, N, K] tensor with transB=true.
  for (int64_t output = 0; output < output_width; ++output) {
    for (int64_t input = 0; input < input_width; ++input) {
      packed.push_back(
          values[static_cast<size_t>(input * output_width + output)]);
    }
  }
}

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
    stream_ = stream;
    if (bytes == 0) {
      return nullptr;
    }
    ptr_ = AllocateDeviceMemoryOnStream(bytes, stream);
    if (ptr_ == nullptr) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL, MusaErrorString(musaErrorMemoryAllocation));
    }
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

struct MoEDeviceConstants {
  ~MoEDeviceConstants() {
    if (ready_event != nullptr) {
      (void)musaEventDestroy(ready_event);
    }
  }

  DeviceBuffer first_weights;
  DeviceBuffer first_biases;
  DeviceBuffer second_weights;
  DeviceBuffer second_biases;
  musaEvent_t ready_event = nullptr;
  int device_id = -1;
  bool ready = false;
};

struct MoERuntimeScratch {
  DeviceBuffer hidden;
  DeviceBuffer expert_output;
  DeviceBuffer pointer_arrays;
};

MoERuntimeScratch& ScratchForStream(const void* owner, musaStream_t stream) {
  using StreamMap =
      std::unordered_map<musaStream_t, std::unique_ptr<MoERuntimeScratch>>;
  thread_local std::unordered_map<const void*, StreamMap> scratch_by_owner;
  auto& scratch = scratch_by_owner[owner][stream];
  if (!scratch) {
    scratch = std::make_unique<MoERuntimeScratch>();
  }
  return *scratch;
}

size_t CheckedBytes(int64_t a, int64_t b, int64_t c) {
  if (a < 0 || b < 0 || c < 0) {
    throw std::runtime_error("MoE runtime dimensions must be non-negative");
  }
  const size_t sa = static_cast<size_t>(a);
  const size_t sb = static_cast<size_t>(b);
  const size_t sc = static_cast<size_t>(c);
  if ((sb != 0 && sa > std::numeric_limits<size_t>::max() / sb) ||
      (sc != 0 && sa * sb > std::numeric_limits<size_t>::max() / sc) ||
      sa * sb * sc > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("MoE runtime tensor size overflowed");
  }
  return sa * sb * sc * sizeof(float);
}

int64_t FlattenedRows(const std::vector<int64_t>& shape) {
  if (shape.size() < 2) {
    throw std::runtime_error("MoE x rank must be at least 2");
  }
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) {
    const int64_t dim = shape[i];
    if (dim < 0 ||
        (dim != 0 && rows > std::numeric_limits<int64_t>::max() / dim)) {
      throw std::runtime_error("MoE x prefix dimensions are invalid");
    }
    rows *= dim;
  }
  return rows;
}

enum class ParsedLinearKind { Gemm, MatMulAdd };

struct ParsedLinear {
  ParsedLinearKind kind = ParsedLinearKind::Gemm;
  Ort::ConstNode linear{nullptr};
  Ort::ConstValueInfo input{nullptr};
  Ort::ConstValueInfo weight{nullptr};
  Ort::ConstValueInfo bias{nullptr};
};

ParsedLinear ParseLinear(Ort::ConstValueInfo output) {
  Ort::ConstNode producer{nullptr};
  if (!musa_ep::GetProducer(output, producer)) {
    throw std::runtime_error("MoE linear output has no producer");
  }
  if (IsOnnxOp(producer, "Gemm")) {
    auto inputs = producer.GetInputs();
    if (inputs.size() != 3) {
      throw std::runtime_error("MoE Gemm input count mismatch");
    }
    return {ParsedLinearKind::Gemm, producer, inputs[0], inputs[1], inputs[2]};
  }
  if (!IsOnnxOp(producer, "Add")) {
    throw std::runtime_error("MoE expected Gemm or MatMul + Add");
  }

  auto add_inputs = producer.GetInputs();
  Ort::ConstNode matmul{nullptr};
  int64_t matmul_input_index = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(add_inputs.size()); ++i) {
    Ort::ConstNode input_producer{nullptr};
    if (musa_ep::GetProducer(add_inputs[static_cast<size_t>(i)],
                             input_producer) &&
        IsOnnxOp(input_producer, "MatMul")) {
      matmul = input_producer;
      matmul_input_index = i;
      break;
    }
  }
  if (!matmul || add_inputs.size() != 2 || matmul_input_index < 0) {
    throw std::runtime_error("MoE MatMul + Add branch is invalid");
  }
  auto matmul_inputs = matmul.GetInputs();
  if (matmul_inputs.size() != 2) {
    throw std::runtime_error("MoE MatMul input count mismatch");
  }
  return {ParsedLinearKind::MatMulAdd, matmul, matmul_inputs[0],
          matmul_inputs[1],
          add_inputs[static_cast<size_t>(1 - matmul_input_index)]};
}

struct ParsedMoEBranch {
  ParsedLinear first;
  ParsedLinear second;
};

struct ParsedMoERouter {
  Ort::ConstValueInfo router{nullptr};
  Ort::ConstValueInfo output{nullptr};
};

struct ParsedMoEGraph {
  Ort::ConstValueInfo x{nullptr};
  std::vector<ParsedMoEBranch> branches;
  std::vector<ParsedMoERouter> routers;
};

ParsedMoEGraph ParseMoEGraph(Ort::ConstGraph graph) {
  ParsedMoEGraph parsed;
  Ort::ConstNode concat{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Concat")) {
      if (concat) {
        throw std::runtime_error("MoE fusion graph has multiple Concat nodes");
      }
      concat = node;
    }
  }
  if (!concat) {
    throw std::runtime_error("MoE fusion graph is missing Concat");
  }

  for (Ort::ConstValueInfo concat_input : concat.GetInputs()) {
    Ort::ConstNode unsqueeze{nullptr};
    Ort::ConstNode second_relu{nullptr};
    Ort::ConstNode first_relu{nullptr};
    if (!musa_ep::GetProducer(concat_input, unsqueeze) ||
        !musa_ep::GetProducer(unsqueeze.GetInputs()[0], second_relu)) {
      throw std::runtime_error("MoE fusion graph has an invalid expert tail");
    }
    ParsedLinear second = ParseLinear(second_relu.GetInputs()[0]);
    if (!musa_ep::GetProducer(second.input, first_relu)) {
      throw std::runtime_error("MoE fusion graph has an invalid first Relu");
    }
    ParsedLinear first = ParseLinear(first_relu.GetInputs()[0]);
    if (!parsed.x) {
      parsed.x = first.input;
    } else if (Name(parsed.x) != Name(first.input)) {
      throw std::runtime_error("MoE expert branches do not share x");
    }
    parsed.branches.push_back({std::move(first), std::move(second)});
  }

  auto concat_outputs = concat.GetOutputs();
  if (concat_outputs.size() != 1) {
    throw std::runtime_error("MoE Concat output count mismatch");
  }
  for (const auto& consumer : concat_outputs[0].GetConsumers()) {
    Ort::ConstNode mul = consumer.node;
    auto mul_inputs = mul.GetInputs();
    auto mul_outputs = mul.GetOutputs();
    if (mul_inputs.size() != 2 || mul_outputs.size() != 1 ||
        consumer.index < 0 || consumer.index > 1) {
      throw std::runtime_error("MoE router Mul is invalid");
    }
    auto mul_consumers = mul_outputs[0].GetConsumers();
    if (mul_consumers.size() != 1) {
      throw std::runtime_error("MoE router Mul must have one ReduceSum");
    }
    Ort::ConstNode reduce_sum = mul_consumers[0].node;
    auto reduce_outputs = reduce_sum.GetOutputs();
    if (reduce_outputs.size() != 1) {
      throw std::runtime_error("MoE ReduceSum output count mismatch");
    }
    parsed.routers.push_back(
        {mul_inputs[static_cast<size_t>(1 - consumer.index)],
         reduce_outputs[0]});
  }
  if (!parsed.x || parsed.branches.size() < 2 || parsed.routers.empty()) {
    throw std::runtime_error("MoE fusion graph is incomplete");
  }
  return parsed;
}

struct LinearDimensions {
  int64_t input_width = 0;
  int64_t output_width = 0;
};

LinearDimensions Dimensions(const ParsedLinear& linear) {
  auto shape = musa_ep::GetStaticShape(linear.weight);
  if (!shape.has_value() || shape->size() != 2) {
    throw std::runtime_error("MoE weight shape is not static");
  }
  return linear.kind == ParsedLinearKind::Gemm
             ? LinearDimensions{(*shape)[1], (*shape)[0]}
             : LinearDimensions{(*shape)[0], (*shape)[1]};
}

}  // namespace

struct MoEFusionCompute : FusionNodeCompute {
  MoEFusionCompute(size_t x_input_index,
                   std::vector<size_t> router_input_indices,
                   std::vector<size_t> output_indices, int64_t expert_count,
                   int64_t input_width, int64_t hidden_width,
                   int64_t output_width, std::vector<float> first_weights,
                   std::vector<float> first_biases,
                   std::vector<float> second_weights,
                   std::vector<float> second_biases)
      : x_input_index(x_input_index),
        router_input_indices(std::move(router_input_indices)),
        output_indices(std::move(output_indices)),
        expert_count(expert_count),
        input_width(input_width),
        hidden_width(hidden_width),
        output_width(output_width),
        host_first_weights(std::move(first_weights)),
        host_first_biases(std::move(first_biases)),
        host_second_weights(std::move(second_weights)),
        host_second_biases(std::move(second_biases)) {}

  OrtStatus* EnsureConstants(musaStream_t stream) const {
    std::lock_guard<std::mutex> lock(constants_mutex);
    int current_device = -1;
    musaError_t device_status = musaGetDevice(&current_device);
    if (device_status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        MusaErrorString(device_status));
    }
    if (device_constants.ready) {
      if (device_constants.device_id != current_device) {
        return Ort::GetApi().CreateStatus(
            ORT_EP_FAIL, "MoE constants cannot be shared across MUSA devices");
      }
      return LaunchStatus(
          musaStreamWaitEvent(stream, device_constants.ready_event, 0));
    }

    const size_t first_weight_bytes = host_first_weights.size() * sizeof(float);
    const size_t first_bias_bytes = host_first_biases.size() * sizeof(float);
    const size_t second_weight_bytes =
        host_second_weights.size() * sizeof(float);
    const size_t second_bias_bytes = host_second_biases.size() * sizeof(float);
    RETURN_IF_ERROR(
        device_constants.first_weights.Resize(first_weight_bytes, stream));
    RETURN_IF_ERROR(
        device_constants.first_biases.Resize(first_bias_bytes, stream));
    RETURN_IF_ERROR(
        device_constants.second_weights.Resize(second_weight_bytes, stream));
    RETURN_IF_ERROR(
        device_constants.second_biases.Resize(second_bias_bytes, stream));
    RETURN_IF_ERROR(CopyTemporaryHostToDevice(
        device_constants.first_weights.data<float>(), host_first_weights.data(),
        first_weight_bytes, stream));
    RETURN_IF_ERROR(CopyTemporaryHostToDevice(
        device_constants.first_biases.data<float>(), host_first_biases.data(),
        first_bias_bytes, stream));
    RETURN_IF_ERROR(CopyTemporaryHostToDevice(
        device_constants.second_weights.data<float>(),
        host_second_weights.data(), second_weight_bytes, stream));
    RETURN_IF_ERROR(CopyTemporaryHostToDevice(
        device_constants.second_biases.data<float>(), host_second_biases.data(),
        second_bias_bytes, stream));
    if (device_constants.ready_event == nullptr) {
      RETURN_IF_ERROR(LaunchStatus(musaEventCreateWithFlags(
          &device_constants.ready_event, musaEventDisableTiming)));
    }
    RETURN_IF_ERROR(
        LaunchStatus(musaEventRecord(device_constants.ready_event, stream)));
    device_constants.device_id = current_device;
    device_constants.ready = true;
    std::vector<float>().swap(host_first_weights);
    std::vector<float>().swap(host_first_biases);
    std::vector<float>().swap(host_second_weights);
    std::vector<float>().swap(host_second_biases);
    return nullptr;
  }

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      musaStream_t stream = GetComputeStream(ctx);
      Ort::ConstValue x = ctx.GetInput(x_input_index);
      ValidateFloat(x, "x");
      const std::vector<int64_t> x_shape = TensorShape(x);
      if (x_shape.size() < 2 || x_shape.back() != input_width) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "MoE runtime x shape mismatch");
      }
      const int64_t rows = FlattenedRows(x_shape);
      std::vector<int64_t> output_shape = x_shape;
      output_shape.back() = output_width;

      std::vector<Ort::ConstValue> routers;
      std::vector<float*> output_pointers;
      routers.reserve(router_input_indices.size());
      output_pointers.reserve(output_indices.size());
      for (size_t i = 0; i < router_input_indices.size(); ++i) {
        Ort::ConstValue router = ctx.GetInput(router_input_indices[i]);
        ValidateFloat(router, "router");
        const std::vector<int64_t> router_shape = TensorShape(router);
        bool prefix_matches = router_shape.size() == x_shape.size() + 1;
        for (size_t dim = 0; prefix_matches && dim + 1 < x_shape.size();
             ++dim) {
          prefix_matches = router_shape[dim] == x_shape[dim];
        }
        const size_t expert_axis = x_shape.size() - 1;
        const bool expert_before_output =
            prefix_matches && router_shape[expert_axis] == expert_count &&
            router_shape[expert_axis + 1] == 1;
        const bool expert_after_output =
            prefix_matches && router_shape[expert_axis] == 1 &&
            router_shape[expert_axis + 1] == expert_count;
        if (!expert_before_output && !expert_after_output) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT, "MoE runtime router shape mismatch");
        }
        routers.push_back(router);

        Ort::UnownedValue output =
            ctx.GetOutput(output_indices[i], output_shape);
        if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
          return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                            "MoE requires MUSA outputs");
        }
        output_pointers.push_back(output.GetTensorMutableData<float>());
      }
      if (rows == 0) {
        return nullptr;
      }

      DeviceInputBuffer x_buffer;
      RETURN_IF_ERROR(x_buffer.Bind(x, stream));
      std::vector<std::unique_ptr<DeviceInputBuffer>> router_buffers;
      std::vector<const float*> router_pointers;
      router_buffers.reserve(routers.size());
      router_pointers.reserve(routers.size());
      for (Ort::ConstValue router : routers) {
        auto buffer = std::make_unique<DeviceInputBuffer>();
        RETURN_IF_ERROR(buffer->Bind(router, stream));
        router_pointers.push_back(static_cast<const float*>(buffer->data()));
        router_buffers.push_back(std::move(buffer));
      }
      RETURN_IF_ERROR(EnsureConstants(stream));

      MoERuntimeScratch& scratch = ScratchForStream(this, stream);
      RETURN_IF_ERROR(scratch.hidden.Resize(
          CheckedBytes(expert_count, rows, hidden_width), stream));
      RETURN_IF_ERROR(scratch.expert_output.Resize(
          CheckedBytes(expert_count, rows, output_width), stream));

      const std::vector<int64_t> first_input_shape = {1, rows, input_width};
      const std::vector<int64_t> first_weight_shape = {
          expert_count, hidden_width, input_width};
      const std::vector<int64_t> hidden_shape = {expert_count, rows,
                                                 hidden_width};
      RETURN_IF_ERROR(ComputeMusaMatMulDevice(
          x_buffer.data(), device_constants.first_weights.data<float>(),
          scratch.hidden.data<float>(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
          first_input_shape, first_weight_shape, hidden_shape, false, true,
          false, false, 1.0f, stream));
      RETURN_IF_ERROR(LaunchStatus(LaunchMoEBiasReluFloatKernel(
          scratch.hidden.data<float>(),
          device_constants.first_biases.data<float>(), rows, expert_count,
          hidden_width, stream)));

      const std::vector<int64_t> second_weight_shape = {
          expert_count, output_width, hidden_width};
      const std::vector<int64_t> expert_output_shape = {expert_count, rows,
                                                        output_width};
      RETURN_IF_ERROR(ComputeMusaMatMulDevice(
          scratch.hidden.data<float>(),
          device_constants.second_weights.data<float>(),
          scratch.expert_output.data<float>(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, hidden_shape,
          second_weight_shape, expert_output_shape, false, true, false, false,
          1.0f, stream));

      if (router_pointers.size() >
              std::numeric_limits<size_t>::max() / (2 * sizeof(void*)) ||
          router_pointers.size() >
              static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "MoE router count overflowed");
      }
      const int64_t router_count = static_cast<int64_t>(router_pointers.size());
      (void)CheckedBytes(router_count, rows, output_width);
      const size_t pointer_bytes = router_pointers.size() * 2 * sizeof(void*);
      RETURN_IF_ERROR(scratch.pointer_arrays.Resize(pointer_bytes, stream));
      std::vector<const void*> host_pointers;
      host_pointers.reserve(router_pointers.size() * 2);
      for (const float* router : router_pointers) {
        host_pointers.push_back(router);
      }
      for (float* output : output_pointers) {
        host_pointers.push_back(output);
      }
      RETURN_IF_ERROR(CopyTemporaryHostToDevice(
          scratch.pointer_arrays.data<void*>(), host_pointers.data(),
          pointer_bytes, stream));
      void** device_pointers = scratch.pointer_arrays.data<void*>();
      const float* const* device_routers =
          reinterpret_cast<const float* const*>(device_pointers);
      float* const* device_outputs = reinterpret_cast<float* const*>(
          device_pointers + router_pointers.size());
      return LaunchStatus(LaunchMoERouterReduceMultiFloatKernel(
          scratch.expert_output.data<float>(),
          device_constants.second_biases.data<float>(), device_routers,
          device_outputs, rows, expert_count, output_width, router_count,
          stream));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t x_input_index;
  std::vector<size_t> router_input_indices;
  std::vector<size_t> output_indices;
  int64_t expert_count;
  int64_t input_width;
  int64_t hidden_width;
  int64_t output_width;
  mutable std::vector<float> host_first_weights;
  mutable std::vector<float> host_first_biases;
  mutable std::vector<float> host_second_weights;
  mutable std::vector<float> host_second_biases;
  mutable MoEDeviceConstants device_constants;
  mutable std::mutex constants_mutex;
};

bool IsMoEFusionGraph(Ort::ConstGraph graph) {
  size_t gemm_count = 0;
  size_t matmul_count = 0;
  size_t add_count = 0;
  size_t relu_count = 0;
  size_t unsqueeze_count = 0;
  size_t concat_count = 0;
  size_t mul_count = 0;
  size_t reduce_sum_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Gemm")) {
      ++gemm_count;
    } else if (IsOnnxOp(node, "MatMul")) {
      ++matmul_count;
    } else if (IsOnnxOp(node, "Add")) {
      ++add_count;
    } else if (IsOnnxOp(node, "Relu")) {
      ++relu_count;
    } else if (IsOnnxOp(node, "Unsqueeze")) {
      ++unsqueeze_count;
    } else if (IsOnnxOp(node, "Concat")) {
      ++concat_count;
    } else if (IsOnnxOp(node, "Mul")) {
      ++mul_count;
    } else if (IsOnnxOp(node, "ReduceSum")) {
      ++reduce_sum_count;
    } else {
      return false;
    }
  }
  return unsqueeze_count >= 2 &&
         gemm_count + matmul_count == unsqueeze_count * 2 &&
         add_count == matmul_count && relu_count == unsqueeze_count * 2 &&
         concat_count == 1 && mul_count >= 1 && reduce_sum_count == mul_count;
}

std::unique_ptr<FusionNodeCompute> CreateMoEFusion(Ort::ConstGraph graph,
                                                   Ort::ConstNode fused_node) {
  ParsedMoEGraph parsed = ParseMoEGraph(graph);
  const int64_t expert_count = static_cast<int64_t>(parsed.branches.size());
  auto fused_input_indices = ValueIndices(fused_node.GetInputs());
  auto fused_output_indices = ValueIndices(fused_node.GetOutputs());

  std::vector<float> first_weights;
  std::vector<float> first_biases;
  std::vector<float> second_weights;
  std::vector<float> second_biases;
  int64_t input_width = 0;
  int64_t hidden_width = 0;
  int64_t output_width = 0;
  for (const ParsedMoEBranch& branch : parsed.branches) {
    const LinearDimensions first = Dimensions(branch.first);
    const LinearDimensions second = Dimensions(branch.second);
    if (first.output_width != second.input_width) {
      throw std::runtime_error("MoE expert hidden dimensions differ");
    }
    if (input_width == 0) {
      input_width = first.input_width;
      hidden_width = first.output_width;
      output_width = second.output_width;
    } else if (first.input_width != input_width ||
               first.output_width != hidden_width ||
               second.input_width != hidden_width ||
               second.output_width != output_width) {
      throw std::runtime_error("MoE expert weight shapes differ");
    }
    AppendWeight(branch.first.weight,
                 branch.first.kind == ParsedLinearKind::MatMulAdd, input_width,
                 hidden_width, first_weights);
    AppendBias(branch.first.bias, hidden_width, first_biases);
    AppendWeight(branch.second.weight,
                 branch.second.kind == ParsedLinearKind::MatMulAdd,
                 hidden_width, output_width, second_weights);
    AppendBias(branch.second.bias, output_width, second_biases);
  }

  std::vector<size_t> router_input_indices;
  std::vector<size_t> output_indices;
  router_input_indices.reserve(parsed.routers.size());
  output_indices.reserve(parsed.routers.size());
  for (const ParsedMoERouter& router : parsed.routers) {
    router_input_indices.push_back(
        IndexOf(fused_input_indices, Name(router.router)));
    output_indices.push_back(
        IndexOf(fused_output_indices, Name(router.output)));
  }

  return std::make_unique<MoEFusionCompute>(
      IndexOf(fused_input_indices, Name(parsed.x)),
      std::move(router_input_indices), std::move(output_indices), expert_count,
      input_width, hidden_width, output_width, std::move(first_weights),
      std::move(first_biases), std::move(second_weights),
      std::move(second_biases));
}
