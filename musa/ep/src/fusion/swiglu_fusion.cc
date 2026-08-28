// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "fusion/swiglu_fusion.h"

#include <musa_runtime.h>

#include <atomic>
#include <climits>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph/graph_utils.h"
#include "kernels/math/matmul.h"
#include "kernels/shared_inc/blas_utils.h"
#include "kernels/shared_inc/kernel_element_types.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "plugin_ep_utils.h"

namespace musa_ep {
namespace {

constexpr size_t kSwiGluMaxInputRank = 8;

struct ParsedSwiGluGraph {
  Ort::ConstValueInfo input{nullptr};
  Ort::ConstValueInfo gate_weight{nullptr};
  Ort::ConstValueInfo up_weight{nullptr};
  Ort::ConstValueInfo output{nullptr};
};

bool InputsMatch(Ort::ConstNode node, const std::string& lhs,
                 const std::string& rhs) {
  const auto inputs = node.GetInputs();
  return inputs.size() == 2 &&
         ((Name(inputs[0]) == lhs && Name(inputs[1]) == rhs) ||
          (Name(inputs[0]) == rhs && Name(inputs[1]) == lhs));
}

bool ParseSwiGluGraph(Ort::ConstGraph graph, ParsedSwiGluGraph& parsed) {
  std::vector<Ort::ConstNode> matmuls;
  std::vector<Ort::ConstNode> muls;
  Ort::ConstNode sigmoid{nullptr};
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "MatMul")) {
      matmuls.push_back(node);
    } else if (IsOnnxOp(node, "Sigmoid")) {
      if (sigmoid) {
        return false;
      }
      sigmoid = node;
    } else if (IsOnnxOp(node, "Mul")) {
      muls.push_back(node);
    } else {
      return false;
    }
  }
  if (matmuls.size() != 2 || muls.size() != 2 || !sigmoid) {
    return false;
  }

  const auto sigmoid_inputs = sigmoid.GetInputs();
  const auto sigmoid_outputs = sigmoid.GetOutputs();
  if (sigmoid_inputs.size() != 1 || sigmoid_outputs.size() != 1) {
    return false;
  }
  const std::string gate_name = Name(sigmoid_inputs[0]);
  const std::string sigmoid_name = Name(sigmoid_outputs[0]);

  Ort::ConstNode gate_matmul{nullptr};
  Ort::ConstNode up_matmul{nullptr};
  for (Ort::ConstNode matmul : matmuls) {
    const auto inputs = matmul.GetInputs();
    const auto outputs = matmul.GetOutputs();
    if (inputs.size() != 2 || outputs.size() != 1 ||
        !inputs[1].IsConstantInitializer()) {
      return false;
    }
    if (Name(outputs[0]) == gate_name) {
      gate_matmul = matmul;
    } else {
      up_matmul = matmul;
    }
  }
  if (!gate_matmul || !up_matmul) {
    return false;
  }

  Ort::ConstNode gate_mul{nullptr};
  for (Ort::ConstNode mul : muls) {
    if (InputsMatch(mul, gate_name, sigmoid_name)) {
      gate_mul = mul;
    }
  }
  if (!gate_mul || gate_mul.GetOutputs().size() != 1) {
    return false;
  }
  const std::string gate_mul_name = Name(gate_mul.GetOutputs()[0]);
  const auto up_outputs = up_matmul.GetOutputs();
  if (up_outputs.size() != 1) {
    return false;
  }
  const std::string up_name = Name(up_outputs[0]);

  Ort::ConstNode output_mul{nullptr};
  for (Ort::ConstNode mul : muls) {
    if (mul.GetId() != gate_mul.GetId() &&
        InputsMatch(mul, gate_mul_name, up_name)) {
      output_mul = mul;
    }
  }
  if (!output_mul || output_mul.GetOutputs().size() != 1) {
    return false;
  }

  const auto gate_inputs = gate_matmul.GetInputs();
  const auto up_inputs = up_matmul.GetInputs();
  if (Name(gate_inputs[0]) != Name(up_inputs[0])) {
    return false;
  }
  const auto gate_shape = GetStaticShape(gate_inputs[1]);
  const auto up_shape = GetStaticShape(up_inputs[1]);
  if (!gate_shape.has_value() || gate_shape != up_shape ||
      gate_shape->size() != 2 || (*gate_shape)[0] <= 0 ||
      (*gate_shape)[1] <= 0) {
    return false;
  }
  const auto input_shape = GetTensorShape(gate_inputs[0]);
  if (!input_shape.has_value() || input_shape->size() < 2 ||
      input_shape->size() > kSwiGluMaxInputRank ||
      (input_shape->back() != -1 && input_shape->back() != (*gate_shape)[0])) {
    return false;
  }

  const auto elem_type = GetTensorElementType(gate_inputs[0]);
  if (!elem_type.has_value() ||
      (*elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
       *elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
       *elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) {
    return false;
  }
  for (Ort::ConstValueInfo value :
       {gate_inputs[1], up_inputs[1], gate_matmul.GetOutputs()[0],
        up_matmul.GetOutputs()[0], sigmoid_outputs[0], gate_mul.GetOutputs()[0],
        output_mul.GetOutputs()[0]}) {
    if (GetTensorElementType(value) != elem_type) {
      return false;
    }
  }

  parsed = {gate_inputs[0], gate_inputs[1], up_inputs[1],
            output_mul.GetOutputs()[0]};
  return true;
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
  const auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error("SwiGlu could not map fused value " + name);
  }
  return it->second;
}

std::vector<uint8_t> ReadInitializerBytes(
    Ort::ConstValueInfo value_info, ONNXTensorElementDataType expected_type,
    const std::vector<int64_t>& expected_shape) {
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("SwiGlu failed to read initializer " +
                             Name(value_info));
  }
  const auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != expected_type ||
      info.GetShape() != expected_shape) {
    throw std::runtime_error("SwiGlu initializer metadata mismatch " +
                             Name(value_info));
  }
  const size_t bytes =
      static_cast<size_t>(info.GetElementCount()) * ElementSize(expected_type);
  const auto* data = static_cast<const uint8_t*>(value.GetTensorRawData());
  return std::vector<uint8_t>(data, data + bytes);
}

int64_t NumElementsChecked(const std::vector<int64_t>& shape) {
  int64_t total = 1;
  bool empty = false;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw std::runtime_error("SwiGlu requires non-negative runtime shapes");
    }
    if (dim == 0) {
      empty = true;
      continue;
    }
    if (total > INT64_MAX / dim) {
      throw std::runtime_error("SwiGlu runtime shape overflows int64");
    }
    total *= dim;
  }
  return empty ? 0 : total;
}

class DeviceBuffer {
 public:
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)musaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer() = default;

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
    ptr_ = AllocateDeviceMemoryOnStream(bytes, stream_);
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

struct SwiGluScratch {
  DeviceBuffer packed_output;
  std::mutex execution_mutex;
};

OrtStatus* MudnnStatus(::musa::dnn::Status status, const char* message) {
  if (status == ::musa::dnn::Status::SUCCESS) {
    return nullptr;
  }
  const std::string error = std::string(message) + ", status=" +
                            std::to_string(static_cast<int>(status));
  return Ort::GetApi().CreateStatus(ORT_EP_FAIL, error.c_str());
}

}  // namespace

struct SwiGluFusionCompute::Impl {
  Impl(size_t input_index, size_t output_index,
       ONNXTensorElementDataType elem_type, std::vector<int64_t> weight_shape,
       std::vector<uint8_t> packed_weights)
      : input_index(input_index),
        output_index(output_index),
        elem_type(elem_type),
        weight_shape(std::move(weight_shape)),
        host_packed_weights(std::move(packed_weights)) {}

  ~Impl() {
    if (weights_ready_event != nullptr) {
      (void)musaEventDestroy(weights_ready_event);
    }
  }

  OrtStatus* EnsureWeights(musaStream_t stream) const {
    if (weights_ready.load(std::memory_order_acquire)) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(weights_mutex);
    if (weights_ready.load(std::memory_order_relaxed)) {
      return nullptr;
    }
    RETURN_IF_ERROR(
        device_packed_weights.Resize(host_packed_weights.size(), stream));
    RETURN_IF_ERROR(CopyTemporaryHostToDevice(
        device_packed_weights.data<void>(), host_packed_weights.data(),
        host_packed_weights.size(), stream));
    RETURN_IF_ERROR(LaunchStatus(musaEventCreateWithFlags(
        &weights_ready_event, musaEventDisableTiming)));
    RETURN_IF_ERROR(LaunchStatus(musaEventRecord(weights_ready_event, stream)));
    RETURN_IF_ERROR(LaunchStatus(musaEventSynchronize(weights_ready_event)));
    RETURN_IF_ERROR(LaunchStatus(musaEventDestroy(weights_ready_event)));
    weights_ready_event = nullptr;
    std::vector<uint8_t>().swap(host_packed_weights);
    weights_ready.store(true, std::memory_order_release);
    return nullptr;
  }

  SwiGluScratch& ScratchForStream(musaStream_t stream) const {
    std::lock_guard<std::mutex> lock(scratch_mutex);
    auto& scratch = scratch_by_stream[stream];
    if (!scratch) {
      scratch = std::make_unique<SwiGluScratch>();
    }
    return *scratch;
  }

  size_t input_index;
  size_t output_index;
  ONNXTensorElementDataType elem_type;
  std::vector<int64_t> weight_shape;
  mutable std::vector<uint8_t> host_packed_weights;
  mutable DeviceBuffer device_packed_weights;
  mutable musaEvent_t weights_ready_event = nullptr;
  mutable std::atomic<bool> weights_ready{false};
  mutable std::mutex weights_mutex;
  mutable std::mutex scratch_mutex;
  mutable std::unordered_map<musaStream_t, std::unique_ptr<SwiGluScratch>>
      scratch_by_stream;
};

SwiGluFusionCompute::SwiGluFusionCompute(size_t input_index,
                                         size_t output_index,
                                         ONNXTensorElementDataType elem_type,
                                         std::vector<int64_t> weight_shape,
                                         std::vector<uint8_t> packed_weights)
    : impl_(std::make_unique<Impl>(input_index, output_index, elem_type,
                                   std::move(weight_shape),
                                   std::move(packed_weights))) {}

SwiGluFusionCompute::~SwiGluFusionCompute() = default;

OrtStatus* SwiGluFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    musaStream_t stream = GetComputeStream(ctx);
    Ort::ConstValue input = ctx.GetInput(impl_->input_index);
    const auto input_info = input.GetTensorTypeAndShapeInfo();
    const auto runtime_elem_type = input_info.GetElementType();
    const auto input_shape = input_info.GetShape();
    if (runtime_elem_type != impl_->elem_type) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "SwiGlu input dtype changed after compilation");
    }
    if (input_shape.size() < 2 || impl_->weight_shape.size() != 2 ||
        input_shape.back() != impl_->weight_shape[0]) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "SwiGlu projection shape mismatch");
    }
    if (runtime_elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        runtime_elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
        runtime_elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "SwiGlu only supports float32/float16/bfloat16");
    }

    const int64_t k = impl_->weight_shape[0];
    const int64_t hidden = impl_->weight_shape[1];
    const int64_t input_elements = NumElementsChecked(input_shape);
    const int64_t rows = input_elements == 0 ? 0 : input_elements / k;
    std::vector<int64_t> output_shape = input_shape;
    output_shape.back() = hidden;
    Ort::UnownedValue output = ctx.GetOutput(impl_->output_index, output_shape);
    if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "SwiGlu requires MUSA output");
    }
    if (output.GetTensorTypeAndShapeInfo().GetElementType() !=
        runtime_elem_type) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "SwiGlu output dtype must match input");
    }
    if (rows == 0) {
      return nullptr;
    }

    DeviceInputBuffer input_buffer;
    RETURN_IF_ERROR(input_buffer.Bind(input, stream));
    RETURN_IF_ERROR(impl_->EnsureWeights(stream));

    if (hidden > INT64_MAX / 2) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "SwiGlu packed width overflows int64");
    }
    const int64_t packed_hidden = hidden * 2;
    const size_t elem_size = ElementSize(runtime_elem_type);
    if (elem_size == 0 ||
        (rows > 0 && static_cast<uint64_t>(rows) >
                         std::numeric_limits<size_t>::max() /
                             static_cast<size_t>(packed_hidden) / elem_size)) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "SwiGlu packed buffer size overflow");
    }
    const size_t packed_bytes = static_cast<size_t>(rows) *
                                static_cast<size_t>(packed_hidden) * elem_size;
    SwiGluScratch& scratch = impl_->ScratchForStream(stream);
    std::lock_guard<std::mutex> execution_lock(scratch.execution_mutex);
    RETURN_IF_ERROR(scratch.packed_output.Resize(packed_bytes, stream));

    std::vector<int64_t> packed_weight_shape = {k, packed_hidden};
    std::vector<int64_t> packed_output_shape = input_shape;
    packed_output_shape.back() = packed_hidden;
    RETURN_IF_ERROR(ComputeMusaMatMulDevice(
        input_buffer.data(), impl_->device_packed_weights.data<void>(),
        scratch.packed_output.data<void>(), runtime_elem_type, input_shape,
        packed_weight_shape, packed_output_shape, false, false, false, false,
        1.0f, stream));

    ::musa::dnn::Handle* handle = nullptr;
    RETURN_IF_ERROR(EnsureMudnnHandle(&handle, stream));
    ::musa::dnn::Tensor packed_tensor;
    ::musa::dnn::Tensor output_tensor;
    if (!SetMudnnTensor(packed_tensor, scratch.packed_output.data<void>(),
                        {rows, packed_hidden}, runtime_elem_type) ||
        !SetMudnnTensor(output_tensor, output.GetTensorMutableRawData(),
                        {rows, hidden}, runtime_elem_type)) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        "failed to setup muDNN SwiGlu tensors");
    }
    ::musa::dnn::SwiGlu op;
    return MudnnStatus(op.Run(*handle, output_tensor, packed_tensor),
                       "muDNN SwiGlu execution failed");
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsSwiGluFusionGraph(Ort::ConstGraph graph) {
  ParsedSwiGluGraph parsed;
  return ParseSwiGluGraph(graph, parsed);
}

std::unique_ptr<FusionNodeCompute> CreateSwiGluFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  ParsedSwiGluGraph parsed;
  if (!ParseSwiGluGraph(graph, parsed)) {
    throw std::runtime_error("invalid SwiGlu fusion graph");
  }
  const auto input_indices = ValueIndices(fused_node.GetInputs());
  const auto output_indices = ValueIndices(fused_node.GetOutputs());
  const size_t input_index = IndexOf(input_indices, Name(parsed.input));
  const size_t output_index = IndexOf(output_indices, Name(parsed.output));

  const auto elem_type = GetTensorElementType(parsed.input);
  const auto weight_shape = GetStaticShape(parsed.gate_weight);
  if (!elem_type.has_value() || !weight_shape.has_value() ||
      weight_shape->size() != 2) {
    throw std::runtime_error("SwiGlu projection metadata is invalid");
  }
  const size_t elem_size = ElementSize(*elem_type);
  const size_t k = static_cast<size_t>((*weight_shape)[0]);
  const size_t hidden = static_cast<size_t>((*weight_shape)[1]);
  if (hidden > SIZE_MAX / 2 || k > SIZE_MAX / (hidden * 2) ||
      k * hidden * 2 > SIZE_MAX / elem_size) {
    throw std::runtime_error("SwiGlu packed weight size overflowed");
  }

  const std::vector<uint8_t> gate =
      ReadInitializerBytes(parsed.gate_weight, *elem_type, *weight_shape);
  const std::vector<uint8_t> up =
      ReadInitializerBytes(parsed.up_weight, *elem_type, *weight_shape);
  const size_t row_bytes = hidden * elem_size;
  const size_t packed_row_bytes = row_bytes * 2;
  std::vector<uint8_t> packed(k * packed_row_bytes);
  for (size_t row = 0; row < k; ++row) {
    std::memcpy(packed.data() + row * packed_row_bytes,
                gate.data() + row * row_bytes, row_bytes);
    std::memcpy(packed.data() + row * packed_row_bytes + row_bytes,
                up.data() + row * row_bytes, row_bytes);
  }

  return std::make_unique<SwiGluFusionCompute>(
      input_index, output_index, *elem_type, *weight_shape, std::move(packed));
}

}  // namespace musa_ep
