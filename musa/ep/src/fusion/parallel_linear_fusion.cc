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

#include "fusion/parallel_linear_fusion.h"

#include <musa_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph/graph_utils.h"
#include "kernels/math/matmul.h"
#include "kernels/nn/parallel_linear_impl.h"
#include "kernels/shared_inc/blas_utils.h"
#include "kernels/shared_inc/kernel_element_types.h"
#include "kernels/shared_inc/op_kernel_common.h"

namespace {

struct BranchInfo {
  size_t bias_byte_offset;
  size_t weight_byte_offset;
  size_t output_index;
};

constexpr size_t kNoBiasInput = std::numeric_limits<size_t>::max();
constexpr size_t kNoOutputIndex = std::numeric_limits<size_t>::max();
constexpr size_t kNoWeightInput = std::numeric_limits<size_t>::max();

bool ParallelLinearGroupedMatMulDisabled() {
  const char* env =
      std::getenv("ORT_MUSA_DISABLE_PARALLEL_LINEAR_GROUPED_MATMUL");
  return env != nullptr && std::strcmp(env, "0") != 0;
}

struct GatedMlpInfo {
  std::string gate_output;
  std::string up_output;
  std::string fused_output;
};

class DeviceBuffer {
 public:
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)musaFree(ptr_);
    }
  }

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

struct ParallelLinearScratch {
  DeviceBuffer merged_output;
  DeviceBuffer pointer_arrays;
};

ParallelLinearScratch& ScratchForStream(const void* owner,
                                        musaStream_t stream) {
  using StreamMap =
      std::unordered_map<musaStream_t, std::unique_ptr<ParallelLinearScratch>>;
  thread_local std::unordered_map<const void*, StreamMap> scratch_by_owner;
  auto& scratch = scratch_by_owner[owner][stream];
  if (!scratch) {
    scratch = std::make_unique<ParallelLinearScratch>();
  }
  return *scratch;
}

bool IsOnnxDomain(const std::string& domain) {
  return domain.empty() || domain == "ai.onnx";
}

bool IsOnnxOp(Ort::ConstNode node, const char* op_type) {
  return node.GetOperatorType() == op_type && IsOnnxDomain(node.GetDomain());
}

std::string Name(Ort::ConstValueInfo value) { return value.GetName(); }

std::vector<int64_t> TensorShape(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

int64_t NumElementsChecked(const std::vector<int64_t>& shape) {
  int64_t total = 1;
  bool empty = false;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw std::runtime_error(
          "ParallelLinear requires non-negative runtime shapes");
    }
    // Empty ONNX tensors are valid inputs.  The fused matmul path must
    // propagate their empty outputs without entering muDNN, which does not
    // accept zero-element tensor descriptors on all supported versions.
    if (dim == 0) {
      empty = true;
      continue;
    }
    if (total > INT64_MAX / dim) {
      throw std::runtime_error(
          "ParallelLinear runtime shape element count overflowed");
    }
    total *= dim;
  }
  return empty ? 0 : total;
}

ONNXTensorElementDataType ValidateParallelLinearTensor(Ort::ConstValue value,
                                                       const char* name) {
  const ONNXTensorElementDataType elem_type =
      value.GetTensorTypeAndShapeInfo().GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    throw std::runtime_error(
        std::string("ParallelLinear only supports float32/float16/bfloat16 ") +
        name);
  }
  return elem_type;
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
    throw std::runtime_error("ParallelLinear could not map fused value " +
                             name);
  }
  return it->second;
}

bool InputsMatch(Ort::ConstNode node, const std::string& lhs,
                 const std::string& rhs) {
  auto inputs = node.GetInputs();
  return inputs.size() == 2 &&
         ((Name(inputs[0]) == lhs && Name(inputs[1]) == rhs) ||
          (Name(inputs[0]) == rhs && Name(inputs[1]) == lhs));
}

std::vector<uint8_t> ReadInitializerBytes(
    Ort::ConstValueInfo value_info, ONNXTensorElementDataType expected_type,
    const std::vector<int64_t>& expected_shape) {
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("ParallelLinear failed to read initializer " +
                             Name(value_info));
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != expected_type ||
      info.GetShape() != expected_shape) {
    throw std::runtime_error("ParallelLinear initializer metadata mismatch " +
                             Name(value_info));
  }
  const size_t bytes =
      static_cast<size_t>(info.GetElementCount()) * ElementSize(expected_type);
  const auto* data = static_cast<const uint8_t*>(value.GetTensorRawData());
  return std::vector<uint8_t>(data, data + bytes);
}

std::optional<GatedMlpInfo> FindGatedMlpInfo(
    Ort::ConstGraph graph, const std::vector<std::string>& branch_outputs) {
  if (branch_outputs.size() != 2) {
    return std::nullopt;
  }

  Ort::ConstNode sigmoid{nullptr};
  std::vector<Ort::ConstNode> muls;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Sigmoid")) {
      if (sigmoid) {
        return std::nullopt;
      }
      sigmoid = node;
    } else if (IsOnnxOp(node, "Mul")) {
      muls.push_back(node);
    }
  }
  if (!sigmoid || muls.size() != 2) {
    return std::nullopt;
  }

  auto sigmoid_inputs = sigmoid.GetInputs();
  auto sigmoid_outputs = sigmoid.GetOutputs();
  if (sigmoid_inputs.size() != 1 || sigmoid_outputs.size() != 1) {
    return std::nullopt;
  }
  const std::string gate_output = Name(sigmoid_inputs[0]);
  size_t gate_index = branch_outputs.size();
  for (size_t i = 0; i < branch_outputs.size(); ++i) {
    if (branch_outputs[i] == gate_output) {
      gate_index = i;
    }
  }
  if (gate_index == branch_outputs.size()) {
    return std::nullopt;
  }
  const std::string& up_output = branch_outputs[1 - gate_index];

  Ort::ConstNode gate_mul{nullptr};
  for (Ort::ConstNode mul : muls) {
    if (InputsMatch(mul, gate_output, Name(sigmoid_outputs[0]))) {
      gate_mul = mul;
    }
  }
  if (!gate_mul || gate_mul.GetOutputs().size() != 1) {
    return std::nullopt;
  }
  const std::string gate_mul_output = Name(gate_mul.GetOutputs()[0]);

  Ort::ConstNode output_mul{nullptr};
  for (Ort::ConstNode mul : muls) {
    if (mul.GetId() != gate_mul.GetId() &&
        InputsMatch(mul, gate_mul_output, up_output)) {
      output_mul = mul;
    }
  }
  if (!output_mul || output_mul.GetOutputs().size() != 1) {
    return std::nullopt;
  }
  return GatedMlpInfo{gate_output, up_output, Name(output_mul.GetOutputs()[0])};
}

}  // namespace

struct ParallelLinearFusionCompute : FusionNodeCompute {
  ParallelLinearFusionCompute(size_t input_index,
                              std::vector<BranchInfo> branches,
                              bool has_activation, bool gated_mlp,
                              size_t gated_output_index,
                              ONNXTensorElementDataType elem_type,
                              std::vector<int64_t> weight_shape,
                              std::vector<uint8_t> host_constants)
      : input_index(input_index),
        branches(std::move(branches)),
        has_activation(has_activation),
        gated_mlp(gated_mlp),
        gated_output_index(gated_output_index),
        elem_type(elem_type),
        weight_shape(std::move(weight_shape)),
        host_constants(std::move(host_constants)) {}

  ~ParallelLinearFusionCompute() override {
    if (constants_ready_event != nullptr) {
      (void)musaEventDestroy(constants_ready_event);
    }
  }

  OrtStatus* EnsureConstants(musaStream_t stream) const {
    if (constants_ready.load(std::memory_order_acquire)) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(constants_mutex);
    if (constants_ready.load(std::memory_order_relaxed)) {
      return nullptr;
    }
    RETURN_IF_ERROR(device_constants.Resize(host_constants.size(), stream));
    RETURN_IF_ERROR(CopyTemporaryHostToDevice(device_constants.data<void>(),
                                              host_constants.data(),
                                              host_constants.size(), stream));
    RETURN_IF_ERROR(LaunchStatus(musaEventCreateWithFlags(
        &constants_ready_event, musaEventDisableTiming)));
    RETURN_IF_ERROR(
        LaunchStatus(musaEventRecord(constants_ready_event, stream)));
    // Complete the one-time initializer upload before publishing the shared
    // pointer.  This keeps all later inference streams off the lock/event path.
    RETURN_IF_ERROR(LaunchStatus(musaEventSynchronize(constants_ready_event)));
    RETURN_IF_ERROR(LaunchStatus(musaEventDestroy(constants_ready_event)));
    constants_ready_event = nullptr;
    std::vector<uint8_t>().swap(host_constants);
    constants_ready.store(true, std::memory_order_release);
    return nullptr;
  }

  bool TryGroupedMatMulDirectOutput(
      const void* input_data, const std::vector<void*>& raw_output_pointers,
      const uint8_t* constants, int64_t rows, int64_t input_width,
      int64_t branch_width, musaStream_t stream) const {
    if (ParallelLinearGroupedMatMulDisabled() || has_activation || gated_mlp ||
        (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
         elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) ||
        (branches.size() != 2 && branches.size() != 3)) {
      return false;
    }

    ::musa::dnn::Handle* handle = nullptr;
    OrtStatus* handle_status = EnsureMudnnHandle(&handle, stream);
    if (handle_status != nullptr) {
      Ort::GetApi().ReleaseStatus(handle_status);
      return false;
    }

    const int groups = static_cast<int>(branches.size());
    std::vector<::musa::dnn::Tensor> a_tensors(groups);
    std::vector<::musa::dnn::Tensor> b_tensors(groups);
    std::vector<::musa::dnn::Tensor> c_tensors(groups);
    std::vector<int64_t> m(groups, rows);
    std::vector<int64_t> n(groups, branch_width);
    std::vector<int64_t> k(groups, input_width);
    std::vector<int64_t> lda(groups, input_width);
    std::vector<int64_t> ldb(groups, input_width);
    std::vector<int64_t> ldc(groups, branch_width);
    for (int index = 0; index < groups; ++index) {
      const BranchInfo& branch = branches[static_cast<size_t>(index)];
      if (branch.weight_byte_offset == kNoWeightInput ||
          !SetMudnnTensor(a_tensors[index], input_data, {rows, input_width},
                          elem_type) ||
          !SetMudnnTensor(b_tensors[index],
                          constants + branch.weight_byte_offset,
                          {branch_width, input_width}, elem_type) ||
          !SetMudnnTensor(c_tensors[index], raw_output_pointers[index],
                          {rows, branch_width}, elem_type)) {
        return false;
      }
    }

    ::musa::dnn::GroupedMatMul op;
    if (op.SetComputeMode(::musa::dnn::GroupedMatMul::ComputeMode::TENSOR) !=
            ::musa::dnn::Status::SUCCESS ||
        op.SetTranspose(false, true) != ::musa::dnn::Status::SUCCESS ||
        op.SetAlpha(1.0) != ::musa::dnn::Status::SUCCESS ||
        op.SetBeta(0.0) != ::musa::dnn::Status::SUCCESS) {
      return false;
    }
    ::musa::dnn::MemoryMaintainer maintainer =
        [stream](size_t bytes) -> ::musa::dnn::MemoryHandler {
      void* ptr = AllocateDeviceMemoryOnStream(bytes, stream);
      return ::musa::dnn::MemoryHandler(ptr, [stream, bytes](void* p) {
        FreeDeviceMemoryOnStream(p, stream, bytes);
      });
    };
    return op.Run(*handle, c_tensors.data(), a_tensors.data(), b_tensors.data(),
                  m.data(), n.data(), k.data(), lda.data(), ldb.data(),
                  ldc.data(), groups,
                  maintainer) == ::musa::dnn::Status::SUCCESS;
  }

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      musaStream_t stream = GetComputeStream(ctx);
      Ort::ConstValue input = ctx.GetInput(input_index);
      const ONNXTensorElementDataType runtime_elem_type =
          ValidateParallelLinearTensor(input, "input");
      if (runtime_elem_type != elem_type) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "ParallelLinear input dtype changed after compilation");
      }
      if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
          (has_activation || gated_mlp)) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "ParallelLinear non-float projections do not support epilogues");
      }
      DeviceInputBuffer input_buffer;
      RETURN_IF_ERROR(input_buffer.Bind(input, stream));
      const std::vector<int64_t> input_shape = TensorShape(input);
      if (input_shape.size() < 2) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ParallelLinear input rank must be >= 2");
      }

      if (weight_shape.size() != 2 || input_shape.back() != weight_shape[0]) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ParallelLinear K dimension mismatch");
      }
      const int64_t branch_count = static_cast<int64_t>(branches.size());
      const int64_t branch_width = weight_shape[1];
      const int64_t rows = NumElementsChecked(input_shape) / input_shape.back();
      std::vector<int64_t> output_shape = input_shape;
      output_shape.back() = branch_width;

      std::vector<float*> output_pointers;
      std::vector<void*> raw_output_pointers;
      float* gated_output = nullptr;
      if (gated_mlp) {
        Ort::UnownedValue output =
            ctx.GetOutput(gated_output_index, output_shape);
        if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "ParallelLinear requires MUSA outputs");
        }
        gated_output = output.GetTensorMutableData<float>();
      } else {
        output_pointers.reserve(branches.size());
        raw_output_pointers.reserve(branches.size());
        for (const BranchInfo& branch : branches) {
          Ort::UnownedValue output =
              ctx.GetOutput(branch.output_index, output_shape);
          if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
            return Ort::GetApi().CreateStatus(
                ORT_NOT_IMPLEMENTED, "ParallelLinear requires MUSA outputs");
          }
          if (output.GetTensorTypeAndShapeInfo().GetElementType() !=
              elem_type) {
            return Ort::GetApi().CreateStatus(
                ORT_INVALID_ARGUMENT,
                "ParallelLinear output dtype must match input");
          }
          void* output_data = output.GetTensorMutableRawData();
          raw_output_pointers.push_back(output_data);
          if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            output_pointers.push_back(static_cast<float*>(output_data));
          }
        }
      }

      // MatMul with zero rows has empty outputs by ONNX semantics.  Keep the
      // allocated output tensors and bypass muDNN/device kernels, which do
      // not consistently accept zero-element descriptors.
      if (rows == 0) {
        return nullptr;
      }

      RETURN_IF_ERROR(EnsureConstants(stream));
      const auto* constants = device_constants.data<uint8_t>();
      const void* merged_weights = constants;
      std::vector<const float*> bias_pointers;
      bias_pointers.reserve(branches.size());
      for (const BranchInfo& branch : branches) {
        bias_pointers.push_back(branch.bias_byte_offset == kNoBiasInput
                                    ? nullptr
                                    : reinterpret_cast<const float*>(
                                          constants + branch.bias_byte_offset));
      }

      ParallelLinearScratch& scratch = ScratchForStream(this, stream);
      if (TryGroupedMatMulDirectOutput(input_buffer.data(), raw_output_pointers,
                                       constants, rows, input_shape.back(),
                                       branch_width, stream)) {
        return nullptr;
      }
      const size_t merged_output_bytes =
          static_cast<size_t>(rows * branch_count * branch_width) *
          ElementSize(elem_type);
      RETURN_IF_ERROR(
          scratch.merged_output.Resize(merged_output_bytes, stream));
      std::vector<int64_t> flat_input_shape = {rows, input_shape.back()};
      std::vector<int64_t> merged_weight_shape = {weight_shape[0],
                                                  branch_count * branch_width};
      std::vector<int64_t> merged_output_shape = {rows,
                                                  branch_count * branch_width};
      RETURN_IF_ERROR(ComputeMusaMatMulDevice(
          input_buffer.data(), merged_weights,
          scratch.merged_output.data<void>(), elem_type, flat_input_shape,
          merged_weight_shape, merged_output_shape, false, false, false, false,
          1.0f, stream));

      musaError_t post_status = musaSuccess;
      if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        if (branches.size() != 2 && branches.size() != 3) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED,
              "ParallelLinear non-float path supports 2 or 3 branches");
        }
        post_status = LaunchParallelLinearPostDirectCopy16Kernel(
            scratch.merged_output.data<void>(), raw_output_pointers[0],
            raw_output_pointers[1],
            branches.size() == 3 ? raw_output_pointers[2] : nullptr, rows,
            branch_count, branch_width, stream);
      } else if (gated_mlp) {
        post_status = LaunchParallelLinearGatedMlpPostFloatKernel(
            scratch.merged_output.data<float>(), gated_output, bias_pointers[0],
            bias_pointers[1], rows, branch_width, stream);
      } else if (branches.size() == 2 || branches.size() == 3) {
        post_status = LaunchParallelLinearPostDirectFloatKernel(
            scratch.merged_output.data<float>(), output_pointers[0],
            output_pointers[1],
            branches.size() == 3 ? output_pointers[2] : nullptr,
            bias_pointers[0], bias_pointers[1],
            branches.size() == 3 ? bias_pointers[2] : nullptr, rows,
            branch_count, branch_width, MusaUnaryOp::Relu, has_activation, 0.0f,
            stream);
      } else {
        const size_t pointer_bytes = branches.size() * 2 * sizeof(const float*);
        RETURN_IF_ERROR(scratch.pointer_arrays.Resize(pointer_bytes, stream));
        std::vector<const float*> host_pointers;
        host_pointers.reserve(branches.size() * 2);
        for (float* output : output_pointers) {
          host_pointers.push_back(output);
        }
        host_pointers.insert(host_pointers.end(), bias_pointers.begin(),
                             bias_pointers.end());
        RETURN_IF_ERROR(CopyTemporaryHostToDevice(
            scratch.pointer_arrays.data<void*>(), host_pointers.data(),
            pointer_bytes, stream));
        float** device_outputs = scratch.pointer_arrays.data<float*>();
        const float* const* device_biases =
            reinterpret_cast<const float* const*>(device_outputs +
                                                  branches.size());
        post_status = LaunchParallelLinearPostFloatKernel(
            scratch.merged_output.data<float>(), device_outputs, device_biases,
            rows, branch_count, branch_width, MusaUnaryOp::Relu, has_activation,
            0.0f, stream);
      }
      if (post_status != musaSuccess) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                          MusaErrorString(post_status));
      }
      return nullptr;
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t input_index;
  std::vector<BranchInfo> branches;
  bool has_activation;
  bool gated_mlp;
  size_t gated_output_index;
  ONNXTensorElementDataType elem_type;
  std::vector<int64_t> weight_shape;
  mutable std::vector<uint8_t> host_constants;
  mutable DeviceBuffer device_constants;
  mutable musaEvent_t constants_ready_event = nullptr;
  mutable std::atomic<bool> constants_ready{false};
  mutable std::mutex constants_mutex;
};

bool IsParallelLinearFusionGraph(Ort::ConstGraph graph) {
  size_t matmul_count = 0;
  size_t gemm_count = 0;
  size_t add_count = 0;
  size_t relu_count = 0;
  size_t sigmoid_count = 0;
  size_t mul_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "MatMul")) {
      ++matmul_count;
    } else if (IsOnnxOp(node, "Gemm")) {
      ++gemm_count;
    } else if (IsOnnxOp(node, "Add")) {
      ++add_count;
    } else if (IsOnnxOp(node, "Relu")) {
      ++relu_count;
    } else if (IsOnnxOp(node, "Sigmoid")) {
      ++sigmoid_count;
    } else if (IsOnnxOp(node, "Mul")) {
      ++mul_count;
    } else {
      return false;
    }
  }
  const size_t linear_count = matmul_count + gemm_count;
  const bool ordinary = linear_count >= 2 && add_count <= matmul_count &&
                        sigmoid_count == 0 && mul_count == 0 &&
                        (relu_count == 0 || relu_count == linear_count);
  const bool gated_mlp = linear_count == 2 && add_count <= matmul_count &&
                         relu_count == 0 && sigmoid_count == 1 &&
                         mul_count == 2;
  return ordinary || gated_mlp;
}

std::unique_ptr<FusionNodeCompute> CreateParallelLinearFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  auto input_indices = ValueIndices(fused_node.GetInputs());
  auto output_indices = ValueIndices(fused_node.GetOutputs());
  std::unordered_map<std::string, Ort::ConstNode> consumer_by_input;
  for (Ort::ConstNode node : graph.GetNodes()) {
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      consumer_by_input.emplace(Name(input), node);
    }
  }

  std::string common_input;
  size_t input_index = 0;
  struct PendingBranch {
    Ort::ConstValueInfo weight{nullptr};
    Ort::ConstValueInfo bias{nullptr};
    std::string output_name;
  };
  std::vector<PendingBranch> pending_branches;
  bool has_activation = false;
  for (Ort::ConstNode matmul : graph.GetNodes()) {
    const bool is_matmul = IsOnnxOp(matmul, "MatMul");
    const bool is_gemm = IsOnnxOp(matmul, "Gemm");
    if (!is_matmul && !is_gemm) {
      continue;
    }
    auto matmul_inputs = matmul.GetInputs();
    auto matmul_outputs = matmul.GetOutputs();
    const std::string input_name = Name(matmul_inputs[0]);
    if (common_input.empty()) {
      common_input = input_name;
      input_index = IndexOf(input_indices, input_name);
    } else if (common_input != input_name) {
      throw std::runtime_error("ParallelLinear inputs are not shared");
    }
    Ort::ConstValueInfo bias{nullptr};
    std::string output_name = Name(matmul_outputs[0]);
    if (is_matmul) {
      auto add_it = consumer_by_input.find(output_name);
      if (add_it != consumer_by_input.end() &&
          IsOnnxOp(add_it->second, "Add")) {
        auto add_inputs = add_it->second.GetInputs();
        auto add_outputs = add_it->second.GetOutputs();
        const size_t bias_position = Name(add_inputs[0]) == output_name ? 1 : 0;
        bias = add_inputs[bias_position];
        output_name = Name(add_outputs[0]);
      }
    } else if (matmul_inputs.size() == 3) {
      bias = matmul_inputs[2];
    }
    auto activation_it = consumer_by_input.find(output_name);
    if (activation_it != consumer_by_input.end() &&
        IsOnnxOp(activation_it->second, "Relu")) {
      has_activation = true;
      output_name = Name(activation_it->second.GetOutputs()[0]);
    }
    pending_branches.push_back({matmul_inputs[1], bias, output_name});
  }

  std::vector<std::string> branch_outputs;
  branch_outputs.reserve(pending_branches.size());
  for (const PendingBranch& branch : pending_branches) {
    branch_outputs.push_back(branch.output_name);
  }
  std::optional<GatedMlpInfo> gated_info =
      FindGatedMlpInfo(graph, branch_outputs);

  std::vector<PendingBranch> ordered_branches;
  ordered_branches.reserve(pending_branches.size());
  size_t gated_output_index = kNoOutputIndex;
  if (gated_info.has_value()) {
    for (const std::string& output_name :
         {gated_info->gate_output, gated_info->up_output}) {
      auto it = std::find_if(pending_branches.begin(), pending_branches.end(),
                             [&](const PendingBranch& branch) {
                               return branch.output_name == output_name;
                             });
      if (it == pending_branches.end()) {
        throw std::runtime_error("ParallelLinear gated branch is missing");
      }
      ordered_branches.push_back(*it);
    }
    gated_output_index = IndexOf(output_indices, gated_info->fused_output);
  } else {
    ordered_branches = pending_branches;
  }

  if (ordered_branches.empty()) {
    throw std::runtime_error("ParallelLinear has no projection branches");
  }
  const auto elem_type =
      musa_ep::GetTensorElementType(ordered_branches.front().weight);
  const auto weight_shape =
      musa_ep::GetStaticShape(ordered_branches.front().weight);
  if (!elem_type.has_value() || !weight_shape.has_value() ||
      weight_shape->size() != 2 || (*weight_shape)[0] <= 0 ||
      (*weight_shape)[1] <= 0) {
    throw std::runtime_error("ParallelLinear weight metadata is invalid");
  }
  const size_t element_size = ElementSize(*elem_type);
  const size_t k = static_cast<size_t>((*weight_shape)[0]);
  const size_t n = static_cast<size_t>((*weight_shape)[1]);
  const size_t branch_count = ordered_branches.size();
  if (n > SIZE_MAX / branch_count || k > SIZE_MAX / (n * branch_count) ||
      k * n * branch_count > SIZE_MAX / element_size) {
    throw std::runtime_error("ParallelLinear packed weight size overflowed");
  }
  const size_t row_bytes = n * element_size;
  const size_t merged_row_bytes = branch_count * row_bytes;
  std::vector<uint8_t> host_constants(k * merged_row_bytes);
  const bool store_grouped_weights =
      (*elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
       *elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) &&
      !has_activation && !gated_info.has_value() &&
      (branch_count == 2 || branch_count == 3);
  std::vector<size_t> branch_weight_offsets(branch_count, kNoWeightInput);
  for (size_t branch_index = 0; branch_index < branch_count; ++branch_index) {
    const PendingBranch& branch = ordered_branches[branch_index];
    if (musa_ep::GetTensorElementType(branch.weight) != elem_type ||
        musa_ep::GetStaticShape(branch.weight) != weight_shape) {
      throw std::runtime_error("ParallelLinear branch weights differ");
    }
    const std::vector<uint8_t> weight =
        ReadInitializerBytes(branch.weight, *elem_type, *weight_shape);
    for (size_t row = 0; row < k; ++row) {
      std::memcpy(host_constants.data() + row * merged_row_bytes +
                      branch_index * row_bytes,
                  weight.data() + row * row_bytes, row_bytes);
    }
    if (store_grouped_weights) {
      std::vector<uint8_t> transposed_weight(weight.size());
      for (size_t row = 0; row < k; ++row) {
        for (size_t column = 0; column < n; ++column) {
          std::memcpy(
              transposed_weight.data() + (column * k + row) * element_size,
              weight.data() + (row * n + column) * element_size, element_size);
        }
      }
      branch_weight_offsets[branch_index] = host_constants.size();
      host_constants.insert(host_constants.end(), transposed_weight.begin(),
                            transposed_weight.end());
    }
  }

  std::vector<BranchInfo> branches;
  branches.reserve(branch_count);
  for (size_t branch_index = 0; branch_index < branch_count; ++branch_index) {
    const PendingBranch& branch = ordered_branches[branch_index];
    size_t bias_byte_offset = kNoBiasInput;
    if (branch.bias != nullptr) {
      const auto bias_shape = musa_ep::GetStaticShape(branch.bias);
      if (musa_ep::GetTensorElementType(branch.bias) != elem_type ||
          !bias_shape.has_value()) {
        throw std::runtime_error("ParallelLinear bias metadata is invalid");
      }
      const std::vector<uint8_t> bias =
          ReadInitializerBytes(branch.bias, *elem_type, *bias_shape);
      if (bias.size() != row_bytes ||
          host_constants.size() > SIZE_MAX - bias.size()) {
        throw std::runtime_error("ParallelLinear bias size is invalid");
      }
      bias_byte_offset = host_constants.size();
      host_constants.insert(host_constants.end(), bias.begin(), bias.end());
    }
    branches.push_back({bias_byte_offset, branch_weight_offsets[branch_index],
                        gated_info.has_value()
                            ? kNoOutputIndex
                            : IndexOf(output_indices, branch.output_name)});
  }
  return std::make_unique<ParallelLinearFusionCompute>(
      input_index, std::move(branches), has_activation, gated_info.has_value(),
      gated_output_index, *elem_type, *weight_shape, std::move(host_constants));
}
