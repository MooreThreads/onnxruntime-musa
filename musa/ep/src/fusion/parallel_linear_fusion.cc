// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "fusion/parallel_linear_fusion.h"

#include <mudnn.h>
#include <musa_runtime.h>

#include <climits>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/math/matmul.h"
#include "kernels/nn/parallel_linear_impl.h"
#include "kernels/shared_inc/blas_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"

namespace {

struct BranchInfo {
  size_t weight_input_index;
  size_t bias_input_index;
  size_t output_index;
};

constexpr size_t kNoBiasInput = std::numeric_limits<size_t>::max();

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
  DeviceBuffer merged_weights;
  DeviceBuffer merged_output;
  DeviceBuffer pointer_arrays;
  size_t merged_weight_bytes = 0;
  bool merged_weights_valid = false;
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
  for (int64_t dim : shape) {
    if (dim <= 0 || total > INT64_MAX / dim) {
      throw std::runtime_error(
          "ParallelLinear requires positive non-overflowing runtime shapes");
    }
    total *= dim;
  }
  return total;
}

void ValidateFloat(Ort::ConstValue value, const char* name) {
  if (value.GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(std::string("ParallelLinear requires float ") +
                             name);
  }
}

std::vector<int64_t> ShapeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

void CheckMudnn(::musa::dnn::Status status, const char* message) {
  if (status != ::musa::dnn::Status::SUCCESS) {
    throw std::runtime_error(std::string(message) + ", status=" +
                             std::to_string(static_cast<int>(status)));
  }
}

void SetupTensor(::musa::dnn::Tensor& tensor, const float* data,
                 const std::vector<int64_t>& shape) {
  CheckMudnn(tensor.SetType(::musa::dnn::Tensor::Type::FLOAT),
             "ParallelLinear tensor type failed");
  CheckMudnn(tensor.SetAddr(data), "ParallelLinear tensor address failed");
  CheckMudnn(tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW),
             "ParallelLinear tensor format failed");
  auto strides = ShapeStrides(shape);
  CheckMudnn(tensor.SetNdInfo(static_cast<int>(shape.size()), shape.data(),
                              strides.data()),
             "ParallelLinear tensor shape failed");
}

void MergeWeights(const std::vector<const float*>& weights,
                  const std::vector<int64_t>& weight_shape,
                  float* merged_weights, musaStream_t stream) {
  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* raw_status = EnsureMudnnHandle(&handle, stream);
  if (raw_status != nullptr) {
    Ort::Status status(raw_status);
    throw std::runtime_error(status.GetErrorMessage());
  }
  std::vector<::musa::dnn::Tensor> inputs(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    SetupTensor(inputs[i], weights[i], weight_shape);
  }
  std::vector<int64_t> merged_shape = {
      weight_shape[0], weight_shape[1] * static_cast<int64_t>(weights.size())};
  ::musa::dnn::Tensor output;
  SetupTensor(output, merged_weights, merged_shape);
  ::musa::dnn::Concat concat;
  CheckMudnn(concat.SetAxis(1), "ParallelLinear concat axis failed");
  CheckMudnn(concat.Run(*handle, output, static_cast<int>(inputs.size()),
                        inputs.data()),
             "ParallelLinear weight concat failed");
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

}  // namespace

struct ParallelLinearFusionCompute : FusionNodeCompute {
  ParallelLinearFusionCompute(size_t input_index,
                              std::vector<BranchInfo> branches,
                              bool has_activation)
      : input_index(input_index),
        branches(std::move(branches)),
        has_activation(has_activation) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      musaStream_t stream = GetComputeStream(ctx);
      Ort::ConstValue input = ctx.GetInput(input_index);
      ValidateFloat(input, "input");
      DeviceInputBuffer input_buffer;
      RETURN_IF_ERROR(input_buffer.Bind(input, stream));
      const std::vector<int64_t> input_shape = TensorShape(input);
      if (input_shape.size() < 2) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "ParallelLinear input rank must be >= 2");
      }

      std::vector<std::unique_ptr<DeviceInputBuffer>> weight_buffers;
      std::vector<std::unique_ptr<DeviceInputBuffer>> bias_buffers;
      std::vector<const float*> weight_pointers;
      std::vector<const float*> bias_pointers;
      weight_buffers.reserve(branches.size());
      bias_buffers.reserve(branches.size());
      weight_pointers.reserve(branches.size());
      bias_pointers.reserve(branches.size());

      std::vector<int64_t> weight_shape;
      for (const BranchInfo& branch : branches) {
        Ort::ConstValue weight = ctx.GetInput(branch.weight_input_index);
        ValidateFloat(weight, "weight");
        if (weight_shape.empty()) {
          weight_shape = TensorShape(weight);
        } else if (TensorShape(weight) != weight_shape) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "ParallelLinear weights must have identical shapes");
        }
        auto weight_buffer = std::make_unique<DeviceInputBuffer>();
        RETURN_IF_ERROR(weight_buffer->Bind(weight, stream));
        weight_pointers.push_back(
            static_cast<const float*>(weight_buffer->data()));
        weight_buffers.push_back(std::move(weight_buffer));
        if (branch.bias_input_index == kNoBiasInput) {
          bias_pointers.push_back(nullptr);
        } else {
          Ort::ConstValue bias = ctx.GetInput(branch.bias_input_index);
          ValidateFloat(bias, "bias");
          auto bias_buffer = std::make_unique<DeviceInputBuffer>();
          RETURN_IF_ERROR(bias_buffer->Bind(bias, stream));
          bias_pointers.push_back(
              static_cast<const float*>(bias_buffer->data()));
          bias_buffers.push_back(std::move(bias_buffer));
        }
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
      output_pointers.reserve(branches.size());
      for (const BranchInfo& branch : branches) {
        Ort::UnownedValue output =
            ctx.GetOutput(branch.output_index, output_shape);
        if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "ParallelLinear requires MUSA outputs");
        }
        output_pointers.push_back(output.GetTensorMutableData<float>());
      }

      ParallelLinearScratch& scratch = ScratchForStream(this, stream);
      const size_t merged_weight_bytes =
          static_cast<size_t>(weight_shape[0] * branch_count * branch_width) *
          sizeof(float);
      if (scratch.merged_weight_bytes != merged_weight_bytes) {
        scratch.merged_weights_valid = false;
        scratch.merged_weight_bytes = merged_weight_bytes;
      }
      RETURN_IF_ERROR(
          scratch.merged_weights.Resize(merged_weight_bytes, stream));
      if (!scratch.merged_weights_valid) {
        MergeWeights(weight_pointers, weight_shape,
                     scratch.merged_weights.data<float>(), stream);
        scratch.merged_weights_valid = true;
      }

      const size_t merged_output_bytes =
          static_cast<size_t>(rows * branch_count * branch_width) *
          sizeof(float);
      RETURN_IF_ERROR(
          scratch.merged_output.Resize(merged_output_bytes, stream));
      std::vector<int64_t> flat_input_shape = {rows, input_shape.back()};
      std::vector<int64_t> merged_weight_shape = {weight_shape[0],
                                                  branch_count * branch_width};
      std::vector<int64_t> merged_output_shape = {rows,
                                                  branch_count * branch_width};
      RETURN_IF_ERROR(ComputeMusaMatMulDevice(
          static_cast<const float*>(input_buffer.data()),
          scratch.merged_weights.data<float>(),
          scratch.merged_output.data<float>(), flat_input_shape,
          merged_weight_shape, merged_output_shape, stream));

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
      const float* const* device_biases = reinterpret_cast<const float* const*>(
          device_outputs + branches.size());
      musaError_t post_status = LaunchParallelLinearPostFloatKernel(
          scratch.merged_output.data<float>(), device_outputs, device_biases,
          rows, branch_count, branch_width, MusaUnaryOp::Relu, has_activation,
          0.0f, stream);
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
};

bool IsParallelLinearFusionGraph(Ort::ConstGraph graph) {
  size_t matmul_count = 0;
  size_t gemm_count = 0;
  size_t add_count = 0;
  size_t relu_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "MatMul")) {
      ++matmul_count;
    } else if (IsOnnxOp(node, "Gemm")) {
      ++gemm_count;
    } else if (IsOnnxOp(node, "Add")) {
      ++add_count;
    } else if (IsOnnxOp(node, "Relu")) {
      ++relu_count;
    } else {
      return false;
    }
  }
  const size_t linear_count = matmul_count + gemm_count;
  return linear_count >= 2 && add_count <= matmul_count &&
         (relu_count == 0 || relu_count == linear_count);
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
  std::vector<BranchInfo> branches;
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
    branches.push_back(
        {IndexOf(input_indices, Name(matmul_inputs[1])),
         bias != nullptr ? IndexOf(input_indices, Name(bias)) : kNoBiasInput,
         IndexOf(output_indices, output_name)});
  }
  return std::make_unique<ParallelLinearFusionCompute>(
      input_index, std::move(branches), has_activation);
}
