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

#include "fusion/split_sequence_moe_fusion.h"

#include <mudnn.h>
#include <musa_runtime.h>

#include <algorithm>
#include <cmath>
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
#include "kernels/math/unary_elementwise_ops_impl.h"
#include "kernels/nn/split_sequence_moe_impl.h"
#include "kernels/shared_inc/blas_utils.h"
#include "kernels/shared_inc/kernel_memory.h"
#include "kernels/shared_inc/op_kernel_common.h"

namespace {

using musa_ep::BuildProducerMap;
using musa_ep::FindProducer;

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

float ReadFloatAttribute(Ort::ConstNode node, const char* name,
                         float default_value) {
  Ort::ConstOpAttr attr;
  float value = default_value;
  return node.GetAttributeByName(name, attr).IsOK() &&
                 attr.GetValue(value).IsOK()
             ? value
             : default_value;
}

std::vector<int64_t> TensorShape(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

void ValidateFloat(Ort::ConstValue value, const char* name) {
  if (value.GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(std::string("SplitSequenceMoE requires float ") +
                             name);
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
    throw std::runtime_error("SplitSequenceMoE could not map fused value " +
                             name);
  }
  return it->second;
}

std::vector<float> ReadFloatInitializer(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) {
    throw std::runtime_error(
        "SplitSequenceMoE weights and norms must be initializers");
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = value_info.GetInitializer(value);
  if (!status.IsOK() || !value) {
    throw std::runtime_error("SplitSequenceMoE failed to read initializer " +
                             Name(value_info));
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("SplitSequenceMoE initializer is not float " +
                             Name(value_info));
  }
  const size_t count = static_cast<size_t>(info.GetElementCount());
  const float* data = value.GetTensorData<float>();
  return std::vector<float>(data, data + count);
}

void AppendInitializer(Ort::ConstValueInfo value_info, size_t expected_count,
                       std::vector<float>& packed) {
  std::vector<float> values = ReadFloatInitializer(value_info);
  if (values.size() != expected_count) {
    throw std::runtime_error("SplitSequenceMoE initializer size mismatch for " +
                             Name(value_info));
  }
  packed.insert(packed.end(), values.begin(), values.end());
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

struct DeviceConstants {
  ~DeviceConstants() {
    if (ready_event != nullptr) {
      (void)musaEventDestroy(ready_event);
    }
  }

  DeviceBuffer values;
  musaEvent_t ready_event = nullptr;
  int device_id = -1;
  bool ready = false;
};

struct RuntimeScratch {
  DeviceBuffer hidden;
  DeviceBuffer group_offsets;
};

RuntimeScratch& ScratchForStream(const void* owner, musaStream_t stream) {
  using StreamMap =
      std::unordered_map<musaStream_t, std::unique_ptr<RuntimeScratch>>;
  thread_local std::unordered_map<const void*, StreamMap> scratch_by_owner;
  auto& scratch = scratch_by_owner[owner][stream];
  if (!scratch) {
    scratch = std::make_unique<RuntimeScratch>();
  }
  return *scratch;
}

size_t CheckedBytes(int64_t rows, int64_t width) {
  if (rows < 0 || width < 0 ||
      (width != 0 && rows > std::numeric_limits<int64_t>::max() / width)) {
    throw std::runtime_error("SplitSequenceMoE tensor size overflowed");
  }
  const uint64_t elements = static_cast<uint64_t>(rows * width);
  if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("SplitSequenceMoE tensor size overflowed");
  }
  return static_cast<size_t>(elements) * sizeof(float);
}

int64_t FlattenedRows(const std::vector<int64_t>& shape) {
  if (shape.size() != 2 || shape[0] < 0 || shape[1] < 0) {
    throw std::runtime_error("SplitSequenceMoE requires a rank-2 tensor");
  }
  return shape[0];
}

struct ParsedSlice {
  Ort::ConstValueInfo start{nullptr};
  Ort::ConstValueInfo end{nullptr};
};

struct ParsedGemm {
  Ort::ConstValueInfo weight{nullptr};
  Ort::ConstValueInfo bias{nullptr};
  int64_t input_width = 0;
  int64_t output_width = 0;
};

struct ParsedBranch {
  ParsedSlice slice;
  ParsedGemm up;
  ParsedGemm down;
  Ort::ConstValueInfo gamma{nullptr};
  Ort::ConstValueInfo beta{nullptr};
};

struct ParsedGraph {
  Ort::ConstValueInfo x{nullptr};
  Ort::ConstValueInfo output{nullptr};
  std::vector<ParsedBranch> branches;
  float epsilon = 0.0f;
};

ParsedGemm ParseGemm(Ort::ConstNode node) {
  if (!IsOnnxOp(node, "Gemm")) {
    throw std::runtime_error("SplitSequenceMoE expected Gemm");
  }
  auto inputs = node.GetInputs();
  if (inputs.size() != 3) {
    throw std::runtime_error("SplitSequenceMoE Gemm input count mismatch");
  }
  auto shape = musa_ep::GetStaticShape(inputs[1]);
  auto bias_shape = musa_ep::GetStaticShape(inputs[2]);
  if (!shape.has_value() || shape->size() != 2 || !bias_shape.has_value()) {
    throw std::runtime_error("SplitSequenceMoE Gemm shape is not static");
  }
  const int64_t input_width = (*shape)[1];
  const int64_t output_width = (*shape)[0];
  if (input_width <= 0 || output_width <= 0 ||
      !((bias_shape->size() == 1 && (*bias_shape)[0] == output_width) ||
        (bias_shape->size() == 2 && (*bias_shape)[0] == 1 &&
         (*bias_shape)[1] == output_width))) {
    throw std::runtime_error("SplitSequenceMoE Gemm shape mismatch");
  }
  return {inputs[1], inputs[2], input_width, output_width};
}

ParsedGraph ParseSplitSequenceMoEGraph(Ort::ConstGraph graph) {
  ParsedGraph parsed;
  const std::vector<Ort::ConstNode> graph_nodes = graph.GetNodes();
  const auto producers = BuildProducerMap(graph_nodes);
  Ort::ConstNode final_concat{nullptr};
  for (Ort::ConstNode node : graph_nodes) {
    if (!IsOnnxOp(node, "Concat")) {
      continue;
    }
    bool has_layer_norm_inputs = node.GetInputs().size() >= 2;
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      Ort::ConstNode producer = FindProducer(producers, input);
      if (!IsOnnxOp(producer, "LayerNormalization")) {
        has_layer_norm_inputs = false;
        break;
      }
    }
    if (has_layer_norm_inputs) {
      if (final_concat) {
        throw std::runtime_error(
            "SplitSequenceMoE graph has multiple final Concat nodes");
      }
      final_concat = node;
    }
  }
  if (!final_concat) {
    throw std::runtime_error("SplitSequenceMoE graph is missing final Concat");
  }

  Ort::ConstNode residual_add{nullptr};
  std::vector<Ort::ConstNode> post_slices;
  std::vector<Ort::ConstNode> layer_norms;
  parsed.branches.reserve(final_concat.GetInputs().size());
  for (Ort::ConstValueInfo input : final_concat.GetInputs()) {
    Ort::ConstNode layer_norm = FindProducer(producers, input);
    if (!layer_norm) {
      throw std::runtime_error("SplitSequenceMoE final branch has no LN");
    }
    auto ln_inputs = layer_norm.GetInputs();
    if (ln_inputs.size() != 3) {
      throw std::runtime_error("SplitSequenceMoE LayerNorm input mismatch");
    }
    Ort::ConstNode post_slice = FindProducer(producers, ln_inputs[0]);
    if (!post_slice) {
      throw std::runtime_error("SplitSequenceMoE final branch has no Slice");
    }
    auto slice_inputs = post_slice.GetInputs();
    if (slice_inputs.size() != 5) {
      throw std::runtime_error("SplitSequenceMoE Slice input mismatch");
    }
    Ort::ConstNode branch_residual = FindProducer(producers, slice_inputs[0]);
    if (!IsOnnxOp(branch_residual, "Add")) {
      throw std::runtime_error("SplitSequenceMoE final Slice has no Add");
    }
    if (residual_add && residual_add.GetId() != branch_residual.GetId()) {
      throw std::runtime_error("SplitSequenceMoE residual Add differs");
    }
    residual_add = branch_residual;
    auto scale_shape = musa_ep::GetStaticShape(ln_inputs[1]);
    auto bias_shape = musa_ep::GetStaticShape(ln_inputs[2]);
    if (!scale_shape.has_value() || scale_shape->size() != 1 ||
        !bias_shape.has_value() || *bias_shape != *scale_shape) {
      throw std::runtime_error("SplitSequenceMoE LayerNorm shape mismatch");
    }
    const float epsilon = ReadFloatAttribute(layer_norm, "epsilon", 1.0e-5f);
    if (!std::isfinite(epsilon) || epsilon <= 0.0f ||
        (parsed.epsilon != 0.0f &&
         std::fabs(parsed.epsilon - epsilon) >
             std::max(1.0e-12f, std::fabs(parsed.epsilon) * 1.0e-6f))) {
      throw std::runtime_error("SplitSequenceMoE epsilon mismatch");
    }
    parsed.epsilon = epsilon;
    post_slices.push_back(post_slice);
    layer_norms.push_back(layer_norm);
    parsed.branches.push_back({ParsedSlice{slice_inputs[1], slice_inputs[2]},
                               {},
                               {},
                               ln_inputs[1],
                               ln_inputs[2]});
  }

  if (!residual_add) {
    throw std::runtime_error("SplitSequenceMoE graph has no residual Add");
  }
  auto residual_inputs = residual_add.GetInputs();
  Ort::ConstNode ffn_concat{nullptr};
  int64_t ffn_input_index = -1;
  for (int64_t i = 0; i < 2; ++i) {
    Ort::ConstNode producer =
        FindProducer(producers, residual_inputs[static_cast<size_t>(i)]);
    if (IsOnnxOp(producer, "Concat")) {
      if (ffn_concat) {
        throw std::runtime_error("SplitSequenceMoE has multiple FFN Concats");
      }
      ffn_concat = producer;
      ffn_input_index = i;
    }
  }
  if (!ffn_concat || ffn_input_index < 0) {
    throw std::runtime_error("SplitSequenceMoE graph has no FFN Concat");
  }
  parsed.x = residual_inputs[static_cast<size_t>(1 - ffn_input_index)];
  parsed.output = final_concat.GetOutputs().at(0);

  if (ffn_concat.GetInputs().size() != parsed.branches.size()) {
    throw std::runtime_error("SplitSequenceMoE group count mismatch");
  }
  for (size_t i = 0; i < parsed.branches.size(); ++i) {
    Ort::ConstNode down = FindProducer(producers, ffn_concat.GetInputs()[i]);
    if (!down) {
      throw std::runtime_error("SplitSequenceMoE FFN branch has no down Gemm");
    }
    auto down_inputs = down.GetInputs();
    if (down_inputs.size() != 3) {
      throw std::runtime_error("SplitSequenceMoE down Gemm mismatch");
    }
    Ort::ConstNode relu = FindProducer(producers, down_inputs[0]);
    if (!IsOnnxOp(relu, "Relu")) {
      throw std::runtime_error("SplitSequenceMoE FFN branch has no Relu");
    }
    Ort::ConstNode up = FindProducer(producers, relu.GetInputs().at(0));
    if (!up) {
      throw std::runtime_error("SplitSequenceMoE FFN branch has no up Gemm");
    }
    auto up_inputs = up.GetInputs();
    if (up_inputs.size() != 3) {
      throw std::runtime_error("SplitSequenceMoE up Gemm mismatch");
    }
    Ort::ConstNode pre_slice = FindProducer(producers, up_inputs[0]);
    if (!pre_slice) {
      throw std::runtime_error(
          "SplitSequenceMoE FFN branch has no input Slice");
    }
    auto pre_inputs = pre_slice.GetInputs();
    if (pre_inputs.size() != 5 || Name(pre_inputs[0]) != Name(parsed.x) ||
        Name(pre_inputs[1]) != Name(parsed.branches[i].slice.start) ||
        Name(pre_inputs[2]) != Name(parsed.branches[i].slice.end)) {
      throw std::runtime_error("SplitSequenceMoE Slice boundary mismatch");
    }
    parsed.branches[i].up = ParseGemm(up);
    parsed.branches[i].down = ParseGemm(down);
    auto x_shape = GetTensorShape(parsed.x);
    if (x_shape.has_value() && x_shape->size() == 2 && x_shape->back() > 0 &&
        parsed.branches[i].up.input_width != x_shape->back()) {
      throw std::runtime_error("SplitSequenceMoE input width mismatch");
    }
    if (parsed.branches[i].up.output_width !=
        parsed.branches[i].down.input_width) {
      throw std::runtime_error("SplitSequenceMoE FFN dimensions mismatch");
    }
  }
  return parsed;
}

struct BoundarySource {
  size_t input_index = 0;
  bool constant = false;
  int64_t constant_value = 0;
};

class SplitSequenceMoEFusionCompute final : public FusionNodeCompute {
 public:
  SplitSequenceMoEFusionCompute(
      size_t x_input_index, size_t output_index,
      std::vector<BoundarySource> boundaries, int64_t group_count,
      int64_t input_width, int64_t hidden_width, int64_t output_width,
      float epsilon, std::vector<float> host_constants,
      size_t first_weight_offset, size_t first_bias_offset,
      size_t second_weight_offset, size_t second_bias_offset,
      size_t gamma_offset, size_t beta_offset)
      : x_input_index(x_input_index),
        output_index(output_index),
        boundaries(std::move(boundaries)),
        group_count(group_count),
        input_width(input_width),
        hidden_width(hidden_width),
        output_width(output_width),
        epsilon(epsilon),
        host_constants(std::move(host_constants)),
        first_weight_offset(first_weight_offset),
        first_bias_offset(first_bias_offset),
        second_weight_offset(second_weight_offset),
        second_bias_offset(second_bias_offset),
        gamma_offset(gamma_offset),
        beta_offset(beta_offset) {}

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
            ORT_EP_FAIL,
            "SplitSequenceMoE constants cannot be shared across MUSA devices");
      }
      return LaunchStatus(
          musaStreamWaitEvent(stream, device_constants.ready_event, 0));
    }

    const size_t bytes = host_constants.size() * sizeof(float);
    RETURN_IF_ERROR(device_constants.values.Resize(bytes, stream));
    RETURN_IF_ERROR(
        CopyTemporaryHostToDevice(device_constants.values.data<float>(),
                                  host_constants.data(), bytes, stream));
    if (device_constants.ready_event == nullptr) {
      RETURN_IF_ERROR(LaunchStatus(musaEventCreateWithFlags(
          &device_constants.ready_event, musaEventDisableTiming)));
    }
    RETURN_IF_ERROR(
        LaunchStatus(musaEventRecord(device_constants.ready_event, stream)));
    device_constants.device_id = current_device;
    device_constants.ready = true;
    std::vector<float>().swap(host_constants);
    return nullptr;
  }

  OrtStatus* ReadBoundaries(Ort::KernelContext& ctx, musaStream_t stream,
                            std::vector<int64_t>& offsets) const {
    offsets.resize(boundaries.size());
    std::vector<int64_t> host_i64(boundaries.size());
    std::vector<int32_t> host_i32(boundaries.size());
    std::vector<bool> is_i32(boundaries.size(), false);
    bool has_device_copy = false;
    for (size_t i = 0; i < boundaries.size(); ++i) {
      if (boundaries[i].constant) {
        offsets[i] = boundaries[i].constant_value;
        continue;
      }
      Ort::ConstValue value = ctx.GetInput(boundaries[i].input_index);
      auto info = value.GetTensorTypeAndShapeInfo();
      if (info.GetElementCount() != 1 ||
          (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
           info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "SplitSequenceMoE boundary must be one int32/int64 value");
      }
      is_i32[i] = info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      if (IsGpuMemory(value.GetTensorMemoryInfo())) {
        has_device_copy = true;
        const size_t bytes = is_i32[i] ? sizeof(int32_t) : sizeof(int64_t);
        void* destination = is_i32[i] ? static_cast<void*>(&host_i32[i])
                                      : static_cast<void*>(&host_i64[i]);
        RETURN_IF_ERROR(LaunchStatus(
            musaMemcpyAsync(destination, value.GetTensorRawData(), bytes,
                            musaMemcpyDeviceToHost, stream)));
      } else if (is_i32[i]) {
        host_i32[i] = value.GetTensorData<int32_t>()[0];
      } else {
        host_i64[i] = value.GetTensorData<int64_t>()[0];
      }
    }
    if (has_device_copy) {
      RETURN_IF_ERROR(LaunchStatus(musaStreamSynchronize(stream)));
    }
    for (size_t i = 0; i < boundaries.size(); ++i) {
      if (!boundaries[i].constant) {
        offsets[i] =
            is_i32[i] ? static_cast<int64_t>(host_i32[i]) : host_i64[i];
      }
    }
    return nullptr;
  }

  OrtStatus* RunGroupedGemm(const float* input, float* output,
                            const float* weights, const float* biases,
                            const std::vector<int64_t>& offsets,
                            int64_t input_width_, int64_t output_width_,
                            musaStream_t stream) const {
    ::musa::dnn::Handle* handle = nullptr;
    RETURN_IF_ERROR(EnsureMudnnHandle(&handle, stream));

    std::vector<::musa::dnn::Tensor> a_tensors;
    std::vector<::musa::dnn::Tensor> b_tensors;
    std::vector<::musa::dnn::Tensor> c_tensors;
    std::vector<::musa::dnn::Tensor> bias_tensors;
    std::vector<int64_t> m;
    std::vector<int64_t> n;
    std::vector<int64_t> k;
    std::vector<int64_t> lda;
    std::vector<int64_t> ldb;
    std::vector<int64_t> ldc;
    for (int64_t group = 0; group < group_count; ++group) {
      const int64_t rows = offsets[static_cast<size_t>(group + 1)] -
                           offsets[static_cast<size_t>(group)];
      if (rows == 0) {
        continue;
      }
      const size_t index = a_tensors.size();
      a_tensors.emplace_back();
      b_tensors.emplace_back();
      c_tensors.emplace_back();
      bias_tensors.emplace_back();
      const float* a =
          input + offsets[static_cast<size_t>(group)] * input_width_;
      const float* b = weights + group * output_width_ * input_width_;
      float* c = output + offsets[static_cast<size_t>(group)] * output_width_;
      const float* bias = biases + group * output_width_;
      if (!SetMudnnTensor(a_tensors[index], a, {rows, input_width_},
                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
          !SetMudnnTensor(b_tensors[index], b, {output_width_, input_width_},
                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
          !SetMudnnTensor(c_tensors[index], c, {rows, output_width_},
                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
          !SetMudnnTensor(bias_tensors[index], bias, {output_width_},
                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
        return Ort::GetApi().CreateStatus(
            ORT_EP_FAIL, "SplitSequenceMoE failed to setup grouped GEMM");
      }
      m.push_back(rows);
      n.push_back(output_width_);
      k.push_back(input_width_);
      lda.push_back(input_width_);
      ldb.push_back(input_width_);
      ldc.push_back(output_width_);
    }
    if (a_tensors.empty()) {
      return nullptr;
    }
    if (a_tensors.size() >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "SplitSequenceMoE group count overflowed");
    }

    ::musa::dnn::GroupedMatMul op;
    if (op.SetComputeMode(::musa::dnn::GroupedMatMul::ComputeMode::TENSOR) !=
            ::musa::dnn::Status::SUCCESS ||
        op.SetTranspose(false, true) != ::musa::dnn::Status::SUCCESS ||
        op.SetAlpha(1.0) != ::musa::dnn::Status::SUCCESS ||
        op.SetBeta(0.0) != ::musa::dnn::Status::SUCCESS ||
        op.SetGamma(1.0) != ::musa::dnn::Status::SUCCESS) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL, "SplitSequenceMoE failed to configure grouped GEMM");
    }
    ::musa::dnn::MemoryMaintainer maintainer =
        [stream](size_t bytes) -> ::musa::dnn::MemoryHandler {
      void* ptr = AllocateDeviceMemoryOnStream(bytes, stream);
      return ::musa::dnn::MemoryHandler(ptr, [stream, bytes](void* p) {
        FreeDeviceMemoryOnStream(p, stream, bytes);
      });
    };
    const int groups = static_cast<int>(a_tensors.size());
    auto status = op.RunWithBiasAdd(*handle, c_tensors.data(), a_tensors.data(),
                                    b_tensors.data(), bias_tensors.data(),
                                    m.data(), n.data(), k.data(), lda.data(),
                                    ldb.data(), ldc.data(), groups, maintainer);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL, "SplitSequenceMoE grouped GEMM execution failed");
    }
    return nullptr;
  }

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      musaStream_t stream = GetComputeStream(ctx);
      Ort::ConstValue x = ctx.GetInput(x_input_index);
      ValidateFloat(x, "x");
      const std::vector<int64_t> x_shape = TensorShape(x);
      if (x_shape.size() != 2 || x_shape[1] != input_width) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "SplitSequenceMoE x shape mismatch");
      }
      const int64_t rows = FlattenedRows(x_shape);
      std::vector<int64_t> offsets;
      RETURN_IF_ERROR(ReadBoundaries(ctx, stream, offsets));
      if (offsets.size() != static_cast<size_t>(group_count + 1) ||
          offsets.front() != 0 || offsets.back() != rows) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "SplitSequenceMoE boundaries do not cover compact rows");
      }
      for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1]) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "SplitSequenceMoE boundaries must be nondecreasing");
        }
      }

      Ort::UnownedValue output = ctx.GetOutput(output_index, x_shape);
      if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "SplitSequenceMoE requires MUSA output");
      }
      if (rows == 0) {
        return nullptr;
      }

      DeviceInputBuffer x_buffer;
      RETURN_IF_ERROR(x_buffer.Bind(x, stream));
      RETURN_IF_ERROR(EnsureConstants(stream));
      RuntimeScratch& scratch = ScratchForStream(this, stream);
      RETURN_IF_ERROR(
          scratch.hidden.Resize(CheckedBytes(rows, hidden_width), stream));
      RETURN_IF_ERROR(scratch.group_offsets.Resize(
          offsets.size() * sizeof(int64_t), stream));
      RETURN_IF_ERROR(CopyTemporaryHostToDevice(
          scratch.group_offsets.data<int64_t>(), offsets.data(),
          offsets.size() * sizeof(int64_t), stream));

      const float* constants = device_constants.values.data<float>();
      const float* first_weights = constants + first_weight_offset;
      const float* first_biases = constants + first_bias_offset;
      const float* second_weights = constants + second_weight_offset;
      const float* second_biases = constants + second_bias_offset;
      const float* gamma = constants + gamma_offset;
      const float* beta = constants + beta_offset;
      RETURN_IF_ERROR(RunGroupedGemm(static_cast<const float*>(x_buffer.data()),
                                     scratch.hidden.data<float>(),
                                     first_weights, first_biases, offsets,
                                     input_width, hidden_width, stream));
      RETURN_IF_ERROR(LaunchStatus(LaunchMusaUnaryFloatKernel(
          scratch.hidden.data<float>(), scratch.hidden.data<float>(),
          rows * hidden_width, MusaUnaryOp::Relu, 0.0f, stream)));

      RETURN_IF_ERROR(RunGroupedGemm(scratch.hidden.data<float>(),
                                     output.GetTensorMutableData<float>(),
                                     second_weights, second_biases, offsets,
                                     hidden_width, output_width, stream));
      return LaunchStatus(LaunchSplitSequenceMoESkipLayerNormFloatKernel(
          static_cast<const float*>(x_buffer.data()),
          output.GetTensorMutableData<float>(), gamma, beta,
          scratch.group_offsets.data<int64_t>(),
          output.GetTensorMutableData<float>(), rows, group_count, output_width,
          epsilon, stream));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  size_t x_input_index;
  size_t output_index;
  std::vector<BoundarySource> boundaries;
  int64_t group_count;
  int64_t input_width;
  int64_t hidden_width;
  int64_t output_width;
  float epsilon;
  mutable std::vector<float> host_constants;
  size_t first_weight_offset;
  size_t first_bias_offset;
  size_t second_weight_offset;
  size_t second_bias_offset;
  size_t gamma_offset;
  size_t beta_offset;
  mutable DeviceConstants device_constants;
  mutable std::mutex constants_mutex;
};

}  // namespace

bool IsSplitSequenceMoEFusionGraph(Ort::ConstGraph graph) {
  size_t gemm_count = 0;
  size_t relu_count = 0;
  size_t slice_count = 0;
  size_t layer_norm_count = 0;
  size_t concat_count = 0;
  size_t add_count = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (IsOnnxOp(node, "Gemm")) {
      ++gemm_count;
    } else if (IsOnnxOp(node, "Relu")) {
      ++relu_count;
    } else if (IsOnnxOp(node, "Slice")) {
      ++slice_count;
    } else if (IsOnnxOp(node, "LayerNormalization")) {
      ++layer_norm_count;
    } else if (IsOnnxOp(node, "Concat")) {
      ++concat_count;
    } else if (IsOnnxOp(node, "Add")) {
      ++add_count;
    } else {
      return false;
    }
  }
  const size_t groups = layer_norm_count;
  return groups >= 2 && slice_count == groups * 2 && gemm_count == groups * 2 &&
         relu_count == groups && concat_count == 2 && add_count == 1;
}

std::unique_ptr<FusionNodeCompute> CreateSplitSequenceMoEFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  ParsedGraph parsed = ParseSplitSequenceMoEGraph(graph);
  if (parsed.branches.size() < 2) {
    throw std::runtime_error("SplitSequenceMoE requires at least two groups");
  }
  const int64_t group_count = static_cast<int64_t>(parsed.branches.size());
  auto fused_input_indices = ValueIndices(fused_node.GetInputs());
  auto fused_output_indices = ValueIndices(fused_node.GetOutputs());
  const size_t x_input_index = IndexOf(fused_input_indices, Name(parsed.x));
  const size_t output_index =
      IndexOf(fused_output_indices, Name(parsed.output));

  const auto x_shape = GetTensorShape(parsed.x);
  if (!x_shape.has_value() || x_shape->size() != 2) {
    throw std::runtime_error("SplitSequenceMoE x shape is invalid");
  }
  const int64_t input_width = x_shape->back() > 0
                                  ? x_shape->back()
                                  : parsed.branches.front().up.input_width;
  int64_t hidden_width = 0;
  int64_t output_width = 0;
  std::vector<float> constants;
  const size_t first_weight_offset = constants.size();
  for (const ParsedBranch& branch : parsed.branches) {
    auto shape = musa_ep::GetStaticShape(branch.up.weight);
    if (!shape.has_value() || shape->size() != 2 ||
        (*shape)[1] != input_width) {
      throw std::runtime_error("SplitSequenceMoE up weight shape mismatch");
    }
    if (hidden_width == 0) {
      hidden_width = (*shape)[0];
    } else if (hidden_width != (*shape)[0]) {
      throw std::runtime_error("SplitSequenceMoE hidden widths differ");
    }
    AppendInitializer(branch.up.weight,
                      static_cast<size_t>(hidden_width * input_width),
                      constants);
  }
  const size_t first_bias_offset = constants.size();
  for (const ParsedBranch& branch : parsed.branches) {
    AppendInitializer(branch.up.bias, static_cast<size_t>(hidden_width),
                      constants);
  }
  const size_t second_weight_offset = constants.size();
  for (const ParsedBranch& branch : parsed.branches) {
    auto shape = musa_ep::GetStaticShape(branch.down.weight);
    if (!shape.has_value() || shape->size() != 2 ||
        (*shape)[1] != hidden_width) {
      throw std::runtime_error("SplitSequenceMoE down weight shape mismatch");
    }
    if (output_width == 0) {
      output_width = (*shape)[0];
    } else if (output_width != (*shape)[0]) {
      throw std::runtime_error("SplitSequenceMoE output widths differ");
    }
    AppendInitializer(branch.down.weight,
                      static_cast<size_t>(output_width * hidden_width),
                      constants);
  }
  if (output_width != input_width) {
    throw std::runtime_error(
        "SplitSequenceMoE skip LayerNorm requires residual width equality");
  }
  const size_t second_bias_offset = constants.size();
  for (const ParsedBranch& branch : parsed.branches) {
    AppendInitializer(branch.down.bias, static_cast<size_t>(output_width),
                      constants);
  }
  const size_t gamma_offset = constants.size();
  for (const ParsedBranch& branch : parsed.branches) {
    AppendInitializer(branch.gamma, static_cast<size_t>(output_width),
                      constants);
  }
  const size_t beta_offset = constants.size();
  for (const ParsedBranch& branch : parsed.branches) {
    AppendInitializer(branch.beta, static_cast<size_t>(output_width),
                      constants);
  }

  std::vector<BoundarySource> boundaries;
  boundaries.reserve(parsed.branches.size() + 1);
  auto add_boundary = [&](Ort::ConstValueInfo value) {
    if (auto constant = musa_ep::ReadScalarIntInitializer(value);
        constant.has_value()) {
      boundaries.push_back({0, true, *constant});
      return;
    }
    boundaries.push_back({IndexOf(fused_input_indices, Name(value)), false, 0});
  };
  add_boundary(parsed.branches.front().slice.start);
  for (const ParsedBranch& branch : parsed.branches) {
    add_boundary(branch.slice.end);
  }

  return std::make_unique<SplitSequenceMoEFusionCompute>(
      x_input_index, output_index, std::move(boundaries), group_count,
      input_width, hidden_width, output_width, parsed.epsilon,
      std::move(constants), first_weight_offset, first_bias_offset,
      second_weight_offset, second_bias_offset, gamma_offset, beta_offset);
}
