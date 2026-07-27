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

#include <limits>

#include "math/topk_impl.h"
#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {

enum class TopKAlgorithm {
  Empty,
  PairReduce,
  MudnnTopKStablePostprocess,
  BlockSort,
  SegmentedRadixSort,
};

bool IsMudnnTopKType(ONNXTensorElementDataType elem_type) {
  switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return true;
    default:
      return false;
  }
}

TopKAlgorithm SelectTopKAlgorithm(ONNXTensorElementDataType elem_type,
                                  const MusaTopKParams& params) {
  if (params.output_elements == 0) {
    return TopKAlgorithm::Empty;
  }
  if (params.k == 1) {
    return TopKAlgorithm::PairReduce;
  }
  if (IsMudnnTopKType(elem_type) &&
      params.k <= kMusaTopKStablePostprocessMaxK) {
    return TopKAlgorithm::MudnnTopKStablePostprocess;
  }
  if (params.dim <= kMusaTopKBlockSortMaxDim) {
    return TopKAlgorithm::BlockSort;
  }
  return TopKAlgorithm::SegmentedRadixSort;
}

bool TryMudnnTopK(Ort::KernelContext& ctx,
                  const std::vector<int64_t>& input_shape,
                  const std::vector<int64_t>& output_shape,
                  ONNXTensorElementDataType elem_type, int64_t axis, int64_t k,
                  int64_t largest, Ort::ConstValue input,
                  Ort::UnownedValue values, Ort::UnownedValue indices,
                  MusaTopKParams params, MusaElementType musa_elem_type) {
  if (!IsMudnnTopKType(elem_type) || axis > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() ||
      !IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(values.GetTensorMemoryInfo()) ||
      !IsGpuMemory(indices.GetTensorMemoryInfo())) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  musaStream_t stream = GetComputeStream(ctx);
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, stream);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor values_tensor;
  ::musa::dnn::Tensor indices_tensor;
  if (!SetMudnnTensor(input_tensor, input.GetTensorRawData(), input_shape,
                      elem_type) ||
      !SetMudnnTensor(values_tensor, values.GetTensorMutableRawData(),
                      output_shape, elem_type) ||
      !SetMudnnTensor(indices_tensor, indices.GetTensorMutableData<int64_t>(),
                      output_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }

  ::musa::dnn::TopK op;
  if (op.SetK(static_cast<int>(k)) != ::musa::dnn::Status::SUCCESS ||
      op.SetDim(static_cast<int>(axis)) != ::musa::dnn::Status::SUCCESS ||
      op.SetLargest(largest != 0) != ::musa::dnn::Status::SUCCESS ||
      op.SetSorted(true) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  ::musa::dnn::MemoryMaintainer maintainer =
      [stream](size_t bytes) -> ::musa::dnn::MemoryHandler {
    void* ptr = AllocateDeviceMemoryOnStream(bytes, stream);
    return ::musa::dnn::MemoryHandler(ptr, [stream, bytes](void* p) {
      FreeDeviceMemoryOnStream(p, stream, bytes);
    });
  };

  if (op.Run(*handle, values_tensor, indices_tensor, input_tensor,
             maintainer) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  return LaunchMusaTopKStablePostprocessKernel(
             input.GetTensorRawData(), values.GetTensorMutableRawData(),
             indices.GetTensorMutableData<int64_t>(), params, musa_elem_type,
             stream) == musaSuccess;
}

class TopK : public OpKernelBase<TopK> {
 public:
  TopK(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", -1);
    largest_ = AttrOrDefault<int64_t>(kernel_info, "largest", 1);
    sorted_ = AttrOrDefault<int64_t>(kernel_info, "sorted", 1);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = -1;
  int64_t largest_ = 1;
  int64_t sorted_ = 1;
};

OrtStatus* TopK::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto input_shape = input_info.GetShape();
  const auto elem_type = input_info.GetElementType();

  if (input_shape.empty()) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "TopK input rank must be at least 1");
  }
  if (largest_ != 0 && largest_ != 1) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "TopK largest must be 0 or 1");
  }
  if (sorted_ != 0 && sorted_ != 1) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "TopK sorted must be 0 or 1");
  }

  const int64_t axis = NormalizeAxis(axis_, input_shape.size());
  if (axis < 0 || axis >= static_cast<int64_t>(input_shape.size())) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "TopK axis out of range");
  }

  Ort::ConstValue k_value = ctx.GetInput(1);
  auto k_info = k_value.GetTensorTypeAndShapeInfo();
  if (k_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      k_info.GetElementCount() != 1) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "TopK K input must be a scalar int64 tensor");
  }
  if (IsGpuMemory(k_value.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "TopK K input must be CPU metadata");
  }
  const int64_t k = k_value.GetTensorData<int64_t>()[0];
  if (k < 0 || k > input_shape[static_cast<size_t>(axis)]) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "TopK K value is outside axis range");
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "TopK unsupported dtype");
  }
  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "TopK requires MUSA input tensor");
  }

  std::vector<int64_t> output_shape = input_shape;
  output_shape[static_cast<size_t>(axis)] = k;
  Ort::UnownedValue values = ctx.GetOutput(0, output_shape);
  Ort::UnownedValue indices = ctx.GetOutput(1, output_shape);
  if (!values || !indices || !IsGpuMemory(values.GetTensorMemoryInfo()) ||
      !IsGpuMemory(indices.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "TopK requires MUSA output tensors");
  }
  int64_t inner = 1;
  for (size_t dim = static_cast<size_t>(axis) + 1; dim < input_shape.size();
       ++dim) {
    inner *= input_shape[dim];
  }
  int64_t rows = 1;
  for (size_t dim = 0; dim < input_shape.size(); ++dim) {
    if (dim == static_cast<size_t>(axis)) {
      continue;
    }
    rows *= input_shape[dim];
  }

  MusaTopKParams params{};
  params.rows = rows;
  params.dim = input_shape[static_cast<size_t>(axis)];
  params.inner = inner;
  params.k = k;
  params.output_elements = NumElements(output_shape);
  params.largest = largest_ == 0 ? 0 : 1;
  params.sorted = sorted_ == 0 ? 0 : 1;

  const void* input_data = input.GetTensorRawData();
  void* values_data = values.GetTensorMutableRawData();
  int64_t* indices_data = indices.GetTensorMutableData<int64_t>();
  musaStream_t stream = GetComputeStream(ctx);
  TopKAlgorithm algorithm = SelectTopKAlgorithm(elem_type, params);
  musaError_t status = musaSuccess;

  if (algorithm == TopKAlgorithm::Empty) {
    return nullptr;
  }
  if (algorithm == TopKAlgorithm::PairReduce) {
    status = LaunchMusaTopKPairReduceKernel(
        input_data, values_data, indices_data, params, musa_elem_type, stream);
  } else if (algorithm == TopKAlgorithm::MudnnTopKStablePostprocess) {
    if (TryMudnnTopK(ctx, input_shape, output_shape, elem_type, axis, k,
                     largest_, input, values, indices, params,
                     musa_elem_type)) {
      return nullptr;
    }
    algorithm = params.dim <= kMusaTopKBlockSortMaxDim
                    ? TopKAlgorithm::BlockSort
                    : TopKAlgorithm::SegmentedRadixSort;
  }

  if (algorithm == TopKAlgorithm::BlockSort) {
    status = LaunchMusaTopKBlockSortKernel(
        input_data, values_data, indices_data, params, musa_elem_type, stream);
  } else if (algorithm == TopKAlgorithm::SegmentedRadixSort) {
    size_t workspace_bytes = 0;
    status = GetMusaTopKRadixSortWorkspaceSize(params, musa_elem_type,
                                               &workspace_bytes);
    if (status == musaSuccess) {
      void* workspace = AllocateDeviceMemoryOnStream(workspace_bytes, stream);
      if (workspace == nullptr) {
        return Ort::GetApi().CreateStatus(
            ORT_FAIL, "TopK failed to allocate radix-sort workspace");
      }
      status = LaunchMusaTopKRadixSortKernel(
          input_data, values_data, indices_data, params, musa_elem_type,
          workspace, workspace_bytes, stream);
      FreeDeviceMemoryOnStream(workspace, stream, workspace_bytes);
    }
  }

  if (status == musaErrorNotSupported) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "TopK unsupported dtype or tensor size");
  }
  return LaunchStatus(status);
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK, kOnnxDomain, 10, 10,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", TopKTensorTypes())
         .AddTypeConstraint("I",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    TopK)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK, kOnnxDomain, 11, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", TopKTensorTypes())
         .AddTypeConstraint("I",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    TopK)
