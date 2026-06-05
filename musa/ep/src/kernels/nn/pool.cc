// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nn/pool_impl.h"
#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
constexpr size_t kMudnnMaxPoolSpatialRank = 3;

OrtStatus* InvalidGlobalAveragePoolStatus(const char* message) {
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message);
}

bool IntShape(const std::vector<int64_t>& shape, size_t begin,
              std::vector<int>& out) {
  out.clear();
  out.reserve(shape.size() - begin);
  for (size_t i = begin; i < shape.size(); ++i) {
    if (shape[i] > std::numeric_limits<int>::max()) {
      return false;
    }
    out.push_back(static_cast<int>(shape[i]));
  }
  return true;
}

bool MudnnPoolFormat(size_t rank, ::musa::dnn::Tensor::Format& format) {
  switch (rank) {
    case 3:
      format = ::musa::dnn::Tensor::Format::NCW;
      return true;
    case 4:
      format = ::musa::dnn::Tensor::Format::NCHW;
      return true;
    case 5:
      format = ::musa::dnn::Tensor::Format::NCDHW;
      return true;
    default:
      return false;
  }
}

bool SetMudnnPoolTensor(::musa::dnn::Tensor& tensor, const void* data,
                        const std::vector<int64_t>& shape,
                        ONNXTensorElementDataType elem_type) {
  ::musa::dnn::Tensor::Type mudnn_type;
  ::musa::dnn::Tensor::Format format;
  if (!MudnnTensorType(elem_type, mudnn_type) ||
      !MudnnPoolFormat(shape.size(), format)) {
    return false;
  }
  return tensor.SetAddr(data) == ::musa::dnn::Status::SUCCESS &&
         tensor.SetType(mudnn_type) == ::musa::dnn::Status::SUCCESS &&
         tensor.SetFormat(format) == ::musa::dnn::Status::SUCCESS &&
         tensor.SetNdInfo(static_cast<int64_t>(shape.size()), shape.data()) ==
             ::musa::dnn::Status::SUCCESS;
}

bool TryMudnnGlobalAveragePool(const void* input_data, void* output_data,
                               const std::vector<int64_t>& input_shape,
                               const std::vector<int64_t>& output_shape,
                               ONNXTensorElementDataType elem_type) {
  const size_t spatial_rank = input_shape.size() - 2;
  if (spatial_rank == 0 || spatial_rank > kMudnnMaxPoolSpatialRank) {
    return false;
  }

  std::vector<int> kernel;
  if (!IntShape(input_shape, 2, kernel)) {
    return false;
  }
  std::vector<int> pads(spatial_rank, 0);
  std::vector<int> strides(spatial_rank, 1);
  std::vector<int> dilations(spatial_rank, 1);

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnPoolTensor(input_tensor, input_data, input_shape, elem_type) ||
      !SetMudnnPoolTensor(output_tensor, output_data, output_shape,
                          elem_type)) {
    return false;
  }

  ::musa::dnn::Pooling pool;
  if (pool.SetMode(::musa::dnn::Pooling::Mode::GLOBAL_AVGPOOL) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  if (pool.SetNdInfo(static_cast<int>(spatial_rank), kernel.data(),
                     pads.data(), strides.data(), dilations.data()) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  // Avg pooling does not consume max-pool indices, but the C++ API requires
  // a Tensor placeholder.
  ::musa::dnn::Tensor indices_tensor;
  if (!SetMudnnTensor(indices_tensor, nullptr, {0},
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }
  return pool.Run(*handle, output_tensor, input_tensor, indices_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

class GlobalAveragePool : public OpKernelBase<GlobalAveragePool> {
 public:
  GlobalAveragePool(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* GlobalAveragePool::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  const auto elem_type = input_info.GetElementType();
  auto input_shape = input_info.GetShape();

  if (input_shape.size() < 3) {
    return InvalidGlobalAveragePoolStatus(
        "GlobalAveragePool input rank must be at least 3");
  }
  if (input_shape.size() > kMusaMaxBroadcastRank) {
    return InvalidGlobalAveragePoolStatus(
        "GlobalAveragePool rank exceeds MUSA device limit");
  }
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    return InvalidGlobalAveragePoolStatus(
        "GlobalAveragePool unsupported dtype");
  }
  for (int64_t dim : input_shape) {
    if (dim < 0) {
      return InvalidGlobalAveragePoolStatus(
          "GlobalAveragePool requires concrete input shape");
    }
  }

  std::vector<int64_t> output_shape = input_shape;
  int64_t spatial_elements = 1;
  for (size_t i = 2; i < input_shape.size(); ++i) {
    spatial_elements *= input_shape[i];
    output_shape[i] = 1;
  }
  if (spatial_elements <= 0) {
    return InvalidGlobalAveragePoolStatus(
        "GlobalAveragePool requires non-empty spatial dimensions");
  }

  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return InvalidGlobalAveragePoolStatus(
        "GlobalAveragePool requires MUSA device tensors");
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return InvalidGlobalAveragePoolStatus(
        "GlobalAveragePool unsupported dtype");
  }

  MusaGlobalAveragePoolParams params{};
  params.channels = input_shape[1];
  params.spatial_elements = spatial_elements;
  params.output_elements = NumElements(output_shape);

  if (TryMudnnGlobalAveragePool(input.GetTensorRawData(),
                                output.GetTensorMutableRawData(), input_shape,
                                output_shape, elem_type)) {
    return nullptr;
  }

  musaError_t status = LaunchMusaGlobalAveragePoolKernel(
      input.GetTensorRawData(), output.GetTensorMutableRawData(), params,
      musa_elem_type, nullptr);
  if (status == musaErrorNotSupported) {
    return InvalidGlobalAveragePoolStatus(
        "GlobalAveragePool unsupported dtype");
  }
  return LaunchStatus(status);
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GlobalAveragePool, kOnnxDomain, 1, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())),
    GlobalAveragePool)
