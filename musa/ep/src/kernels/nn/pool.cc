// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nn/pool_impl.h"
#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
constexpr size_t kMudnnMaxPoolSpatialRank = 3;

OrtStatus* PoolStatus(const char* op_name, const char* message,
                      OrtErrorCode code = ORT_NOT_IMPLEMENTED) {
  std::string full = std::string(op_name) + " " + message;
  return Ort::GetApi().CreateStatus(code, full.c_str());
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

bool IntVector(const std::vector<int64_t>& values, std::vector<int>& out) {
  out.clear();
  out.reserve(values.size());
  for (int64_t value : values) {
    if (value > std::numeric_limits<int>::max() ||
        value < std::numeric_limits<int>::min()) {
      return false;
    }
    out.push_back(static_cast<int>(value));
  }
  return true;
}

bool MudnnPoolFormat(size_t rank, ::musa::dnn::Tensor::Format& format) {
  switch (rank) {
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

bool TryMudnnPool(const void* input_data, void* output_data,
                  const std::vector<int64_t>& input_shape,
                  const std::vector<int64_t>& output_shape,
                  const std::vector<int64_t>& kernel_shape,
                  const std::vector<int64_t>& pads,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& dilations,
                  ONNXTensorElementDataType elem_type,
                  ::musa::dnn::Pooling::Mode mode, musaStream_t stream) {
  const size_t spatial_rank = input_shape.size() - 2;
  if (spatial_rank == 0 || spatial_rank > kMudnnMaxPoolSpatialRank ||
      input_shape.size() < 4) {
    return false;
  }

  std::vector<int> kernel;
  std::vector<int> pads_begin;
  std::vector<int> stride_values;
  std::vector<int> dilation_values;
  std::vector<int64_t> begin_pads(spatial_rank, 0);
  for (size_t i = 0; i < spatial_rank; ++i) {
    if (pads[i] != pads[i + spatial_rank]) {
      return false;
    }
    begin_pads[i] = pads[i];
  }
  if (!IntVector(kernel_shape, kernel) || !IntVector(begin_pads, pads_begin) ||
      !IntVector(strides, stride_values) ||
      !IntVector(dilations, dilation_values)) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, stream);
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
  if (pool.SetMode(mode) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  if (pool.SetNdInfo(static_cast<int>(spatial_rank), kernel.data(),
                     pads_begin.data(), stride_values.data(),
                     dilation_values.data()) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  ::musa::dnn::Tensor indices_tensor;
  if (!SetMudnnTensor(indices_tensor, nullptr, {0},
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }
  return pool.Run(*handle, output_tensor, input_tensor, indices_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

bool TryMudnnGlobalAveragePool(const void* input_data, void* output_data,
                               const std::vector<int64_t>& input_shape,
                               const std::vector<int64_t>& output_shape,
                               ONNXTensorElementDataType elem_type,
                               musaStream_t stream) {
  const size_t spatial_rank = input_shape.size() - 2;
  std::vector<int64_t> kernel_shape(input_shape.begin() + 2, input_shape.end());
  std::vector<int64_t> pads(spatial_rank * 2, 0);
  std::vector<int64_t> strides(spatial_rank, 1);
  std::vector<int64_t> dilations(spatial_rank, 1);
  return TryMudnnPool(input_data, output_data, input_shape, output_shape,
                      kernel_shape, pads, strides, dilations, elem_type,
                      ::musa::dnn::Pooling::Mode::GLOBAL_AVGPOOL, stream);
}

bool TryMudnnMaxPool(const void* input_data, void* output_data,
                     const std::vector<int64_t>& input_shape,
                     const std::vector<int64_t>& output_shape,
                     const std::vector<int64_t>& kernel_shape,
                     const std::vector<int64_t>& pads,
                     const std::vector<int64_t>& strides,
                     const std::vector<int64_t>& dilations,
                     ONNXTensorElementDataType elem_type, musaStream_t stream) {
  return TryMudnnPool(input_data, output_data, input_shape, output_shape,
                      kernel_shape, pads, strides, dilations, elem_type,
                      ::musa::dnn::Pooling::Mode::MAXPOOL, stream);
}

bool HfdType(ONNXTensorElementDataType elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

std::vector<int64_t> NormalizeSpatialAttr(std::vector<int64_t> values,
                                          size_t spatial_rank,
                                          int64_t default_value) {
  if (values.empty()) {
    values.assign(spatial_rank, default_value);
  }
  return values;
}

std::vector<int64_t> NormalizePads(std::vector<int64_t> pads,
                                   size_t spatial_rank) {
  if (pads.empty()) {
    pads.assign(spatial_rank * 2, 0);
  }
  return pads;
}

OrtStatus* ValidatePoolCommon(const char* op_name,
                              const std::vector<int64_t>& input_shape,
                              const std::vector<int64_t>& kernel_shape,
                              const std::vector<int64_t>& pads,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& dilations) {
  if (input_shape.size() < 3) {
    return PoolStatus(op_name, "input rank must be at least 3",
                      ORT_INVALID_ARGUMENT);
  }
  if (input_shape.size() > kMusaMaxBroadcastRank) {
    return PoolStatus(op_name, "rank exceeds MUSA device limit");
  }
  for (int64_t dim : input_shape) {
    if (dim < 0) {
      return PoolStatus(op_name, "requires concrete input shape",
                        ORT_INVALID_ARGUMENT);
    }
  }

  const size_t spatial_rank = input_shape.size() - 2;
  if (kernel_shape.size() != spatial_rank || pads.size() != spatial_rank * 2 ||
      strides.size() != spatial_rank || dilations.size() != spatial_rank) {
    return PoolStatus(op_name, "attribute rank mismatch", ORT_INVALID_ARGUMENT);
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    if (kernel_shape[i] <= 0 || strides[i] <= 0 || dilations[i] <= 0) {
      return PoolStatus(op_name,
                        "kernel_shape, strides and dilations must be positive",
                        ORT_INVALID_ARGUMENT);
    }
    if (pads[i] < 0 || pads[i + spatial_rank] < 0) {
      return PoolStatus(op_name, "pads must be non-negative",
                        ORT_INVALID_ARGUMENT);
    }
  }
  return nullptr;
}

std::vector<int64_t> PoolOutputShape(const std::vector<int64_t>& input_shape,
                                     const std::vector<int64_t>& kernel_shape,
                                     const std::vector<int64_t>& pads,
                                     const std::vector<int64_t>& strides,
                                     const std::vector<int64_t>& dilations) {
  std::vector<int64_t> output_shape = input_shape;
  const size_t spatial_rank = input_shape.size() - 2;
  for (size_t i = 0; i < spatial_rank; ++i) {
    const int64_t effective_kernel = (kernel_shape[i] - 1) * dilations[i] + 1;
    const int64_t numerator = input_shape[i + 2] + pads[i] +
                              pads[i + spatial_rank] - effective_kernel;
    output_shape[i + 2] = numerator < 0 ? 0 : numerator / strides[i] + 1;
  }
  return output_shape;
}

MusaMaxPoolParams MakeMaxPoolParams(const std::vector<int64_t>& input_shape,
                                    const std::vector<int64_t>& output_shape,
                                    const std::vector<int64_t>& kernel_shape,
                                    const std::vector<int64_t>& pads,
                                    const std::vector<int64_t>& strides,
                                    const std::vector<int64_t>& dilations,
                                    bool has_indices) {
  MusaMaxPoolParams params{};
  params.rank = static_cast<int32_t>(input_shape.size());
  params.spatial_rank = static_cast<int32_t>(input_shape.size() - 2);
  params.has_indices = has_indices ? 1 : 0;
  params.output_elements = NumElements(output_shape);

  auto input_strides = Strides(input_shape);
  auto output_strides = Strides(output_shape);
  for (size_t i = 0; i < input_shape.size(); ++i) {
    params.input_dims[i] = input_shape[i];
    params.input_strides[i] = input_strides[i];
    params.output_dims[i] = output_shape[i];
    params.output_strides[i] = output_strides[i];
  }
  for (int32_t i = 0; i < params.spatial_rank; ++i) {
    const auto idx = static_cast<size_t>(i);
    params.kernel_shape[i] = kernel_shape[idx];
    params.pads_begin[i] = pads[idx];
    params.strides[i] = strides[idx];
    params.dilations[i] = dilations[idx];
  }
  return params;
}

class GlobalAveragePool : public OpKernelBase<GlobalAveragePool> {
 public:
  GlobalAveragePool(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* GlobalAveragePool::Compute(Ort::KernelContext& ctx) const {
  musaStream_t stream = GetComputeStream(ctx);
  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  const auto elem_type = input_info.GetElementType();
  auto input_shape = input_info.GetShape();

  if (!HfdType(elem_type)) {
    return PoolStatus("GlobalAveragePool", "unsupported dtype");
  }
  if (input_shape.size() < 3) {
    return PoolStatus("GlobalAveragePool", "input rank must be at least 3",
                      ORT_INVALID_ARGUMENT);
  }

  const size_t spatial_rank = input_shape.size() - 2;
  std::vector<int64_t> kernel_shape(input_shape.begin() + 2, input_shape.end());
  std::vector<int64_t> pads(spatial_rank * 2, 0);
  std::vector<int64_t> strides(spatial_rank, 1);
  std::vector<int64_t> dilations(spatial_rank, 1);
  RETURN_IF_ERROR(ValidatePoolCommon("GlobalAveragePool", input_shape,
                                     kernel_shape, pads, strides, dilations));

  std::vector<int64_t> output_shape = input_shape;
  int64_t spatial_elements = 1;
  for (size_t i = 2; i < input_shape.size(); ++i) {
    spatial_elements *= input_shape[i];
    output_shape[i] = 1;
  }
  if (spatial_elements <= 0) {
    return PoolStatus("GlobalAveragePool",
                      "requires non-empty spatial dimensions",
                      ORT_INVALID_ARGUMENT);
  }

  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return PoolStatus("GlobalAveragePool", "requires MUSA device tensors");
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return PoolStatus("GlobalAveragePool", "unsupported dtype");
  }

  if (TryMudnnGlobalAveragePool(input.GetTensorRawData(),
                                output.GetTensorMutableRawData(), input_shape,
                                output_shape, elem_type, stream)) {
    return nullptr;
  }

  MusaGlobalAveragePoolParams params{};
  params.channels = input_shape[1];
  params.spatial_elements = spatial_elements;
  params.output_elements = NumElements(output_shape);
  musaError_t status = LaunchMusaGlobalAveragePoolKernel(
      input.GetTensorRawData(), output.GetTensorMutableRawData(), params,
      musa_elem_type, stream);
  if (status == musaErrorNotSupported) {
    return PoolStatus("GlobalAveragePool", "unsupported dtype");
  }
  return LaunchStatus(status);
}

class GlobalMaxPool : public OpKernelBase<GlobalMaxPool> {
 public:
  GlobalMaxPool(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* GlobalMaxPool::Compute(Ort::KernelContext& ctx) const {
  musaStream_t stream = GetComputeStream(ctx);
  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  const auto elem_type = input_info.GetElementType();
  auto input_shape = input_info.GetShape();

  if (!HfdType(elem_type)) {
    return PoolStatus("GlobalMaxPool", "unsupported dtype");
  }
  if (input_shape.size() < 3) {
    return PoolStatus("GlobalMaxPool", "input rank must be at least 3",
                      ORT_INVALID_ARGUMENT);
  }

  const size_t spatial_rank = input_shape.size() - 2;
  std::vector<int64_t> kernel_shape(input_shape.begin() + 2, input_shape.end());
  std::vector<int64_t> pads(spatial_rank * 2, 0);
  std::vector<int64_t> strides(spatial_rank, 1);
  std::vector<int64_t> dilations(spatial_rank, 1);
  RETURN_IF_ERROR(ValidatePoolCommon("GlobalMaxPool", input_shape, kernel_shape,
                                     pads, strides, dilations));

  std::vector<int64_t> output_shape = input_shape;
  for (size_t i = 2; i < output_shape.size(); ++i) {
    output_shape[i] = 1;
  }

  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return PoolStatus("GlobalMaxPool", "requires MUSA device tensors");
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return PoolStatus("GlobalMaxPool", "unsupported dtype");
  }

  musaError_t status = LaunchMusaMaxPoolKernel(
      input.GetTensorRawData(), output.GetTensorMutableRawData(), nullptr,
      MakeMaxPoolParams(input_shape, output_shape, kernel_shape, pads, strides,
                        dilations, false),
      musa_elem_type, stream);
  if (status == musaErrorNotSupported) {
    return PoolStatus("GlobalMaxPool", "unsupported dtype");
  }
  return LaunchStatus(status);
}

class MaxPool : public OpKernelBase<MaxPool> {
 public:
  MaxPool(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    auto_pad_ = AttrOrDefault<std::string>(kernel_info, "auto_pad", "NOTSET");
    ceil_mode_ = AttrOrDefault<int64_t>(kernel_info, "ceil_mode", 0);
    storage_order_ = AttrOrDefault<int64_t>(kernel_info, "storage_order", 0);
    kernel_shape_ = AttrsOrEmpty(kernel_info, "kernel_shape");
    pads_ = AttrsOrEmpty(kernel_info, "pads");
    strides_ = AttrsOrEmpty(kernel_info, "strides");
    dilations_ = AttrsOrEmpty(kernel_info, "dilations");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::string auto_pad_;
  int64_t ceil_mode_ = 0;
  int64_t storage_order_ = 0;
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilations_;
};

OrtStatus* MaxPool::Compute(Ort::KernelContext& ctx) const {
  musaStream_t stream = GetComputeStream(ctx);
  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  const auto elem_type = input_info.GetElementType();
  auto input_shape = input_info.GetShape();
  const size_t spatial_rank =
      input_shape.size() >= 2 ? input_shape.size() - 2 : 0;

  if (auto_pad_ != "NOTSET") {
    return PoolStatus("MaxPool", "only supports auto_pad=NOTSET");
  }
  if (ceil_mode_ != 0) {
    return PoolStatus("MaxPool", "ceil_mode is not supported");
  }
  if (storage_order_ != 0) {
    return PoolStatus("MaxPool", "only supports storage_order=0");
  }
  if (kernel_shape_.empty()) {
    return PoolStatus("MaxPool", "requires kernel_shape", ORT_INVALID_ARGUMENT);
  }

  std::vector<int64_t> pads = NormalizePads(pads_, spatial_rank);
  std::vector<int64_t> strides =
      NormalizeSpatialAttr(strides_, spatial_rank, 1);
  std::vector<int64_t> dilations =
      NormalizeSpatialAttr(dilations_, spatial_rank, 1);
  RETURN_IF_ERROR(ValidatePoolCommon("MaxPool", input_shape, kernel_shape_,
                                     pads, strides, dilations));

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return PoolStatus("MaxPool", "unsupported dtype");
  }

  std::vector<int64_t> output_shape =
      PoolOutputShape(input_shape, kernel_shape_, pads, strides, dilations);
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return PoolStatus("MaxPool", "requires MUSA device tensors");
  }

  Ort::UnownedValue indices_output(nullptr);
  int64_t* indices_data = nullptr;
  bool has_indices = false;
  if (ctx.GetOutputCount() > 1) {
    indices_output = ctx.GetOutput(1, output_shape);
    if (indices_output) {
      if (!IsGpuMemory(indices_output.GetTensorMemoryInfo())) {
        return PoolStatus("MaxPool", "indices output must be MUSA tensor");
      }
      indices_data = indices_output.GetTensorMutableData<int64_t>();
      has_indices = true;
    }
  }

  if (!has_indices && HfdType(elem_type) &&
      TryMudnnMaxPool(input.GetTensorRawData(),
                      output.GetTensorMutableRawData(), input_shape,
                      output_shape, kernel_shape_, pads, strides, dilations,
                      elem_type, stream)) {
    return nullptr;
  }

  musaError_t status = LaunchMusaMaxPoolKernel(
      input.GetTensorRawData(), output.GetTensorMutableRawData(), indices_data,
      MakeMaxPoolParams(input_shape, output_shape, kernel_shape_, pads, strides,
                        dilations, has_indices),
      musa_elem_type, stream);
  if (status == musaErrorNotSupported) {
    return PoolStatus("MaxPool", "unsupported dtype");
  }
  return LaunchStatus(status);
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GlobalAveragePool, kOnnxDomain, 1, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())),
    GlobalAveragePool)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GlobalMaxPool, kOnnxDomain, 1, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())),
    GlobalMaxPool)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MaxPool, kOnnxDomain, 1, 7,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())), MaxPool)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MaxPool, kOnnxDomain, 8, 11,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", HfdTensorTypes())
         .AddTypeConstraint(
             "I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))),
    MaxPool)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MaxPool, kOnnxDomain, 12, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", MaxPoolOpset12TensorTypes())
         .AddTypeConstraint(
             "I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))),
    MaxPool)
