// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/gather_nd_impl.h"

namespace {

class GatherND : public OpKernelBase<GatherND> {
 public:
  GatherND(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    batch_dims_ = AttrOrDefault<int64_t>(kernel_info, "batch_dims", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t batch_dims_ = 0;
};

OrtStatus* GatherND::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  Ort::ConstValue indices = ctx.GetInput(1);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto indices_info = indices.GetTensorTypeAndShapeInfo();
  auto input_shape = input_info.GetShape();
  auto indices_shape = indices_info.GetShape();

  if (indices_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "GatherND only supports int64 indices");
  }
  if (indices_shape.empty()) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "GatherND indices rank must be larger than 0");
  }
  if (batch_dims_ < 0 ||
      batch_dims_ >= static_cast<int64_t>(indices_shape.size())) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "GatherND invalid batch_dims");
  }
  if (input_shape.size() > kMusaMaxBroadcastRank ||
      indices_shape.size() > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "GatherND rank exceeds MUSA device limit");
  }
  for (int64_t dim : input_shape) {
    if (dim < 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "GatherND requires concrete input shape");
    }
  }
  for (int64_t dim : indices_shape) {
    if (dim < 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT, "GatherND requires concrete indices shape");
    }
  }

  const int64_t num_slice_dims = indices_shape.back();
  const int64_t last_indices_dimension = batch_dims_ + num_slice_dims;
  if (num_slice_dims < 0 ||
      last_indices_dimension > static_cast<int64_t>(input_shape.size())) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "GatherND last dimension of indices exceeds input rank");
  }
  for (int64_t dim = 0; dim < batch_dims_; ++dim) {
    if (input_shape[static_cast<size_t>(dim)] !=
        indices_shape[static_cast<size_t>(dim)]) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "GatherND batch dimensions differ");
    }
  }

  const size_t elem_size = ElementSize(input_info.GetElementType());
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "GatherND unsupported dtype");
  }
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(indices.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "GatherND requires MUSA input tensors");
  }

  std::vector<int64_t> output_shape(indices_shape.begin(),
                                    indices_shape.end() - 1);
  output_shape.insert(output_shape.end(),
                      input_shape.begin() + last_indices_dimension,
                      input_shape.end());
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "GatherND requires MUSA output tensor");
  }

  MusaGatherNDParams params{};
  params.input_rank = static_cast<int32_t>(input_shape.size());
  params.indices_rank = static_cast<int32_t>(indices_shape.size());
  params.batch_dims = static_cast<int32_t>(batch_dims_);
  params.num_slice_dims = static_cast<int32_t>(num_slice_dims);
  params.output_elements = NumElements(output_shape);
  params.slice_size = NumElements(std::vector<int64_t>(
      input_shape.begin() + last_indices_dimension, input_shape.end()));
  const int64_t num_slices = NumElements(
      std::vector<int64_t>(indices_shape.begin(), indices_shape.end() - 1));
  const int64_t num_batches = NumElements(std::vector<int64_t>(
      input_shape.begin(), input_shape.begin() + batch_dims_));
  params.num_slices_per_batch = num_batches == 0 ? 0 : num_slices / num_batches;
  params.input_batch_stride = NumElements(std::vector<int64_t>(
      input_shape.begin() + batch_dims_, input_shape.end()));

  for (size_t i = 0; i < input_shape.size(); ++i) {
    params.input_dims[i] = input_shape[i];
  }
  int64_t running_product = params.slice_size;
  for (int64_t i = num_slice_dims - 1; i >= 0; --i) {
    params.sizes_from_slice_dims[static_cast<size_t>(i)] = running_product;
    running_product *= input_shape[static_cast<size_t>(batch_dims_ + i)];
  }

  return LaunchStatus(LaunchMusaGatherNDKernel(
      input.GetTensorRawData(), indices.GetTensorData<int64_t>(),
      output.GetTensorMutableRawData(), static_cast<int32_t>(elem_size), params,
      GetComputeStream(ctx)));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GatherND, kOnnxDomain, 11, 12,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GatherNDTensorTypes())
         .AddTypeConstraint(
             "indices", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))),
    GatherND)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GatherND, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GatherNDTensorTypes())
         .AddTypeConstraint(
             "indices", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))),
    GatherND)
