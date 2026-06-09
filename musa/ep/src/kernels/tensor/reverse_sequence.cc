// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/reverse_sequence_impl.h"

namespace {

class ReverseSequence : public OpKernelBase<ReverseSequence> {
 public:
  ReverseSequence(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    batch_axis_ = kernel_info.GetAttribute<int64_t>("batch_axis");
    time_axis_ = kernel_info.GetAttribute<int64_t>("time_axis");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t batch_axis_ = 0;
  int64_t time_axis_ = 1;
};

OrtStatus* ReverseSequence::Compute(Ort::KernelContext& ctx) const {
  if (batch_axis_ < 0 || time_axis_ < 0 || batch_axis_ > 1 || time_axis_ > 1 ||
      batch_axis_ == time_axis_) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "ReverseSequence requires batch_axis/time_axis to be 0 and 1");
  }

  Ort::ConstValue input = ctx.GetInput(0);
  Ort::ConstValue sequence_lens = ctx.GetInput(1);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto seq_info = sequence_lens.GetTensorTypeAndShapeInfo();
  auto input_shape = input_info.GetShape();
  auto seq_shape = seq_info.GetShape();

  if (input_shape.size() < 2) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "ReverseSequence input rank must be at least 2");
  }
  if (seq_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "ReverseSequence only supports int64 sequence_lens");
  }
  const int64_t batch_size = input_shape[static_cast<size_t>(batch_axis_)];
  const int64_t max_seq_len = input_shape[static_cast<size_t>(time_axis_)];
  if (seq_shape.size() != 1 || seq_shape[0] != batch_size) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "ReverseSequence sequence_lens shape must be {batch_size}");
  }
  for (int64_t dim : input_shape) {
    if (dim < 0) {
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "ReverseSequence requires concrete input shape");
    }
  }

  const size_t elem_size = ElementSize(input_info.GetElementType());
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "ReverseSequence unsupported dtype");
  }
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(sequence_lens.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ReverseSequence requires MUSA input tensors");
  }

  Ort::UnownedValue output = ctx.GetOutput(0, input_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "ReverseSequence requires MUSA output tensor");
  }

  int64_t element_size = 1;
  for (size_t dim = 2; dim < input_shape.size(); ++dim) {
    element_size *= input_shape[dim];
  }

  MusaReverseSequenceParams params{};
  params.batch_size = batch_size;
  params.max_seq_len = max_seq_len;
  params.element_size = element_size;
  params.total_elements = NumElements(input_shape);
  params.time_major = time_axis_ == 0 ? 1 : 0;

  return LaunchStatus(LaunchMusaReverseSequenceKernel(
      input.GetTensorRawData(), sequence_lens.GetTensorData<int64_t>(),
      output.GetTensorMutableRawData(), static_cast<int32_t>(elem_size), params,
      GetComputeStream(ctx)));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ReverseSequence, kOnnxDomain, 10, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllFixedSizeTensorTypes())),
    ReverseSequence)
