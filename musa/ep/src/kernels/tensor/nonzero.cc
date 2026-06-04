// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/nonzero_impl.h"

namespace {

class MusaDeviceBuffer {
 public:
  explicit MusaDeviceBuffer(size_t bytes) {
    if (bytes != 0) {
      Ort::ThrowOnError(LaunchStatus(musaMalloc(&ptr_, bytes)));
    }
  }
  ~MusaDeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)musaFree(ptr_);
    }
  }
  MusaDeviceBuffer(const MusaDeviceBuffer&) = delete;
  MusaDeviceBuffer& operator=(const MusaDeviceBuffer&) = delete;
  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

MusaNonZeroParams MakeNonZeroParams(const std::vector<int64_t>& shape,
                                    int64_t nonzero_elements) {
  MusaNonZeroParams params{};
  const std::vector<int64_t> effective_shape =
      shape.empty() ? std::vector<int64_t>{1} : shape;
  auto strides = Strides(effective_shape);
  params.rank = static_cast<int32_t>(effective_shape.size());
  params.total_elements = NumElements(effective_shape);
  params.nonzero_elements = nonzero_elements;
  for (size_t i = 0; i < effective_shape.size(); ++i) {
    params.input_strides[i] = strides[i];
  }
  return params;
}

class NonZero : public OpKernelBase<NonZero> {
 public:
  NonZero(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* NonZero::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto elem_type = input_info.GetElementType();
  auto input_shape = input_info.GetShape();
  const std::vector<int64_t> effective_shape =
      input_shape.empty() ? std::vector<int64_t>{1} : input_shape;
  if (effective_shape.size() > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "NonZero rank exceeds MUSA kernel limit");
  }
  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "NonZero requires MUSA input");
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return UnsupportedDeviceElementwiseStatus("NonZero", elem_type);
  }

  const int64_t total_elements = NumElements(effective_shape);
  int64_t nonzero_elements = 0;
  MusaDeviceBuffer device_counts(
      static_cast<size_t>(NonZeroBlockCount(total_elements)) * sizeof(int));
  if (total_elements > 0) {
    musaError_t status = LaunchMusaNonZeroCountKernel(
        input.GetTensorRawData(), total_elements,
        static_cast<int*>(device_counts.get()), musa_elem_type, nullptr);
    if (status == musaErrorNotSupported) {
      return UnsupportedDeviceElementwiseStatus("NonZero", elem_type);
    }
    RETURN_IF_ERROR(LaunchStatus(status));

    std::vector<int> prefix_counts(
        static_cast<size_t>(NonZeroBlockCount(total_elements)));
    RETURN_IF_ERROR(LaunchStatus(musaMemcpy(
        prefix_counts.data(), device_counts.get(),
        prefix_counts.size() * sizeof(int), musaMemcpyDeviceToHost)));
    for (size_t i = 1; i < prefix_counts.size(); ++i) {
      prefix_counts[i] += prefix_counts[i - 1];
    }
    nonzero_elements = prefix_counts.empty() ? 0 : prefix_counts.back();
    RETURN_IF_ERROR(LaunchStatus(musaMemcpy(
        device_counts.get(), prefix_counts.data(),
        prefix_counts.size() * sizeof(int), musaMemcpyHostToDevice)));
  }

  const int64_t rank = static_cast<int64_t>(effective_shape.size());
  Ort::UnownedValue output = ctx.GetOutput(0, {rank, nonzero_elements});
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "NonZero requires MUSA output");
  }
  if (nonzero_elements == 0) {
    return nullptr;
  }

  musaError_t status = LaunchMusaNonZeroOutputKernel(
      input.GetTensorRawData(), static_cast<const int*>(device_counts.get()),
      output.GetTensorMutableData<int64_t>(),
      MakeNonZeroParams(input_shape, nonzero_elements), musa_elem_type,
      nullptr);
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus("NonZero", elem_type);
  }
  return LaunchStatus(status);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    NonZero, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", NonZeroTensorTypes())),
    NonZero)
