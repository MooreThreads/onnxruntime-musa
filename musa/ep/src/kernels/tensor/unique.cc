// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>

#include "shared_inc/op_kernel_common.h"
#include "tensor/unique_impl.h"

namespace {

class MusaDeviceBuffer {
 public:
  MusaDeviceBuffer(size_t bytes, musaStream_t stream)
      : bytes_(bytes), stream_(stream) {
    if (bytes_ != 0) {
      ptr_ = AllocateDeviceMemoryOnStream(bytes_, stream_);
      status_ = ptr_ != nullptr ? musaSuccess : musaErrorMemoryAllocation;
    }
  }
  ~MusaDeviceBuffer() {
    if (ptr_ != nullptr) {
      FreeDeviceMemoryOnStream(ptr_, stream_, bytes_);
    }
  }
  MusaDeviceBuffer(const MusaDeviceBuffer&) = delete;
  MusaDeviceBuffer& operator=(const MusaDeviceBuffer&) = delete;

  bool OK() const { return status_ == musaSuccess; }
  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t stream_ = nullptr;
  musaError_t status_ = musaSuccess;
};

class Unique : public OpKernelBase<Unique> {
 public:
  Unique(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis",
                                   std::numeric_limits<int64_t>::min());
    sorted_ = AttrOrDefault<int64_t>(kernel_info, "sorted", 1);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_;
  int64_t sorted_;
};

OrtStatus* Unique::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto input_shape = input_info.GetShape();
  const auto elem_type = input_info.GetElementType();
  const int64_t input_count = input_info.GetElementCount();

  if (sorted_ != 0 && sorted_ != 1) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Unique sorted must be 0 or 1");
  }
  if (axis_ != std::numeric_limits<int64_t>::min()) {
    const int64_t axis = NormalizeAxis(axis_, input_shape.size());
    if (axis != 0 || input_shape.size() != 1) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "Unique only supports flattened input or axis=0 for 1D tensors");
    }
  }
  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Unique requires MUSA input");
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      (musa_elem_type != MusaElementType::Int32 &&
       musa_elem_type != MusaElementType::Int64)) {
    return UnsupportedDeviceElementwiseStatus("Unique", elem_type);
  }

  musaStream_t stream = GetComputeStream(ctx);
  const int block_count = UniqueBlockCount(input_count);
  MusaDeviceBuffer first_flags(static_cast<size_t>(input_count) * sizeof(int),
                               stream);
  MusaDeviceBuffer block_counts(static_cast<size_t>(block_count) * sizeof(int),
                                stream);
  if (!first_flags.OK() || !block_counts.OK()) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Unique temporary allocation failed");
  }

  int64_t unique_count = 0;
  if (input_count > 0) {
    musaError_t status = LaunchMusaUniqueCountKernel(
        input.GetTensorRawData(), input_count,
        static_cast<int*>(first_flags.get()),
        static_cast<int*>(block_counts.get()), musa_elem_type, stream);
    if (status == musaErrorNotSupported) {
      return UnsupportedDeviceElementwiseStatus("Unique", elem_type);
    }
    RETURN_IF_ERROR(LaunchStatus(status));

    std::vector<int> counts(static_cast<size_t>(block_count));
    RETURN_IF_ERROR(LaunchStatus(musaMemcpyAsync(
        counts.data(), block_counts.get(), counts.size() * sizeof(int),
        musaMemcpyDeviceToHost, stream)));
    RETURN_IF_ERROR(LaunchStatus(musaStreamSynchronize(stream)));
    for (int count : counts) {
      unique_count += count;
    }
  }

  Ort::UnownedValue values = ctx.GetOutput(0, {unique_count});
  if (!values || !IsGpuMemory(values.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Unique requires MUSA output");
  }

  Ort::UnownedValue indices = ctx.GetOutputCount() > 1
                                  ? ctx.GetOutput(1, {unique_count})
                                  : Ort::UnownedValue(nullptr);
  Ort::UnownedValue inverse_indices = ctx.GetOutputCount() > 2
                                          ? ctx.GetOutput(2, {input_count})
                                          : Ort::UnownedValue(nullptr);
  Ort::UnownedValue counts = ctx.GetOutputCount() > 3
                                 ? ctx.GetOutput(3, {unique_count})
                                 : Ort::UnownedValue(nullptr);

  if ((indices && !IsGpuMemory(indices.GetTensorMemoryInfo())) ||
      (inverse_indices &&
       !IsGpuMemory(inverse_indices.GetTensorMemoryInfo())) ||
      (counts && !IsGpuMemory(counts.GetTensorMemoryInfo()))) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Unique requires MUSA optional outputs");
  }
  if (input_count == 0) {
    return nullptr;
  }

  musaError_t status = LaunchMusaUniqueOutputKernel(
      input.GetTensorRawData(), input_count, unique_count,
      static_cast<const int*>(first_flags.get()),
      values.GetTensorMutableRawData(),
      indices ? indices.GetTensorMutableData<int64_t>() : nullptr,
      inverse_indices ? inverse_indices.GetTensorMutableData<int64_t>()
                      : nullptr,
      counts ? counts.GetTensorMutableData<int64_t>() : nullptr,
      sorted_ == 0 ? 0 : 1, musa_elem_type, stream);
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus("Unique", elem_type);
  }
  return LaunchStatus(status);
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unique, kOnnxDomain, 11, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", UniqueTensorTypes())),
    Unique)
