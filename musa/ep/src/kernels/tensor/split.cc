// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/split_impl.h"

namespace {

constexpr size_t kSplitManySmallOutputCount = 2;
constexpr size_t kSplitManySmallMaxWidthBytes = 32 * 1024;
constexpr size_t kSplitManySmallMaxMapBytes = 4 * 1024 * 1024;

class Split : public OpKernelBase<Split> {
 public:
  Split(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
};

OrtStatus* Split::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  int64_t axis = NormalizeAxis(axis_, shape0.size());
  std::vector<int64_t> splits;
  if (ctx.GetInputCount() > 1) {
    splits = ReadIntTensor(ctx, 1);
  } else {
    size_t count = ctx.GetOutputCount();
    splits.assign(
        count, shape0[static_cast<size_t>(axis)] / static_cast<int64_t>(count));
  }
  size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Split unsupported dtype");
  }
  int64_t axis_start = 0;
  const int64_t outer =
      axis == 0 ? 1
                : std::accumulate(shape0.begin(), shape0.begin() + axis,
                                  int64_t{1}, std::multiplies<int64_t>());
  const int64_t inner =
      axis + 1 == static_cast<int64_t>(shape0.size())
          ? 1
          : std::accumulate(shape0.begin() + axis + 1, shape0.end(), int64_t{1},
                            std::multiplies<int64_t>());
  const bool gpu_input = IsGpuMemory(input0.GetTensorMemoryInfo());
  if (!gpu_input) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Split requires MUSA input and outputs");
  }
  std::vector<void*> output_data(splits.size());
  for (size_t out_idx = 0; out_idx < splits.size(); ++out_idx) {
    std::vector<int64_t> out_shape = shape0;
    out_shape[static_cast<size_t>(axis)] = splits[out_idx];
    Ort::UnownedValue y = ctx.GetOutput(out_idx, out_shape);
    if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "Split requires MUSA input and outputs");
    }
    output_data[out_idx] = y.GetTensorMutableRawData();
    axis_start += splits[out_idx];
  }
  if (axis_start != shape0[static_cast<size_t>(axis)]) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Split sizes do not match input axis");
  }
  musaStream_t stream = GetComputeStream(ctx);
  const int64_t max_split =
      splits.empty() ? 0 : *std::max_element(splits.begin(), splits.end());
  const size_t max_width_bytes =
      static_cast<size_t>(max_split) * static_cast<size_t>(inner) * elem_size;
  if (output_data.size() >= kSplitManySmallOutputCount &&
      max_width_bytes <= kSplitManySmallMaxWidthBytes) {
    const int64_t input_row_elements =
        shape0[static_cast<size_t>(axis)] * inner;
    const size_t element_descriptor_bytes =
        static_cast<size_t>(input_row_elements) * sizeof(MusaSplitElementDesc);
    if (input_row_elements > 0 &&
        element_descriptor_bytes <= kSplitManySmallMaxMapBytes) {
      std::vector<MusaSplitElementDesc> element_descriptors(
          static_cast<size_t>(input_row_elements));
      int64_t split_start = 0;
      for (size_t out_idx = 0; out_idx < output_data.size(); ++out_idx) {
        const int64_t output_width = splits[out_idx] * inner;
        for (int64_t local_element = 0; local_element < output_width;
             ++local_element) {
          element_descriptors[static_cast<size_t>(
              split_start * inner + local_element)] = MusaSplitElementDesc{
              output_data[out_idx], output_width, local_element};
        }
        split_start += splits[out_idx];
      }

      MusaSplitElementDesc* device_element_descriptors = nullptr;
      musaError_t status =
          musaMalloc(reinterpret_cast<void**>(&device_element_descriptors),
                     element_descriptor_bytes);
      if (status != musaSuccess) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
      }

      OrtStatus* copy_status = CopyTemporaryHostToDevice(
          device_element_descriptors, element_descriptors.data(),
          element_descriptor_bytes, stream);
      if (copy_status != nullptr) {
        (void)musaFree(device_element_descriptors);
        return copy_status;
      }

      OrtStatus* launch_status = LaunchStatus(LaunchMusaSplitManySmallRows(
          input0.GetTensorRawData(), device_element_descriptors, outer,
          input_row_elements, static_cast<int32_t>(elem_size), stream));
      FreeDeviceMemoryOnStream(device_element_descriptors, stream);
      return launch_status;
    }

    std::vector<MusaSplitCopyDesc> descriptors(output_data.size());
    int64_t split_start = 0;
    for (size_t out_idx = 0; out_idx < output_data.size(); ++out_idx) {
      descriptors[out_idx] =
          MusaSplitCopyDesc{output_data[out_idx], split_start, splits[out_idx]};
      split_start += splits[out_idx];
    }

    MusaSplitCopyDesc* device_descriptors = nullptr;
    const size_t descriptor_bytes =
        descriptors.size() * sizeof(MusaSplitCopyDesc);
    musaError_t status = musaMalloc(
        reinterpret_cast<void**>(&device_descriptors), descriptor_bytes);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }

    OrtStatus* copy_status = CopyTemporaryHostToDevice(
        device_descriptors, descriptors.data(), descriptor_bytes, stream);
    if (copy_status != nullptr) {
      (void)musaFree(device_descriptors);
      return copy_status;
    }

    OrtStatus* launch_status = LaunchStatus(LaunchMusaSplitManySmallCopies(
        input0.GetTensorRawData(), device_descriptors,
        static_cast<int64_t>(output_data.size()), outer, inner,
        shape0[static_cast<size_t>(axis)], static_cast<int32_t>(elem_size),
        stream));
    FreeDeviceMemoryOnStream(device_descriptors, stream);
    return launch_status;
  }

  return LaunchStatus(LaunchMusaSplitCopies(
      input0.GetTensorRawData(), output_data.data(), splits.data(),
      static_cast<int64_t>(output_data.size()), outer, inner,
      shape0[static_cast<size_t>(axis)], static_cast<int32_t>(elem_size),
      stream));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    Split)
