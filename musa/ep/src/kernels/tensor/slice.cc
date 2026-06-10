// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/slice_impl.h"

namespace {
class Slice : public OpKernelBase<Slice> {
 public:
  Slice(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Slice::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  std::vector<int64_t> starts = ReadIntTensor(ctx, 1);
  std::vector<int64_t> ends = ReadIntTensor(ctx, 2);
  std::vector<int64_t> axes;
  std::vector<int64_t> steps(starts.size(), 1);
  if (ctx.GetInputCount() > 3) axes = ReadIntTensor(ctx, 3);
  if (ctx.GetInputCount() > 4) steps = ReadIntTensor(ctx, 4);
  if (axes.empty()) {
    axes.resize(starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  }
  std::vector<int64_t> out_shape = shape0;
  std::vector<int64_t> norm_starts(shape0.size(), 0),
      norm_steps(shape0.size(), 1);
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = NormalizeAxis(axes[i], shape0.size());
    int64_t dim = shape0[static_cast<size_t>(axis)];
    int64_t step = steps[i];
    if (step <= 0)
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "Slice negative step not implemented");
    int64_t start = starts[i] < 0 ? starts[i] + dim : starts[i];
    int64_t end = ends[i] < 0 ? ends[i] + dim : ends[i];
    start = std::max<int64_t>(0, std::min(start, dim));
    end = std::max<int64_t>(0, std::min(end, dim));
    norm_starts[static_cast<size_t>(axis)] = start;
    norm_steps[static_cast<size_t>(axis)] = step;
    out_shape[static_cast<size_t>(axis)] =
        std::max<int64_t>(0, (end - start + step - 1) / step);
  }
  size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Slice unsupported dtype");
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  const bool full_slice =
      out_shape == shape0 &&
      std::all_of(norm_starts.begin(), norm_starts.end(),
                  [](int64_t start) { return start == 0; }) &&
      std::all_of(norm_steps.begin(), norm_steps.end(),
                  [](int64_t step) { return step == 1; });
  if (full_slice) {
    const bool src_gpu = IsGpuMemory(input0.GetTensorMemoryInfo());
    const bool dst_gpu = IsGpuMemory(y.GetTensorMemoryInfo());
    if (src_gpu && dst_gpu) {
      return CopyRawTensor(input0, y, input0.GetTensorSizeInBytes(),
                           GetComputeStream(ctx));
    }
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Slice requires MUSA input and output");
  }

  if (shape0.size() == 2 && norm_steps[0] == 1 && norm_steps[1] == 1 &&
      IsGpuMemory(input0.GetTensorMemoryInfo()) &&
      IsGpuMemory(y.GetTensorMemoryInfo())) {
    const int64_t width_elems = out_shape[1];
    const int64_t height = out_shape[0];
    if (width_elems > 0 && height > 0) {
      const auto* src_base =
          static_cast<const uint8_t*>(input0.GetTensorRawData());
      auto* dst_base = static_cast<uint8_t*>(y.GetTensorMutableRawData());
      const size_t src_pitch = static_cast<size_t>(shape0[1]) * elem_size;
      const size_t dst_pitch = static_cast<size_t>(width_elems) * elem_size;
      const size_t width_bytes = static_cast<size_t>(width_elems) * elem_size;
      const size_t src_offset =
          static_cast<size_t>(norm_starts[0] * shape0[1] + norm_starts[1]) *
          elem_size;
      if (width_bytes >= 256) {
        return DeviceMemcpy2D(
            dst_base, dst_pitch, src_base + src_offset, src_pitch, width_bytes,
            static_cast<size_t>(height), GetComputeStream(ctx));
      }
    }
  }

  const bool all_unit_steps =
      std::all_of(norm_steps.begin(), norm_steps.end(),
                  [](int64_t step) { return step == 1; });
  if (all_unit_steps && IsGpuMemory(input0.GetTensorMemoryInfo()) &&
      IsGpuMemory(y.GetTensorMemoryInfo())) {
    const size_t last_dim = shape0.empty() ? 0 : shape0.size() - 1;
    const bool last_axis_slice =
        shape0.size() >= 2 && norm_steps[last_dim] == 1 &&
        out_shape[last_dim] > 0 &&
        norm_starts[last_dim] + out_shape[last_dim] <= shape0[last_dim] &&
        [&]() {
          for (size_t dim = 0; dim < last_dim; ++dim) {
            if (norm_starts[dim] != 0 || out_shape[dim] != shape0[dim]) {
              return false;
            }
          }
          return true;
        }();
    if (last_axis_slice) {
      musaError_t status = LaunchMusaSliceLastAxisKernel(
          input0.GetTensorRawData(), y.GetTensorMutableRawData(),
          static_cast<int32_t>(elem_size), NumElements(out_shape),
          shape0[last_dim], out_shape[last_dim], norm_starts[last_dim],
          GetComputeStream(ctx));
      if (status != musaErrorNotSupported) {
        return LaunchStatus(status);
      }
    }

    size_t copy_dim = shape0.size();
    for (size_t dim = 0; dim < shape0.size(); ++dim) {
      if (norm_starts[dim] != 0 || out_shape[dim] != shape0[dim]) {
        copy_dim = dim;
        break;
      }
    }
    if (copy_dim < shape0.size()) {
      bool suffix_full = true;
      for (size_t dim = copy_dim + 1; dim < shape0.size(); ++dim) {
        suffix_full &= norm_starts[dim] == 0 && out_shape[dim] == shape0[dim];
      }
      if (suffix_full) {
        int64_t height = 1;
        for (size_t dim = 0; dim < copy_dim; ++dim) {
          height *= out_shape[dim];
        }
        int64_t suffix = 1;
        for (size_t dim = copy_dim + 1; dim < shape0.size(); ++dim) {
          suffix *= shape0[dim];
        }
        const int64_t width_elems = out_shape[copy_dim] * suffix;
        if (width_elems > 0 && height > 0) {
          const auto* src_base =
              static_cast<const uint8_t*>(input0.GetTensorRawData());
          auto* dst_base = static_cast<uint8_t*>(y.GetTensorMutableRawData());
          const size_t src_pitch =
              static_cast<size_t>(shape0[copy_dim] * suffix) * elem_size;
          const size_t dst_pitch = static_cast<size_t>(width_elems) * elem_size;
          const size_t width_bytes =
              static_cast<size_t>(width_elems) * elem_size;
          const size_t src_offset =
              static_cast<size_t>(norm_starts[copy_dim] * suffix) * elem_size;
          if (width_bytes >= 256) {
            return DeviceMemcpy2D(dst_base, dst_pitch, src_base + src_offset,
                                  src_pitch, width_bytes,
                                  static_cast<size_t>(height),
                                  GetComputeStream(ctx));
          }
        }
      }
    }
  }

  if (shape0.size() <= kMusaMaxBroadcastRank &&
      IsGpuMemory(input0.GetTensorMemoryInfo()) &&
      IsGpuMemory(y.GetTensorMemoryInfo())) {
    auto in_strides = Strides(shape0);
    MusaSliceParams params{};
    params.rank = static_cast<int32_t>(shape0.size());
    params.total_elements = NumElements(out_shape);
    for (size_t dim = 0; dim < shape0.size(); ++dim) {
      params.input_strides[dim] = in_strides[dim];
      params.output_dims[dim] = out_shape[dim];
      params.starts[dim] = norm_starts[dim];
      params.steps[dim] = norm_steps[dim];
    }
    return LaunchStatus(LaunchMusaSliceKernel(
        input0.GetTensorRawData(), y.GetTensorMutableRawData(),
        static_cast<int32_t>(elem_size), params, GetComputeStream(ctx)));
  }

  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                    "Slice requires MUSA input and output");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Slice, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddTypeConstraint("Tind", IntTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)
         .SetInputMemType(2, OrtMemTypeCPUInput)
         .SetInputMemType(3, OrtMemTypeCPUInput)
         .SetInputMemType(4, OrtMemTypeCPUInput)),
    Slice)
