// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
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
          : std::accumulate(shape0.begin() + axis + 1, shape0.end(),
                            int64_t{1}, std::multiplies<int64_t>());
  const bool gpu_input = IsGpuMemory(input0.GetTensorMemoryInfo());
  std::vector<uint8_t> in;
  auto in_strides = Strides(shape0);
  for (size_t out_idx = 0; out_idx < splits.size(); ++out_idx) {
    std::vector<int64_t> out_shape = shape0;
    out_shape[static_cast<size_t>(axis)] = splits[out_idx];
    Ort::UnownedValue y = ctx.GetOutput(out_idx, out_shape);
    if (gpu_input && IsGpuMemory(y.GetTensorMemoryInfo())) {
      const auto* src_base = static_cast<const uint8_t*>(input0.GetTensorRawData());
      auto* dst_base = static_cast<uint8_t*>(y.GetTensorMutableRawData());
      const size_t width_bytes = static_cast<size_t>(splits[out_idx] * inner) * elem_size;
      const size_t src_pitch =
          static_cast<size_t>(shape0[static_cast<size_t>(axis)] * inner) * elem_size;
      const size_t dst_pitch = width_bytes;
      const size_t src_offset = static_cast<size_t>(axis_start * inner) * elem_size;
      RETURN_IF_ERROR(DeviceMemcpy2D(dst_base, dst_pitch, src_base + src_offset,
                                     src_pitch, width_bytes,
                                     static_cast<size_t>(outer)));
      axis_start += splits[out_idx];
      continue;
    }
    if (in.empty()) {
      RETURN_IF_ERROR(CopyToHost(input0, in));
    }
    std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                             elem_size);
    for (int64_t i = 0; i < NumElements(out_shape); ++i) {
      auto oc = Coordinates(i, out_shape);
      auto ic = oc;
      ic[static_cast<size_t>(axis)] += axis_start;
      std::memcpy(
          out.data() + static_cast<size_t>(i) * elem_size,
          in.data() + static_cast<size_t>(Offset(ic, in_strides)) * elem_size,
          elem_size);
    }
    RETURN_IF_ERROR(CopyFromHost(y, out.data(), out.size()));
    axis_start += splits[out_idx];
  }
  return nullptr;
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", TensorTypesWithBool())), Split)
