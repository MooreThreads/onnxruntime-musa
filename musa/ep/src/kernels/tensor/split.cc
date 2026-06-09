// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/split_impl.h"

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
  return LaunchStatus(LaunchMusaSplitCopies(
      input0.GetTensorRawData(), output_data.data(), splits.data(),
      static_cast<int64_t>(output_data.size()), outer, inner,
      shape0[static_cast<size_t>(axis)], static_cast<int32_t>(elem_size),
      GetComputeStream(ctx)));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    Split)
