// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

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
        static_cast<int32_t>(elem_size), params, nullptr));
  }


  std::vector<uint8_t> in;
  RETURN_IF_ERROR(CopyToHost(input0, in));
  std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                           elem_size);
  auto in_strides = Strides(shape0);
  for (int64_t i = 0; i < NumElements(out_shape); ++i) {
    auto oc = Coordinates(i, out_shape);
    auto ic = oc;
    for (size_t d = 0; d < ic.size(); ++d)
      ic[d] = norm_starts[d] + oc[d] * norm_steps[d];
    std::memcpy(
        out.data() + static_cast<size_t>(i) * elem_size,
        in.data() + static_cast<size_t>(Offset(ic, in_strides)) * elem_size,
        elem_size);
  }
  return CopyFromHost(y, out.data(), out.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Slice, kOnnxDomain, 13, 17,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T", TensorTypesWithBool())
                                       .AddTypeConstraint("Tind",
                                                          IntTensorTypes())),
                                  Slice)
