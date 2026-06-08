// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/pad_impl.h"

namespace {

uint64_t ScalarInputOrZero(Ort::KernelContext& ctx, size_t index,
                           size_t elem_size) {
  uint64_t bits = 0;
  if (ctx.GetInputCount() <= index || ctx.GetInput(index) == nullptr) {
    return bits;
  }
  std::vector<uint8_t> bytes;
  Ort::ThrowOnError(CopyToHost(ctx.GetInput(index), bytes));
  if (bytes.empty()) {
    return bits;
  }
  if (bytes.size() != elem_size) {
    throw std::runtime_error("Pad constant_value must be a scalar");
  }
  std::memcpy(&bits, bytes.data(), bytes.size());
  return bits;
}

MusaPadParams MakePadParams(const std::vector<int64_t>& input_shape,
                            const std::vector<int64_t>& output_shape,
                            const std::vector<int64_t>& pads_begin) {
  MusaPadParams params{};
  params.rank = static_cast<int32_t>(input_shape.size());
  params.total_elements = NumElements(output_shape);
  auto input_strides = Strides(input_shape);
  for (size_t dim = 0; dim < input_shape.size(); ++dim) {
    params.input_dims[dim] = input_shape[dim];
    params.input_strides[dim] = input_strides[dim];
    params.output_dims[dim] = output_shape[dim];
    params.pads_begin[dim] = pads_begin[dim];
  }
  return params;
}

class Pad : public OpKernelBase<Pad> {
 public:
  Pad(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    mode_ = AttrOrDefault<std::string>(kernel_info, "mode", "constant");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::string mode_ = "constant";
};

OrtStatus* Pad::Compute(Ort::KernelContext& ctx) const {
  if (mode_ != "constant") {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Pad only supports constant mode");
  }

  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto elem_type = input_info.GetElementType();
  auto input_shape = input_info.GetShape();
  if (input_shape.size() > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Pad rank exceeds MUSA kernel limit");
  }

  const size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0 || elem_size > sizeof(uint64_t)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Pad unsupported dtype");
  }

  std::vector<int64_t> pads = ReadIntTensor(ctx, 1);
  std::vector<int64_t> axes;
  if (ctx.GetInputCount() > 3 && ctx.GetInput(3) != nullptr) {
    axes = ReadIntTensor(ctx, 3);
  } else {
    axes.resize(input_shape.size());
    std::iota(axes.begin(), axes.end(), 0);
  }
  if (pads.size() != axes.size() * 2) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Pad pads/axes shape mismatch");
  }

  std::vector<int64_t> pads_begin(input_shape.size(), 0);
  std::vector<int64_t> pads_end(input_shape.size(), 0);
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = NormalizeAxis(axes[i], input_shape.size());
    if (axis < 0 || axis >= static_cast<int64_t>(input_shape.size()) ||
        pads[i] < 0 || pads[i + axes.size()] < 0) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "Pad only supports non-negative pads");
    }
    pads_begin[axis] = pads[i];
    pads_end[axis] = pads[i + axes.size()];
  }

  std::vector<int64_t> output_shape(input_shape.size(), 0);
  for (size_t dim = 0; dim < input_shape.size(); ++dim) {
    output_shape[dim] = input_shape[dim] + pads_begin[dim] + pads_end[dim];
  }

  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Pad requires MUSA input");
  }

  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Pad requires MUSA output");
  }

  const uint64_t constant_value = ScalarInputOrZero(ctx, 2, elem_size);
  const bool zero_constant = constant_value == 0;
  const bool right_pad_last_axis_only =
      input_shape.size() >= 2 &&
      std::all_of(pads_begin.begin(), pads_begin.end(),
                  [](int64_t pad) { return pad == 0; }) &&
      [&]() {
        for (size_t dim = 0; dim + 1 < pads_end.size(); ++dim) {
          if (pads_end[dim] != 0) {
            return false;
          }
        }
        return pads_end.back() > 0;
      }();
  if (zero_constant && right_pad_last_axis_only) {
    void* output_data = output.GetTensorMutableRawData();
    const size_t output_bytes =
        static_cast<size_t>(NumElements(output_shape)) * elem_size;
    musaError_t status = musaMemsetAsync(output_data, 0, output_bytes, nullptr);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }

    const int64_t rows = NumElements(input_shape) / input_shape.back();
    const size_t src_pitch = static_cast<size_t>(input_shape.back()) * elem_size;
    const size_t dst_pitch = static_cast<size_t>(output_shape.back()) * elem_size;
    return DeviceMemcpy2D(output_data, dst_pitch, input.GetTensorRawData(),
                          src_pitch, src_pitch, static_cast<size_t>(rows));
  }

  return LaunchStatus(LaunchMusaPadKernel(
      input.GetTensorRawData(), output.GetTensorMutableRawData(), constant_value,
      static_cast<int32_t>(elem_size),
      MakePadParams(input_shape, output_shape, pads_begin), nullptr));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pad, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)
         .SetInputMemType(2, OrtMemTypeCPUInput)
         .SetInputMemType(3, OrtMemTypeCPUInput)),
    Pad)
