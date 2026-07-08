// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/tile_impl.h"

namespace {

MusaTileParams MakeTileParams(const std::vector<int64_t>& input_shape,
                              const std::vector<int64_t>& output_shape) {
  MusaTileParams params{};
  params.rank = static_cast<int32_t>(output_shape.size());
  params.total_elements = NumElements(output_shape);

  std::vector<int64_t> padded_input(output_shape.size(), 1);
  const size_t offset = output_shape.size() - input_shape.size();
  for (size_t i = 0; i < input_shape.size(); ++i) {
    padded_input[offset + i] = input_shape[i];
  }
  std::vector<int64_t> input_strides = Strides(input_shape);

  for (size_t dim = 0; dim < output_shape.size(); ++dim) {
    params.input_dims[dim] = padded_input[dim];
    params.output_dims[dim] = output_shape[dim];
    if (dim < offset) {
      params.input_strides[dim] = 0;
    } else {
      params.input_strides[dim] = input_strides[dim - offset];
    }
  }
  return params;
}

class Tile : public OpKernelBase<Tile> {
 public:
  Tile(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Tile::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto elem_type = input_info.GetElementType();
  auto input_shape = input_info.GetShape();
  std::vector<int64_t> repeats = ReadIntTensor(ctx, 1);
  if (repeats.size() < input_shape.size()) {
    repeats.insert(repeats.begin(), input_shape.size() - repeats.size(), 1);
  }
  if (repeats.size() > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Tile rank exceeds MUSA kernel limit");
  }

  std::vector<int64_t> padded_input(repeats.size(), 1);
  const size_t offset = repeats.size() - input_shape.size();
  for (size_t i = 0; i < input_shape.size(); ++i) {
    padded_input[offset + i] = input_shape[i];
  }

  std::vector<int64_t> output_shape(repeats.size(), 1);
  for (size_t i = 0; i < repeats.size(); ++i) {
    if (repeats[i] < 0) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "Tile requires non-negative repeats");
    }
    output_shape[i] = padded_input[i] * repeats[i];
  }

  const size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Tile unsupported dtype");
  }
  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Tile requires MUSA input");
  }

  Ort::UnownedValue y = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Tile requires MUSA output");
  }

  if (output_shape == input_shape) {
    return CopyRawTensor(input, y, input.GetTensorSizeInBytes(),
                         GetComputeStream(ctx));
  }

  if (!input_shape.empty() && input_shape.back() > 0 && repeats.back() > 1 &&
      std::all_of(repeats.begin(), repeats.end() - 1,
                  [](int64_t repeat) { return repeat == 1; })) {
    const int64_t cols = input_shape.back();
    const int64_t rows = NumElements(input_shape) / cols;
    return LaunchStatus(LaunchMusaTileLastDimKernel(
        input.GetTensorRawData(), y.GetTensorMutableRawData(),
        static_cast<int32_t>(elem_size), rows, cols, repeats.back(),
        GetComputeStream(ctx)));
  }

  return LaunchStatus(LaunchMusaTileKernel(
      input.GetTensorRawData(), y.GetTensorMutableRawData(),
      static_cast<int32_t>(elem_size),
      MakeTileParams(input_shape, output_shape), GetComputeStream(ctx)));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Tile, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    Tile)
