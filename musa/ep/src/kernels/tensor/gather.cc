// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/gather_impl.h"

namespace {
constexpr int64_t kMaxCpuMetadataGatherElements = kMusaMaxBroadcastRank;

template <typename T>
OrtStatus* GatherCpuMetadataTyped(Ort::ConstValue input,
                                  Ort::ConstValue indices_value,
                                  Ort::UnownedValue output,
                                  ONNXTensorElementDataType index_type,
                                  int64_t input_count, int64_t output_count) {
  std::vector<T> input_data = ReadTyped<T>(input);
  std::vector<int64_t> indices;
  if (index_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    indices = ReadTyped<int64_t>(indices_value);
  } else if (index_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> vals = ReadTyped<int32_t>(indices_value);
    indices.assign(vals.begin(), vals.end());
  } else {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Gather metadata path unsupported index dtype");
  }

  std::vector<T> output_data;
  output_data.reserve(static_cast<size_t>(output_count));
  for (int64_t index : indices) {
    int64_t normalized = index < 0 ? index + input_count : index;
    if (normalized < 0 || normalized >= input_count) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Gather metadata index out of range");
    }
    output_data.push_back(input_data[static_cast<size_t>(normalized)]);
  }
  return WriteTyped<T>(output, output_data);
}

OrtStatus* GatherCpuMetadata(Ort::ConstValue input,
                             Ort::ConstValue indices_value,
                             Ort::UnownedValue output,
                             ONNXTensorElementDataType elem_type,
                             ONNXTensorElementDataType index_type,
                             const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& indices_shape,
                             int64_t axis) {
  const int64_t input_count = NumElements(input_shape);
  const int64_t output_count = NumElements(indices_shape);
  if (axis != 0 || input_shape.size() != 1 ||
      input_count > kMaxCpuMetadataGatherElements ||
      output_count > kMaxCpuMetadataGatherElements) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Gather CPU metadata path only supports small rank-1 shape tensors");
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return GatherCpuMetadataTyped<int64_t>(
        input, indices_value, output, index_type, input_count, output_count);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return GatherCpuMetadataTyped<int32_t>(
        input, indices_value, output, index_type, input_count, output_count);
  }
  return Ort::GetApi().CreateStatus(
      ORT_NOT_IMPLEMENTED,
      "Gather CPU metadata path only supports int32/int64 tensors");
}

class Gather : public OpKernelBase<Gather> {
 public:
  Gather(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
};

OrtStatus* Gather::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  int64_t axis = NormalizeAxis(axis_, shape0.size());
  Ort::ConstValue indices_value = ctx.GetInput(1);
  auto indices_info = indices_value.GetTensorTypeAndShapeInfo();
  auto indices_shape = indices_info.GetShape();
  std::vector<int64_t> out_shape;
  out_shape.insert(out_shape.end(), shape0.begin(), shape0.begin() + axis);
  out_shape.insert(out_shape.end(), indices_shape.begin(), indices_shape.end());
  out_shape.insert(out_shape.end(), shape0.begin() + axis + 1, shape0.end());
  size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gather unsupported dtype");
  size_t index_elem_size = ElementSize(indices_info.GetElementType());
  if (index_elem_size != sizeof(int32_t) &&
      index_elem_size != sizeof(int64_t)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gather unsupported index dtype");
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (IsGpuMemory(input0.GetTensorMemoryInfo()) &&
      IsGpuMemory(indices_value.GetTensorMemoryInfo()) &&
      IsGpuMemory(y.GetTensorMemoryInfo())) {
    int64_t block_size = 1;
    for (size_t dim = static_cast<size_t>(axis) + 1; dim < shape0.size(); ++dim)
      block_size *= shape0[dim];
    int64_t indices_count = NumElements(indices_shape);
    int64_t indices_max = shape0[static_cast<size_t>(axis)];
    int64_t prefix_count = 1;
    for (int64_t dim = 0; dim < axis; ++dim)
      prefix_count *= shape0[static_cast<size_t>(dim)];
    int64_t input_block_size = indices_max * block_size;
    int64_t output_block_size = indices_count * block_size;
    return LaunchStatus(LaunchMusaGatherKernel(
        input0.GetTensorRawData(), indices_value.GetTensorRawData(),
        y.GetTensorMutableRawData(), static_cast<int32_t>(elem_size),
        static_cast<int32_t>(index_elem_size), input_block_size, indices_max,
        output_block_size, block_size, prefix_count * output_block_size,
        GetComputeStream(ctx)));
  }

  // Shape outputs are CPU metadata; small constant indices may be placed on
  // MUSA.
  if (!IsGpuMemory(input0.GetTensorMemoryInfo())) {
    return GatherCpuMetadata(input0, indices_value, y, elem_type,
                             indices_info.GetElementType(), shape0,
                             indices_shape, axis);
  }

  return Ort::GetApi().CreateStatus(
      ORT_NOT_IMPLEMENTED,
      "Gather requires MUSA tensors except CPU shape metadata inputs");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddTypeConstraint("Tind", IntTensorTypes())),
    Gather)
