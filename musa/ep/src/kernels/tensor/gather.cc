// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/gather_impl.h"

namespace {
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
  auto indices_type = indices_info.GetElementType();
  auto indices_shape = indices_info.GetShape();
  std::vector<int64_t> out_shape;
  out_shape.insert(out_shape.end(), shape0.begin(), shape0.begin() + axis);
  out_shape.insert(out_shape.end(), indices_shape.begin(), indices_shape.end());
  out_shape.insert(out_shape.end(), shape0.begin() + axis + 1, shape0.end());
  size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gather unsupported dtype");
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (IsGpuMemory(input0.GetTensorMemoryInfo()) &&
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
    int64_t output_count = prefix_count * output_block_size;
    int32_t index_element_size =
        indices_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ? 4 : 8;
    if (IsGpuMemory(indices_value.GetTensorMemoryInfo())) {
      return LaunchStatus(LaunchMusaGatherKernel(
          input0.GetTensorRawData(), indices_value.GetTensorRawData(),
          y.GetTensorMutableRawData(), static_cast<int32_t>(elem_size),
          index_element_size, input_block_size, indices_max, output_block_size,
          block_size, output_count, nullptr));
    }

    std::vector<int64_t> indices = ReadIntTensor(ctx, 1);
    const auto* input_bytes = static_cast<const uint8_t*>(input0.GetTensorRawData());
    auto* output_bytes = static_cast<uint8_t*>(y.GetTensorMutableRawData());
    const size_t copy_bytes = static_cast<size_t>(block_size) * elem_size;
    for (int64_t prefix = 0; prefix < prefix_count; ++prefix) {
      for (int64_t idx_offset = 0; idx_offset < indices_count; ++idx_offset) {
        int64_t gather_idx = indices[static_cast<size_t>(idx_offset)];
        if (gather_idx < 0) gather_idx += indices_max;
        size_t dst_offset =
            static_cast<size_t>(prefix * output_block_size + idx_offset * block_size) *
            elem_size;
        if (gather_idx < 0 || gather_idx >= indices_max) {
          musaError_t memset_status =
              musaMemsetAsync(output_bytes + dst_offset, 0, copy_bytes, nullptr);
          if (memset_status != musaSuccess) {
            return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                              MusaErrorString(memset_status));
          }
          continue;
        }
        size_t src_offset =
            static_cast<size_t>(prefix * input_block_size + gather_idx * block_size) *
            elem_size;
        RETURN_IF_ERROR(DeviceMemcpy(output_bytes + dst_offset,
                                     input_bytes + src_offset, copy_bytes));
      }
    }
    return nullptr;
  }

  std::vector<int64_t> indices = ReadIntTensor(ctx, 1);
  std::vector<uint8_t> in;
  RETURN_IF_ERROR(CopyToHost(input0, in));
  std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                           elem_size);
  auto in_strides = Strides(shape0);
  for (int64_t i = 0; i < NumElements(out_shape); ++i) {
    auto oc = Coordinates(i, out_shape);
    std::vector<int64_t> ic(shape0.size(), 0);
    for (int64_t d = 0; d < axis; ++d)
      ic[static_cast<size_t>(d)] = oc[static_cast<size_t>(d)];
    int64_t idx_offset = 0;
    auto idx_strides = Strides(indices_shape);
    for (size_t j = 0; j < indices_shape.size(); ++j)
      idx_offset += oc[static_cast<size_t>(axis) + j] * idx_strides[j];
    int64_t gather_idx = indices[static_cast<size_t>(idx_offset)];
    if (gather_idx < 0) gather_idx += shape0[static_cast<size_t>(axis)];
    ic[static_cast<size_t>(axis)] = gather_idx;
    for (size_t d = static_cast<size_t>(axis) + 1; d < shape0.size(); ++d) {
      ic[d] = oc[d - 1 + indices_shape.size()];
    }
    std::memcpy(
        out.data() + static_cast<size_t>(i) * elem_size,
        in.data() + static_cast<size_t>(Offset(ic, in_strides)) * elem_size,
        elem_size);
  }
  return CopyFromHost(y, out.data(), out.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Gather, kOnnxDomain, 13, 17,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T", TensorTypesWithBool())
                                       .AddTypeConstraint("Tind",
                                                          IntTensorTypes())),
                                  Gather)
