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
    if (IsGpuMemory(indices_value.GetTensorMemoryInfo())) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "Gather with MUSA indices requires device-side bounds validation");
    }

    std::vector<int64_t> indices = ReadIntTensor(ctx, 1);
    for (int64_t index : indices) {
      int64_t normalized_index = index;
      if (normalized_index < 0) normalized_index += indices_max;
      if (normalized_index < 0 || normalized_index >= indices_max) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "Gather index is out of bounds");
      }
    }
    const auto* input_bytes =
        static_cast<const uint8_t*>(input0.GetTensorRawData());
    auto* output_bytes = static_cast<uint8_t*>(y.GetTensorMutableRawData());
    const size_t copy_bytes = static_cast<size_t>(block_size) * elem_size;
    for (int64_t prefix = 0; prefix < prefix_count; ++prefix) {
      for (int64_t idx_offset = 0; idx_offset < indices_count; ++idx_offset) {
        int64_t gather_idx = indices[static_cast<size_t>(idx_offset)];
        if (gather_idx < 0) gather_idx += indices_max;
        size_t dst_offset = static_cast<size_t>(prefix * output_block_size +
                                                idx_offset * block_size) *
                            elem_size;
        if (gather_idx < 0 || gather_idx >= indices_max) {
          musaError_t memset_status = musaMemsetAsync(output_bytes + dst_offset,
                                                      0, copy_bytes, nullptr);
          if (memset_status != musaSuccess) {
            return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                              MusaErrorString(memset_status));
          }
          continue;
        }
        size_t src_offset = static_cast<size_t>(prefix * input_block_size +
                                                gather_idx * block_size) *
                            elem_size;
        RETURN_IF_ERROR(DeviceMemcpy(output_bytes + dst_offset,
                                     input_bytes + src_offset, copy_bytes));
      }
    }
    return nullptr;
  }

  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                    "Gather requires MUSA input and output");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", TensorTypesWithBool())
         .AddTypeConstraint("Tind", IntTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    Gather)
