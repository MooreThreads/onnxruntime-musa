// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
::musa::dnn::Tensor::Format MudnnFormatForShape(
    const std::vector<int64_t>& shape) {
  if (shape.empty() || shape.size() == 1 || shape.size() == 3) {
    return ::musa::dnn::Tensor::Format::NCW;
  }
  if (shape.size() == 4) {
    return ::musa::dnn::Tensor::Format::NCHW;
  }
  if (shape.size() == 5) {
    return ::musa::dnn::Tensor::Format::NCDHW;
  }
  return ::musa::dnn::Tensor::Format::NCHW;
}

bool SetupMudnnFloatTensor(::musa::dnn::Tensor& tensor, const void* data,
                           const std::vector<int64_t>& shape) {
  if (tensor.SetType(::musa::dnn::Tensor::Type::FLOAT) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  if (tensor.SetAddr(data) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  if (tensor.SetFormat(MudnnFormatForShape(shape)) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  std::vector<int64_t> dims = shape.empty() ? std::vector<int64_t>{1} : shape;
  std::vector<int64_t> strides =
      shape.empty() ? std::vector<int64_t>{1} : Strides(shape);
  return tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data(),
                          strides.data()) == ::musa::dnn::Status::SUCCESS;
}

bool TryMudnnConcatFloat(Ort::KernelContext& ctx,
                         const std::vector<std::vector<int64_t>>& shapes,
                         const std::vector<int64_t>& out_shape, int64_t axis,
                         Ort::UnownedValue y) {
  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  std::vector<::musa::dnn::Tensor> input_tensors(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (!SetupMudnnFloatTensor(input_tensors[i],
                               ctx.GetInput(i).GetTensorData<float>(),
                               shapes[i])) {
      return false;
    }
  }

  ::musa::dnn::Tensor output_tensor;
  if (!SetupMudnnFloatTensor(output_tensor, y.GetTensorMutableData<float>(),
                             out_shape)) {
    return false;
  }

  ::musa::dnn::Concat concat_op;
  if (concat_op.SetAxis(static_cast<int>(axis)) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return concat_op.Run(*handle, output_tensor,
                       static_cast<int>(input_tensors.size()),
                       input_tensors.data()) == ::musa::dnn::Status::SUCCESS;
}

class Concat : public OpKernelBase<Concat> {
 public:
  Concat(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
};

OrtStatus* Concat::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  int64_t axis = NormalizeAxis(axis_, shape0.size());
  std::vector<std::vector<int64_t>> shapes;
  size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Concat unsupported dtype");
  }
  std::vector<int64_t> out_shape = shape0;
  out_shape[static_cast<size_t>(axis)] = 0;
  for (size_t i = 0; i < ctx.GetInputCount(); ++i) {
    auto v = ctx.GetInput(i);
    shapes.push_back(v.GetTensorTypeAndShapeInfo().GetShape());
    out_shape[static_cast<size_t>(axis)] +=
        shapes.back()[static_cast<size_t>(axis)];
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (AllGpuInputs(ctx) && IsGpuMemory(y.GetTensorMemoryInfo())) {
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        TryMudnnConcatFloat(ctx, shapes, out_shape, axis, y)) {
      return nullptr;
    }

    const int64_t outer =
        axis == 0 ? 1
                  : std::accumulate(out_shape.begin(), out_shape.begin() + axis,
                                    int64_t{1}, std::multiplies<int64_t>());
    const int64_t inner =
        axis + 1 == static_cast<int64_t>(out_shape.size())
            ? 1
            : std::accumulate(out_shape.begin() + axis + 1, out_shape.end(),
                              int64_t{1}, std::multiplies<int64_t>());
    const int64_t output_axis = out_shape[static_cast<size_t>(axis)];
    auto* dst_base = static_cast<uint8_t*>(y.GetTensorMutableRawData());
    int64_t dst_axis_offset = 0;
    for (size_t input_idx = 0; input_idx < shapes.size(); ++input_idx) {
      Ort::ConstValue v = ctx.GetInput(input_idx);
      const int64_t input_axis = shapes[input_idx][static_cast<size_t>(axis)];
      const size_t width_bytes =
          static_cast<size_t>(input_axis * inner) * elem_size;
      const size_t src_pitch = width_bytes;
      const size_t dst_pitch =
          static_cast<size_t>(output_axis * inner) * elem_size;
      const auto* src = static_cast<const uint8_t*>(v.GetTensorRawData());
      auto* dst =
          dst_base + static_cast<size_t>(dst_axis_offset * inner) * elem_size;
      RETURN_IF_ERROR(DeviceMemcpy2D(dst, dst_pitch, src, src_pitch,
                                     width_bytes, static_cast<size_t>(outer)));
      dst_axis_offset += input_axis;
    }
    return nullptr;
  }

  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                    "Concat requires MUSA inputs and output");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", TensorTypesWithBool())),
    Concat)
