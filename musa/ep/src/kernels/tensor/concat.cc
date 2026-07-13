// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"
#include "tensor/concat_impl.h"

namespace {

constexpr size_t kConcatManySmallInputCount = 32;
constexpr size_t kConcatManySmallMaxWidthBytes = 4096;
constexpr size_t kConcatRelaxedSmallInputCount = 2;
constexpr size_t kConcatRelaxedSmallMaxWidthBytes = 32 * 1024;
constexpr size_t kConcatManySmallMaxMapBytes = 4 * 1024 * 1024;
constexpr int64_t kMaxCpuMetadataConcatElements = kMusaMaxBroadcastRank;

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
  if (NumElements(out_shape) == 0) {
    return false;
  }
  for (const auto& shape : shapes) {
    if (NumElements(shape) == 0) {
      return false;
    }
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, GetComputeStream(ctx));
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

bool ShouldUseConcatSmallRows(size_t input_count, size_t min_input_count,
                              size_t max_width_bytes, size_t width_limit_bytes,
                              int64_t output_row_elements,
                              size_t element_descriptor_bytes) {
  return input_count >= min_input_count &&
         max_width_bytes <= width_limit_bytes && output_row_elements > 0 &&
         element_descriptor_bytes <= kConcatManySmallMaxMapBytes;
}

bool RangesOverlap(const void* lhs, size_t lhs_size, const void* rhs,
                   size_t rhs_size) {
  if (lhs == nullptr || rhs == nullptr || lhs_size == 0 || rhs_size == 0) {
    return false;
  }
  const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
  const auto lhs_end = lhs_begin + lhs_size;
  const auto rhs_end = rhs_begin + rhs_size;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

OrtStatus* LaunchConcatSmallRows(void* output,
                                 const std::vector<const void*>& input_data,
                                 const std::vector<int64_t>& input_axis_dims,
                                 int64_t outer, int64_t inner,
                                 int64_t output_row_elements, int32_t elem_size,
                                 musaStream_t stream) {
  if (input_data.size() <= static_cast<size_t>(kMusaConcatSmallRowsMaxInputs)) {
    return LaunchStatus(LaunchMusaConcatManySmallRowsDirect(
        output, input_data.data(), input_axis_dims.data(),
        static_cast<int64_t>(input_data.size()), outer, inner,
        output_row_elements, elem_size, stream));
  }

  std::vector<MusaConcatElementDesc> element_descriptors(
      static_cast<size_t>(output_row_elements));
  int64_t output_offset = 0;
  for (size_t input_idx = 0; input_idx < input_data.size(); ++input_idx) {
    const int64_t input_width = input_axis_dims[input_idx] * inner;
    for (int64_t local_element = 0; local_element < input_width;
         ++local_element) {
      element_descriptors[static_cast<size_t>(output_offset + local_element)] =
          MusaConcatElementDesc{input_data[input_idx], input_width,
                                local_element};
    }
    output_offset += input_width;
  }

  MusaConcatElementDesc* device_element_descriptors = nullptr;
  const size_t element_descriptor_bytes =
      static_cast<size_t>(output_row_elements) * sizeof(MusaConcatElementDesc);
  device_element_descriptors = reinterpret_cast<MusaConcatElementDesc*>(
      AllocateDeviceMemoryOnStream(element_descriptor_bytes, stream));
  if (device_element_descriptors == nullptr) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL, MusaErrorString(musaErrorMemoryAllocation));
  }

  OrtStatus* copy_status = CopyTemporaryHostToDevice(
      device_element_descriptors, element_descriptors.data(),
      element_descriptor_bytes, stream);
  if (copy_status != nullptr) {
    (void)musaFree(device_element_descriptors);
    return copy_status;
  }

  OrtStatus* launch_status = LaunchStatus(
      LaunchMusaConcatManySmallRows(output, device_element_descriptors, outer,
                                    output_row_elements, elem_size, stream));
  FreeDeviceMemoryOnStream(device_element_descriptors, stream,
                           element_descriptor_bytes);
  return launch_status;
}

template <typename T>
OrtStatus* ConcatCpuMetadataTyped(Ort::KernelContext& ctx,
                                  Ort::UnownedValue output,
                                  int64_t output_count, musaStream_t stream) {
  std::vector<T> output_data;
  output_data.reserve(static_cast<size_t>(output_count));
  for (size_t i = 0; i < ctx.GetInputCount(); ++i) {
    std::vector<T> input_data = ReadTyped<T>(ctx.GetInput(i), stream);
    output_data.insert(output_data.end(), input_data.begin(), input_data.end());
  }
  return WriteTyped<T>(output, output_data, stream);
}

OrtStatus* ConcatCpuMetadata(Ort::KernelContext& ctx, Ort::UnownedValue output,
                             ONNXTensorElementDataType elem_type,
                             const std::vector<std::vector<int64_t>>& shapes,
                             const std::vector<int64_t>& out_shape,
                             int64_t axis, musaStream_t stream) {
  const int64_t output_count = NumElements(out_shape);
  if (axis != 0 || out_shape.size() != 1 ||
      output_count > kMaxCpuMetadataConcatElements) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Concat CPU metadata path only supports small rank-1 shape tensors");
  }
  for (const auto& shape : shapes) {
    if (shape.size() != 1) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "Concat CPU metadata path only supports rank-1 inputs");
    }
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ConcatCpuMetadataTyped<int64_t>(ctx, output, output_count, stream);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return ConcatCpuMetadataTyped<int32_t>(ctx, output, output_count, stream);
  }
  return Ort::GetApi().CreateStatus(
      ORT_NOT_IMPLEMENTED,
      "Concat CPU metadata path only supports int32/int64 tensors");
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
    const int64_t outer =
        axis == 0 ? 1
                  : std::accumulate(out_shape.begin(), out_shape.begin() + axis,
                                    int64_t{1}, std::multiplies<int64_t>());
    const int64_t inner =
        axis + 1 == static_cast<int64_t>(out_shape.size())
            ? 1
            : std::accumulate(out_shape.begin() + axis + 1, out_shape.end(),
                              int64_t{1}, std::multiplies<int64_t>());
    std::vector<const void*> input_data(shapes.size());
    std::vector<int64_t> input_axis_dims(shapes.size());
    int64_t max_input_axis = 0;
    void* output_data = y.GetTensorMutableRawData();
    const size_t output_bytes =
        static_cast<size_t>(NumElements(out_shape)) * elem_size;
    bool output_overlaps_input = false;
    for (size_t input_idx = 0; input_idx < shapes.size(); ++input_idx) {
      Ort::ConstValue v = ctx.GetInput(input_idx);
      input_data[input_idx] = v.GetTensorRawData();
      input_axis_dims[input_idx] = shapes[input_idx][static_cast<size_t>(axis)];
      max_input_axis = std::max(max_input_axis, input_axis_dims[input_idx]);
      const size_t input_bytes =
          static_cast<size_t>(NumElements(shapes[input_idx])) * elem_size;
      output_overlaps_input |= RangesOverlap(
          output_data, output_bytes, input_data[input_idx], input_bytes);
    }

    musaStream_t stream = GetComputeStream(ctx);
    void* concat_output = output_data;
    void* temp_output = nullptr;
    if (output_overlaps_input && output_bytes > 0) {
      temp_output = AllocateDeviceMemoryOnStream(output_bytes, stream);
      if (temp_output == nullptr) {
        return Ort::GetApi().CreateStatus(
            ORT_EP_FAIL, MusaErrorString(musaErrorMemoryAllocation));
      }
      concat_output = temp_output;
    }
    auto finish_concat = [&](OrtStatus* status) -> OrtStatus* {
      if (temp_output == nullptr) {
        return status;
      }
      if (status == nullptr) {
        musaError_t copy_status =
            musaMemcpyAsync(output_data, temp_output, output_bytes,
                            musaMemcpyDeviceToDevice, stream);
        if (copy_status != musaSuccess) {
          status = Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                              MusaErrorString(copy_status));
        }
      }
      FreeDeviceMemoryOnStream(temp_output, stream, output_bytes);
      return status;
    };

    const int64_t output_row_elements =
        out_shape[static_cast<size_t>(axis)] * inner;
    const size_t max_width_bytes = static_cast<size_t>(max_input_axis) *
                                   static_cast<size_t>(inner) * elem_size;
    const size_t element_descriptor_bytes =
        static_cast<size_t>(output_row_elements) *
        sizeof(MusaConcatElementDesc);
    if (ShouldUseConcatSmallRows(input_data.size(), kConcatManySmallInputCount,
                                 max_width_bytes, kConcatManySmallMaxWidthBytes,
                                 output_row_elements,
                                 element_descriptor_bytes)) {
      return finish_concat(LaunchConcatSmallRows(
          concat_output, input_data, input_axis_dims, outer, inner,
          output_row_elements, static_cast<int32_t>(elem_size), stream));
    }

    if (ShouldUseConcatSmallRows(
            input_data.size(), kConcatRelaxedSmallInputCount, max_width_bytes,
            kConcatRelaxedSmallMaxWidthBytes, output_row_elements,
            element_descriptor_bytes)) {
      return finish_concat(LaunchConcatSmallRows(
          concat_output, input_data, input_axis_dims, outer, inner,
          output_row_elements, static_cast<int32_t>(elem_size), stream));
    }

    if (!output_overlaps_input &&
        elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        TryMudnnConcatFloat(ctx, shapes, out_shape, axis, y)) {
      return nullptr;
    }

    return finish_concat(LaunchStatus(LaunchMusaConcatCopies(
        concat_output, input_data.data(), input_axis_dims.data(),
        static_cast<int64_t>(input_data.size()), outer, inner,
        out_shape[static_cast<size_t>(axis)], static_cast<int32_t>(elem_size),
        stream)));
  }

  return ConcatCpuMetadata(ctx, y, elem_type, shapes, out_shape, axis,
                           GetComputeStream(ctx));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllFixedSizeTensorTypes())),
    Concat)
