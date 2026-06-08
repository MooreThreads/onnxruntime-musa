// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/cast_op_impl.h"

namespace {
constexpr int64_t kMaxCpuMetadataCastElements = kMusaMaxBroadcastRank;

template <typename SrcT, typename DstT>
OrtStatus* CastCpuMetadataTyped(Ort::ConstValue input,
                                Ort::UnownedValue output) {
  std::vector<SrcT> src = ReadTyped<SrcT>(input);
  std::vector<DstT> dst;
  dst.reserve(src.size());
  for (SrcT value : src) {
    if constexpr (std::is_integral_v<SrcT> && std::is_integral_v<DstT>) {
      long double v = static_cast<long double>(value);
      if (v > static_cast<long double>(std::numeric_limits<DstT>::max()) ||
          v < static_cast<long double>(std::numeric_limits<DstT>::min())) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "Cast metadata integer value overflows");
      }
    }
    dst.push_back(static_cast<DstT>(value));
  }
  return WriteTyped<DstT>(output, dst);
}

template <typename SrcT>
OrtStatus* CastCpuMetadataFromSrc(Ort::ConstValue input,
                                  Ort::UnownedValue output,
                                  ONNXTensorElementDataType dst_type) {
  switch (dst_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return CastCpuMetadataTyped<SrcT, int32_t>(input, output);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return CastCpuMetadataTyped<SrcT, int64_t>(input, output);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return CastCpuMetadataTyped<SrcT, float>(input, output);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return CastCpuMetadataTyped<SrcT, double>(input, output);
    default:
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "Cast CPU metadata path unsupported destination dtype");
  }
}

OrtStatus* CastCpuIntMetadata(Ort::ConstValue input,
                              Ort::UnownedValue output,
                              ONNXTensorElementDataType src_type,
                              ONNXTensorElementDataType dst_type,
                              int64_t count) {
  if (count > kMaxCpuMetadataCastElements) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Cast CPU metadata path only supports small shape tensors");
  }
  if (src_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return CastCpuMetadataFromSrc<int64_t>(input, output, dst_type);
  }
  if (src_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return CastCpuMetadataFromSrc<int32_t>(input, output, dst_type);
  }
  return Ort::GetApi().CreateStatus(
      ORT_NOT_IMPLEMENTED,
      "Cast CPU metadata path only supports int32/int64 sources");
}

class Cast : public OpKernelBase<Cast> {
 public:
  Cast(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    to_ = AttrOrDefault<int64_t>(kernel_info, "to", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t to_ = 0;
};

OrtStatus* Cast::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  const size_t src_elem_size = ElementSize(elem_type);
  const size_t dst_elem_size =
      ElementSize(static_cast<ONNXTensorElementDataType>(to_));
  if (src_elem_size == 0 || dst_elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Cast unsupported dtype");
  }

  Ort::UnownedValue y = ctx.GetOutput(0, shape0);
  if (!IsGpuMemory(input0.GetTensorMemoryInfo()) ||
      !IsGpuMemory(y.GetTensorMemoryInfo())) {
    if (!IsGpuMemory(input0.GetTensorMemoryInfo())) {
      return CastCpuIntMetadata(input0, y, elem_type,
                                static_cast<ONNXTensorElementDataType>(to_),
                                NumElements(shape0));
    }
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Cast requires MUSA tensors except CPU shape metadata input");
  }

  const int64_t n = NumElements(shape0);
  if (static_cast<int64_t>(elem_type) == to_) {
    return DeviceMemcpy(y.GetTensorMutableRawData(), input0.GetTensorRawData(),
                        static_cast<size_t>(n) * src_elem_size);
  }

  return LaunchStatus(LaunchMusaCastKernel(
      input0.GetTensorRawData(), y.GetTensorMutableRawData(),
      static_cast<int32_t>(elem_type), static_cast<int32_t>(to_), n, nullptr));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T1", AllFixedSizeTensorTypes())
         .AddTypeConstraint("T2", AllFixedSizeTensorTypes())),
    Cast)
