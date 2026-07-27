// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/cast_op_impl.h"

namespace {
constexpr int64_t kMaxCpuMetadataCastElements = kMusaMaxBroadcastRank;

OrtStatus* CastCpuIntMetadata(Ort::ConstValue input, Ort::UnownedValue output,
                              ONNXTensorElementDataType src_type,
                              ONNXTensorElementDataType dst_type, int64_t count,
                              musaStream_t stream) {
  if (count > kMaxCpuMetadataCastElements) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Cast CPU metadata path only supports small shape tensors");
  }
  if (src_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
      dst_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int64_t> src = ReadTyped<int64_t>(input, stream);
    std::vector<int32_t> dst;
    dst.reserve(src.size());
    for (int64_t value : src) {
      if (value > std::numeric_limits<int32_t>::max() ||
          value < std::numeric_limits<int32_t>::min()) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "Cast metadata int64 value overflows int32");
      }
      dst.push_back(static_cast<int32_t>(value));
    }
    return WriteTyped<int32_t>(output, dst, stream);
  }
  if (src_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      dst_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    std::vector<int32_t> src = ReadTyped<int32_t>(input, stream);
    std::vector<int64_t> dst(src.begin(), src.end());
    return WriteTyped<int64_t>(output, dst, stream);
  }
  if (src_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
      dst_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return WriteTyped<int64_t>(output, ReadTyped<int64_t>(input, stream),
                               stream);
  }
  if (src_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      dst_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return WriteTyped<int32_t>(output, ReadTyped<int32_t>(input, stream),
                               stream);
  }
  if (dst_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::vector<float> dst;
    if (src_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      std::vector<int64_t> src = ReadTyped<int64_t>(input, stream);
      dst.reserve(src.size());
      for (int64_t value : src) {
        dst.push_back(static_cast<float>(value));
      }
      return WriteTyped<float>(output, dst, stream);
    }
    if (src_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      std::vector<int32_t> src = ReadTyped<int32_t>(input, stream);
      dst.reserve(src.size());
      for (int32_t value : src) {
        dst.push_back(static_cast<float>(value));
      }
      return WriteTyped<float>(output, dst, stream);
    }
  }
  return Ort::GetApi().CreateStatus(
      ORT_NOT_IMPLEMENTED,
      "Cast CPU metadata path only supports int32/int64 casts");
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
                                NumElements(shape0), GetComputeStream(ctx));
    }
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Cast requires MUSA tensors except CPU shape metadata input");
  }

  const int64_t n = NumElements(shape0);
  if (elem_type == static_cast<ONNXTensorElementDataType>(to_)) {
    return CopyRawTensor(input0, y, input0.GetTensorSizeInBytes(),
                         GetComputeStream(ctx));
  }

  return LaunchStatus(LaunchMusaCastKernel(
      input0.GetTensorRawData(), y.GetTensorMutableRawData(),
      static_cast<int32_t>(elem_type), static_cast<int32_t>(to_), n,
      GetComputeStream(ctx)));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T1", AllFixedSizeTensorTypes())
         .AddTypeConstraint("T2", AllFixedSizeTensorTypes())),
    Cast)
