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

#pragma once

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace musa_logical_ops {

constexpr size_t kMudnnMaxElementwiseRank = 5;

inline bool TryMudnnLogicalBinary(Ort::KernelContext& ctx,
                                  const std::vector<int64_t>& lhs_shape,
                                  const std::vector<int64_t>& rhs_shape,
                                  ::musa::dnn::Binary::Mode mode) {
  const std::vector<int64_t> out_shape = BroadcastShape(lhs_shape, rhs_shape);
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  if (out_shape.size() > kMudnnMaxElementwiseRank ||
      lhs_shape.size() > kMudnnMaxElementwiseRank ||
      rhs_shape.size() > kMudnnMaxElementwiseRank ||
      !IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
    return false;
  }

  Ort::UnownedValue output = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return false;
  }
  if (NumElements(out_shape) == 0) {
    return true;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, GetComputeStream(ctx));
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor lhs_tensor;
  ::musa::dnn::Tensor rhs_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnTensor(lhs_tensor, lhs.GetTensorRawData(), lhs_shape,
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) ||
      !SetMudnnTensor(rhs_tensor, rhs.GetTensorRawData(), rhs_shape,
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) ||
      !SetMudnnTensor(output_tensor, output.GetTensorMutableRawData(),
                      out_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)) {
    return false;
  }

  ::musa::dnn::Binary op;
  if (op.SetMode(mode) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, lhs_tensor, rhs_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

}  // namespace musa_logical_ops
