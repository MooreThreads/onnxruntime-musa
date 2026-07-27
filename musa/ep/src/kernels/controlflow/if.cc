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

namespace {

class If {
 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* /*state*/,
                                     OrtKernelImpl*& kernel) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    kernel = nullptr;
    RETURN_IF_ERROR(Ort::GetEpApi().CreateIfKernel(info, &kernel));
    return nullptr;
    EXCEPTION_TO_RETURNED_STATUS_END
  }
};

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    If, kOnnxDomain, 13, 15,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("B",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", AllFixedSizeTensorTypesNoBFloat16())
         .SetInputMemType(0, OrtMemTypeCPUInput)),
    If)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    If, kOnnxDomain, 16, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("B",
                            GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", AllFixedSizeTensorTypes())
         .SetInputMemType(0, OrtMemTypeCPUInput)),
    If)
