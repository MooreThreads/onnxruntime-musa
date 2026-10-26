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

#include <cstdlib>

#include "ep_data_transfer.h"
#include "shared_inc/op_kernel_common.h"

namespace {

// MemcpyFromHost / MemcpyToHost boundary copies.
//
// ORT's MemcpyTransformer inserts these nodes whenever a tensor crosses a
// CPU/device placement boundary and assigns them to the device EP. If the EP
// does not register a kernel for the op, ORT falls back to the generic CPU EP
// memcpy (session_state.cc) and the copy destination stays in host memory --
// the device-side consumer then rejects it (e.g. "Flatten requires MUSA device
// tensors"). Registering these kernels keeps the node on the plugin's device
// allocator (the MemcpyFromHost destination is allocated as GPU/DEFAULT) and
// runs the transfer on the plugin's own compute stream.
//
// Both directions share one implementation and defer the actual transfer to
// the EP's OrtDataTransferImpl (created by the factory and handed to kernels as
// their create state) so pageable-H2D staging and D2H stream ordering match the
// rest of the EP. Tensor-only, mirroring the CUDA plugin kernels: non-tensor
// values (sequences, optionals) do not match this type constraint and keep
// falling back to the CPU EP memcpy, which handles the sequence cases.
class MemcpyBoundary final : public OpKernelBase<MemcpyBoundary> {
 public:
  MemcpyBoundary(const OrtKernelInfo* /*info*/, void* state)
      : data_transfer_impl_(static_cast<OrtDataTransferImpl*>(state)) {}

  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  OrtDataTransferImpl* data_transfer_impl_ = nullptr;
};

OrtStatus* MemcpyBoundary::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  Ort::UnownedValue output =
      ctx.GetOutput(0, input.GetTensorTypeAndShapeInfo().GetShape());

  const void* src = input.GetTensorRawData();
  void* dst = output.GetTensorMutableRawData();
  const size_t bytes = input.GetTensorSizeInBytes();
  if (bytes == 0 || src == dst) {
    return nullptr;
  }

  if (std::getenv("ORT_MUSA_TEST_FAIL_MEMCPY_KERNEL") != nullptr) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL, "test-only MUSA memcpy kernel route reached");
  }

  const bool src_is_gpu = IsGpuMemory(input.GetTensorMemoryInfo());
  const bool dst_is_gpu = IsGpuMemory(output.GetTensorMemoryInfo());
  if (src_is_gpu == dst_is_gpu) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL, "Memcpy expects exactly one MUSA device tensor");
  }

  if (data_transfer_impl_ == nullptr) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL, "Memcpy: data transfer implementation is missing");
  }

  return MusaCopyTensorOnStream(*data_transfer_impl_, input, output,
                                GetComputeStream(ctx));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MemcpyFromHost, kOnnxDomain, 1, 1,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .SetInputMemType(0, OrtMemTypeCPUInput)),
    MemcpyBoundary)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MemcpyToHost, kOnnxDomain, 1, 1,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .SetOutputMemType(0, OrtMemTypeCPUOutput)),
    MemcpyBoundary)
