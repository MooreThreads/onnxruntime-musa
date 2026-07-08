// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <typeinfo>

#include "runtime/ep_musa_utils.h"
#include "runtime_graph_dump.h"
#include "utils.h"

// CRTP base shared by every elementary kernel. Derived classes only need to
// provide:
//   - a constructor `Derived(const OrtKernelInfo* info, void* state)` that
//     reads the attributes it cares about, and
//   - `OrtStatus* Compute(Ort::KernelContext& ctx) const`.
template <typename Derived>
class OpKernelBase : public OrtKernelImpl {
 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* state,
                                     /*out*/ OrtKernelImpl*& kernel) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    auto k = std::make_unique<Derived>(info, state);
    kernel = k.release();
    return nullptr;
    EXCEPTION_TO_RETURNED_STATUS_END
  }

  static OrtStatus* ORT_API_CALL
  ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    auto* k = static_cast<Derived*>(this_ptr);
    if (std::getenv("MUSA_EP_TRACE_KERNELS") != nullptr) {
      std::fprintf(stderr, "MUSA_KERNEL_BEGIN %s impl=%p\n",
                   typeid(Derived).name(), static_cast<void*>(this_ptr));
      std::fflush(stderr);
    }
    Ort::KernelContext ctx(kernel_ctx);
    OrtStatus* status = nullptr;
    if (RuntimeGraphDumpEnabled()) {
      struct RuntimeComputeScope {
        explicit RuntimeComputeScope(const void* impl,
                                     const char* fallback_label)
            : id(BeginRuntimeCompute(impl, fallback_label)) {}
        ~RuntimeComputeScope() { EndRuntimeCompute(id); }
        uint64_t id;
      } runtime_compute_scope(this_ptr, typeid(Derived).name());
      status = k->Compute(ctx);
    } else {
      status = k->Compute(ctx);
    }
    if (std::getenv("MUSA_EP_TRACE_KERNELS") != nullptr) {
      if (status == nullptr && std::getenv("MUSA_EP_TRACE_SYNC") != nullptr) {
        musaError_t sync_status = musaStreamSynchronize(
            static_cast<musaStream_t>(ctx.GetGPUComputeStream()));
        if (sync_status != musaSuccess) {
          status = Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                              MusaErrorString(sync_status));
        }
      }
      std::fprintf(stderr, "MUSA_KERNEL_END %s impl=%p status=%p\n",
                   typeid(Derived).name(), static_cast<void*>(this_ptr),
                   static_cast<void*>(status));
      std::fflush(stderr);
    }
    return status;
    EXCEPTION_TO_RETURNED_STATUS_END
  }

  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    UnregisterRuntimeKernelInstance(this_ptr);
    delete static_cast<Derived*>(this_ptr);
  }

 protected:
  OpKernelBase() : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
  }
};
