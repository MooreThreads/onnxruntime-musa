#pragma once

#include "plugin_ep_utils.h"

class AddKernel : public OrtKernelImpl {
 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* state,
                                     OrtKernelImpl*& kernel) noexcept;

 private:
  struct PrivateTag {};
  AddKernel(const OrtKernelInfo* info, PrivateTag);

  static OrtStatus* ORT_API_CALL
  ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

  const OrtKernelInfo* info_;
};
