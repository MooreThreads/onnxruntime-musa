#include "ep_kernel_registration.h"

#include "kernels/utils.h"

size_t GetNumKernels() { return RegisteredKernelCreateInfoFuncs().size(); }

static OrtStatus* RegisterKernels(Ort::KernelRegistry& kernel_registry,
                                  const char* ep_name,
                                  void* create_kernel_state) {
  for (auto build_func : RegisteredKernelCreateInfoFuncs()) {
    KernelCreateInfo kernel_create_info = {};
    RETURN_IF_ERROR(
        build_func(ep_name, create_kernel_state, &kernel_create_info));

    if (kernel_create_info.kernel_def != nullptr) {
      RETURN_IF_ERROR(kernel_registry.AddKernel(
          kernel_create_info.kernel_def, kernel_create_info.kernel_create_func,
          kernel_create_info.kernel_create_func_state));
    }
  }

  return nullptr;
}

OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state,
                                OrtKernelRegistry** out_kernel_registry) {
  *out_kernel_registry = nullptr;

  if (GetNumKernels() == 0) {
    return nullptr;
  }

  try {
    Ort::KernelRegistry kernel_registry;
    Ort::Status status{
        RegisterKernels(kernel_registry, ep_name, create_kernel_state)};

    *out_kernel_registry = status.IsOK() ? kernel_registry.release() : nullptr;
    return status.release();
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}
