#include "musa/musa_backend.h"

#if ORT_MUSA_HAS_MUSA
#include <musa_runtime.h>
#endif

namespace ort_musa {

MusaRuntimeInfo QueryMusaRuntime() {
  MusaRuntimeInfo info;
#if ORT_MUSA_HAS_MUSA
  int count = 0;
  const musaError_t status = musaGetDeviceCount(&count);
  if (status == musaSuccess) {
    info.available = count > 0;
    info.device_count = count;
    info.description = info.available ? "MUSA runtime available"
                                      : "MUSA runtime has no visible devices";
  } else {
    info.description = musaGetErrorString(status);
  }
#else
  info.description = "Built without MUSA runtime";
#endif
  return info;
}

}  // namespace ort_musa
