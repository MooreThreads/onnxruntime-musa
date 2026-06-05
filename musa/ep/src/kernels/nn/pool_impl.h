#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaGlobalAveragePoolKernel(const void* input,
                                              void* output,
                                              MusaGlobalAveragePoolParams params,
                                              MusaElementType elem_type,
                                              musaStream_t stream);
