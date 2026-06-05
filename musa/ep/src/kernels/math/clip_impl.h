#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaClipKernel(const void* input,
                                 void* output,
                                 MusaClipParams params,
                                 MusaElementType elem_type,
                                 musaStream_t stream);
