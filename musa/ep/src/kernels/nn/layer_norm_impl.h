#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaLayerNormKernel(const void* input,
                                      const float* scale,
                                      const float* bias,
                                      void* output,
                                      float* mean,
                                      float* inv_std,
                                      MusaLayerNormParams params,
                                      float epsilon,
                                      MusaElementType elem_type,
                                      musaStream_t stream);
