#pragma once

#include "shared_inc/device_kernel_types.h"

// `scale`/`bias` carry `scale_elem_type` elements (float32 or float16); the
// device kernel reads them through that type, so no host-side conversion is
// needed.
musaError_t LaunchMusaLayerNormKernel(const void* input, const void* scale,
                                      const void* bias, void* output,
                                      float* mean, float* inv_std,
                                      MusaLayerNormParams params, float epsilon,
                                      MusaElementType elem_type,
                                      MusaElementType scale_elem_type,
                                      musaStream_t stream);
