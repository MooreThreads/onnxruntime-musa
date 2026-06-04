#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaUnaryFloatKernel(const float* input,
                                       float* output,
                                       int64_t count,
                                       MusaUnaryOp op,
                                       float alpha,
                                       musaStream_t stream);

musaError_t LaunchMusaUnaryKernel(const void* input,
                                  void* output,
                                  int64_t count,
                                  MusaUnaryOp op,
                                  float alpha,
                                  MusaElementType elem_type,
                                  musaStream_t stream);

musaError_t LaunchMusaIsNaNKernel(const void* input,
                                  uint8_t* output,
                                  int64_t count,
                                  MusaElementType elem_type,
                                  musaStream_t stream);
