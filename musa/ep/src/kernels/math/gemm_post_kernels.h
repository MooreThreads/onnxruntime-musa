#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaGemmPostFloatKernel(float* output,
                                          const float* bias,
                                          MusaBroadcastParams params,
                                          bool has_bias,
                                          float beta,
                                          MusaUnaryOp activation,
                                          bool has_activation,
                                          float activation_alpha,
                                          musaStream_t stream);

musaError_t LaunchMusaGemmPostKernel(void* output,
                                     const void* bias,
                                     MusaBroadcastParams params,
                                     bool has_bias,
                                     float beta,
                                     MusaUnaryOp activation,
                                     bool has_activation,
                                     float activation_alpha,
                                     MusaElementType elem_type,
                                     musaStream_t stream);
