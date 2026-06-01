#pragma once

#include "basic_kernels.h"

musaError_t LaunchMusaGemmPostFloatKernel(float* output,
                                          const float* bias,
                                          MusaBroadcastParams params,
                                          bool has_bias,
                                          float beta,
                                          MusaUnaryOp activation,
                                          bool has_activation,
                                          float activation_alpha,
                                          musaStream_t stream);
