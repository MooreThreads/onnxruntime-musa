#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaNotBoolKernel(const uint8_t* input,
                                    uint8_t* output,
                                    int64_t count,
                                    musaStream_t stream);

musaError_t LaunchMusaOrBoolKernel(const uint8_t* lhs,
                                   const uint8_t* rhs,
                                   uint8_t* output,
                                   MusaBroadcastParams params,
                                   musaStream_t stream);

musaError_t LaunchMusaAndBoolKernel(const uint8_t* lhs,
                                    const uint8_t* rhs,
                                    uint8_t* output,
                                    MusaBroadcastParams params,
                                    musaStream_t stream);

musaError_t LaunchMusaBitwiseAndKernel(const void* lhs,
                                       const void* rhs,
                                       void* output,
                                       MusaBroadcastParams params,
                                       MusaElementType elem_type,
                                       musaStream_t stream);
