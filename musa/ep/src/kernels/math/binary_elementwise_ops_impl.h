#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaBinaryFloatKernel(const float* lhs, const float* rhs,
                                        float* output,
                                        MusaBroadcastParams params,
                                        MusaBinaryOp op, musaStream_t stream);

musaError_t LaunchMusaBinaryKernel(const void* lhs, const void* rhs,
                                   void* output, MusaBroadcastParams params,
                                   MusaBinaryOp op, MusaElementType elem_type,
                                   musaStream_t stream);

musaError_t LaunchMusaPowKernel(const void* lhs, const void* rhs, void* output,
                                MusaBroadcastParams params,
                                MusaElementType lhs_elem_type,
                                MusaElementType rhs_elem_type,
                                musaStream_t stream);

musaError_t LaunchMusaCompareFloatKernel(const float* lhs, const float* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op, musaStream_t stream);

musaError_t LaunchMusaCompareInt32Kernel(const int32_t* lhs, const int32_t* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op, musaStream_t stream);

musaError_t LaunchMusaCompareInt64Kernel(const int64_t* lhs, const int64_t* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op, musaStream_t stream);

musaError_t LaunchMusaCompareKernel(const void* lhs, const void* rhs,
                                    uint8_t* output, MusaBroadcastParams params,
                                    MusaCompareOp op, MusaElementType elem_type,
                                    musaStream_t stream);
