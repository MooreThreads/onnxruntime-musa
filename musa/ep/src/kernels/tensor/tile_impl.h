#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaTileKernel(const void* input,
                                 void* output,
                                 int32_t element_size,
                                 MusaTileParams params,
                                 musaStream_t stream);

musaError_t LaunchMusaTileLastDimKernel(const void* input,
                                        void* output,
                                        int32_t element_size,
                                        int64_t rows,
                                        int64_t cols,
                                        int64_t repeats,
                                        musaStream_t stream);
