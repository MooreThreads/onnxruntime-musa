#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaWhereKernel(const uint8_t* condition,
                                  const void* x,
                                  const void* y,
                                  void* output,
                                  int32_t element_size,
                                  MusaWhereParams params,
                                  musaStream_t stream);

musaError_t LaunchMusaWhereSameShapeFastKernel(const uint8_t* condition,
                                               const void* x,
                                               const void* y,
                                               void* output,
                                               int32_t element_size,
                                               int64_t total_elements,
                                               musaStream_t stream);

musaError_t LaunchMusaWhereRowwiseFastKernel(const uint8_t* condition,
                                             const void* x,
                                             const void* y,
                                             void* output,
                                             int32_t element_size,
                                             int64_t rows,
                                             int64_t inner_size,
                                             bool x_broadcast_rows,
                                             bool y_broadcast_rows,
                                             musaStream_t stream);
