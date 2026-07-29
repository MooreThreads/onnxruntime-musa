#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMusaConstantOfShapeKernel(void* output, uint64_t value,
                                            int32_t element_size, int64_t count,
                                            musaStream_t stream);
