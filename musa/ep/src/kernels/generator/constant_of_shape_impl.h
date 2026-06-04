#pragma once

#include <cstdint>

#include <musa_runtime.h>

musaError_t LaunchMusaConstantOfShapeKernel(void* output,
                                            uint64_t value,
                                            int32_t element_size,
                                            int64_t count,
                                            musaStream_t stream);
