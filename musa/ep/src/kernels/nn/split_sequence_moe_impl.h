// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchSplitSequenceMoESkipLayerNormFloatKernel(
    const float* residual_input, const float* ffn_output, const float* gamma,
    const float* beta, const int64_t* group_offsets, float* output,
    int64_t rows, int64_t group_count, int64_t width, float epsilon,
    musaStream_t stream);
