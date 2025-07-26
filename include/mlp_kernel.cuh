#pragma once

__global__ void mlp_kernel(
    const float* __restrict__ input,
    float* __restrict__ weightsLayer1,
    float* __restrict__ biasLayer1,
    float* __restrict__ weightsLayer2,
    float* __restrict__ biasLayer2,
    float* output,
    int input_size,
    int hidden_size,
    int output_size,
    float learning_rate,
    const float* __restrict__ target,
    bool training
);