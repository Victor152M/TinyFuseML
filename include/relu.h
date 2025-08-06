#pragma once

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float relu_derivative(float x){
    return x > 0.0f ? 1.0f : 0.0f;
}

__device__ __forceinline__ float leaky_relu(float x) {
    return x > 0 ? x : 0.1f * x;
}

__device__ inline float leaky_relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.1f;
}
