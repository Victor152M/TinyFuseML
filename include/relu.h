#pragma once

__device__ __forceinline__ float Relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float ReluDerivative(float x){
    return x > 0.0f ? 1.0f : 0.0f;
}

__device__ __forceinline__ float LeakyRelu(float x) {
    return x > 0 ? x : 0.1f * x;
}

__device__ inline float LeakyReluDerivative(float x) {
    return x > 0.0f ? 1.0f : 0.1f;
}
