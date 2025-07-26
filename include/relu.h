#pragma once

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float reluDerivative(float x){
    return x > 0.0f ? 1.0f : 0.0f;
}
