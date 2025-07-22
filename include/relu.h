#pragma once

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}
