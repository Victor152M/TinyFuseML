#pragma once
#include <vec3.h>


__global__ void RenderKernel(
    unsigned char* outputRGBA, int width, int height, float time,
    const float* weightsLayer1, const float* biasLayer1,
    const float* weightsLayer2, const float* biasLayer2,
    const float* weightsLayer3, const float* biasLayer3,
    float* hashTable, int baseHashResolution);

__device__ float SceneSDF(Vec3 p, float time, Vec3& color,
                          const float* weightsLayer1,
                          const float* biasLayer1,
                          const float* weightsLayer2,
                          const float* biasLayer2,
                          const float* weightsLayer3,
                          const float* biasLayer3,
                          float* hashTable,
                          int baseHashResolution);

__global__ void GenerateSDFTargetsKernel(const float* positions, float* targets, int N, float time);


