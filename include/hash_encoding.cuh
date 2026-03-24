#pragma once

#include <cuda_runtime.h>

// Hash table hyperparameters
constexpr int N_LEVELS = 16; // L
constexpr int FEATURES_PER_LEVEL = 4; // F
constexpr int LOG2_HASHMAP_SIZE = 24; // T
constexpr float SCALE_FACTOR = 1.5f;
constexpr int HASHMAP_SIZE = 1 << LOG2_HASHMAP_SIZE;
// baseHashResolution - declared in main.cpp

constexpr int HASH_ENCODED_SIZE = N_LEVELS * FEATURES_PER_LEVEL;


void AllocateHashTable(float** deviceHashTable);

__device__ void HashEncode(float x, float y, float* output, float* hashTable, int* hashIndices, int baseHashResolution);
__device__ void HashEncode3D(float x, float y, float z, float* output, float* hashTable, int* hashIndices, int baseHashResolution);
