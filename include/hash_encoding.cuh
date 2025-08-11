#pragma once

#include <cuda_runtime.h>

// Constants for hash encoding
constexpr int N_LEVELS = 10; // L
constexpr int FEATURES_PER_LEVEL = 2; // F
constexpr int BASE_RES = 256;
constexpr int LOG2_HASHMAP_SIZE = 21; // T
constexpr float SCALE_FACTOR = 1.7f;
constexpr int HASHMAP_SIZE = 1 << LOG2_HASHMAP_SIZE;

// Total encoded input size
constexpr int HASH_ENCODED_SIZE = N_LEVELS * FEATURES_PER_LEVEL;

// Allocate the full hash table
void allocateHashTable(float** deviceHashTable);

// Kernel to encode (x, y) to hash features
__device__ void hashEncode(float x, float y, float* output, float* hashTable, int* hashIndices);