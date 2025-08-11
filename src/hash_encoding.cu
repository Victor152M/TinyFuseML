#include <hash_encoding.cuh>
#include <vector>


__device__ __forceinline__ unsigned int spatialHash(int x, int y, int level) {
    const unsigned int PRIME1 = 1u; //works surpsingly well to use a 1 instead
    const unsigned int PRIME2 = 2654435761u;
    const unsigned int PRIME3 = 805459861u;

    unsigned int hash = (x * PRIME1) ^ (y * PRIME2) ^ (level * PRIME3);
    return hash % (HASHMAP_SIZE - 1);
}

// Hash encoding with bilinear interpolation + index tracking for training
__device__ void hashEncode(
    float x, float y,
    float* output,
    float* hashTable,
    int* hashIndices  // Store which indices were accessed
) {
    int indexOuput = 0;
    int indexHashIndices = 0;


    for (int level = 0; level < N_LEVELS; ++level) {
        int resolution = static_cast<int>(BASE_RES * powf(SCALE_FACTOR, level));
        int nVertices = (resolution + 1) * (resolution + 1);

        float fx = x * resolution;
        float fy = y * resolution;
        int x0 = floorf(fx);
        int y0 = floorf(fy);
        float dx = fx - x0;
        float dy = fy - y0;
        
        auto getIndex = [&](int xi, int yi, int feature) -> int {
            if (nVertices <= HASHMAP_SIZE) {
                // Direct mapping for coarse levels (no hashing)
                int baseIndex = level * HASHMAP_SIZE * FEATURES_PER_LEVEL + (yi * (resolution + 1) + xi) * FEATURES_PER_LEVEL;
                return baseIndex + feature;
            } else {
                // Hashing for fine levels
                int hashVal = spatialHash(xi, yi, level);
                int baseIndex = level * HASHMAP_SIZE * FEATURES_PER_LEVEL + hashVal * FEATURES_PER_LEVEL;
                return baseIndex + feature;
            }
        };

        for (int f = 0; f < FEATURES_PER_LEVEL; ++f) {
            int levelOffset = level * (1 << LOG2_HASHMAP_SIZE) * FEATURES_PER_LEVEL;

            int idx00 = getIndex(x0,     y0,     f);
            int idx10 = getIndex(x0 + 1, y0,     f);
            int idx01 = getIndex(x0,     y0 + 1, f);
            int idx11 = getIndex(x0 + 1, y0 + 1, f);

            float feat00 = hashTable[idx00];
            float feat10 = hashTable[idx10];
            float feat01 = hashTable[idx01];
            float feat11 = hashTable[idx11];

            float top = (1 - dx) * feat00 + dx * feat10;
            float bottom = (1 - dx) * feat01 + dx * feat11;
            float interpolated = (1 - dy) * top + dy * bottom;

            output[indexOuput++] = interpolated;

            // Save indices for backprop
            hashIndices[indexHashIndices++] = idx00;
            hashIndices[indexHashIndices++] = idx10;
            hashIndices[indexHashIndices++] = idx01;
            hashIndices[indexHashIndices++] = idx11;
        }
    }
}


void allocateHashTable(float** deviceHashTable) {
    const int levelSize = (1 << LOG2_HASHMAP_SIZE) * FEATURES_PER_LEVEL; // Each hash coresspond to a tile
    const size_t tableSize = N_LEVELS * levelSize;

    std::vector<float> tempTable(tableSize);
    for (size_t i = 0; i < tableSize; ++i) {
        tempTable[i] = ((float) rand() / RAND_MAX - 0.5f) * 0.01f;
    }

    cudaMalloc(deviceHashTable, tableSize * sizeof(float));
    cudaMemcpy(*deviceHashTable, tempTable.data(), tableSize * sizeof(float), cudaMemcpyHostToDevice);
}

