#include <hash_encoding.cuh>
#include <vector>


__device__ __forceinline__ unsigned int SpatialHash(int x, int y, int level) 
{
    const unsigned int PRIME1 = 1u; // cache locality -> efficient lookup 
                                   // (still achieves pseudo-independence with d-1 permutated dimensions)
    const unsigned int PRIME2 = 2654435761u;
    const unsigned int PRIME3 = 805459861u;

    unsigned int hash = (x * PRIME1) ^ (y * PRIME2) ^ (level * PRIME3);
    return hash % (HASHMAP_SIZE - 1);
}

__device__ __forceinline__ unsigned int SpatialHash(int x, int y, int z, int level) 
{
    const unsigned int PRIME1 = 1;
    const unsigned int PRIME2 = 2654435761u;
    const unsigned int PRIME3 = 805459861u;
    const unsigned int PRIME4 = 1610919749u;

    unsigned int hash = (x * PRIME1) ^ (y * PRIME2) ^ (z * PRIME3) ^ (level * PRIME4);
    return hash % (HASHMAP_SIZE - 1);
}


// Hash encoding with bilinear interpolation + index tracking for training
__device__ void HashEncode(
    float x, float y,
    float* output,
    float* hashTable,
    int* hashIndices,  // Store which indices were accessed
    int baseHashResolution
) 
{
    int indexOuput = 0;
    int indexHashIndices = 0;

    for (int level = 0; level < N_LEVELS; ++level) 
    {
        int resolution = static_cast<int>(baseHashResolution * powf(SCALE_FACTOR, level));
        int nVertices = (resolution + 1) * (resolution + 1);

        float fx = x * resolution;
        float fy = y * resolution;
        int x0 = floorf(fx);
        int y0 = floorf(fy);
        float dx = fx - x0;
        float dy = fy - y0;
        
        auto GetIndex = [&](int xi, int yi, int feature) -> int {
            int hashVal;
            if (nVertices <= HASHMAP_SIZE) 
            {
                // Wrap linear index to avoid going out of bounds
                int linearIdx = yi * (resolution + 1) + xi;
                hashVal = linearIdx % HASHMAP_SIZE;
            } else 
            {
                hashVal = SpatialHash(xi, yi, level);
            }
            int baseIndex = level * HASHMAP_SIZE * FEATURES_PER_LEVEL + hashVal * FEATURES_PER_LEVEL;
            return baseIndex + feature;
        };

        for (int f = 0; f < FEATURES_PER_LEVEL; ++f) 
        {
            int idx00 = GetIndex(x0,     y0,     f);
            int idx10 = GetIndex(x0 + 1, y0,     f);
            int idx01 = GetIndex(x0,     y0 + 1, f);
            int idx11 = GetIndex(x0 + 1, y0 + 1, f);

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
};


__device__ void HashEncode3D(
    float x, float y, float z,
    float* output,
    float* hashTable,
    int* hashIndices,   // Store which indices were accessed
    int baseHashResolution
) 
{
    int indexOutput = 0;
    int indexHashIndices = 0;

    for (int level = 0; level < N_LEVELS; ++level) 
    {
        int resolution = static_cast<int>(baseHashResolution * powf(SCALE_FACTOR, level));
        int nVertices = (resolution + 1) * (resolution + 1) * (resolution + 1);

        float fx = x * resolution;
        float fy = y * resolution;
        float fz = z * resolution;
        int x0 = floorf(fx);
        int y0 = floorf(fy);
        int z0 = floorf(fz);
        float dx = fx - x0;
        float dy = fy - y0;
        float dz = fz - z0;

        auto GetIndex = [&](int xi, int yi, int zi, int feature) -> int 
        {
            int hashVal;
            if (nVertices <= HASHMAP_SIZE) {
                // Wrap linear index to avoid going out of bounds
                int linearIdx = (zi * (resolution + 1) + yi) * (resolution + 1) + xi;

                // Ensure non-negative
                linearIdx = linearIdx % HASHMAP_SIZE;
                if (linearIdx < 0) linearIdx += HASHMAP_SIZE;

                hashVal = linearIdx;
            } else 
            {
                hashVal = SpatialHash(xi, yi, zi, level);
            }
            int baseIndex = level * HASHMAP_SIZE * FEATURES_PER_LEVEL + hashVal * FEATURES_PER_LEVEL;
            return baseIndex + feature;
        };

        for (int f = 0; f < FEATURES_PER_LEVEL; ++f) 
        {
            int idx000 = GetIndex(x0,     y0,     z0,     f);
            int idx100 = GetIndex(x0 + 1, y0,     z0,     f);
            int idx010 = GetIndex(x0,     y0 + 1, z0,     f);
            int idx110 = GetIndex(x0 + 1, y0 + 1, z0,     f);
            int idx001 = GetIndex(x0,     y0,     z0 + 1, f);
            int idx101 = GetIndex(x0 + 1, y0,     z0 + 1, f);
            int idx011 = GetIndex(x0,     y0 + 1, z0 + 1, f);
            int idx111 = GetIndex(x0 + 1, y0 + 1, z0 + 1, f);

            float c000 = hashTable[idx000];
            float c100 = hashTable[idx100];
            float c010 = hashTable[idx010];
            float c110 = hashTable[idx110];
            float c001 = hashTable[idx001];
            float c101 = hashTable[idx101];
            float c011 = hashTable[idx011];
            float c111 = hashTable[idx111];

            float c00 = c000 * (1 - dx) + c100 * dx;
            float c01 = c001 * (1 - dx) + c101 * dx;
            float c10 = c010 * (1 - dx) + c110 * dx;
            float c11 = c011 * (1 - dx) + c111 * dx;

            float c0 = c00 * (1 - dy) + c10 * dy;
            float c1 = c01 * (1 - dy) + c11 * dy;

            float interpolated = c0 * (1 - dz) + c1 * dz;

            output[indexOutput++] = interpolated;

            hashIndices[indexHashIndices++] = idx000;
            hashIndices[indexHashIndices++] = idx100;
            hashIndices[indexHashIndices++] = idx010;
            hashIndices[indexHashIndices++] = idx110;
            hashIndices[indexHashIndices++] = idx001;
            hashIndices[indexHashIndices++] = idx101;
            hashIndices[indexHashIndices++] = idx011;
            hashIndices[indexHashIndices++] = idx111;
        }
    }
};


void AllocateHashTable(float** deviceHashTable) 
{
    const int levelSize = (1 << LOG2_HASHMAP_SIZE) * FEATURES_PER_LEVEL; // Each hash coresspond to a tile
    const size_t tableSize = N_LEVELS * levelSize;

    std::vector<float> tempTable(tableSize);
    for (size_t i = 0; i < tableSize; ++i) {
        tempTable[i] = ((float) rand() / RAND_MAX - 0.5f) * 0.01f;
    }

    cudaMalloc(deviceHashTable, tableSize * sizeof(float));
    cudaMemcpy(*deviceHashTable, tempTable.data(), tableSize * sizeof(float), cudaMemcpyHostToDevice);
};

