#include "relu.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vec3.h>
#include <hash_encoding.cuh>


#define PI 3.14159265359f
#define MAX_STEPS 1024
#define SURF_EPS 0.01f
#define MAX_DIST 1.4f


__device__ float SDSphere(const Vec3& p, float r) { return p.Length() - r; };

__device__ float SDGround(Vec3 p) { return p.y + 1.0f; }// plane at y = -1 

__device__ float SceneSDFSphere(Vec3 p, float time, Vec3& color) 
{ 
    float r = 0.3f; 
    color = Vec3(0.8f, 0.5f, 0.3f);
    return SDSphere(p, r);  
};


__global__ void GenerateSDFTargetsKernel( const float* positions, // [N,3] sampled points 
                                            float* targets, // [N] SDF values to write 
                                            int N, 
                                            float time
                                            )
{ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= N) return; 
    Vec3 p(positions[idx*3 + 0], positions[idx*3 + 1], positions[idx*3 + 2]); 
    Vec3 colorDummy;
    targets[idx] = SceneSDFSphere(p, time, colorDummy); 
};


__device__ float SceneSDF(Vec3 p, float time, Vec3& color,
                          const float* weightsLayer1,
                          const float* biasLayer1,
                          const float* weightsLayer2,
                          const float* biasLayer2,
                          const float* weightsLayer3,
                          const float* biasLayer3,
                          float* hashTable,
                          int baseHashResolution)
{
    // 1) Hash encode the point
    float encodedInput[HASH_ENCODED_SIZE];
    int hashIndices[HASH_ENCODED_SIZE*8];
    HashEncode3D(p.x, p.y, p.z, encodedInput, hashTable, hashIndices, baseHashResolution);

    float activation1[16];
    float activation2[16];
    float output[1];

    // Forward pass
    for(int i=0;i<16;i++)
    {
        float sum = biasLayer1[i];
        for(int j=0;j<HASH_ENCODED_SIZE;j++) 
            sum += weightsLayer1[i*HASH_ENCODED_SIZE + j]*encodedInput[j];
        activation1[i] = Relu(sum);
    }

    for(int i=0;i<16;i++)
    {
        float sum = biasLayer2[i];
        for(int j=0;j<16;j++) 
            sum += weightsLayer2[i*16 + j]*activation1[j];
        activation2[i] = Relu(sum);
    }

    float sum = biasLayer3[0];
    for(int j=0;j<16;j++)
        sum += weightsLayer3[j]*activation2[j];
    output[0] = sum;

    return output[0];
}


__device__ Vec3 EstimateNormal(const Vec3& p, float time,
                               const float* weightsLayer1,
                               const float* biasLayer1,
                               const float* weightsLayer2,
                               const float* biasLayer2,
                               const float* weightsLayer3,
                               const float* biasLayer3,
                               float* hashTable, int baseHashResolution)
{
    const float h = 0.001f;
    Vec3 colorDummy;

    float dx = SceneSDF(p + Vec3(h,0,0), time, colorDummy,
                        weightsLayer1, biasLayer1,
                        weightsLayer2, biasLayer2,
                        weightsLayer3, biasLayer3,
                        hashTable, baseHashResolution)
             - SceneSDF(p - Vec3(h,0,0), time, colorDummy,
                        weightsLayer1, biasLayer1,
                        weightsLayer2, biasLayer2,
                        weightsLayer3, biasLayer3,
                        hashTable, baseHashResolution);
    float dy = SceneSDF(p + Vec3(0,h,0), time, colorDummy,
                        weightsLayer1, biasLayer1,
                        weightsLayer2, biasLayer2,
                        weightsLayer3, biasLayer3,
                        hashTable, baseHashResolution)
             - SceneSDF(p - Vec3(0,h,0), time, colorDummy,
                        weightsLayer1, biasLayer1,
                        weightsLayer2, biasLayer2,
                        weightsLayer3, biasLayer3,
                        hashTable, baseHashResolution);
    float dz = SceneSDF(p + Vec3(0,0,h), time, colorDummy,
                        weightsLayer1, biasLayer1,
                        weightsLayer2, biasLayer2,
                        weightsLayer3, biasLayer3,
                        hashTable, baseHashResolution)
             - SceneSDF(p - Vec3(0,0,h), time, colorDummy,
                        weightsLayer1, biasLayer1,
                        weightsLayer2, biasLayer2,
                        weightsLayer3, biasLayer3,
                        hashTable, baseHashResolution);
    return Vec3(dx, dy, dz).Normalize();
};


__device__ bool RayMarch(const Vec3& ro, const Vec3& rd, float time, 
                         Vec3& hitPos, Vec3& hitColor, float& ao,
                         const float* weightsLayer1,
                         const float* biasLayer1,
                         const float* weightsLayer2,
                         const float* biasLayer2,
                         const float* weightsLayer3,
                         const float* biasLayer3,
                         float* hashTable,
                         int baseHashResolution)
{
    float t = 0.0f;
    ao = 1.0f;
    for(int i=0;i<MAX_STEPS;i++)
    {
        Vec3 p = ro + rd*t;
        Vec3 colorDummy;
        float dist = SceneSDF(p, time, colorDummy,
                              weightsLayer1, biasLayer1,
                              weightsLayer2, biasLayer2,
                              weightsLayer3, biasLayer3,
                              hashTable, baseHashResolution);
        float safeDist = fabsf(dist);

        if (fabsf(dist) < SURF_EPS) 
        {
            hitPos = p;
            hitColor = Vec3(0.75f, 0.75f, 0.85f);
            return true;
        }

        float safety = 0.5f;
        t += max(dist * safety, 1e-4f);

        if(t > MAX_DIST) break;
    }
    return false;
}


__device__ float SoftShadow(const Vec3& ro, const Vec3& rd, float time,
                            const float* weightsLayer1,
                            const float* biasLayer1,
                            const float* weightsLayer2,
                            const float* biasLayer2,
                            const float* weightsLayer3,
                            const float* biasLayer3,
                            float* hashTable,
                            float mint, float maxt, float k,
                            int baseHashResolution)
{
    float res=1.0f, t=fmaxf(mint,1e-4f);
    Vec3 colorDummy;
    for(int i=0;i<32;i++)
    {
        float h = fabsf(SceneSDF(ro + rd*t, time, colorDummy,
                                 weightsLayer1, biasLayer1,
                                 weightsLayer2, biasLayer2,
                                 weightsLayer3, biasLayer3,
                                 hashTable, baseHashResolution));
        float safeT = fmaxf(t,1e-4f);
        res = fminf(res, k*h/safeT);
        t += fmaxf(h, SURF_EPS)*0.5f;
        if(res < 0.001f || t > maxt) break;
    }
    return fmaxf(res, 0.0f);
};


__global__ void RenderKernel(
    unsigned char* outputRGBA, int width, int height, float time,
    const float* weightsLayer1, const float* biasLayer1,
    const float* weightsLayer2, const float* biasLayer2,
    const float* weightsLayer3, const float* biasLayer3,
    float* hashTable, int baseHashResolution)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    int idx = (y*width + x)*4;

    // Camera setup
    float aspectRatio = (float)width/(float)height;
    float fov = 60.0f;
    float tanFov = tanf(fov*0.5f*PI/180.0f);
    float u = (2.0f*(x+0.5f)/width - 1.0f) * tanFov * aspectRatio;
    float v = (1.0f - 2.0f*(y+0.5f)/height) * tanFov;

    float camDist = 0.7f;  // Changed from 0.9f to 0.7f to not show artifacts from outside the training domain
    float camAngle = time*0.3f;
    Vec3 camPos(cosf(camAngle)*camDist, 0.3f, sinf(camAngle)*camDist);
    Vec3 target(0,0,0);
    Vec3 forward = (target - camPos).Normalize();
    Vec3 worldUp(0,1,0);
    Vec3 right = forward.Cross(worldUp).Normalize();
    Vec3 up = right.Cross(forward).Normalize();
    Vec3 rayDir = (forward + right*u + up*v).Normalize();

    Vec3 hitPos, hitColor;
    float ao;
    if(RayMarch(camPos, rayDir, time, hitPos, hitColor, ao,
                weightsLayer1, biasLayer1,
                weightsLayer2, biasLayer2,
                weightsLayer3, biasLayer3,
                hashTable, baseHashResolution))
    {
        Vec3 n = EstimateNormal(hitPos, time,
                                 weightsLayer1, biasLayer1,
                                 weightsLayer2, biasLayer2,
                                 weightsLayer3, biasLayer3,
                                 hashTable, baseHashResolution);
        Vec3 lightDir = Vec3(-1,1,-1).Normalize();
        float diff = fmaxf(0.0f, n.Dot(-lightDir));
        Vec3 viewDir = (camPos-hitPos).Normalize();
        Vec3 halfDir = (viewDir+lightDir).Normalize();
        float spec = powf(fmaxf(0.0f, n.Dot(halfDir)),64.0f);
        float rim = powf(1.0f - fmaxf(0.0f, viewDir.Dot(n)),3.0f);
        float shadow = SoftShadow(hitPos+n*0.01f, lightDir*-1.0f, time,
                                 weightsLayer1, biasLayer1,
                                 weightsLayer2, biasLayer2,
                                 weightsLayer3, biasLayer3,
                                 hashTable, 0.01f, 5.0f, 8.0f, baseHashResolution);

        Vec3 baseColor = hitColor;
        Vec3 litColor = baseColor*0.9f + baseColor*diff*shadow +
                        Vec3(1,1,1)*spec*0.4f + Vec3(1,1,1)*rim*0.2f;
        litColor = litColor*(0.8f + 0.2f*ao);

        float dist = (hitPos-camPos).Length();
        float fogStrength = fminf(fmaxf((dist-5.0f)/30.0f,0.0f),1.0f);
        litColor = Lerp(litColor, Vec3(0,0,0), fogStrength);
        litColor = Clamp(litColor, 0.0f, 1.0f);

        outputRGBA[idx+0] = (unsigned char)(litColor.x*255.0f);
        outputRGBA[idx+1] = (unsigned char)(litColor.y*255.0f);
        outputRGBA[idx+2] = (unsigned char)(litColor.z*255.0f);
        outputRGBA[idx+3] = 255;
    } else {
        // Sky
        float t_sky = powf(0.5f*(rayDir.y+1.0f),0.5f);
        Vec3 sky = Lerp(Vec3(0.6f,0.8f,1.0f), Vec3(0.7f,0.9f,1.0f), t_sky);
        outputRGBA[idx+0] = (unsigned char)(sky.x*255.0f);
        outputRGBA[idx+1] = (unsigned char)(sky.y*255.0f);
        outputRGBA[idx+2] = (unsigned char)(sky.z*255.0f);
        outputRGBA[idx+3] = 255;
    }
}


void RenderMLPSDF(unsigned char* outputRGBA, int width, int height, float time,
                  const float* weightsLayer1, const float* biasLayer1,
                  const float* weightsLayer2, const float* biasLayer2,
                  const float* weightsLayer3, const float* biasLayer3,
                  float* hashTable, int baseHashResolution)
{
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width+15)/16, (height+15)/16);
    RenderKernel<<<numBlocks, threadsPerBlock>>>(outputRGBA, width, height, time,
                                                 weightsLayer1, biasLayer1,
                                                 weightsLayer2, biasLayer2,
                                                 weightsLayer3, biasLayer3,
                                                 hashTable, baseHashResolution);
    cudaDeviceSynchronize();
}