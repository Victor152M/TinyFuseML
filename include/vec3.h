#pragma once
#include <cmath>
#include <cuda_runtime.h>


struct Vec3 {
    float x, y, z;

    __host__ __device__ __inline__ Vec3(float a = 0.0f) : x(a), y(a), z(a) {}
    __host__ __device__ __inline__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ __inline__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ __inline__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ __inline__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__ __inline__ Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ __inline__ Vec3 operator/(const Vec3& v) const { return Vec3(x / v.x, y / v.y, z / v.z); }

    __host__ __device__ __inline__ float Dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ __inline__ Vec3 Cross(const Vec3& v) const 
    {
        return Vec3(y * v.z - z * v.y,
                    z * v.x - x * v.z,
                    x * v.y - y * v.x);
    }

    __host__ __device__ __inline__ float Length() const { return sqrtf(x * x + y * y + z * z); }
    __host__ __device__ __inline__ Vec3 Normalize() const 
    {
        float l = Length();
        return l > 0 ? (*this) * (1.0f / l) : Vec3(0.0f);
    }

    __host__ __device__ __inline__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
};


__host__ __device__ __inline__ Vec3 Abs(const Vec3& v) { return Vec3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
__host__ __device__ __inline__ Vec3 Fract(const Vec3& v) { return Vec3(v.x - floorf(v.x), v.y - floorf(v.y), v.z - floorf(v.z)); }
__host__ __device__ __inline__ float Fract(float x) { return x - floorf(x); }
__host__ __device__ __inline__ float Mix(float a, float b, float t) { return a + (b - a) * t; }
__host__ __device__ __inline__ Vec3 Mix(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }


__host__ __device__ __inline__ Vec3 RotateY(const Vec3& p, float angle) 
{
    float c = cosf(angle);
    float s = sinf(angle);
    return Vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

__host__ __device__ __inline__ Vec3 RotateX(const Vec3& p, float angle) 
{
    float c = cosf(angle);
    float s = sinf(angle);
    return Vec3(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

__host__ __device__ __inline__ Vec3 RotateZ(const Vec3& p, float angle) 
{
    float c = cosf(angle);
    float s = sinf(angle);
    return Vec3(p.x * c - p.y * s, p.x * s + p.y * c, p.z);
}

__host__ __device__ __inline__ Vec3 Clamp(const Vec3& v, float minVal, float maxVal) 
{
    return Vec3(fminf(fmaxf(v.x, minVal), maxVal),
                fminf(fmaxf(v.y, minVal), maxVal),
                fminf(fmaxf(v.z, minVal), maxVal));
}

__host__ __device__ __inline__ Vec3 Lerp(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }

