#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include "vec3.h"

#ifndef DISTANCEFUNCTION
#define DISTANCEFUNCTION

class DistanceFunction
{
public:
    CUDA_HOSTDEV virtual float distance(vec3 pos) = 0;
};

#endif

#ifndef SPHERE
#define SPHERE

class Sphere : public DistanceFunction
{
public:
    float radius;
    CUDA_HOSTDEV Sphere() {}
    CUDA_HOSTDEV Sphere(float r)
    {
        radius = r;
    }
    CUDA_HOSTDEV float distance(vec3 pos)
    {
        return pos.length() - radius;
    }
};

#endif
