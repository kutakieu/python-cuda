#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifndef MATHUTILS
#define MATHUTILS

class MathUtils
{
public:
    static CUDA_HOSTDEV float clamp(float in, float low, float high)
    {
        if (low < in && in < high)
            return in;
        else if (in < low)
            return low;
        else
            return high;
    }
    // static CUDA_HOSTDEV vec3 get_normal(vec3 pos, distance_function)
};

#endif
