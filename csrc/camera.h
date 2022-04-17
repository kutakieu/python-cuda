#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifndef CAMERA
#define CAMERA

#include "vec3.h"

class camera
{
public:
    vec3 up;
    vec3 forward;
    vec3 right;
    vec3 position;
    float focal_length;
    float horizontal_fov;
    float horizontal_fov_rad;

    CUDA_HOSTDEV camera() {}
    CUDA_HOSTDEV camera(vec3 _up, vec3 _forward, vec3 _position, float _horizontal_fov)
    {
        up = _up;
        forward = _forward;
        right = cross(up, forward);
        position = _position;
        horizontal_fov = _horizontal_fov;
        horizontal_fov_rad = horizontal_fov * M_PI / 180;
        focal_length = 1 / tan(horizontal_fov_rad / 2);
    }
    CUDA_HOSTDEV inline vec3 make_ray(float x, float y)
    {
        return (right * x + up * y + forward * focal_length).normalize();
    }
};

#endif
