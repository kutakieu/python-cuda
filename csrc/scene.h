#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifndef SCENE
#define SCENE

#include <vector>
#include "vec3.h"
#include "distance_functions.h"
#include <camera.h>

class Scene
{
public:
    camera cam;
    Sphere objects;
    int n_objects;
    // thrust::device_vector<Sphere *> objects_device;
    // std::vector<Sphere *> objects;

    CUDA_HOSTDEV Scene()
    {
    }
    CUDA_HOSTDEV Scene(camera _cam, Sphere _objects, int _n_objects)
    {
        cam = _cam;
        objects = _objects;
        n_objects = _n_objects;
    }
    // CUDA_HOSTDEV Scene(std::vector<Sphere *> distance_functions)
    // {
    //     objects = distance_functions;
    // }
    // CUDA_HOSTDEV inline void add_object(Sphere &obj)
    // {
    //     objects_device.push_back(&obj);
    // }
    // __host__ inline void copy_to_device()
    // {
    //     objects_device = objects_host;
    // }
    CUDA_HOSTDEV inline float distance(vec3 position)
    {
        float distance = 100000;
        for (int i = 0; i < n_objects; i++)
        {
            Sphere obj = objects;
            float current_distance = obj.distance(position);
            if (current_distance < distance)
                distance = current_distance;
        }
        return distance;
    }
};

#endif
