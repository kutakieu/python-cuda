#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifndef SCENE
#define SCENE

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include "vec3.h"
#include "geometry.h"
#include <camera.h>

class Scene
{
public:
    camera cam;
    Sphere *objects;
    int n_objects;

    CUDA_HOSTDEV Scene()
    {
    }
    CUDA_HOSTDEV Scene(camera _cam, Sphere *_objects, int _n_objects)
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
    // CUDA_HOSTDEV inline void copy_objects_to_device()
    // {
    //     cudaMemcpyToSymbol("objects_device", objects, sizeof(Sphere) * n_objects);
    //     // objects_device = objects;
    // }
    CUDA_HOSTDEV inline float distance(vec3 position)
    {
        float distance = 100000;
        for (int i = 0; i < n_objects; i++)
        {
            // Sphere *obj = objects;
            // float current_distance = objects[i].radius;
            float current_distance = objects[i].distance(position);
            // float current_distance = 100000;
            if (current_distance < distance)
                distance = current_distance;
        }
        return distance;
    }
};

#endif
