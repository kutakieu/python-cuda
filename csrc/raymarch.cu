#include <stdio.h>
#include "vec3.h"
#include "camera.h"
#include "scene.h"
#include "math_utils.h"
#include <iostream>
using namespace std;

#define PI 3.14159265
#define ANGLE 90.0
#define FOV ANGLE * 0.5 * PI / 180
#define CAM_POS vec3(0, 0, -5)
#define SPHERE_SIZE 1.0
#define LIGHT_DIR vec3(-0.577, 0.577, 0.577)

__device__ float distance_function(vec3 pos)
{
    return pos.length() - SPHERE_SIZE;
}

__device__ vec3 get_normal(vec3 pos)
{
    float d = 0.0001;
    return vec3(
               distance_function(pos + vec3(d, 0, 0)) - distance_function(pos + vec3(-d, 0, 0)),
               distance_function(pos + vec3(0, d, 0)) - distance_function(pos + vec3(0, -d, 0)),
               distance_function(pos + vec3(0, 0, d)) - distance_function(pos + vec3(0, 0, -d)))
        .normalize01();
}

__device__ vec3 ray_march(vec3 ray, Scene *scene)
{
    float distance = 0.0;
    float rLen = 0.0;
    vec3 rPos = scene->cam.position;
    for (int i = 0; i < 64; i++)
    {
        distance = scene->distance(rPos);
        rLen += distance;
        rPos = scene->cam.position + ray * rLen;
    }

    if (abs(distance) < 0.001)
    {
        vec3 normal = get_normal(rPos);
        float diff = MathUtils::clamp(abs(dot(LIGHT_DIR, normal)), 0.1, 1.0);
        return vec3(diff, diff, diff);
        // return vec3(normal.x(), normal.y(), normal.z());
    }
    else
    {
        return vec3(0, 0, 0);
    }
}

__global__ void kernel_ray_marching(vec3 *fb, int max_x, int max_y, Scene *scene)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    float focus_length = 1 / tan(FOV);
    vec3 ray = scene->cam.make_ray(float(i) / max_x * 2 - 1, float(j) / max_y * 2 - 1);
    fb[pixel_index] = ray_march(ray, scene);
}
