#include <geometry.h>

__global__ void copy_objects(Sphere *objects_device, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i == 0 && j == 0)
    {
    }
    return;
}
