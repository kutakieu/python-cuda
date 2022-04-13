#include <stdio.h>
#include "vec3.h"

void __global__ kernel_add_one(int *a, int length)
{
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    while (gid < length)
    {
        a[gid] += 1;
        gid += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_render(vec3 *fb, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    fb[pixel_index] = vec3(float(i) / max_x, float(j) / max_y, 0.2f);
}
