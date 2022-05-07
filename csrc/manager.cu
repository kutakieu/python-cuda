#include <raymarch.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include <vec3.h>
#include <camera.h>
#include <scene.h>
#include <geometry.h>
#include <thrust/device_vector.h>

#define cudaCheckErrors(msg)                             \
  do                                                     \
  {                                                      \
    cudaError_t __err = cudaGetLastError();              \
    if (__err != cudaSuccess)                            \
    {                                                    \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
              msg, cudaGetErrorString(__err),            \
              __FILE__, __LINE__);                       \
      fprintf(stderr, "*** FAILED - ABORTING\n");        \
      exit(1);                                           \
    }                                                    \
  } while (0)

using namespace std;

GPURenderer::GPURenderer(int *array_host_, int canvas_height_, int canvas_width_)
{
  array_host = array_host_;
  canvas_height = canvas_height_;
  canvas_width = canvas_width_;
  cudaMallocManaged((void **)&fb, sizeof(vec3) * canvas_height * canvas_width);

  Sphere s = Sphere(1.0);
  int n_objects = 1;
  camera cam = camera(vec3(0, 1, 0), vec3(0, 0, -1), vec3(0, 0, 2), 90);
  scene = Scene(cam, &s, n_objects);
  cudaMalloc((void **)&(scene_device), sizeof(Scene));
  cudaCheckErrors("cudaMalloc1 fail");
  cudaMemcpy(scene_device, &scene, sizeof(Scene), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMalloc1 fail");

  Sphere *host_object;
  cudaMalloc((void **)&host_object, sizeof(Sphere) * n_objects);
  cudaCheckErrors("cudaMalloc1 fail");
  cudaMemcpy(host_object, scene.objects, sizeof(Sphere) * n_objects, cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMalloc1 fail");
  cudaMemcpy(&(scene_device->objects), &host_object, sizeof(Sphere *), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMalloc1 fail");

  cout
      << "n_objects: " << n_objects << endl;
}

void GPURenderer::render()
{
  cout << "render" << endl;
  int tx = 8;
  int ty = 8;
  dim3 blocks(canvas_width / tx + 1, canvas_height / ty + 1);
  dim3 threads(tx, ty);

  kernel_ray_marching<<<blocks, threads>>>(fb, canvas_height, canvas_width, scene_device);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  assert(err == 0);
  fb2img();
  cudaFree(fb);
}

// Scene GPURenderer::make_scene()
// {
//   Sphere s = Sphere(1.0);
//   scene.add_object(s);
//   // scene.copy_to_device();
//   return scene;
// }

void GPURenderer::fb2img()
{
  for (int j = canvas_height - 1; j >= 0; j--)
  {
    for (int i = 0; i < canvas_width; i++)
    {
      size_t pixel_index = j * canvas_width + i;
      int ir = int(255.99 * fb[pixel_index].r());
      int ig = int(255.99 * fb[pixel_index].g());
      int ib = int(255.99 * fb[pixel_index].b());
      array_host[pixel_index * 3] = ir;
      array_host[pixel_index * 3 + 1] = ig;
      array_host[pixel_index * 3 + 2] = ib;
    }
  }
}

GPURenderer::~GPURenderer()
{
  cudaFree(array_device);
}
