/*
This is the central piece of code. This file implements a class
(interface in GPURenderer.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include <vec3.h>
using namespace std;

GPURenderer::GPURenderer(int *array_host_, int canvas_height_, int canvas_width_)
{
  array_host = array_host_;
  canvas_height = canvas_height_;
  canvas_width = canvas_width_;
  // length = length_;
  // int size = length * sizeof(int);
  // cudaError_t err = cudaMalloc((void **)&array_device, size);
  // assert(err == 0);
  // err = cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
  // assert(err == 0);

  // size_t fb_size = canvas_height * canvas_width * sizeof(vec3);
  // cudaError_t err = cudaMalloc((void **)&fb, fb_size);
  // assert(err == 0);
  // err = cudaMemcpy(array_device, array_host, fb_size, cudaMemcpyHostToDevice);
  // assert(err == 0);

  cudaMallocManaged((void **)&fb, sizeof(vec3) * canvas_height * canvas_width);
}

void GPURenderer::render()
{
  int tx = 8;
  int ty = 8;
  dim3 blocks(canvas_width / tx + 1, canvas_height / ty + 1);
  dim3 threads(tx, ty);
  kernel_render<<<blocks, threads>>>(fb, canvas_height, canvas_width);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
  cudaDeviceSynchronize();
  // size_t fb_size = canvas_height * canvas_width * sizeof(vec3);
  // cudaMemcpy(array_host, array_device, fb_size, cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  assert(err == 0);

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
  cudaFree(fb);
}

void GPURenderer::increment()
{
  kernel_add_one<<<64, 64>>>(array_device, length);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

void GPURenderer::retreive()
{
  int size = length * sizeof(int);
  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if (err != 0)
  {
    cout << err << endl;
    assert(0);
  }
}

void GPURenderer::retreive_to(int *array_host_, int length_)
{
  assert(length == length_);
  int size = length * sizeof(int);
  cudaMemcpy(array_host_, array_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

GPURenderer::~GPURenderer()
{
  cudaFree(array_device);
}
