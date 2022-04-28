#include <vec3.h>
#include <camera.h>
#include <scene.h>
class GPURenderer
{
  // pointer to the GPU memory where the array is stored
  int *array_device;
  // pointer to the CPU memory where the array is stored
  int *array_host;
  // length of the array (number of elements)
  int length;
  vec3 *fb;
  Scene scene;
  int canvas_height, canvas_width;

public:
  GPURenderer(int *INPLACE_ARRAY1, int DIM1, int DIM2);
  ~GPURenderer();
  void render();
  Scene make_scene();
  void raymarch();
  void fb2img();
};
