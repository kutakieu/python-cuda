#include <vec3.h>
#include <camera.h>
#include <scene.h>
class GPURenderer
{
  int *array_device;
  int *array_host;
  int length;
  vec3 *fb;
  Scene scene;
  Scene *scene_device;
  Sphere *sphere;
  int canvas_height, canvas_width;

public:
  GPURenderer(){};
  GPURenderer(int *INPLACE_ARRAY1, int DIM1, int DIM2);
  ~GPURenderer();
  void render();
  Scene make_scene();
  void raymarch();
  void fb2img();
};
