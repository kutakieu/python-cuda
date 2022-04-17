#include <vec3.h>
#include <camera.h>
class GPURenderer
{
  // pointer to the GPU memory where the array is stored
  int *array_device;
  // pointer to the CPU memory where the array is stored
  int *array_host;
  // length of the array (number of elements)
  int length;
  vec3 *fb;
  camera cam;
  int canvas_height, canvas_width;

public:
  /* By using the swig default names INPLACE_ARRAY1, DIM1 in the header
     file (these aren't the names in the implementation file), we're giving
     swig the info it needs to cast to and from numpy arrays.

     If instead the constructor line said
       GPURenderer(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  GPURenderer(int *INPLACE_ARRAY1, int DIM1, int DIM2); // constructor (copies to GPU)

  ~GPURenderer(); // destructor

  void render();

  void raymarch();

  void fb2img();

  void increment(); // does operation inplace on the GPU

  void retreive(); // gets results back from GPU, putting them in the memory that was passed in
  // the constructor

  // gets results back from the gpu, putting them in the supplied memory location
  void retreive_to(int *INPLACE_ARRAY1, int DIM1);
};
