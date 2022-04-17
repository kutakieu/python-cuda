import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "csrc/manager.hh":
    cdef cppclass C_GPURenderer "GPURenderer":
        C_GPURenderer(np.int32_t*, int, int)
        void render()
        void raymarch()
        void fb2img()
        void increment()
        void retreive()
        void retreive_to(np.int32_t*, int)

cdef class GPURenderer:
    cdef C_GPURenderer* g
    cdef int canvas_height
    cdef int canvas_width

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.int32_t] arr, canvas_height, canvas_width):
        # self.canvas_height, self.canvas_width = arr.shape[:2]
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.g = new C_GPURenderer(&arr[0], self.canvas_height, self.canvas_width)

    def render(self):
        self.g.render()
