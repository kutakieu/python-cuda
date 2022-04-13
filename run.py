import gpurenderer
import numpy as np
import numpy.testing as npt


def test():
    canvas_width = 300
    canvas_height = 300
    arr = np.zeros([canvas_height, canvas_width, 3], dtype=np.int32)
    adder = gpurenderer.GPURenderer(arr.reshape(-1), canvas_height, canvas_width)
    adder.render()

    print(arr)


test()
