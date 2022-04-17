import cudapkg.gpurenderer
import numpy as np
from PIL import Image

canvas_width = 300
canvas_height = 300
arr = np.zeros([canvas_height, canvas_width, 3], dtype=np.int32)
adder = cudapkg.gpurenderer.GPURenderer(arr.reshape(-1), canvas_height, canvas_width)
adder.render()

Image.fromarray(arr.astype(np.uint8)).save("test.jpg")
