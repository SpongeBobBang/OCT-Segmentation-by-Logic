from PIL import Image
import numpy
im = Image.open("Stefan with Art.jpg")
np_im = numpy.array(im)
print(np_im.shape)
np_im = np_im - 18
print(np_im)
new_im = Image.fromarray(np_im)
new_im.save("numpy_altered_sample2.png")
