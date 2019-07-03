""" to test python generic methods """

from itertools import combinations
from PIL import Image, ImageFilter
import numpy as np
import cv2
from scipy import ndimage

import law_texture_energy
import kmeans_clustering

# IMG = cv2.imread('Stefan with Art.jpg')
# a = np.array([[1, 2], [3, 4]])
# print(IMG.shape)
# IMG = Image.open('Stefan with Art.jpg')
# print(IMG.size)

# print(list(range(5, 9)))

# def make_tuple():
#     t = (1, 2, 3)
#     return t[0], t[1]

# mt = make_tuple()
# a, b = mt
# print(mt)
# print(mt[0])
# print(a)

# ternary = (1, 2)[True]
# print(ternary)

# NAME_IMG = "Stefan with Art"
# K = 6
# MAX_NUM_ITERATION = 10
# MAP_FEATURES = law_texture_energy.extract_law_texture_features(NAME_IMG+".jpg")
# kmeans_clustering.cluster_output_image_label(\
#     NAME_IMG+"_law's", MAP_FEATURES, K, MAX_NUM_ITERATION)

# img = Image.open('Stefan with Art.jpg')
# img.filter(ImageFilter.GaussianBlur(10)).show()

# print(len(["ee"]))
# print(('f', 't')[True])

# tupl = (1, 2), 3
# print(tupl[0])

# map = law_texture_energy.extract_laws_texture_features("Tiger.png")
# # print(map)
# kmeans_clustering.cluster_output_image_label("Tiger_kc.png", map, 4, 5)

# NAME = "Stefan with Art.jpg"
# IMG = Image.open(NAME)
# A_IMG = np.array(IMG)
# A_IMG2 = cv2.imread(NAME)
# print(IMG.mode, IMG.size)
# print(A_IMG.shape)
# print(A_IMG2.shape)

# T = (2, 3, 4, 5)
# print(law_texture_energy.product(T))


# IMG = cv2.imread("Tiger.png")
# b, g, r = cv2.split(IMG)
# print(isinstance(b, np.ndarray))

# a = np.array([3, 4, 5, 6, 7])
# print(a-1)E)

# e = [1, 2, 3, 4, 5, 6]
# a = np.array([e, e, e, e, e, e])
# b = np.array([e, e, e, e, e, e])
# print(np.subtract(a, b))

# NAME = "test2.png"
# img = Image.open(NAME)
# a_img = np.array(img)
# as_img = np.split(a_img, [1, 2], axis=2)
# a_img_r = as_img[0].reshape(img.size)
# i = 20
# # print(a_img_r[i][i])
# kmeans_clustering.print_matrix_elements(a_img_r)
# k = np.ones(25, dtype=np.float).reshape(5, 5)
# o = ndimage.convolve(a_img_r, k)
# print()
# kmeans_clustering.print_matrix_elements(o)
# # print(o[i][i])
# print()
# d = np.subtract(a_img_r[0], o[0], dtype=np.int16)
# print(d)
# print(np.clip(d, 0, 255))
# # print(d[i][i])
# # kmeans_clustering.print_matrix_elements(d)

# NAME = "Tiger.png"
# # a_img = np.array(Image.open(NAME))
# # print(a_img.shape)
# # print(np.array(Image.open(NAME)))
# # kmeans_clustering.print_matrix_elements(np.array(Image.open(NAME)))
# MAP = law_texture_energy.extract_laws_texture_features(NAME)

# NAME = "test2.png"
# img = Image.open(NAME)
# a_img = np.array(img)[..., :3]
# as_img = np.split(a_img, [1, 2], axis=2)
# a_img_r = as_img[0].reshape(img.size)
# i = 20
# # print(a_img_r[i][i])
# kmeans_clustering.print_matrix_elements(a_img_r)
# print()
# k = np.ones(25, dtype=np.float).reshape(5, 5)
# o = ndimage.convolve(a_img_r, k)
# o = o/25
# kmeans_clustering.print_matrix_elements(o)
# print()
# o = np.subtract(a_img_r, o, dtype=np.float32)
# kmeans_clustering.print_matrix_elements(o)
# print()
# d = np.clip(o, 0, 255)
# d = d.astype(np.uint8)
# kmeans_clustering.print_matrix_elements(d)
# print()

# NAME ="Stefan with Art.jpg"
# # MAP = law_texture_energy.extract_laws_texture_features(NAME)
# img = law_texture_energy.remove_illumination(Image.open(NAME), 15)
# img.save("sdad.png")

t = (1, 2, 3)
one, two = (1, 2, 3)[:2]
print(one)
print(two)
