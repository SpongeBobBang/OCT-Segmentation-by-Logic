""" to test python generic methods """

from itertools import combinations
from PIL import Image, ImageFilter
import numpy as np
import cv2

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

map = law_texture_energy.extract_laws_texture_features("Tiger.png")
kmeans_clustering.cluster_output_image_label("Tiger_kc.png", map, 4, 5)
