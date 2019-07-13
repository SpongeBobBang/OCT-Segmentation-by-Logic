""" to test python generic methods """

from itertools import combinations
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

import laws_texture_energy
import kmeans_clustering
import image_factory

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

# t = (1, 2, 3)
# one, two = (1, 2, 3)[:2]
# print(one)
# print(two)

# NAME = "Stefan with Art"
# MAP = laws_texture_energy.extract_laws_texture_features(NAME+".jpg")
# kmeans_clustering.cluster_output_image_label(NAME+"_laws", MAP, 4, 10)

PATH_FOLDER = laws_texture_energy.URI_SAMPLES


def main():
    """ random temproary tests """
    # name = "Abrams_Post_114_1_1_0_1"
    # img = laws_texture_energy.remove_illumination(Image.open(PATH_FOLDER_ORI+name+".jpg"), 15)
    # img.save(name+"_ri.png")

    # name = "Stefan with Art"
    # img = laws_texture_energy.remove_illumination(Image.open(name+".jpg"), 15)
    # img.save(name+"_ri.png")

    # name = "Tiger"
    # map_features = laws_texture_energy.extract_laws_texture_features(name+".png")
    # kmeans_clustering.cluster_output_image_label(name+"_laws", map_features, 4, 30)

    # img = plt.imread("img_sample/Stefan with Art.jpg")
    # plt.figure(figsize=(10, 15))
    # plt.imshow(img)
    # plt.show()

    # a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    # print(np.mean(a))
    # print(np.mean(a, axis=0))
    # print(np.mean(a, axis=1))

    # img = Image.open("img_sample/Leaves.png")
    # print(img.mode)
    # mean_features = laws_texture_energy.extract_laws_texture_mean("img_sample/Leaves.png")
    # print('ee:', mean_features[6], 'ss:', mean_features[3], 'rr:', mean_features[4])

    # a = np.array([[1], [1], [1], [1], [1], [1], [1]])
    # a = a.flatten()
    # print(a)

    # a = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3,3 ], [4, 4, 4, 4]])
    # print(a[0:3, 0:2])

    # a = np.array([[13, 11,  8,  7], [11,  3, 11, 14], [16, 12, 14, 10], [15,  6, 10,  5]],
    #   \dtype=np.float64)
    # kernel = np.ones(9, dtype=np.float64).reshape(3, 3)
    # kernel = kernel*(-2)
    # print(ndimage.convolve(a, kernel))

    # a_img = np.array(Image.open("img_sample/Leaves.png"), dtype=np.float64)
    # as_img = np.split(a_img, [1, 2], axis=2)
    # kernel = np.ones(9, dtype=np.float64).reshape(3, 3)
    # kernel = kernel*10
    # # print(laws_texture_energy.convolve(a_img, kernel))
    # height, width = a_img.shape[:2]
    # print(ndimage.convolve(as_img[0].reshape(height, width), kernel))

    # arrays = [np.random.randn(20, 10) for _ in range(3)]
    # print(np.stack(arrays, axis=2).shape)

    # print(type(np.array(1)))

    # samples = ["Leaves", "Leaves2", "Grass", "Brick", "Brick2", "Stone"]
    # for sample in samples:
    #     img = Image.open("img_sample/"+sample+".png")
    #     mtrx_img = np.array(img)
    #     mean_features = laws_texture_energy.extract_laws_energy_mean(mtrx_img)
    #     print(sample, 'ee:', mean_features[6], 'ss:', mean_features[3], 'rr:', mean_features[4])

    # print("dshajd.jgp"[:-4])

    # img = Image.open("Tiger_laws_kc.png")
    # print(img.mode)

    # ary = np.arange(99)
    # # ary = [1, 2, 3, 4, 5]
    # ary = ary[2:]
    # ary = ary[0::3]
    # print(ary)

    # a = np.array([-1, -2, 0, 2, 1])
    # b = np.array([1, 4, 6, 4, 1])
    # print(np.dot(a, b))

    # img = Image.open("img_sample/Stefan with Art.jpg")
    # print(img.mode)

    # a = np.ones((3,9,3))
    # a[0] = a[0].ravel()
    # print(a.flatten())
    # print(isinstance(r, np.ndarray))

    # name_img = "Abrams_Post_114_1_1_0_1.jpg"
    # img = Image.open("img_sample/" + name_img)
    # mtrx_img = np.array(img)
    # mtrx_features = laws_texture_energy.get_laws_energy_matrix(mtrx_img)
    # k = 6
    # max_num_iteration = 30
    # mtrx_cluster = kmeans_clustering.get_kmeans_cluster_matrix(mtrx_features, k, max_num_iteration)
    # image_factory.write_image_by_matrix(mtrx_cluster, name_img, tag="laws_kc")

    # name_img = "Abrams_Post_114_1_1_0_1_laws_kc.png"
    # mtrx = image_factory.get_matrix_from_uri(name_img)
    # mtrx = image_factory.filter_by_gaussian(mtrx, 5)
    # image_factory.write_image_by_matrix(mtrx, name_img, tag="gaussian")

    # name_img = "Abrams_Post_114_1_1_0_1_laws_kc_gaussian.png"
    # mtrx_img = image_factory.get_matrix_from_uri(name_img)
    # mtrx_features = laws_texture_energy.get_laws_energy_matrix(mtrx_img)
    # mtrx_cluster = kmeans_clustering.get_kmeans_cluster_matrix(mtrx_features, 6, 30)
    # image_factory.write_image_by_matrix(mtrx_cluster, name_img, tag="laws_kc")

    # name_img = "Abrams_Post_114_1_1_0_1_laws_kc_gaussian.png"
    # mtrx_img = image_factory.get_matrix_from_uri(name_img)
    # mtrx_features = image_factory.sum_region(mtrx_img, 20)
    # mtrx_cluster = kmeans_clustering.get_kmeans_cluster_matrix(mtrx_features, 3, 30)
    # image_factory.write_image_by_matrix(mtrx_cluster, name_img, tag="sum_kc")

    # print(3 in [2, 3, 5])

    # mtrx = image_factory.get_matrix_from_uri("Abrams_Post_114_1_1_0_1.jpg")
    # image_factory.show_matrix(mtrx)

    # import matplotlib.image as mpimg
    # img = mpimg.imread('Abrams_Post_114_1_1_0_1.jpg')
    # plt.imshow(img)
    # plt.show()

    name = "Abrams_Post_114_1_1_0_1_laws_kc.png"
    mtrx = image_factory.get_matrix_from_uri(name)
    mtrx_avg = image_factory.average_region(mtrx, 20)
    image_factory.show_matrix(mtrx_avg)
    mtrx_cluster = kmeans_clustering.get_kmeans_cluster_matrix(mtrx_avg, 3, 30)
    image_factory.write_image_by_matrix(mtrx_cluster, name, tag="avg_kc")

    # mtrx = image_factory.get_matrix_from_uri("Abrams_Post_114_1_1_0_1.jpg")
    # mtrx = image_factory.threshold(mtrx, 127)
    # print(type(mtrx))
    # image_factory.show_matrix(mtrx)

if __name__ == "__main__":
    main()
