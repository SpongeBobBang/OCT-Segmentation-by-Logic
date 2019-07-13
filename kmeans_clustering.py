"""
K-means clustering algorithm
Stefan, Yuzhao Heng
"""

import numpy as np
import cv2

import laws_texture_energy
import image_factory

TAG = "kc"

class KmeansClustering():
    """ Implements k-means clustering
        Works on a matrix of feature vectors. Returns a labeled grayscale image of clusters """
    def __init__(self, matrix_features, k, \
        max_num_iteration, epsilon_change=0.01, num_attempt=5):
        self.shp_mtrx = matrix_features.shape
        self.k = k
        smpls = linearize_matrix_features(matrix_features, self.shp_mtrx)
        crtr = get_criteria(max_num_iteration, epsilon_change)
        cmpctnss, arry_lbl, cntrs = kmeans_cluster(smpls, k, crtr, num_attempt)
        self.compatness_normalized = self.normalize_compactness(cmpctnss)
        self.arry_lbl = arry_lbl
        self.cntrs = cntrs
    def normalize_compactness(self, compactness):
        """ Normalize compactness of clustering result, based on dimensions of matrix """
        return compactness / (self.shp_mtrx[0] * self.shp_mtrx[1])
    def get_matrix_label(self, normalized=True):
        """ Get the matrix of labels, equivalent to dimension of a grayscale image """
        arry_lbl = self.get_array_label(normalized)
        return arry_lbl.reshape(self.get_shape_matrix())
    def get_array_label(self, normalized, color_depth=8):
        """ Make the matrix of labels more distinctive by distributing labels evenly across a \
            grayscale color range.
            Returns a grayscale matrix """
        arry_lbl = self.arry_lbl.flatten()
        if normalized:
            lmt_top = 2**color_depth
            step = int(lmt_top / self.k)
            lbls_nrmlzd = np.array(list(range(0, lmt_top, step)))
            arry_lbl = lbls_nrmlzd[arry_lbl]
        else:
            arry_lbl = self.cntrs[arry_lbl]
        return np.uint8(arry_lbl)
    def get_shape_matrix(self):
        """ Get the shape of the original image from the map of features """
        if is_grayscale_matrix_shape(self.shp_mtrx):
            return self.shp_mtrx[:2]
        else:
            return self.shp_mtrx[0], self.shp_mtrx[1], self.shp_mtrx[3]

def linearize_matrix_features(matrix_features, shape_matrix):
    """ Linearize the first 2 dimensions of the matrix, so that each entry consisting of a feature \
        vector """
    hght, wdth = shape_matrix[:2]
    # Has more than 1 channels
    # Only changes shape of ndarray when each pixel element is more than 1 dimension
    if not is_grayscale_matrix_shape(shape_matrix):
        mtrx_ftrs = [[None for x in range(wdth)] for y in range(hght)]
        for y, row in enumerate(matrix_features):
            for x, e in enumerate(row):
                mtrx_ftrs[y][x] = e.flatten()
        matrix_features = np.array(mtrx_ftrs)
    return matrix_features.reshape(hght*wdth, -1).astype(np.float32)

def is_grayscale_matrix_shape(shape_matrix):
    """ Checks if the image, from shape of a given matrix is in grayscale or expected RGB """
    return len(shape_matrix) in [2, 3]

def get_criteria(max_num_iteration, epsilon_change):
    """ Get the criteria for clustering, namely set stop condition to either maximum iteration \
        reached, or change of vectors is small, and the values of both """
    return cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_num_iteration, epsilon_change

def kmeans_cluster(samples, k, criteria, num_attempt):
    """ Perform k-means clustering """
    return cv2.kmeans(samples, k, None, criteria, num_attempt, cv2.KMEANS_RANDOM_CENTERS)

def get_kmeans_cluster_matrix(matrix_features, k, max_num_iteration, \
    epsilon_change=0.01, num_attempt=5, label_normalized=True):
    """ Write a image file of labels, given k-means clustering parameters """
    return KmeansClustering(matrix_features, k, max_num_iteration, epsilon_change, num_attempt).\
        get_matrix_label(label_normalized)

URI_SAMPLES = laws_texture_energy.URI_SAMPLES

def main():
    """ module test """
    name = "Headshot.jpg"
    normalized = False
    mtrx = get_kmeans_cluster_matrix(name[:-4], cv2.imread(URI_SAMPLES+name), 4, 30, normalized)
    image_factory.write_image_by_matrix(mtrx, name)

if __name__ == "__main__":
    main()
