"""
K-means clustering algorithm
Stefan, Yuzhao Heng
"""

from PIL import Image
import numpy as np
import cv2
import laws_texture_energy

TAG = "kc"

class KmeansClustering():
    """ Implements k-means clustering
        Works on a matrix of feature vectors. Returns a labeled grayscale image of clusters """
    def __init__(self, name_image, matrix_features, k, \
        max_num_iteration, epsilon_change=0.01, num_attempt=5):
        self.name_img = name_image
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
    def output_image_label(self, path, normalized=True):
        """ Write a image file of labels, normalized or not, given current clustering object """
        arry_lbl = self.get_array_label(normalized)
        write_image_by_matrix(self.get_matrix_label(arry_lbl), self.name_img, path)
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
    def get_matrix_label(self, array_label):
        """ Get the matrix of labels, equivalent to dimension of a grayscale image """
        return array_label.reshape(self.get_shape_matrix())
    def get_shape_matrix(self):
        """ Get the shape of the original image from the map of features """
        if is_grayscale_matrix_shape(self.shp_mtrx): 
            return self.shp_mtrx[:2]
        else: # RGB image with 3 color channel
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
    return len(shape_matrix) == 3

def get_criteria(max_num_iteration, epsilon_change):
    """ Get the criteria for clustering, namely set stop condition to either maximum iteration \
        reached, or change of vectors is small, and the values of both """
    return cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_num_iteration, epsilon_change

def kmeans_cluster(samples, k, criteria, num_attempt):
    """ Perform k-means clustering """
    return cv2.kmeans(samples, k, None, criteria, num_attempt, cv2.KMEANS_RANDOM_CENTERS)

def write_image_by_matrix(matrix, name_img, path):
    """ Write a image file of labels, given a matrix of label values """
    img_lbl = Image.fromarray(matrix)
    img_lbl.save(path+name_img+ "_"+TAG +".png")

def cluster_output_image_label(name_image, matrix_features, k, \
    max_num_iteration, epsilon_change=0.01, num_attempt=5, label_normalized=True, path=""):
    """ Write a image file of labels, given k-means clustering parameters """
    clstrng = KmeansClustering(\
        name_image, matrix_features, k, max_num_iteration, epsilon_change, num_attempt)
    clstrng.output_image_label(path, label_normalized)

URI_SAMPLES = laws_texture_energy.URI_SAMPLES

def main():
    """ test """
    name = "Headshot.jpg"
    normalized = False
    cluster_output_image_label(name[:-4], cv2.imread(URI_SAMPLES+name), 4, 30, normalized)

if __name__ == "__main__":
    main()
