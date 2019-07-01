"""
K-means clustering algorithm
Stefan, Yuzhao Heng
"""

from PIL import Image
import numpy as np
import cv2

class KmeansClustering():
    """
    Implements k-means clustering
    Works on a matrix of feature vectors
    Returns a labeled grayscale image of clusters
    """
    def __init__(self, name_img, map_features, k, \
        max_num_iteration, epsilon_change=0.01, num_attempt=5):
        self.name_img = name_img
        self.k = k
        self.max_num_iteration = max_num_iteration
        self.epsilon_change = epsilon_change
        self.num_attempt = num_attempt
        self.shape_map = map_features.shape
        self.samples = linearize_map_features(map_features)
        self.criteria = self.get_criteria()
        compactness, array_label, centers = self.cluster()
        self.compatness_normalized = self.normalize_compactness(compactness)
        self.array_label = array_label
        self.centers = centers
    def get_criteria(self):
        """
        Get the criteria for clustering, namely set stop condition to either maximum iteration
        reached, or change of vectors is small, and the values of both
        """
        return cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, \
            self.max_num_iteration, self.epsilon_change
    def cluster(self):
        """ Perform the clustering """
        return cv2.kmeans(\
            self.samples, self.k, None, self.criteria, self.num_attempt, cv2.KMEANS_RANDOM_CENTERS)
    def normalize_compactness(self, compactness):
        """ Normalize compactness of clustering result, based on dimensions of matrix """
        return compactness / (self.shape_map[0] * self.shape_map[1])
    def get_normalized_array_label(self, color_depth=8):
        """
        Make the matrix of labels more distinctive by distributing labels evenly across a color
        range
        """
        magnitude_top = 2**color_depth
        step = int(magnitude_top / len(self.centers))
        labels_normalized = np.array(list(range(0, magnitude_top, step)))
        return np.uint8(labels_normalized[self.array_label.flatten()])
    def get_matrix_label(self, array_label):
        """ Get the matrix of labels, equivalent to dimension of a grayscale image """
        # width, height = self.get_shape_matrix
        return array_label.reshape(self.get_shape_matrix())
    def get_shape_matrix(self):
        """ Get the shape of the original image from the map of features """
        if len(self.shape_map) == 3: # Grayscale image with 1 color channel
            return self.shape_map[0], self.shape_map[1]
        else: # RGB image with 3 color channel
            return self.shape_map[0], self.shape_map[1], self.shape_map[3]
    def output_image_label(self, path="", normalized=True):
        """ Write a image file of labels, normalized or not, given current clustering object """
        array_label = (self.array_label, self.get_normalized_array_label())[normalized]
        write_image_label(self.get_matrix_label(array_label), self.name_img, path)

def linearize_map_features(map_features, i_dimension=2):
    """ Linearize the first 2 dimensions of the map to make a 2D table,
    each entry consisting of a feature vector """
    map_features = np.array(map_features)
    dimension = map_features.shape[i_dimension]
    samples = map_features.reshape(-1, dimension)
    return np.float32(samples)

def write_image_label(matrix_label, name_img, path=""):
    """ Write a image file of labels, given a matrix of label values """
    img_label = Image.fromarray(matrix_label)
    img_label.save(path + name_img+"_kc.png")

def cluster_output_image_label(name_img, map_features, k, \
    max_num_iteration, epsilon_change=0.01, num_attempt=5):
    """ Write a image file of labels, given k-means clustering parameters """
    clustering = KmeansClustering(\
        name_img, map_features, k, max_num_iteration, epsilon_change, num_attempt)
    clustering.output_image_label()

def print_matrix_elements(matrix):
    """ Print individual elements for a matrix """
    for row in matrix:
        for e in row:
            print(e, end=" ")
        print()

if __name__ == "__main__":
    NAME = "Stefan with Art.jpg"
    CLUSTER = KmeansClustering(NAME[:-4], cv2.imread(NAME), 4, 10)
    CLUSTER.output_image_label()

    # matrix = CLUSTER.get_matrix_label(CLUSTER.get_normalized_array_label())
    # print(matrix.shape)
    # print(len(matrix))
    # img_label = Image.fromarray(matrix)
    # img_label.save(NAME+"_main.png")
