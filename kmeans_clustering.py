"""
K-means clustering algorithm
Stefan, Yuzhao Heng
"""

from PIL import Image
import numpy as np
import cv2

TAG = "kc"

class KmeansClustering():
    """ Implements k-means clustering
        Works on a matrix of feature vectors. Returns a labeled grayscale image of clusters """
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
        # print(centers)
    def get_criteria(self):
        """ Get the criteria for clustering, namely set stop condition to either maximum iteration \
            reached, or change of vectors is small, and the values of both """
        return cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, \
            self.max_num_iteration, self.epsilon_change
    def cluster(self):
        """ Perform the clustering """
        return cv2.kmeans(\
            self.samples, self.k, None, self.criteria, self.num_attempt, cv2.KMEANS_RANDOM_CENTERS)
    def normalize_compactness(self, compactness):
        """ Normalize compactness of clustering result, based on dimensions of matrix """
        return compactness / (self.shape_map[0] * self.shape_map[1])
    def get_centers(self):
        """ Get the values of each cluster center """
        return self.centers
    def get_normalized_array_label(self, color_depth=8):
        """ Make the matrix of labels more distinctive by distributing labels evenly across a \
            color range.
            Returns a grayscale matrix """
        magnitude_top = 2**color_depth
        step = int(magnitude_top / len(self.centers))
        labels_normalized = np.array(list(range(0, magnitude_top, step)))
        print(self.array_label.shape)
        print(self.array_label[330:345])
        array_label = labels_normalized[self.array_label.flatten()]
        if not self.is_grayscale_matrix():
            array_label = array_label[2:]
            array_label = array_label[0::3] # Keep label of one channel so that color stands out
            print(array_label[110:115])
        return np.uint8(array_label)
    def get_matrix_label(self, array_label, normalized=True):
        """ Get the matrix of labels, equivalent to dimension of a grayscale image """
        # width, height = self.get_shape_matrix
        return array_label.reshape(self.get_shape_matrix(normalized))
    def get_shape_matrix(self, normalized):
        """ Get the shape of the original image from the map of features """
        if normalized:
            return self.shape_map[0], self.shape_map[1]
        else:
            if self.is_grayscale_matrix(): # Grayscale image with 1 color channel
                return self.shape_map[0], self.shape_map[1]
            else: # RGB image with 3 color channel
                return self.shape_map[0], self.shape_map[1], self.shape_map[3]
    def is_grayscale_matrix(self):
        return len(self.shape_map) == 3
    def output_image_label(self, path="", normalized=True):
        """ Write a image file of labels, normalized or not, given current clustering object """
        array_label = (self.array_label, self.get_normalized_array_label())[normalized]
        write_image_label(self.get_matrix_label(array_label, normalized), self.name_img, path)

def linearize_map_features(map_features, i_dimension=2):
    """ Linearize the first 2 dimensions of the map to make a 2D table, each entry consisting of a \
        feature vector """
    map_features = np.array(map_features)
    dimension = map_features.shape[i_dimension]
    samples = map_features.reshape(-1, dimension)
    return np.float32(samples)

def write_image_label(matrix_label, name_img, path=""):
    """ Write a image file of labels, given a matrix of label values """
    # print(matrix_label.shape)
    # print(matrix_label[0:15, 35:50])
    img_label = Image.fromarray(matrix_label)
    img_label.save(path + name_img+ "_"+TAG +".png")

def cluster_output_image_label(name_img, map_features, k, \
    max_num_iteration, epsilon_change=0.01, num_attempt=5, label_normalized=True):
    """ Write a image file of labels, given k-means clustering parameters """
    clustering = KmeansClustering(\
        name_img, map_features, k, max_num_iteration, epsilon_change, num_attempt)
    clustering.output_image_label(normalized=label_normalized)

def matrix_elements_to_str(matrix):
    """ Print individual elements for a matrix """
    string = ""
    for row in matrix:
        for e in row:
            string += str(e)+", "
        string += "\n"
    return string

PATH_FOLDER = "D:/UMD/Career/Research Assistant/Segmentation by Logic/Code/img_sample/"

def main():
    """ test """
    name = "Stefan with Art.jpg"
    normalized = False
    cluster_output_image_label(name, cv2.imread(PATH_FOLDER+name), 4, 30, normalized)

if __name__ == "__main__":
    main()
