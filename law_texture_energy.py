"""
Quantify texture by feature vector, from Law's texture energy measures
Stefan, Yuzhao Heng
"""
import numpy as np
from PIL import Image, ImageFilter

class TextureEnergyByLaw():
    """ Implements Law's texture energy masks for quantifying texture """
    def __init__(self, path):
        self.init_masks()
        self.load_img(path)
        self.filter_img()
        self.get_feature_map()
    def init_masks(self):
        """ Get the 9 convolution masks to compute texture energy """
        vector_level = [1, 4, 6, 4, 1]
        vector_edge = [-1, -2, 0, 2, 1]
        vector_spot = [-1, 0, 2, 0, -1]
        vector_ripple = [1, -4, 6, -4, 1]
        self.size_vector = 5
        self.size_mask = (self.size_vector, self.size_vector)
        self.i_name_vert = 0
        self.i_name_hori = 1
        self.vectors = {"l": vector_level, "e": vector_edge, "s": vector_spot, "r": vector_ripple}
        self.names_mask = ["ee", "ss", "rr", "el", "el", "rl", "se", "re", "rs"]
        self.dimension_feature = len(self.names_mask)
        self.masks = [None for i in range(self.dimension_feature)]
        self.maps_features = [None for i in range(self.dimension_feature)]
        for i in range(len(self.names_mask)):
            self.masks[i] = self.get_mask(self.names_mask[i])
    def get_mask(self, name):
        """ Get the mask matrix based on name, corresponding to the vertical and horizontal matrix
        """
        vector_vert = self.vectors[name[self.i_name_vert]]
        vector_hori = self.vectors[name[self.i_name_hori]]
        print(make_matrix_vertical(vector_vert))
        return np.dot(make_matrix_vertical(vector_vert), make_matrix_horizontal(vector_hori))
    def load_img(self, path):
        """ Load the image to get pixel values """
        self.img = Image.open(path)
        self.size_img = self.img.size
        width, height = self.size_img
        self.map_features = [[[] for y in range(height)] for x in range(width)]
        self.features_filtered = [None for i in range(self.dimension_feature)]
    def filter_img(self):
        """ Get texture feature vector for each pixel """
        for i, mask in enumerate(self.masks):
            kernel = ImageFilter.Kernel(self.size_mask, linearize_matrix(mask))
            self.maps_features[i] = self.img.filter(kernel)
    def get_feature_map(self):
        """ Combine the filtered images by masks, into feature vectors of one matrix """
        for i in range(self.dimension_feature):
            matrix_features = self.maps_features[i].load()
            for x in range(len(self.map_features)):
                for y in range(len(self.map_features[x])):
                    self.map_features[x][y].append(matrix_features[x, y])

def make_matrix_vertical(l):
    """ Get the matrix with dimension m*1 needed for m*m masks, from an 1D list """
    # matrix = [[0] for i in range(len(l))]
    # for i, value in enumerate(l):
    #     matrix[i][0] = value
    # return matrix
    a = np.array(l)
    return a.reshape(-1, 1)

def make_matrix_horizontal(l):
    """ Get the matrix with dimension 1*n needed for n*n masks, from an 1D list """
    matrix = [[0 for i in range(len(l))]]
    for i, value in enumerate(l):
        matrix[0][i] = value
    return matrix

def linearize_matrix(matrix):
    """ Convert a matrix to 1D list, to be fed into a Kernel object """
    return np.array(matrix).flatten()

def extract_features(img):
    """ Get the feature vector map, using law's texture energy measures """
    tebl = TextureEnergyByLaw(img)
    return tebl.map_features

if __name__ == "__main__":
    PATH_FOLDER = "D:/UMD/Career/Research Assistant/Segmentation by Logic/Code/Image/ori/"
    NAME_IMG = "Abrams_Post_114_1_1_0_1.jpg"
    TEBL = TextureEnergyByLaw(PATH_FOLDER+NAME_IMG)
