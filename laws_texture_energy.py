""" Quantify texture by feature vector, from Laws' texture energy measures
    Stefan, Yuzhao Heng """

from PIL import Image
import numpy as np
from scipy import ndimage

import kmeans_clustering

class TextureEnergyByLaws():
    """ Implements Laws; texture energy masks for quantifying texture for a single image. Works on \
        grayscale images.
        Returns a matrix of dimension of image, each element as vector """
    def __init__(self, path):
        self.img = remove_illumination(Image.open(path), 15)
        self.a_img = np.array(self.img)
        self.init_masks()
        self.init_map_features()
        self.filter_img()
        self.get_feature_map()
    def init_masks(self):
        """ Get the 9 convolution masks to compute texture energy """
        vector_level = [1, 4, 6, 4, 1]
        vector_edge = [-1, -2, 0, 2, 1]
        vector_spot = [-1, 0, 2, 0, -1]
        vector_ripple = [1, -4, 6, -4, 1]
        self.vectors = {"l": vector_level, "e": vector_edge, "s": vector_spot, "r": vector_ripple}
        self.names_mask = [["le", "el"], ["lr", "rl"], ["es", "se"], ["ss"], ["rr"], ["ls", "sl"], \
            ["ee"], ["er", "re"], ["sr", "rs"]]
        self.dimension_feature = len(self.names_mask)
        self.masks = [None for i in range(len(self.names_mask))]
        self.imgs_feature = [None for i in range(self.dimension_feature)]
        for i, name_mask in enumerate(self.names_mask):
            self.masks[i] = self.get_mask(name_mask)
    def get_mask(self, name):
        """ Get the mask matrix based on name, corresponding to the vertical and horizontal matrix \
            """
        return self.get_mask_single(name) if self.is_single_mask(name) \
            else self.get_mask_double(name)
    def is_single_mask(self, e):
        """ Check if the name of the mask makes a single mask or a mask pair """
        # Left boolean for name of mask, right boolean for mask matrix
        return len(e) == 1 or len(e) == 5
    def get_mask_single(self, name):
        """ Get the mask matrix if the name contrains 1 instance """
        if not isinstance(name, str):
            name = name[0]
        vector_vert = self.vectors[name[0]]
        vector_hori = self.vectors[name[1]]
        return np.dot(make_matrix_vertical(vector_vert), make_matrix_horizontal(vector_hori))
    def get_mask_double(self, name):
        """ Get the mask matrix if the name contrains 2 instance """
        return self.get_mask_single(name[0]), self.get_mask_single(name[1])
    def init_map_features(self):
        """ Initialize the dimension of map of features with dimensions of image """
        width, height = self.img.size
        self.map_features = [[[] for x in range(width)] for y in range(height)]
    def filter_img(self):
        """ Get texture feature vector for each pixel """
        for i, mask in enumerate(self.masks):
            self.imgs_feature[i] = self.filter_by_mask(mask)
    def filter_by_mask(self, mask):
        """ Filter the image with mask """
        if self.is_single_mask(mask):
            return self.filter_by_1_mask(mask)
        else:
            img1 = self.filter_by_1_mask(mask[0])
            img2 = self.filter_by_1_mask(mask[1])
            return get_average_img(self.img, [img1, img2])
    def filter_by_1_mask(self, mask):
        """ Filter the image with 1 mask """
        return Image.fromarray(convolve(self.a_img, mask))
    def get_feature_map(self):
        """ Combine the filtered images by masks, into feature vectors of one matrix """
        for i in range(self.dimension_feature):
            matrix_feature = self.imgs_feature[i].load()
            for y in range(len(self.map_features)):
                for x in range(len(self.map_features[y])):
                    self.map_features[y][x].append(matrix_feature[x, y])
        self.map_features = np.array(self.map_features)

def get_pixel_values(img):
    """ Get the pixel values of an image """
    pixel_access = img.load()
    width, height = img.size
    pixels = [[None for y in range(height)] for x in range(width)]
    for x, row in enumerate(pixels):
        for y, e in enumerate(row):
            pixels[x][y] = pixel_access[x, y]
    return pixels

def make_matrix_vertical(l):
    """ Get the matrix with dimension m*1 needed for m*m masks, from an 1D list """
    return np.array(l).reshape(-1, 1)

def make_matrix_horizontal(l):
    """ Get the matrix with dimension 1*n needed for n*n masks, from an 1D list """
    return np.array(l).reshape(1, -1)

def linearize_matrix(matrix):
    """ Convert a matrix to 1D list, to be fed into a Kernel object """
    return np.array(matrix).flatten()

def is_grayscale_img(img):
    """ Checks if an Image or numpy array of an image is in grayscale or expected RGB """
    if isinstance(img, Image.Image):
        return img.mode == "L"
    elif isinstance(img, np.ndarray):
        return len(img.shape) == 2
    else:
        return False

def remove_illumination(img, dimension_kernel):
    """ Remove the effect of intensity in an image by subtracting the average of pixels in \
        proximity """
    a_img = np.array(img, dtype=np.float32)
    if not is_grayscale_img(a_img):
        a_img = a_img[..., :3] # Remove 4th transparency layer
    shape_kernel = dimension_kernel, dimension_kernel
    n_element = product(shape_kernel)
    kernel = np.ones(n_element, dtype=np.float32).reshape(shape_kernel)
    a_img_sum = convolve(a_img, kernel)
    a_img_avg = a_img_sum/n_element
    a_img = np.subtract(a_img, a_img_avg, dtype=np.float32)
    a_img = np.clip(a_img, 0, 2**8 - 1)
    a_img = a_img.astype(np.uint8)
    return Image.fromarray(a_img)

def convolve(a_img, kernel):
    """ Convoluve a matrix of potentially multiple dimension vectors by a kernel, returns a np \
        array. Typically to an image """
    if is_grayscale_img(a_img):
        return convolve_1_channel(a_img, kernel)
    else:
        height, width = a_img.shape[:2]
        as_img = np.split(a_img, [1, 2], axis=2)
        for a_img_channel in as_img:
            a_img_channel = a_img_channel.reshape(height, width)
            a_img_channel = convolve_1_channel(a_img_channel, kernel)
        return np.dstack(as_img)

def convolve_1_channel(a_img, kernel):
    """ Convolve a matrix with elements of single dimension """
    return ndimage.convolve(a_img, kernel)

def product(l):
    """ Get the product of a list or tuple """
    prod = 1
    for e in l:
        prod *= e
    return prod

def get_average_img(img_skeleton, list_img):
    """ Get the average image given a list of images of same shape """
    l = len(list_img)
    mode = img_skeleton.mode
    width, height = img_skeleton.size
    shape = ((height, width, 3), (height, width))[mode == "L"]
    a_img_avg = np.zeros(shape, np.float)
    for img in list_img:
        a_img = np.array(img, dtype=np.float)
        a_img_avg += a_img/l
    a_img_avg = np.array(np.round(a_img_avg), dtype=np.uint8)
    return Image.fromarray(a_img_avg, mode=mode)

def extract_laws_texture_features(img):
    """ Get the feature vector map, using laws texture energy measures """
    tebl = TextureEnergyByLaws(img)
    return tebl.map_features

if __name__ == "__main__":
    # PATH_FOLDER = "D:/UMD/Career/Research Assistant/Segmentation by Logic/Code/Image/ori/"
    # NAME_IMG = "Abrams_Post_114_1_1_0_1.jpg"
    # # TEBL = TextureEnergyByLaws(PATH_FOLDER+NAME_IMG)
    # MAP_FEATURES = extract_laws_texture_features(PATH_FOLDER+NAME_IMG)
    NAME = "Stefan with Art.jpg"
    MAP = extract_laws_texture_features(NAME)
    # print("map", MAP)
    # print(MAP.shape)
