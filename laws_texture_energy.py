""" Quantify texture by feature vector, from Laws' texture energy measures
    Stefan, Yuzhao Heng """

from PIL import Image
import numpy as np
from scipy import ndimage

TAG = "laws"

class TextureEnergyByLaws():
    """ Implements Laws; texture energy masks for quantifying texture for a single image. Works on \
        grayscale images.
        Returns a matrix of dimension of image, each element as vector """
    def __init__(self, uri, radius_illumination=15):
        self.img = remove_illumination(Image.open(uri), radius_illumination)
        # self.img = Image.open(uri)
        self.a_img = np.array(self.img)
        self.init_masks()
        self.init_map_features()
        self.filter_img()
        self.sum_texture_energy()
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
        self.maps_energy = [None for i in range(self.dimension_feature)]
        for i, name_mask in enumerate(self.names_mask):
            self.masks[i] = self.get_mask(name_mask)
    def get_mask(self, name):
        """ Get the mask matrix based on name, corresponding to the vertical and horizontal matrix \
            """
        return self.get_mask_single(name) if self.is_single_mask(name) \
            else self.get_mask_double(name)
    def is_single_mask(self, e):
        """ Check if the name of the mask makes a single mask or a mask pair """
        # For name of mask, or mask matrix
        if isinstance(e, list):
            return len(e) == 1
        elif isinstance(e, np.ndarray):
            return len(e) == 5
        else:
            return False
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
            # return get_average_img(self.img, [img1, img2])
            return [img1, img2]
    def filter_by_1_mask(self, mask):
        """ Filter the image with 1 mask """
        a_img = self.a_img.astype(int)
        a_img = convolve(a_img, mask) # Results can be negative and should be 
        # a_img = np.clip(a_img, 0, 2**8 - 1)
        # print(a_img[20:25, 20:25])
        # Image.fromarray(a_img.astype(np.uint8)).show()
        # print(a_img[20:25, 20:25])
        # print(a_img.shape)
        return a_img
    def sum_texture_energy(self):
        """ Get the texture energy map by getting the sum of pixel values in proximity """
        for i, img_feature in enumerate(self.imgs_feature):
            # print(i, img_feature)
            self.maps_energy[i] = self.filter_by_sum(img_feature, 15)
            # print(self.maps_energy[i].shape)
    def is_single_img_feature(self, img_feature):
        """ Check if the element of filtered images by mask contains one image """
        # Can be either an Image object, or a list of Image objects
        return isinstance(img_feature, np.ndarray)
        # return len(img_feature) != 2
    def filter_by_sum(self, img_feature, dimension_kernel):
        """ Filter each texture feature by getting regional sum """
        if self.is_single_img_feature(img_feature):
            # print(sum_region(img_feature, dimension_kernel).shape)
            return sum_region(img_feature, dimension_kernel)
        else:
            a_img1 = sum_region(img_feature[0], dimension_kernel)
            a_img2 = sum_region(img_feature[1], dimension_kernel)
            return get_average_matrix(a_img1.shape, [a_img1, a_img2])
    def get_feature_map(self):
        """ Combine the filtered images by masks, into feature vectors of one matrix """
        for i in range(self.dimension_feature):
            matrix_feature = self.maps_energy[i]
            # print(i, matrix_feature == None)
            # print(matrix_feature.shape)
            for y in range(len(self.map_features)):
                for x in range(len(self.map_features[y])):
                    # print(self.map_features[y][x])
                    self.map_features[y][x].append(matrix_feature[y][x])
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

def sum_region(img, dimension_kernel):
    """ Filter an image where pixel values are regional sum """
    a_img = np.array(img, dtype=np.float64)
    if not is_grayscale_img(a_img):
        a_img = a_img[..., :3] # Remove 4th transparency layer for RGB image
    shape_kernel = dimension_kernel, dimension_kernel
    n_element = product(shape_kernel)
    kernel = np.ones(n_element, dtype=np.float64).reshape(shape_kernel)
    # kernel = kernel*67
    a = convolve(a_img, kernel)
    # print(a[55:60, 55:60])
    # kernel2 = np.array([[0, 0, 0], [0, 100, 0], [0, 0, 0]])
    # print(kernel2.dtype)
    p = convolve(a_img, kernel)
    # if p.dtype == np.float64:
    #     print(p[20:25, 20:25])
    return convolve(a_img, kernel)

def convolve(a_img, kernel):
    # if a_img.dtype == np.float64:
    #     print(a_img.dtype)
    """ Convoluve an image matrix of potentially multiple dimension vectors by a kernel, returns a \
        np array """
    # if not with_limit:
    #     a_img = np.array(a_img, dtype=np.float64)
    if is_grayscale_img(a_img):
        return convolve_1_channel(a_img, kernel)
    else:
        height, width = a_img.shape[:2]
        as_img = np.split(a_img, [1, 2], axis=2)
        for i, a_img_channel in enumerate(as_img):
            as_img[i] = a_img_channel.flatten()
            as_img[i] = a_img_channel.reshape(height, width)
            as_img[i] = convolve_1_channel(as_img[i], kernel)
            # # print(as_img[i][20:25, 20:25])
            # if as_img[i].dtype == np.float64:
            #     print(as_img[i].dtype, "asda")
            #     print(as_img[i][20:25, 20:25])
        # if np.dstack(as_img).dtype == np.float64:
            # print("dsajkd", np.stack(as_img, axis=2).reshape(height, width, 3)[0][20:25, 20:25])
        return np.stack(as_img, axis=2).reshape(height, width, 3)

def convolve_1_channel(a_img, kernel):
    # if a_img.dtype == np.float64:
    #     print(a_img.dtype)
    #     print(kernel)
    #     print(ndimage.convolve(a_img, kernel)[20:25, 20:25])
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

def get_average_matrix(shape, list_img):
    """ Get the average image given a list of images of same shape """
    l = len(list_img)
    # mode = img_skeleton.mode
    # width, height = img_skeleton.size
    # shape = ((height, width, 3), (height, width))[mode == "L"]
    a_img_avg = np.zeros(shape, np.float)
    for img in list_img:
        a_img = np.array(img, dtype=np.float)
        a_img_avg += a_img/l
    a_img_avg = np.array(np.round(a_img_avg), dtype=np.uint8)
    return a_img_avg

def extract_laws_texture_map(uri_img, radius_illumination=15):
    """ Get the feature vector map, using laws texture energy measures """
    tebl = TextureEnergyByLaws(uri_img, radius_illumination)
    return tebl.map_features

def extract_laws_texture_mean(uri_img, radius_illumination=15):
    """ Get the average for each feature vector dimension across an image """
    map_features = extract_laws_texture_map(uri_img, radius_illumination)
    if is_grayscale_img(Image.open(uri_img)):
        return extract_laws_texture_mean_1_channel(map_features)
    else:
        map_features_channels = np.split(map_features, [1, 2], axis=3)
        means_channel = []
        for map_features_channel in map_features_channels:
            means_channel.append(extract_laws_texture_mean_1_channel(map_features_channel))
        return np.mean(means_channel, axis=0)

def extract_laws_texture_mean_1_channel(map_features):
    """ Get the average for each feature vector dimension across one channel of an image """
    map_features = np.absolute(map_features)
    height, width, dimension = map_features.shape[:3]
    map_features = map_features.flatten()
    map_features = map_features.reshape(height*width, dimension)
    return np.mean(map_features, axis=1)

PATH_FOLDER = "D:/UMD/Career/Research Assistant/Segmentation by Logic/Code/img_sample/"

def main():
    """ test """
    name_img = "Abrams_Post_114_1_1_0_1.jpg"
    map_features = extract_laws_texture_map(PATH_FOLDER+name_img)
    print("map", map_features)
    print(map_features.shape)

if __name__ == "__main__":
    main()
