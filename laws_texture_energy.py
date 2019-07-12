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
    def __init__(self, matrix_image, radius_illumination=15):
        self.mtrx_img = remove_illumination(matrix_image, radius_illumination)
        self.init_waveforms()
        self.filter_waveforms()
        self.filter_sum()
        self.get_matrix_energies()
    def init_waveforms(self):
        """ Get the 9 convolution masks of waveforms to compute texture energy """
        vctr_level = [1, 4, 6, 4, 1]
        vctr_edge = [-1, -2, 0, 2, 1]
        vctr_spot = [-1, 0, 2, 0, -1]
        vctr_ripple = [1, -4, 6, -4, 1]
        self.vctrs = {"l": vctr_level, "e": vctr_edge, "s": vctr_spot, "r": vctr_ripple}
        names_wvfrm = [["le", "el"], ["lr", "rl"], ["es", "se"], ["ss"], ["rr"], ["ls", "sl"], \
            ["ee"], ["er", "re"], ["sr", "rs"]]
        self.dmnsn_ftr = len(names_wvfrm) # Dimension/Number of features
        self.wvfrms = []
        for name_wvfrm in names_wvfrm:
            self.wvfrms.append(self.get_waveforms(name_wvfrm))
        self.mtrxs_ftr = [] # List of matrices by features
        self.mtrxs_enrg = [] # List of matrices of energy measures
    def get_waveforms(self, name):
        """ Get the 2D mask matrix based on name, corresponding to the vertical and horizontal \
            vector """
        return self.get_waveform_single(name[0]) if self.is_single_waveform_name(name) \
            else self.get_waveform_double(name)
    def is_single_waveform_name(self, l):
        """ Check if the name of the mask makes a single mask or a mask pair, given the name list \
            """
        return len(l) == 1
    def get_waveform_single(self, name):
        """ Get the 2D convolutional waveform mask from the name string """
        return get_mask_2d(self.vctrs[name[0]], self.vctrs[name[1]])
    def get_waveform_double(self, name):
        """ Get the mask matrix if the name contrains 2 instance """
        return self.get_waveform_single(name[0]), self.get_waveform_single(name[1])
    def filter_waveforms(self):
        """ Get texture feature vector for each pixel """
        for wvfrm in self.wvfrms:
            self.mtrxs_ftr.append(self.filter_by_waveform(wvfrm))
    def filter_by_waveform(self, waveform):
        """ Filter the image with laws masks """
        if self.is_single_waveform(waveform):
            return self.filter_by_1_waveform(waveform)
        else:
            return self.filter_by_1_waveform(waveform[0]), self.filter_by_1_waveform(waveform[1])
    def is_single_waveform(self, e):
        """ Check if the name of the mask makes a single mask or a mask pair, given the mask \
            matrix """
        # The element of masks can be either ndarray or tuple of ndarray
        return isinstance(e, np.ndarray)
    def filter_by_1_waveform(self, waveform):
        """ Filter the image with 1 mask """
        mtrx_img = self.mtrx_img.astype(int) # Results can be negative and should be
        mtrx_img = convolve(mtrx_img, waveform)
        return mtrx_img
    def filter_sum(self):
        """ Get the texture energy map by getting the sum of pixel values in proximity """
        for mtrx_ftr in self.mtrxs_ftr:
            self.mtrxs_enrg.append(self.filter_by_sum(mtrx_ftr, 15))
    def filter_by_sum(self, matrix_feature, dimension_kernel):
        """ Filter each texture feature by getting regional sum """
        if self.is_single_matrix_feature(matrix_feature):
            return sum_region(matrix_feature, dimension_kernel)
        else:
            mtrx1_img = sum_region(matrix_feature[0], dimension_kernel)
            mtrx2_img = sum_region(matrix_feature[1], dimension_kernel)
            mtrx1_img = mtrx1_img.astype(np.float64)
            mtrx2_img = mtrx2_img.astype(np.float64)
            return get_average_matrix(mtrx1_img.shape, [mtrx1_img, mtrx2_img])
    def is_single_matrix_feature(self, matrix_feature):
        """ Check if the element of filtered images by mask contains one image """
        # Can be either an Image object, or a list of Image objects
        return isinstance(matrix_feature, np.ndarray)
    def get_matrix_energies(self):
        """ Combine the filtered images by masks, into feature vectors of one matrix """
        hght, wdth = self.mtrx_img.shape[:2]
        self.mtrx_enrgs = [[[] for x in range(wdth)] for y in range(hght)]
        for mtrx_enrg in self.mtrxs_enrg:
            for y in range(len(self.mtrx_enrgs)):
                for x in range(len(self.mtrx_enrgs[y])):
                    self.mtrx_enrgs[y][x].append(mtrx_enrg[y][x])
        self.mtrx_enrgs = np.array(self.mtrx_enrgs)

def make_matrix_vertical(l):
    """ Get the matrix with dimension m*1 needed for m*m masks, from an 1D list """
    return np.array(l).reshape(-1, 1)

def make_matrix_horizontal(l):
    """ Get the matrix with dimension 1*n needed for n*n masks, from an 1D list """
    return np.array(l).reshape(1, -1)

def get_mask_2d(vector1_1d, vector2_1d):
    """ Get the 2D matrix product from 2 1D vectors """
    return np.dot(make_matrix_vertical(vector1_1d), make_matrix_horizontal(vector2_1d))

def convolve(matrix_image, kernel):
    """ Convoluve an image matrix of potentially multiple dimension vectors by a kernel, returns a \
        np array """
    if is_grayscale_matrix(matrix_image):
        return convolve_1_channel(matrix_image, kernel)
    else:
        hght, wdth = matrix_image.shape[:2]
        mtrxs_img = np.split(matrix_image, [1, 2], axis=2)
        for i, mtrx_img_channel in enumerate(mtrxs_img):
            mtrx_img_channel = mtrx_img_channel.flatten().reshape(hght, wdth)
            mtrxs_img[i] = convolve_1_channel(mtrx_img_channel, kernel)
        return np.stack(mtrxs_img, axis=2).reshape(hght, wdth, 3)

def is_grayscale_matrix(matrix_image):
    """ Checks if an Image or numpy array of an image is in grayscale or expected RGB """
    if isinstance(matrix_image, Image.Image):
        return matrix_image.mode == "L"
    elif isinstance(matrix_image, np.ndarray):
        return len(matrix_image.shape) == 2
    else:
        return False

def convolve_1_channel(a_img, kernel):
    """ Convolve a matrix with elements of single dimension """
    return ndimage.convolve(a_img, kernel)

def remove_illumination(matrix_image, dimension_kernel):
    """ Remove the effect of intensity in an image by subtracting the average of pixels in \
        proximity """
    matrix_image = np.array(matrix_image, dtype=np.float64)
    if not is_grayscale_matrix(matrix_image):
        matrix_image = matrix_image[..., :3] # Remove 4th transparency layer
    shp_krnl = dimension_kernel, dimension_kernel
    num_elmnt = product(shp_krnl)
    krnl = np.ones(num_elmnt, dtype=np.float64).reshape(shp_krnl)
    mtrx_img_sum = convolve(matrix_image, krnl)
    mtrx_img_avg = mtrx_img_sum/num_elmnt
    matrix_image = np.subtract(matrix_image, mtrx_img_avg, dtype=np.float64)
    matrix_image = np.clip(matrix_image, 0, 2**8 - 1)
    return matrix_image.astype(np.uint8)

def product(l):
    """ Get the product of a list or tuple """
    prod = 1
    for e in l:
        prod *= e
    return prod

def sum_region(matrix_image, dimension_kernel):
    """ Filter a matrix of image where pixel values are regional sum """
    if not is_grayscale_matrix(matrix_image):
        matrix_image = matrix_image[..., :3] # Remove 4th transparency layer for RGB image
    shp_kernel = dimension_kernel, dimension_kernel
    num_elmnt = product(shp_kernel)
    krnl = np.ones(num_elmnt).reshape(shp_kernel)
    return convolve(matrix_image, krnl)

def get_average_matrix(shape, matrices):
    """ Get the average matrix given a list of matrices images of same shape """
    num_elmnt = len(matrices)
    mtrx_avg = np.zeros(shape)
    for mtrx in matrices:
        mtrx = np.array(mtrx)
        mtrx_avg += mtrx/num_elmnt
    mtrx_avg = np.array(np.round(mtrx_avg))
    return mtrx_avg

def extract_laws_energy_matrix(matrix_image, radius_illumination=15):
    """ Get the feature vector map, using laws texture energy measures """
    tebl = TextureEnergyByLaws(matrix_image, radius_illumination)
    return tebl.mtrx_enrgs

def extract_laws_energy_mean(matrix_image, radius_illumination=15):
    """ Get the average for each feature vector dimension across an image """
    mtrx_enrgs = extract_laws_energy_matrix(matrix_image, radius_illumination)
    if is_grayscale_matrix(matrix_image):
        return extract_laws_enery_mean_1_channel(mtrx_enrgs)
    else:
        mtrx_enrgs_chnnls = np.split(mtrx_enrgs, [1, 2], axis=3)
        means_chnnl = []
        for mtrx_enrgs_chnnl in mtrx_enrgs_chnnls:
            means_chnnl.append(extract_laws_enery_mean_1_channel(mtrx_enrgs_chnnl))
        return np.mean(means_chnnl, axis=0)

def extract_laws_enery_mean_1_channel(matrix_energies):
    """ Get the average for each feature vector dimension across one channel of an image """
    matrix_energies = np.absolute(matrix_energies)
    hght, wdth, dmnsn = matrix_energies.shape[:3]
    matrix_energies = matrix_energies.flatten()
    matrix_energies = matrix_energies.reshape(hght*wdth, dmnsn)
    return np.mean(matrix_energies, axis=1)

URI_SAMPLES = "D:/UMD/Career/Research Assistant/Segmentation by Logic/Code/img_sample/"

def main():
    """ test """
    name_img = "Abrams_Post_114_1_1_0_1.jpg"
    mtrx_enrg = extract_laws_energy_matrix(URI_SAMPLES+name_img)
    print("map", mtrx_enrg)
    print(mtrx_enrg.shape)

if __name__ == "__main__":
    main()
