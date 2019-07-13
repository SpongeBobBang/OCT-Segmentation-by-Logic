""" For image processing, handles interface to disk and convolutional filters """

from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plot
import cv2

class _ImageConvolution():
    """ Handles convolution and common variants """
    def convolve(self, matrix_image, kernel):
        """ Convoluve an image matrix of potentially multiple dimension vectors by a kernel """
        if is_grayscale_matrix(matrix_image):
            return self.convolve_1_channel(matrix_image, kernel)
        else:
            hght, wdth = matrix_image.shape[:2]
            mtrxs_img = np.split(matrix_image, [1, 2], axis=2)
            for i, mtrx_img_channel in enumerate(mtrxs_img):
                mtrx_img_channel = mtrx_img_channel.flatten().reshape(hght, wdth)
                mtrxs_img[i] = self.convolve_1_channel(mtrx_img_channel, kernel)
            return np.stack(mtrxs_img, axis=2).reshape(hght, wdth, 3)
    def convolve_1_channel(self, matrix, kernel):
        """ Convolve a matrix with elements of single dimension """
        return ndimage.convolve(matrix, kernel)
    def remove_illumination(self, matrix_image, dimension_kernel):
        """ Remove the effect of intensity in an image by subtracting the average of pixels in \
            proximity """
        matrix_image = np.array(matrix_image, dtype=np.float64)
        if not is_grayscale_matrix(matrix_image):
            matrix_image = matrix_image[..., :3] # Remove 4th transparency layer
        shp_krnl = dimension_kernel, dimension_kernel
        num_elmnt = dimension_kernel * dimension_kernel
        krnl = np.ones(num_elmnt, dtype=np.float64).reshape(shp_krnl)
        mtrx_img_sum = self.convolve(matrix_image, krnl)
        mtrx_img_avg = mtrx_img_sum/num_elmnt
        matrix_image = np.subtract(matrix_image, mtrx_img_avg, dtype=np.float64)
        matrix_image = np.clip(matrix_image, 0, 2**8 - 1)
        return matrix_image.astype(np.uint8)
    def sum_region(self, matrix_image, dimension_kernel):
        """ Filter a matrix of image where pixel values are regional sum """
        if not is_grayscale_matrix(matrix_image):
            matrix_image = matrix_image[..., :3] # Remove possible 4th transparency layer
        shp_kernel = dimension_kernel, dimension_kernel
        num_elmnt = dimension_kernel * dimension_kernel
        krnl = np.ones(num_elmnt).reshape(shp_kernel)
        return self.convolve(matrix_image, krnl)
    def average_region(self, matrix_image, dimension_kernel):
        """ Filter a matrix of image where pixel values are regional sum """
        if not is_grayscale_matrix(matrix_image):
            matrix_image = matrix_image[..., :3] # Remove possible 4th transparency layer
        shp_kernel = dimension_kernel, dimension_kernel
        num_elmnt = dimension_kernel * dimension_kernel
        krnl = np.ones(num_elmnt).reshape(shp_kernel)
        krnl /= num_elmnt
        return self.convolve(matrix_image, krnl)
    def get_average_matrix(self, shape, matrices):
        """ Get the average matrix given a list of matrices images of same shape """
        num_elmnt = len(matrices)
        mtrx_avg = np.zeros(shape)
        for mtrx in matrices:
            mtrx = np.array(mtrx)
            mtrx_avg += mtrx/num_elmnt
        mtrx_avg = np.array(np.round(mtrx_avg))
        return mtrx_avg

def is_grayscale_matrix(matrix_image):
    """ Checks if a matrix of an image is in grayscale or expected RGB """
    return len(matrix_image.shape) == 2

def convolve(matrix_image, kernel):
    """ Convole matrix of an image """
    return _ImageConvolution().convolve(matrix_image, kernel)

def remove_illumination(matrix_image, dimension_kernel):
    """ Remove illumination by region """
    return _ImageConvolution().convolve(matrix_image, dimension_kernel)

def sum_region(matrix_image, dimension_kernel):
    """ Convolve image by summing region """
    return _ImageConvolution().sum_region(matrix_image, dimension_kernel)

def average_region(matrix_image, dimension_kernel):
    """ Convolve image by taking average of region """
    return _ImageConvolution().average_region(matrix_image, dimension_kernel)

def get_average_matrix(shape, matrices):
    """ Take the average matrix by a list of matrices of same shape """
    return _ImageConvolution().get_average_matrix(shape, matrices)

def threshold(matrix, value_threshold):
    """ Binarize a matrix of image by thresholding """
    return cv2.threshold(matrix, value_threshold, 255, cv2.THRESH_BINARY)

class _GaussianFilter():
    """ Operations with Gausisan filter """
    def filter_matrix(self, matrix, sigma):
        """ Filter the matrix, given sigma of gaussian filter """
        return ndimage.gaussian_filter(matrix, sigma=sigma)

def filter_by_gaussian(matrix, sigma):
    """ Convolve a matrix with Gaussian filter """
    return _GaussianFilter().filter_matrix(matrix, sigma)

class _LawsWaveforms():
    """ Get a list of the convolution masks of waveforms to compute texture energy, order of 5 """
    def __init__(self):
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

def get_mask_2d(vector1_1d, vector2_1d):
    """ Get the 2D matrix product from 2 1D vectors """
    return np.dot(make_matrix_vertical(vector1_1d), make_matrix_horizontal(vector2_1d))

def make_matrix_vertical(l):
    """ Get the matrix with dimension m*1 needed for m*m masks, from an 1D list """
    return np.array(l).reshape(-1, 1)

def make_matrix_horizontal(l):
    """ Get the matrix with dimension 1*n needed for n*n masks, from an 1D list """
    return np.array(l).reshape(1, -1)

def get_laws_waveforms():
    """ Get the convolutional wavrforms for laws texture analysis, order of 5 """
    return _LawsWaveforms().wvfrms

class _ImageInterface():
    """ Handles interface between fiiles and runtime matrices of images """
    def show_matrix(self, matrix):
        """ Show the image equivalent of a matrix """
        plot.figure(figsize=(21.6, 10.8))
        if is_grayscale_matrix(matrix):
            plot.imshow(matrix, cmap='gray')
        else:
            plot.imshow(matrix)
        plot.show()
    def get_matrix_from_uri(self, name_image, path):
        """ Get ndarray from uri of an image """
        img = Image.open(path+name_image)
        return np.array(img)
    def write_image_by_matrix(self, matrix, name_image, path, tag):
        """ Write an png image file """
        img_lbl = Image.fromarray(matrix)
        uri = path+name_image
        if self.__has_tag(tag):
            uri += "_"+tag
        img_lbl.save(uri+".png")
    def __has_tag(self, tag):
        """ Check if the string is a valid tag """
        return tag != ''

def show_matrix(matrix):
    """ Show image from a matrix """
    _ImageInterface().show_matrix(matrix)

def get_matrix_from_uri(name_image, path=""):
    """ Get the matrix from a uri of an image """
    return _ImageInterface().get_matrix_from_uri(name_image, path)

def write_image_by_matrix(matrix, name_image, path='', tag=''):
    """ Write a image file of labels, given a matrix of label values """
    # Take out image file extension
    return _ImageInterface().write_image_by_matrix(matrix, name_image[:-4], path, tag)

def main():
    """ unit test """
    name_img = "Stefan with Art.jpg"
    mtrx = get_matrix_from_uri(name_img, "img_sample/")
    mtrx = filter_by_gaussian(mtrx, 5)
    write_image_by_matrix(mtrx, name_img)

if __name__ == "__main__":
    main()
