""" Quantify texture by feature vector, from Laws' texture energy measures
    Stefan, Yuzhao Heng """

import numpy as np

import image_factory

TAG = "laws"

class TextureEnergyByLaws():
    """ Implements Laws; texture energy masks for quantifying texture for a single image. Works on \
        grayscale images.
        Returns a matrix of dimension of image, each element as vector """
    def __init__(self, matrix_image, radius_illumination=15):
        self.mtrx_img = image_factory.remove_illumination(matrix_image, radius_illumination)
        self.wvfrms = image_factory.get_laws_waveforms()
        self.mtrxs_ftr = [] # List of matrices by features
        self.mtrxs_enrg = [] # List of matrices of energy measures
        self.filter_waveforms()
        self.filter_sum()
        self.get_matrix_energies()
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
        return image_factory.convolve(mtrx_img, waveform)
    def filter_sum(self):
        """ Get the texture energy map by getting the sum of pixel values in proximity """
        for mtrx_ftr in self.mtrxs_ftr:
            self.mtrxs_enrg.append(self.filter_by_sum(mtrx_ftr, 15))
    def filter_by_sum(self, matrix_feature, dimension_kernel):
        """ Filter each texture feature by getting regional sum """
        if self.is_single_matrix_feature(matrix_feature):
            return image_factory.sum_region(matrix_feature, dimension_kernel)
        else:
            mtrx1_img = image_factory.sum_region(matrix_feature[0], dimension_kernel)
            mtrx2_img = image_factory.sum_region(matrix_feature[1], dimension_kernel)
            mtrx1_img = mtrx1_img.astype(np.float64)
            mtrx2_img = mtrx2_img.astype(np.float64)
            return image_factory.get_average_matrix(mtrx1_img.shape, [mtrx1_img, mtrx2_img])
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

def get_laws_energy_matrix(matrix_image, radius_illumination=15):
    """ Get the feature vector map, using laws texture energy measures """
    return TextureEnergyByLaws(matrix_image, radius_illumination).mtrx_enrgs

def get_laws_energy_mean(matrix_image, radius_illumination=15):
    """ Get the average for each feature vector dimension across an image """
    mtrx_enrgs = get_laws_energy_matrix(matrix_image, radius_illumination)
    if image_factory.is_grayscale_matrix(matrix_image):
        return get_laws_enery_mean_1_channel(mtrx_enrgs)
    else:
        mtrx_enrgs_chnnls = np.split(mtrx_enrgs, [1, 2], axis=3)
        means_chnnl = []
        for mtrx_enrgs_chnnl in mtrx_enrgs_chnnls:
            means_chnnl.append(get_laws_enery_mean_1_channel(mtrx_enrgs_chnnl))
        return np.mean(means_chnnl, axis=0)

def get_laws_enery_mean_1_channel(matrix_energies):
    """ Get the average for each feature vector dimension across one channel of an image """
    matrix_energies = np.absolute(matrix_energies)
    hght, wdth, dmnsn = matrix_energies.shape[:3]
    matrix_energies = matrix_energies.flatten()
    matrix_energies = matrix_energies.reshape(hght*wdth, dmnsn)
    return np.mean(matrix_energies, axis=1)

URI_SAMPLES = "img_sample/"

def main():
    """ module test """
    name_img = "Abrams_Post_114_1_1_0_1.jpg"
    mtrx_enrg = get_laws_energy_matrix(URI_SAMPLES+name_img)
    print("map", mtrx_enrg)
    print(mtrx_enrg.shape)

if __name__ == "__main__":
    main()
