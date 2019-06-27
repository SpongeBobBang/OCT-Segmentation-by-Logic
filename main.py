"""
Segmentation by Logic
For segmenting lumen from OCT images
Stefan, Yuzhao Heng
"""

import numpy as np
import cv2

import law_texture_energy

IMG = cv2.imread('Stefan with Art.jpg')
