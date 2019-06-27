"""
K-means clustering algorithm
Stefan, Yuzhao Heng
"""

import numpy as np
import cv2

IMG = cv2.imread('Stefan with Art.jpg') # Matrix with elements of 3 dimension for RGB image / 3D array
Z = IMG.reshape((-1, 3)) # 2D Array, row number: width * height, column number 3
print(Z)

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
RET, LABEL, CENTER = cv2.kmeans(Z, K, None, CRITERIA, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
CENTER = np.uint8(CENTER)
RES = CENTER[LABEL.flatten()]
RES2 = RES.reshape((IMG.shape))

cv2.imshow('res2', RES2)
cv2.waitKey(0)
cv2.destroyAllWindows()
