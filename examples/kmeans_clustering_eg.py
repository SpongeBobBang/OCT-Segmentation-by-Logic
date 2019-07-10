""" OpenCV python tutorial on k-means clustering """

from PIL import Image
import numpy as np
import cv2

IMG = cv2.imread('D:/UMD/Career/Research Assistant/Segmentation by Logic/Code/img_sample/Stefan with Art.jpg')
Z = IMG.reshape((-1, 3))
Z = np.float32(Z)

CRITERIA = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01
K = 4
RET, LABEL, CENTERS = cv2.kmeans(Z, K, None, CRITERIA, 10, cv2.KMEANS_RANDOM_CENTERS)

CENTERS = np.uint8(CENTERS)
RES = CENTERS[LABEL.flatten()]
print(LABEL)
a = np.array([[1, 2, 3, 0], [3, 2, 1, 0]])
print(CENTERS[a.flatten()])
print(CENTERS[a].flatten())
# print(LABEL.flatten())
RES2 = RES.reshape((IMG.shape))

cv2.imshow('res2', RES2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image.fromarray(RES2).save("kc.jpg")
