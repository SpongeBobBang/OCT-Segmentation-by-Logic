from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import imageio
pic = imageio.imread('D:/UMD/Career/Research Assistant/Segmentation by Logic/Code/img_sample/Headshot.jpg')
# pic = imageio.imread('img_sample/Headshot.jpg')
h, w = pic.shape[:2]
im_small_long = pic.reshape((h * w, 3))
im_small_wide = im_small_long.reshape((h, w, 3))
km = KMeans(n_clusters=2)
km.fit(im_small_long)
seg = np.asarray([(1 if i == 1 else 0)
                  for i in km.labels_]).reshape((h,w))
contours = measure.find_contours(seg, 0.5, fully_connected="high")
simplified_contours = [measure.approximate_polygon(c, tolerance=5)
                       for c in contours]
plt.figure(figsize=(5, 10))
for n, contour in enumerate(simplified_contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.ylim(h, 0)
plt.axes().set_aspect('equal')
plt.show()
