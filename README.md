# OCT-Segmentation-by-Logic
Segmentation by Logic

Stefan, Yuzhao Heng

For segmenting lumen from OCT images.

User's guide

1. Open python file test.py. 

2. Run corresponding tests that operate on 1 image. The Result will be in current folder. 

test_laws_kc: 
Quantify an image by texture within 5*5 region and segment using k-means clustering. 

test_avg:
Smooth an image by taking average. 

test_visualize_threshold: 
Find a optimal threshold value by toggleing the value, binarizing clustered image. 
