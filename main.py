"""
Segmentation by Logic
For segmenting lumen from OCT images
Stefan, Yuzhao Heng
"""

import law_texture_energy
import kmeans_clustering

# Try result, quantifying texture with law's texture energy measure, segmenting image with k-means
# clusterting

if __name__ == "__main__":
    PATH_FOLDER = "D:/UMD/Career/Research Assistant/Segmentation by Logic/Code/Image/ori/"
    NAME_IMG = "Abrams_Post_114_1_1_0_1"
    FILE_EXTENSION_OCT = ".jpg"
    SUFFIX_TEXTURE_LAW = "_law's"
    K = 6
    MAX_NUM_ITERATION = 20
    MAP_FEATURES = law_texture_energy.extract_laws_texture_features(\
        PATH_FOLDER + NAME_IMG+FILE_EXTENSION_OCT)
    kmeans_clustering.cluster_output_image_label(\
        NAME_IMG+SUFFIX_TEXTURE_LAW, MAP_FEATURES, K, MAX_NUM_ITERATION)
    