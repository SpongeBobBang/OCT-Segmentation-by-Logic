"""
Segmentation by Logic
For segmenting lumen from OCT images
Stefan, Yuzhao Heng
"""

import laws_texture_energy
import kmeans_clustering

FILE_EXTENSION_OCT = ".jpg"
PATH_FOLDER = laws_texture_energy.PATH_FOLDER

SUFFIX_TEXTURE_LAW = "_" + laws_texture_energy.TAG

NAMES_IMG = {"OCT": "Abrams_Post_114_1_1_0_1", "test": "Stefan with Art", "benchmark": "Tiger", \
    "benchmark_small": "Tiger_Small"}

def main():
    """ Try result, quantifying texture with law's texture energy measure, segmenting image with \
        k-means clusterting """
    name_img = NAMES_IMG["OCT"]
    k = 6
    max_num_iteration = 20
    map_features = laws_texture_energy.extract_laws_texture_features(\
        PATH_FOLDER + name_img+FILE_EXTENSION_OCT)
    kmeans_clustering.cluster_output_image_label(\
        name_img+SUFFIX_TEXTURE_LAW, map_features, k, max_num_iteration)

if __name__ == "__main__":
    main()
