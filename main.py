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

NAMES_IMG = {"OCT": "Abrams_Post_114_1_1_0_1.jpg", "Me": "Stefan with Art.jpg", \
    "Benchmark": "Tiger.png"}

def main():
    """ Try result, quantifying texture with law's texture energy measure, segmenting image with \
        k-means clusterting """
    name_img = NAMES_IMG["OCT"]
    k = 4
    max_num_iteration = 30
    map_features = laws_texture_energy.extract_laws_texture_map(PATH_FOLDER + name_img)
    # print(map_features.shape)
    # print(map_features)
    kmeans_clustering.cluster_output_image_label(\
        name_img[:-4]+SUFFIX_TEXTURE_LAW, map_features, k, max_num_iteration)

if __name__ == "__main__":
    main()
