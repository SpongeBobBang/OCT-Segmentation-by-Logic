"""
Segmentation by Logic
For segmenting lumen from OCT images
Stefan, Yuzhao Heng
"""

from PIL import Image
import numpy as np
import laws_texture_energy
import kmeans_clustering

FILE_EXTENSION_OCT = ".jpg"
URI_SAMPLES = laws_texture_energy.URI_SAMPLES

SUFFIX_TEXTURE_LAW = "_" + laws_texture_energy.TAG

NAMES_IMG = {"OCT": "Abrams_Post_114_1_1_0_1.jpg", "Me": "Stefan with Art.jpg", \
    "Benchmark_t": "Tiger.png", "Benchmark_f": "Flag.png", "Benchmark_s": "Sunflower.png", }

def main():
    """ Try result, quantifying texture with law's texture energy measure, segmenting image with \
        k-means clusterting """
    name_img = NAMES_IMG["Benchmark_t"]
    img = Image.open(URI_SAMPLES + name_img)
    mtrx_img = np.array(img)
    map_features = laws_texture_energy.extract_laws_energy_matrix(mtrx_img)
    k = 4
    max_num_iteration = 30
    kmeans_clustering.cluster_output_image_label(\
        name_img[:-4]+SUFFIX_TEXTURE_LAW, map_features, k, max_num_iteration)

if __name__ == "__main__":
    main()
