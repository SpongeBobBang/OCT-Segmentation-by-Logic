"""
Segmentation by Logic
For segmenting lumen from OCT kidney images
"""

import unittest

import laws_texture_energy
import kmeans_clustering
import image_factory

URI_SAMPLES = laws_texture_energy.URI_SAMPLES

TAG_LAWS = laws_texture_energy.TAG
TAG_KMEANS = kmeans_clustering.TAG

NAMES_IMG = {"OCT": "Abrams_Post_114_1_1_0_1.jpg", "Me": "Stefan with Art.jpg", \
    "Benchmark_t": "Tiger.png", "Benchmark_f": "Flag.png", "Benchmark_s": "Sunflower.png", }

class Test_VisualizeResults(unittest.TestCase):
    """ Try quantify texture and segment, given sample OCT image """
    def test_laws_kc(self):
        """ Quantify texture with laws' texture energy measure, segment with k-means clusterting """
        name_img = NAMES_IMG["OCT"]
        k = 6
        max_num_iteration = 30
        mtrx_img = image_factory.get_matrix_from_uri(name_img, URI_SAMPLES)
        mtrx_features = laws_texture_energy.get_laws_energy_matrix(mtrx_img)
        mtrx_cluster = kmeans_clustering.get_kmeans_cluster_matrix(mtrx_features, \
            k, max_num_iteration)
        image_factory.write_image_by_matrix(mtrx_cluster, name_img, tag=TAG_LAWS+"_"+TAG_KMEANS)

if __name__ == "__main__":
    unittest.main()
