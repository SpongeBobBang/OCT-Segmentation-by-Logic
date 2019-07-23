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

OCT = "Abrams_Post_114_1_1_0_1"
OCT_LK = "Abrams_Post_114_1_1_0_1_laws_kc"
OCT_LK5 = "Abrams_Post_114_1_1_0_1_laws_kc5"
ME = "Stefan with Art"
BENCH_T = "Tiger"
BENCH_F = "Flag"
BENCH_S = "Benchmark_s"
URI_IMG = {OCT: "Abrams_Post_114_1_1_0_1.jpg", OCT_LK: "Abrams_Post_114_1_1_0_1_laws_kc.png", \
    OCT_LK5: "Abrams_Post_114_1_1_0_1_laws_kc5.png", \
    ME: "Stefan with Art.jpg", BENCH_T: "Tiger.png", BENCH_F: "Flag.png", BENCH_S: "Sunflower.png"}

class Test_VisualizeResults(unittest.TestCase):
    """ Try quantify texture and segment, given sample OCT image """
    def test_laws_kc(self):
        """ Quantify texture with laws' texture energy measure, segment with k-means clusterting """
        name = OCT_LK5
        uri = URI_IMG[name]
        k = 5
        max_num_iteration = 30
        mtrx_img = image_factory.get_matrix_from_uri(uri, URI_SAMPLES)
        mtrx_features = laws_texture_energy.get_laws_energy_matrix(mtrx_img)
        mtrx_cluster = kmeans_clustering.get_kmeans_cluster_matrix(mtrx_features, \
            k, max_num_iteration)
        image_factory.show_matrix(mtrx_cluster)
        image_factory.write_image_by_matrix(mtrx_cluster, name, tag=TAG_LAWS+"_"+TAG_KMEANS)
        print("written")
    def test_avg(self):
        """ Try process result of texture anlysis by taking average, for further segmentation """
        name = OCT_LK
        uri = URI_IMG[name]
        mtrx = image_factory.get_matrix_from_uri(uri)
        mtrx_avg = image_factory.average_region(mtrx, 20)
        image_factory.show_matrix(mtrx_avg)
        image_factory.write_image_by_matrix(mtrx_avg, name, tag="avg")
    def test_visualize_threshold(self):
        """ Toggle threshold value to see which gives good segmentation of noise region """
        mtrx = image_factory.get_matrix_from_uri("Abrams_Post_114_1_1_0_1_laws_kc_avg.png")
        image_factory.visualize_threshold(mtrx)

if __name__ == "__main__":
    unittest.main()
