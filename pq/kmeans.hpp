#pragma once

#include "loss.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <random>
#include <unordered_set>
#include "math.h"
#include <vector>
#include <assert.h>
#include <cfloat>
#include <iostream>
#include <inttypes.h>

#if defined(__AVX2__) || defined(__AVX__)
    #include <immintrin.h>
#endif

namespace PQ{

    // The implementation of this class has taken parts from: https://github.com/yahoojapan/NGT/blob/master/lib/NGT/Clustering.h
    /// Class for performing k-means clustering on a given dataset
    template <typename TLoss = EuclideanLoss>
    class KMeans
    {
    public:
        struct Cluster {
            std::vector<float> centroid;
            std::vector<unsigned int> members;
            Cluster(){}
            Cluster(const Cluster &c1){ // Is this still needed
                centroid = c1.centroid;
                members = c1.members;
            }
        };
        std::vector<Cluster> clusters;
    private:
        unsigned int padding = 0; // Padding at the end of each vector

        // Clustering configuration
        const unsigned int K;
        const double TOL; // Minium relative change between two iteration, otherwise stop training
        const unsigned int MAX_ITER; // Maximum number of iterations
        loss MODE;
        double inertia = DBL_MAX;

    public:
        /// Constructs an KMeans instance with chosen parameters
        ///
        /// @param K_clusters Number of clusters to fit the data to (default: 256)
        /// @param max_iter The number of llyod iteration to at most perform before terminating (default: 100)
        /// @param tol The minimum relative change between two iterations. If this change is not achieved then training terminates (default: 0.001)
        KMeans(unsigned int K_clusters = 256,
               unsigned int max_iter = 100, double tol = 0.001)
            : K(K_clusters),
            TOL(tol),
            MAX_ITER(max_iter)
        {
            std::cout << "Creating Kmeans with K=" << K << std::endl; // debug
            assert(K <= 256);
            assert(K > 0);
        }

        // Finds the best set of clusters that minimizes the inertia
        void fit(dataT data)
        {
            assert(K <= data.size());

            TLoss::padData(data);
        #if __AVX2__
            padData(data);
        #endif

            inertia = DBL_MAX; // Such that previous calls to fit doesn't affect current 
            unsigned int iteration = 0;
            double inertia_delta = 1.0;
            clusters = initCentroids(data);

            //std::cout << "Kmeans total iterations " << max_iter << " : ";
            while (inertia_delta >= TOL && iteration < MAX_ITER)
            {
                if (iteration%10 == 0) std::cout << iteration << "-" << std::flush; // debug
                // Step 1 in llyod: assign points to clusters
                double current_inertia = assignToClusters(data);

                // Step 2 in llyod: Set clusters to be center of points in cluster
                assignCentroids(data);

                // Calculate inertia difference
                inertia_delta = (inertia - current_inertia)/inertia;
                inertia = current_inertia;
                iteration++;
            }
            std::cout << std::endl; // debug
            return inertia;
        }


    private:

        // Assign all points to their cluster with the nearest centroid according to distance type
        // Currently does not handle if any clusters are empty
        double assignToClusters(const dataT &data)
        {
            // Clear member variable for each cluster
            for (auto cit = clusters.begin(); cit != clusters.end(); cit++) {
                (*cit).members.clear();
            }

            double current_inertia = 0;

            for (unsigned int i = 0; i < data.size(); i++) {
                double min_dist = DBL_MAX;
                unsigned int min_label = data.size() + 1;
                for (unsigned int c_i = 0; c_i < K; c_i++) {
                    double d = TLoss::distance(data[i], clusters[c_i].centroid);
                    if (d < min_dist) {
                        min_label = c_i;
                        min_dist = d;
                    }
                }
                clusters[min_label].members.push_back(i);
                current_inertia += min_dist;
            }
            return current_inertia;
        }


        // Assigns centroids according to the loss function
        void assignCentroids(const dataT& data)
        {
            void (*assignFunc)(const dataT&, Cluster&);
            // Determine the function to update centroids with
            switch (MODE) {
                case euclidean {
                    #if __AVX2__
                        assignFunc = &setCentroidMean_avx2;
                    #else
                        assignFunc = &setCentroidMean;
                    #endif
                    break;
                }
                case inner_product {
                    std::cout << "Not supported yet" << std::endl;
                    assert(false);
                    break;
                }
            }
            // Assigns all centriods
            for (auto cit = clusters.begin(); cit != clusters.end(); cit++) {
                assignFunc(data, cit);
            }
        }

        // Sets a cluster centroid to the mean of all points inside the cluster
        // This is independent of the distance type
        void setCentroidMean(const dataT &data, Cluster &c)
        {
            fill(c.centroid.begin(), c.centroid.end(), 0.0f);
            for (unsigned int idx : c.members) {
                for (unsigned int d = 0; d < c.centroid.size(); d++) {
                    c.centroid[d] += data[idx][d];
                }
            }
            assert(c.members.size() != 0);
            for (unsigned int d = 0; d < c.centroid.size(); d++) {
                c.centroid[d] /= c.members.size();
            }
        }

        // Kmeans++ initialization of centroids
        std::vector<Cluster> initCentroids(const dataT &data)
        {
            std::vector<Cluster> clusters(K);
            // Pick first centroids uniformly
            auto &rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, data.size()-1);
            clusters[0].centroid = data[random_idx(rand_gen)];

            // Choose the rest proportional to their distance to already chosen centroids
            std::vector<double> distances(data.size(), DBL_MAX);
            for (unsigned int c_i = 1; c_i < K; c_i++) {
                // Calculate distances to last chosen centroid
                for (unsigned int i = 0; i < data.size(); i++) {
                    double new_dist = sumOfSquares(data[i], clusters[c_i-1].centroid);
                    distances[i] = std::min(distances[i], new_dist);
                }

                std::discrete_distribution<int> rng(distances.begin(), distances.end());
                clusters[c_i].centroid = data[rng(rand_gen)];
            }
            return clusters;
        }

        // Calculates the distance between v1 and v2 according to the distance mode specified constructor
        double distance(const std::vector<float> &v1,const std::vector<float> &v2){
            switch (MODE) {
                case euclidean {
                #if __AVX2__
                    return sumOfSquares_avx(v1, v2);
                #else
                    return sumOfSquares(v1, v2);
                #endif
                }
                case inner_product {
                    std::cout << "Not supported" << std::endl;
                    assert(false);
                    break;
                }
            }
            return MAX_DBL;
        }

        // Calculates the sum of squares of the difference between two vectors
        // (v1-v2) @ (v1-v2)
        // This is analagous to euclidean distance
        double sumOfSquares(const std::vector<float> &v1, const std::vector<float> &v2)
        {
            double csum = 0.0;
            float *a = &v1[0];
            float *b = &v2[0];
            float *a_end = &v1[v1.size()];
            while (a != a_end) {
                double d = (double)*a++ - (double)*b++;
                csum += d * d;
            }
            return csum;
        }


    #ifdef __AVX2__
        // Does the same as sumOfSquares but uses avx2 instructions
        double sumOfSquares_avx(const std::vector<float> &v1, const std::vector<float> &v2)
        {
            __m256 sum = _mm256_setzero_ps();
            float *a = &v1[0];
            float *b = &v2[0];
            float *a_end = &v1[v1.size()];
            while (a != a_end) {
                __m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(v, v));
                a += 8;
                b += 8;
            }

            __attribute__((aligned(32))) float f[8];
            _mm256_store_ps(f, sum);
            double s = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
            return s;
        }

        // Assign a centroid `c` to the euclidean mean of all the points assigned to it using avx2
        void setCentroidMean_avx2(const dataT &data, Cluster &c)
        {
            unsigned int n256 = data[0].size()/8;
            __m256 sum[n256];
            for (unsigned int n = 0; n < n256; n++) {
                sum[n] = _mm256_setzero_ps();
            }
            for (unsigned int idx : c.members) {
                float *a = &data[idx][0];
                for (unsigned int n = 0; n < n256; n++, a+=8) {
                    sum[n] = _mm256_add_ps(sum[n], _mm256_loadu_ps(a));
                }
            }
            assert(c.members.size() != 0);
            float div = 1.0/c.members.size();
            alignas(32) float div_a[8] = {div, div, div, div, div, div, div, div};
            __m256 div_v = _mm256_load_ps(div_a);
            float *cen_s = &c.centroid[0];
            for (unsigned int n = 0; n < n256; n++, cen_s += 8) {
                _mm256_storeu_ps(cen_s,
                    _mm256_mul_ps(sum[n], div_v));
            }
        }

        // Adds padding such that each vector is a multiple of 8
        void padData(dataT &data) {
            padding = (8 - (data[0].size() % 8))%8;
            for (std::vector<float> &vec : data) {
                for (unsigned int p = 0; p < padding; p++) {
                    vec.push_back(0);
                }
            }
        }
    #endif
    }

} // PQ