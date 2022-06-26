#pragma once

#include "loss.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <random>
#include <unordered_set>
#include <math.h>
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
            Cluster(const std::vector<float> &cen): centroid(cen){}
            Cluster() {}
            Cluster(const Cluster &c1): centroid(c1.centroid), members(c1.members){} // Is this still needed
        };
        std::vector<Cluster> clusters;
    private:
        unsigned int padding = 0; // Padding at the end of each vector
        bool is_fitted = false;

        // Clustering configuration
        TLoss loss;
        const unsigned int K;
        const double TOL; // Minium relative change between two iteration, otherwise stop training
        const unsigned int MAX_ITER; // Maximum number of iterations
        double inertia = DBL_MAX;

    public:
        /// Constructs an KMeans instance with chosen parameters
        /// @param TLoss Type of loss function used for optimizing the clusters (default: euclidean)
        /// @param K_clusters Number of clusters to fit the data to (default: 256)
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
            data_t v = {{1.0,0.1}};
            //std::cout << "loss padd" << std::endl;
            //std::cout << "loss padd" << loss.padData(v) << std::endl;
        }

        // Finds the best set of clusters that minimizes the inertia
        void fit(data_t &data)
        {
            if (is_fitted) {
                return;
            }
            assert(K <= data.size());
            padding = loss.padData(data);
            //std::cout << padding << std::endl;
            unsigned int iteration = 0;
            double inertia_delta = 1.0;

            // Initalize clusters
            //std::cout << "Init centroids K = " << K << std::endl;
            //std::cout << "data size = " << data.size() << "," << data[0].size() << std::endl;
            data_t centroids = loss.initCentroids(data, K);
            //std::cout << "Init centroids DONE" << std::endl;
            for (std::vector<float> cen : centroids) {
                //std::cout << "pushing back" << std::endl;
                clusters.push_back(Cluster(cen));
            }

            //std::cout << "Kmeans total iterations " << MAX_ITER << " : ";
            while (inertia_delta >= TOL && iteration < MAX_ITER)
            {
                if (iteration%10 == 0) std::cout << iteration << "-" << std::flush; // debug
                // Step 1 in llyod: assign points to clusters
                double current_inertia = assignToClusters(data);

                // Step 2 in llyod: Set clusters to be center of points in cluster
                // Assigns all centriods
                for (Cluster &c : clusters) {
                    c.centroid = loss.getCentroid(data, c.members);
                }

                // Calculate inertia difference
                inertia_delta = (inertia - current_inertia)/inertia;
                inertia = current_inertia;
                iteration++;
            }
            std::cout << std::endl; // debug
            is_fitted = true;
        }

    private:
        // Assign all points to their cluster with the nearest centroid according to distance type
        // Currently does not handle if any clusters are empty
        double assignToClusters(const data_t &data)
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
                    double d = loss.distance(data[i], clusters[c_i].centroid);
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

    };

} // PQ