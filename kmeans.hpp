#pragma once
#include "puffinn/dataset.hpp"
#include "puffinn/format/unit_vector.hpp"
#include "puffinn/format/real_vector.hpp"
#include "puffinn/math.hpp"

#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <array>
#include <random>
#include <unordered_set>
#include <cfloat>

#if defined(__AVX2__) || defined(__AVX__)
    #include <immintrin.h>
#endif

namespace puffinn
{
    // The implementation of this class is highly inspired by: https://github.com/yahoojapan/NGT/blob/master/lib/NGT/Clustering.h
    /// Class for performing k-means clustering on a given dataset
    class KMeans
    {
    public:
        struct Cluster {
            std::vector<float> centroid;
            std::vector<unsigned int> members;
            Cluster(){}
            Cluster(const Cluster &c1){
                centroid = c1.centroid;
                members = c1.members;
            }
        };
        enum distanceType {euclidean, mahalanobis, none};
        std::vector<float> covarianceMatrix; // Currently only public for testing purposes
        std::vector<Cluster> gb_clusters;
    private:

        using dataType = std::vector<std::vector<float>>; // Type of the data which the kmeans works with
        unsigned int padding = 0; // Padding at the end of each vector

        // Clustering configuration
        const unsigned int K;
        const float TOL;
        const uint16_t MAX_ITER;
        const unsigned int N_RUNS; // Perhaps not needed for final version
        distanceType MODE;

        // gb are the global best results
        // These will also be removed if N_RUNS is removed

    public:
        double gb_inertia = DBL_MAX;
        /// Constructs an KMeans instance with chosen parameters
        ///
        /// @param K_clusters Number of clusters to fit the data to (default: 256)
        /// @param mode The type of distance to minimize when assigning points to clusters (default: euclidean)
        /// @param runs The number of times it tries to find optimal clusters, the final set of clusters is the best of all runs (default: 3)
        /// @param max_iter The number of llyod iteration to at most perform before terminating (default: 100)
        /// @param tol The minimum inertia difference before the algorithm is terminated
        KMeans(unsigned int K_clusters = 256, distanceType mode = euclidean,
            unsigned int runs = 3, unsigned int max_iter = 100, float tol = 0.0f)
            : K(K_clusters),
            TOL(tol),
            MAX_ITER(max_iter),
            N_RUNS(runs),
            MODE(mode)
        {
            std::cout << "Creating Kmeans with K=" << K << std::endl;
            assert(K <= 256);
            assert(K > 0);
        }

        ~KMeans() {}

        // Finds the best set of clusters that minimizes the inertia
        // Runs N_RUNS times and then chooses the best of those runs
        void fit(dataType &data)
        {
            assert(K <= data.size());

        #if __AVX2__
            padData(data);
        #endif

            if (MODE == mahalanobis){
                createCovarianceMatrix(data);
            }

            gb_inertia = DBL_MAX; // Such that previous calls to fit doesn't affect current 
            unsigned int to_run = N_RUNS, max_iter = MAX_ITER;
            if (data.size() >= 100000) {
                // When dataset is large enough there is no need to run multiple times
                // Instead we increase iterations
                to_run = 1; 
                max_iter = std::max(max_iter, 100u);
            } 

            for(unsigned int run=0; run < to_run; run++) {
                std::vector<Cluster> clusters = init_centroids_kpp(data);
                double run_inertia = lloyd(data, clusters, max_iter);
                if (run_inertia < gb_inertia) {
                    gb_inertia = run_inertia;
                    gb_clusters =  clusters; // Copies the whole Class 
                }
            }
        }

        std::vector<unsigned int> getGBMembers(size_t c_i){
            return gb_clusters[c_i].members;
        } 

        dataType getAllCentroids(){
            dataType all_centroids(K);
            for (unsigned int c_i = 0; c_i < K; c_i++){
                all_centroids[c_i] = getCentroid(c_i);
            }
            return all_centroids;
        }

        std::vector<float> getCentroid(size_t c_i) {
            // Removes padding from centroid
            return std::vector<float>(&*gb_clusters[c_i].centroid.begin(), (&*gb_clusters[c_i].centroid.end())-padding);
        }


        // ************************************************************************************
        // Here comes functions that are only public because we need to access them for testing
        // ************************************************************************************

        // Performs a single kmeans clustering using the lloyd algorithm
        double lloyd(dataType &data, std::vector<Cluster> &clusters, unsigned int max_iter) 
        {
            double inertia_delta = DBL_MAX;
            double inertia = DBL_MAX;
            unsigned int iteration = 0;

            //std::cout << "Kmeans total iterations " << max_iter << " : ";
            while (inertia_delta > TOL && iteration < max_iter )
            {
                if (iteration%10 == 0) std::cout << iteration << "-" << std::flush;
                // Step 1 in llyod: assign points to clusters
                double current_inertia = assignToClusters(data, clusters);
                // Step 2 in llyod: Set clusters to be center of points in cluster
                for (auto cit = clusters.begin(); cit != clusters.end(); cit++) {
                #if __AVX2__
                    setCentroidMean_avx2(data, *cit);
                #else
                    setCentroidMean(data, *cit);
                #endif
                }

                // Calculate inertia difference
                inertia_delta = (inertia - current_inertia);
                inertia = current_inertia;
                iteration++;
            }
            std::cout << std::endl;
            return inertia;

        }

        // Creates the non-centered covariance matrix of data
        // Stores the matrix in the member covarianceMatrix
        // Build covariance matrix by covariance[i][j] = avg((v[i] * v[j])) over all v in dataset
        void createCovarianceMatrix(dataType &data){
            covarianceMatrix.resize(data[0].size()*data[0].size());

            // Add all v[i] * v[j]
            for(unsigned int i = 0;  i < data.size(); i++){
                for(unsigned int d1= 0; d1 < data[0].size(); d1++){
                    for(unsigned int d2 = 0; d2 < data[0].size(); d2++){  
                        covarianceMatrix[(data[0].size() * d1) + d2] += data[i][d1] * data[i][d2];
                    }
                }
            }
            // Get average by dividing by n 
            for(unsigned int cov = 0; cov < (data[0].size()*data[0].size()); cov++){
                covarianceMatrix[cov] /= (data.size());
            }
        }

        // Assign all points to their cluster with the nearest centroid according to distance type
        // Currently does not handle if any clusters are empty
        double assignToClusters(dataType &data, std::vector<Cluster> &clusters)
        {
            // Clear member variable for each cluster
            for (auto cit = clusters.begin(); cit != clusters.end(); cit++) {
                (*cit).members.clear();
            }

            double inertia = 0;

            for (unsigned int i = 0; i < data.size(); i++) {
                double min_dist = DBL_MAX;
                unsigned int min_label = data.size() + 1;
                for (unsigned int c_i = 0; c_i < K; c_i++) {
                    double d = distance(data[i], clusters[c_i].centroid);
                    if (d < min_dist) {
                        min_label = c_i;
                        min_dist = d;
                    }
                }
                clusters[min_label].members.push_back(i);
                inertia += min_dist;
            }
            return inertia;
        }

        // Sets a cluster centroid to the mean of all points inside the cluster
        // This is independent of the distance type
        void setCentroidMean(dataType &data, Cluster &c)
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

        // Samples K random points and uses those as starting centers for clusters
        std::vector<Cluster> init_centroids_random(dataType &data)
        {
            std::vector<Cluster> clusters(K);
            auto &rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, data.size()-1);

            unsigned int c_i = 0;
            std::unordered_set<unsigned int> used; // To ensure the same point isn't picked twice
            while(c_i < K) {
                unsigned int sample_idx = random_idx(rand_gen);
                if (used.find(sample_idx) == used.end()) {
                    used.insert(sample_idx);
                    clusters[c_i++].centroid = data[sample_idx];
                }
            }
            return clusters;

        }

        // Kmeans++ initialization of centroids
        std::vector<Cluster> init_centroids_kpp(dataType &data)
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
        double distance(std::vector<float> &v1, std::vector<float> &v2){
            if(MODE == euclidean){
            #if __AVX2__
                return sumOfSquares_avx(v1, v2);
            #else
                return sumOfSquares(v1, v2);
            #endif
            }
            
            if(MODE == mahalanobis){
            #if __AVX2__
                return mahaDistance_avx(v1,v2);
            #else
                return mahaDistance(v1,v2);
            #endif
            }
        }

        // Calculates the sum of squares of the difference between two vectors
        // (v1-v2) @ (v1-v2)
        // This is analagous to euclidean distance
        double sumOfSquares(std::vector<float> &v1, std::vector<float> &v2)
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


        // Calculates the 'mahalanobis' distance between v1 and v2
        // This is not the standard mahalanobis distance but the distance
        // d = (v1-v2) @ M @ (v1-v2)
        // Where M is the non-centered covariance matrix and @ symbolizes matrix multiplication
        double mahaDistance(std::vector<float> &v1, std::vector<float> &v2)
        {
                // Calculates difference (v1 - v2)
                std::vector<float> delta(v1.size());
                for(unsigned int d = 0; d < v1.size(); d++){
                    delta[d] = v1[d] - v2[d];
                }

                // Multiplies difference with matrix ((v1-v2) @ M)
                std::vector<float> temp(v1.size(), 0);
                for(unsigned int d = 0; d < v1.size(); d++){
                    for(unsigned int delta_i = 0; delta_i < v1.size(); delta_i++){
                        temp[d] += delta[delta_i] * covarianceMatrix[(v1.size() * delta_i) + d]; 
                    }    
                }

                double distance = 0.0;
                // Multiplies previous result with (v1-v2)
                // essentially the inner product between ((v1-v2) @ M) and (v1-v2)
                for(unsigned int delta_i = 0; delta_i < v1.size(); delta_i++){
                    distance += ((double) delta[delta_i]) * ((double) temp[delta_i]);
                }
                return distance;                
        }


    #ifdef __AVX2__
        // Does the same as sumOfSquares but uses avx2 instructions
        double sumOfSquares_avx(std::vector<float> &v1, std::vector<float> &v2)
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

        // The same as mahaDistance but uses avx2 instructions
        double mahaDistance_avx(std::vector<float> &v1, std::vector<float> &v2)
        {

            unsigned int dim = v1.size();
            unsigned int n256 = dim/8;
            float *a = &v1[0];
            float *b = &v2[0];
            __m256 diff[n256];
            // Calculates difference (v1-v2)
            for (unsigned int n= 0; n < n256; n++) {
                diff[n] = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                a += 8;
                b += 8;
            }
            float *c = &covarianceMatrix[0];
            __m256 sum;
            __attribute__((aligned(32))) float res_v[dim];
            __attribute__((aligned(32))) float f[8];

            // Multiplies difference with matrix ((v1-v2) @ M)
            // Because the covariance matrix is symmetric when can go row by row instead of column by column
            // This should lead to fewer cache-misses
            for (unsigned int d = 0; d < dim; d++) {
                sum = _mm256_setzero_ps();
                for (unsigned int n= 0; n < n256; n++) {
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff[n], _mm256_loadu_ps(c)));
                    c+=8;
                }
                _mm256_store_ps(f, sum);
                res_v[d] = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
            }

            sum = _mm256_setzero_ps();
            float *d = res_v;
            // Multiplies previous result with (v1-v2)
            // essentially the inner product between ((v1-v2) @ M) and (v1-v2)
            for (unsigned int n= 0; n < n256; n++) {
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff[n], _mm256_loadu_ps(d)));
                d += 8;
            }
            _mm256_store_ps(f, sum);
            double s = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
            return s;
        }

        void setCentroidMean_avx2(dataType &data, Cluster &c)
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
        void padData(dataType &data) {
            padding = (8 - (data[0].size() % 8))%8;
            for (std::vector<float> &vec : data) {
                for (unsigned int p = 0; p < padding; p++) {
                    vec.push_back(0);
                }
            }
        }
    #endif

        // ONLY FOR TESTING AND DEBUG

        std::vector<float> getCovarianceMatrix(){
            return covarianceMatrix;
        }

        double totalError(dataType &data, distanceType mode = none) {
            padData(data);
            if (mode == none) {
                mode = MODE;
            }
            //double (KMeans::* d_ptr)(std::vector<float>&, std::vector<float>&);
            //if (mode == euclidean) d_ptr = &KMeans::sumOfSquares;
            //if (mode == mahalanobis) d_ptr = &KMeans::mahaDistance;
            double total_err = 0;
            for (Cluster &c : gb_clusters) {
                for (unsigned int idx : c.members) {
                    if (mode == euclidean)        total_err += sumOfSquares(data[idx], c.centroid);
                    else if (mode == mahalanobis) total_err += mahaDistance(data[idx], c.centroid);
                    //total_err += (*d_ptr)(data[idx], c.centroid);
                }
                //std::cerr << std::endl;
            }
            //std::cerr << std::endl << std::endl;
            return total_err;
            

        }
    };
}