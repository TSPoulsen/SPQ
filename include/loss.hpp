#pragma once

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
#include <vector>
#include <cassert>
#include <cfloat>
#include <algorithm>
#include <random>

namespace PQ {

    using data_t = std::vector<std::vector<float>>;

    struct ILoss {
        // Default is to not pad
        unsigned int padData(data_t&) const { return 0u; } ; 

        // Default initialization method (KMeans++)
        data_t initCentroids(const data_t &data, const unsigned int K) const {
            data_t centroids(K);
            // Pick first centroids uniformly
            std::default_random_engine gen;
            std::uniform_int_distribution<unsigned int> random_idx(0, data.size()-1);
            centroids[0] = data[random_idx(gen)];

            // Choose the rest proportional to their distance to already chosen centroids
            std::vector<double> distances(data.size(), DBL_MAX);
            for (unsigned int c_i = 1; c_i < K; c_i++) {
                // Calculate distances to last chosen centroid
                for (unsigned int i = 0; i < data.size(); i++) {
                    double new_dist = distance(data[i], centroids[c_i-1]); // Default use euclidean distance
                    distances[i] = std::min(distances[i], new_dist);
                }
                std::discrete_distribution<int> rng(distances.begin(), distances.end());
                centroids[c_i] = data[rng(gen)];
            }
            return centroids;
        }

        virtual double distance(const std::vector<float> &v1, const std::vector<float> &v2) const = 0; 
        virtual std::vector<float> getCentroid(const data_t &data, const std::vector<unsigned int> &members) const = 0;

    };

#ifdef __AVX2__
    #include <immintrin.h>
    float inner_product(const float *v1, const float *v2, const size_t size)
    {
        __m256 sum = _mm256_setzero_ps();

        for (size_t i = 0; i < size; i += 8) {
            __m256 mv1 = _mm256_loadu_ps(v1);
            __m256 mv2 = _mm256_loadu_ps(v2);
            sum = _mm256_add_ps(sum, 
                _mm256_mul_ps(
                    mv1,
                    mv2));

            v1+=8;
            v2+=8;
        }
        alignas(32) float stored[8];
        _mm256_store_ps(stored, sum);
        int16_t ip = 0;
        for (unsigned i=0; i<8; i++) { ip += stored[i]; }
        return ip;
    }

    struct LossAVX: ILoss {
        // Overload of ILoss which adds padding such that each vector is a multiple of 8
        size_t padData(data_t &data) const
        {
            size_t padding = (8 - (data[0].size() % 8))%8;
            for (std::vector<float> &vec : data) {
                for (size_t p = 0; p < padding; p++) {
                    vec.push_back(0);
                }
            }
            return padding;
        }

        // Assign a centroid `c` to the euclidean mean of all the points assigned to it using avx2
        std::vector<float> getCentroid(const data_t &data, const std::vector<unsigned int> &members) const
        {
            unsigned int n256 = data[0].size()/8;
            std::vector<__m256> sum(n256, _mm256_setzero_ps());

            for (unsigned int idx : members) {
                const float *a = &data[idx][0];
                for (unsigned int n = 0; n < n256; n++, a+=8) {
                    sum[n] = _mm256_add_ps(sum[n], _mm256_loadu_ps(a));
                }
            }
            assert(members.size() != 0);
            float div = 1.0/members.size();
            alignas(32) float div_a[8] = {div, div, div, div, div, div, div, div};
            __m256 div_v = _mm256_load_ps(div_a);
            std::vector<float> centroid(data[0].size());
            float *cen_s = &centroid[0];
            for (unsigned int n = 0; n < n256; n++, cen_s += 8) {
                _mm256_storeu_ps(cen_s,
                    _mm256_mul_ps(sum[n], div_v));
            }
            return centroid;
        }
    };

    struct EuclideanLoss: LossAVX {
        // Calculates the sum of squares between two vectors using avx2
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) const
        {
            __m256 sum = _mm256_setzero_ps();
            const float *a = &v1[0];
            const float *b = &v2[0];
            const float *a_end = &v1[v1.size()];
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
    };

/*
    class IPLoss: ILossAVX {
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) = null;
        std::vector<float> getCentroid(const data_t &data, std::unsigned<int> &members) = null;
    };

    class WeightedIPLoss: ILossAVX {
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) = null;
        std::vector<float> getCentroid(const data_t &data, std::unsigned<int> &members) = null;

    };
*/

#else

    float inner_product(const float *v1, const float *v2, const size_t size)
    {
        float sum = 0;
        for (size_t i = 0; i < size; i++) {
            sum += *(v1++) * *(v2++);
        }
        return sum;
    }

    struct LossDefault: ILoss {
        // Sets a cluster centroid to the mean of all points inside the cluster
        // This is independent of the distance type
        std::vector<float> getCentroid(const data_t &data, const std::vector<unsigned int> &members) const
        {
            std::vector<float> centroid(data[0].size());
            std::fill(centroid.begin(), centroid.end(), 0.0f);
            for (unsigned int idx : members) {
                for (unsigned int d = 0; d < centroid.size(); d++) {
                    centroid[d] += data[idx][d];
                }
            }
            assert(members.size() != 0);
            for (unsigned int d = 0; d < centroid.size(); d++) {
                centroid[d] /= members.size();
            }
            return centroid;
        }
    };

    struct EuclideanLoss: LossDefault {
        // Calculates the sum of squares of the difference between two vectors
        // This is analagous to euclidean distance
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) const
        {
            double csum = 0.0;
            const float *a = &v1[0];
            const float *b = &v2[0];
            const float *a_end = &v1[v1.size()];
            while (a != a_end) {
                double d = (double)*a++ - (double)*b++;
                csum += d * d;
            }
            return csum;
        }
    };
    /*
    
    class IPLoss: LossBase {
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) = null;
        std::vector<float> getCentroid(const data_t &data, std::unsigned<int> &members) = null;
    };

    class WeightedIPLoss: LossBase {
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) = null;
        std::vector<float> getCentroid(const data_t &data, std::unsigned<int> &members) = null;
    };

    */

#endif



} // namespace PQ