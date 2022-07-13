#pragma once
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <cassert>
#include <random>
#include <vector>
#include <cfloat>

#include "math_utils.hpp"
#include "loss/loss_base.hpp"


namespace PQ
{
    using data_t = std::vector<std::vector<float>>;

    class ILoss
    {
    protected:
        const data_t *data_;
        size_t N_;
        size_t dim_; // including padding
        size_t padding_;
    public:
        virtual void init(data_t &data)
        {
            //std::cout << "instead here" << std::endl;
            padData(data);
            this->data_ = &data;
            N_ = data.size();
            dim_ = data[0].size();
        }
#ifdef __AVX2__
        // Overload of ILoss which adds padding such that each vector is a multiple of 8
        size_t padData(data_t &data)
        {
            size_t dim = data[0].size();
            //std::cout << "padding data" << std::endl;
            padding_ = (8 - (dim % 8)) % 8;
            //std::cout << "pad size: " << padding << std::endl; 
            for (std::vector<float> &vec : data)
            {
                for (size_t p = 0; p < padding_; p++)
                {
                    vec.push_back(0);
                }
            }
            return padding_;
        }
#else

        // Default is to not pad
        size_t padData(data_t &)
        {
            padding_ = 0u;
            return padding_;
        }
#endif

        // Default initialization method (KMeans++)
        data_t initCentroids(const size_t K) const
        {
            data_t centroids(K);
            if (K == 0u)
                return centroids;
            // Pick first centroids uniformly
            std::default_random_engine gen;
            std::uniform_int_distribution<unsigned int> random_idx(0, N_ - 1);
            centroids[0] = (*data_)[random_idx(gen)];

            // Choose the rest proportional to their distance to already chosen centroids
            std::vector<double> distances(N_, DBL_MAX);
            for (size_t c_i = 1; c_i < K; c_i++)
            {
                // Calculate distances to last chosen centroid
                for (unsigned int i = 0; i < N_; i++)
                {
                    double new_dist = distance(i, centroids[c_i - 1]); // Default use euclidean distance
                    distances[i] = std::min(distances[i], new_dist);
                }
                std::discrete_distribution<int> rng(distances.begin(), distances.end());
                centroids[c_i] = (*data_)[rng(gen)];
            }
            return centroids;
        }

        virtual double distance(const size_t idx, const std::vector<float> &v2) const = 0;
        virtual std::vector<float> getCentroid(const std::vector<unsigned int> &members) const = 0;
    };

#ifdef __AVX2__
    class LossAVX : public ILoss
    {
    public:


        // Assign a centroid `c` to the euclidean mean of all the points assigned to it using avx2
        std::vector<float> getCentroid(const std::vector<unsigned int> &members) const
        {
            unsigned int n256 = dim_ / 8;
            std::vector<__m256> sum(n256, _mm256_setzero_ps());

            for (unsigned int idx : members)
            {
                const float *a = PTR_START(data_, idx);
                for (unsigned int n = 0; n < n256; n++, a += 8)
                {
                    sum[n] = _mm256_add_ps(sum[n], _mm256_loadu_ps(a));
                }
            }
            assert(members.size() != 0);
            float div = 1.0 / members.size();
            alignas(32) float div_a[8] = {div, div, div, div, div, div, div, div};
            __m256 div_v = _mm256_load_ps(div_a);
            std::vector<float> centroid(dim_);
            float *cen_s = &centroid[0];
            for (unsigned int n = 0; n < n256; n++, cen_s += 8)
            {
                _mm256_storeu_ps(cen_s,
                                 _mm256_mul_ps(sum[n], div_v));
            }
            return centroid;
        }
    };
#else
    class LossDefault : public ILoss
    {
    public:
        // Sets a cluster centroid to the mean of all points inside the cluster
        // This is independent of the distance type
        std::vector<float> getCentroid(const std::vector<unsigned int> &members) const
        {
            std::vector<float> centroid(dim_);
            std::fill(centroid.begin(), centroid.end(), 0.0f);
            for (unsigned int idx : members)
            {
                const float *a = PTR_START(data_, idx);
                for (unsigned int d = 0; d < centroid.size(); d++)
                {
                    centroid[d] += *a++;
                }
            }
            assert(members.size() != 0);
            for (unsigned int d = 0; d < centroid.size(); d++)
            {
                centroid[d] /= members.size();
            }
            return centroid;
        }
    };

#endif
} // namespace PQ 