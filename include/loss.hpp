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

#include "math_utils.hpp"

#define PTR_START(data_ptr, idx) &(*data_ptr)[idx][0]

namespace PQ
{
    using data_t = std::vector<std::vector<float>>;

    class ILoss
    {
    protected:
        const data_t *data_;
        size_t N_;
        size_t dim_;
    public:
        virtual void init(data_t &data)
        {
            padData(data);
            this->data_ = &data;
            N_ = data.size();
            dim_ = data[0].size();
        }

        // Default is to not pad
        virtual size_t padData(data_t &) const
        {
            return 0u;
        }

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
#include <immintrin.h>

    class LossAVX : public ILoss
    {
    public:
        // Overload of ILoss which adds padding such that each vector is a multiple of 8
        size_t padData(data_t &data) const
        {
            size_t padding = (8 - (dim_ % 8)) % 8;
            for (std::vector<float> &vec : data)
            {
                for (size_t p = 0; p < padding; p++)
                {
                    vec.push_back(0);
                }
            }
            return padding;
        }

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

    class EuclideanLoss : public LossAVX
    {
    public:
        // Calculates the sum of squares between two vectors using avx2
        double distance(const size_t idx, const std::vector<float> &v2) const
        {
            __m256 sum = _mm256_setzero_ps();
            const float *a = PTR_START(data_, idx);
            const float *b = &v2[0];
            const float *a_end = a + v2.size();
            while (a != a_end)
            {
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

    class ProductLoss : public LossAVX
    {
    public:
        // Non-centered covariance matrix of data
        data_t cov;

        // Initializes the covariance matrix
        void init(data_t &data)
        {
            LossAVX::init(data);
            setCov(data_, cov);
        }

        double distance(const size_t idx, const std::vector<float> &v2) const
        {
            unsigned int n256 = dim_ / 8;
            const float *a = PTR_START(data_, idx);
            const float *b = &v2[0];
            __m256 diff[n256];
            // Calculates difference (v1-v2)
            for (unsigned int n = 0; n < n256; n++)
            {
                diff[n] = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                a += 8;
                b += 8;
            }
            __m256 sum;
            __attribute__((aligned(32))) float res_v[dim_];
            __attribute__((aligned(32))) float f[8];

            // Multiplies difference with matrix ((v1-v2) @ M)
            // Because the covariance matrix is symmetric when can go row by row instead of column by column
            // This should lead to fewer cache-misses
            for (unsigned int d = 0; d < dim_; d++)
            {
                sum = _mm256_setzero_ps();
                const float *c = &cov[d][0];
                for (unsigned int n = 0; n < n256; n++)
                {
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff[n], _mm256_loadu_ps(c)));
                    c += 8;
                }
                _mm256_store_ps(f, sum);
                res_v[d] = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
            }

            sum = _mm256_setzero_ps();
            float *d = res_v;
            // Multiplies previous result with (v1-v2)
            // essentially the inner product between ((v1-v2) @ M) and (v1-v2)
            for (unsigned int n = 0; n < n256; n++)
            {
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff[n], _mm256_loadu_ps(d)));
                d += 8;
            }
            _mm256_store_ps(f, sum);
            double s = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
            return s;
        }
    };

    /*
        class WeightedIPLoss: ILossAVX {
            double distance(const std::vector<float> &v1, const std::vector<float> &v2) = null;
            std::vector<float> getCentroid(const data_t &data, std::unsigned<int> &members) = null;

        };
    */

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

    class EuclideanLoss : public LossDefault
    {
    public:
        // Calculates the sum of squares of the difference between two vectors
        // This is analagous to euclidean distance
        double distance(const size_t idx, const std::vector<float> &v2) const
        {
            double csum = 0.0;
            const float *a = PTR_START(data_, idx);
            const float *b = &v2[0];
            const float *a_end = a + v2.size();
            while (a != a_end)
            {
                double d = (double)*a++ - (double)*b++;
                csum += d * d;
            }
            return csum;
        }
    };

    class ProductLoss : public LossDefault
    {
    public:
        // Non-centered covariance matrix of data
        data_t cov;

        // Initializes the covariance matrix
        void init(data_t &data)
        {
            LossDefault::init(data);
            Math::setCov(data, cov);
        }

        // Calculates the 'mahalanobis' distance between v1 and v2
        // This is not the standard mahalanobis distance but the distance
        // d = (v1-v2) @ M @ (v1-v2)
        // Where M is the non-centered covariance matrix and @ symbolizes matrix multiplication
        double distance(const size_t idx, const std::vector<float> &v2) const
        {
            const std::vector<float> v1 = (*data_)[idx];
            // Calculates difference (v1 - v2)
            std::vector<float> delta(dim_);
            for (unsigned int d = 0; d < dim_; d++)
            {
                delta[d] = v1[d] - v2[d];
            }

            // Multiplies difference with matrix ((v1-v2) @ M)
            std::vector<float> temp(dim_, 0);
            for (unsigned int d1 = 0; d1 < dim_; d1++)
            {
                for (unsigned int d2 = 0; d2 < dim_; d2++)
                {
                    temp[d1] += delta[d2] * cov[d2][d1];
                }
            }

            double distance = 0.0;
            // Multiplies previous result with (v1-v2)
            // essentially the inner product between ((v1-v2) @ M) and (v1-v2)
            for (unsigned int delta_i = 0; delta_i < v1.size(); delta_i++)
            {
                distance += ((double)delta[delta_i]) * ((double)temp[delta_i]);
            }
            return distance;
        }
    };

    class WeightedProductLoss : ILoss
    {
    private:
        // Referenced as T in paper for I(t>T) weight function
        // A weight of 0 weights parallel and orthogonal errors equally
        double weight;
        // Same size as dataset, which contains the scaling of parallel error for each data point
        std::vector<double> parallel_scalings;

        // It is the same for all points
        // When the parallel scaling is approximated, the orthogonal error has to be divided by (d-1)
        // as this is done in the approximation
        double orthogonal_scaling;

        // This is an approximation of the true scaling, see equation 3 (Theorem 3.4 in original paper)
        double scalingApproximation(const std::vector<float> &v1)
        {
            float v1_norm = Math::innerProduct(&v1[0], &v1[0], v1.size());
            double frac = (weight * weight) / v1_norm;
            return frac / (1.0 - frac);
        }

        // calculates the parallel and the orthogonal error and returns in a pair
        // The first element of the returned pair is the parallel error and the second is therefore the orthogonal
        std::pair<double, double> calculateErrors(const std::vector<float> &v1, const std::vector<float> &v2) const
        {
            const float *v1p = &v1[0];
            const float *v2p = &v2[0];

            float ip = Math::innerProduct(v1p, v2p, dim_);
            float v1_norm = Math::innerProduct(v1p, v1p, dim_);
            float v2_norm = Math::innerProduct(v2p, v2p, dim_);
            float frac = (ip * ip) / (v1_norm);

            double para_err = v1_norm + frac - (2 * ip);
            double ortho_err = v2_norm - frac;
            return std::make_pair(para_err, ortho_err);
        }

    public:
        void init(data_t &data, double w = 0.0)
        {
            ILoss::init(data);
            weight = w;
            orthogonal_scaling = 1.0 / (data_[0].size() - 1);
            for (const std::vector<float> &v1 : *data_)
            {
                parallel_scalings.push_back(scalingApproximation(v1));
            }
        }

        // This is purely used for testing purposes
        // Any code inside this loss class should just access parallel_scalings
        double getScaling(const size_t idx) const
        {
            assert(idx <= parallel_scalings.size());
            return parallel_scalings[idx];
        }

        double distance(const size_t idx, const std::vector<float> &v2) const
        {
            std::pair<double, double> errors = calculateErrors((*data_)[idx], v2);
            return parallel_scalings[idx] * errors.first + orthogonal_scaling * errors.second;
        }
    };
    /*

    class WeightedIPLoss: LossBase {
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) = null;
        std::vector<float> getCentroid(const data_t &data, std::unsigned<int> &members) = null;
    };

    */

#endif

} // namespace PQ