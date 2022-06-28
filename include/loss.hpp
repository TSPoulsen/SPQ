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

namespace PQ
{
    using data_t = std::vector<std::vector<float>>;

    // Calculates the non-centered covariance and sets it to cov
    void setCov(const data_t &data, data_t &cov)
    {
        size_t n = data.size();
        size_t dim = data[0].size();
        cov = std::vector<std::vector<float>>(dim, std::vector<float>(dim, 0));

        // Add all v[i] * v[j]
        for (unsigned int i = 0; i < n; i++)
        {
            for (unsigned int d1 = 0; d1 < dim; d1++)
            {
                for (unsigned int d2 = 0; d2 < dim; d2++)
                {
                    cov[d1][d2] += data[i][d1] * data[i][d2];
                }
            }
        }
        // Get average by dividing by n
        for (size_t i = 0; i < dim; i++)
        {
            for (size_t j = 0; j < dim; j++)
            {
                cov[i][j] /= n;
            }
        }
    }

    class ILoss
    {
    public:
        virtual void init(data_t &data)
        {
            padData(data);
        }

        // Default is to not pad
        virtual size_t padData(data_t &) const
        {
            return 0u;
        }

        // Default initialization method (KMeans++)
        data_t initCentroids(const data_t &data, const size_t K) const
        {
            data_t centroids(K);
            if (K == 0u)
                return centroids;
            // Pick first centroids uniformly
            std::default_random_engine gen;
            std::uniform_int_distribution<unsigned int> random_idx(0, data.size() - 1);
            centroids[0] = data[random_idx(gen)];

            // Choose the rest proportional to their distance to already chosen centroids
            std::vector<double> distances(data.size(), DBL_MAX);
            for (size_t c_i = 1; c_i < K; c_i++)
            {
                // Calculate distances to last chosen centroid
                for (unsigned int i = 0; i < data.size(); i++)
                {
                    double new_dist = distance(data[i], centroids[c_i - 1]); // Default use euclidean distance
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

        for (size_t i = 0; i < size; i += 8)
        {
            __m256 mv1 = _mm256_loadu_ps(v1);
            __m256 mv2 = _mm256_loadu_ps(v2);
            sum = _mm256_add_ps(sum,
                                _mm256_mul_ps(
                                    mv1,
                                    mv2));

            v1 += 8;
            v2 += 8;
        }
        alignas(32) float stored[8];
        _mm256_store_ps(stored, sum);
        int16_t ip = 0;
        for (unsigned i = 0; i < 8; i++)
        {
            ip += stored[i];
        }
        return ip;
    }

    class LossAVX : public ILoss
    {
    public:
        // Overload of ILoss which adds padding such that each vector is a multiple of 8
        size_t padData(data_t &data) const
        {
            size_t padding = (8 - (data[0].size() % 8)) % 8;
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
        std::vector<float> getCentroid(const data_t &data, const std::vector<unsigned int> &members) const
        {
            unsigned int n256 = data[0].size() / 8;
            std::vector<__m256> sum(n256, _mm256_setzero_ps());

            for (unsigned int idx : members)
            {
                const float *a = &data[idx][0];
                for (unsigned int n = 0; n < n256; n++, a += 8)
                {
                    sum[n] = _mm256_add_ps(sum[n], _mm256_loadu_ps(a));
                }
            }
            assert(members.size() != 0);
            float div = 1.0 / members.size();
            alignas(32) float div_a[8] = {div, div, div, div, div, div, div, div};
            __m256 div_v = _mm256_load_ps(div_a);
            std::vector<float> centroid(data[0].size());
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
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) const
        {
            __m256 sum = _mm256_setzero_ps();
            const float *a = &v1[0];
            const float *b = &v2[0];
            const float *a_end = &v1[v1.size()];
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
            setCov(data, cov);
        }

        double distance(const std::vector<float> &v1, const std::vector<float> &v2) const
        {
            unsigned int dim = v1.size();
            unsigned int n256 = dim / 8;
            const float *a = &v1[0];
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
            __attribute__((aligned(32))) float res_v[dim];
            __attribute__((aligned(32))) float f[8];

            // Multiplies difference with matrix ((v1-v2) @ M)
            // Because the covariance matrix is symmetric when can go row by row instead of column by column
            // This should lead to fewer cache-misses
            for (unsigned int d = 0; d < dim; d++)
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

    float inner_product(const float *v1, const float *v2, const size_t size)
    {
        float sum = 0;
        for (size_t i = 0; i < size; i++)
        {
            sum += *(v1++) * *(v2++);
        }
        return sum;
    }

    class LossDefault : public ILoss
    {
    public:
        // Sets a cluster centroid to the mean of all points inside the cluster
        // This is independent of the distance type
        std::vector<float> getCentroid(const data_t &data, const std::vector<unsigned int> &members) const
        {
            std::vector<float> centroid(data[0].size());
            std::fill(centroid.begin(), centroid.end(), 0.0f);
            for (unsigned int idx : members)
            {
                for (unsigned int d = 0; d < centroid.size(); d++)
                {
                    centroid[d] += data[idx][d];
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
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) const
        {
            double csum = 0.0;
            const float *a = &v1[0];
            const float *b = &v2[0];
            const float *a_end = &v1[v1.size()];
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
            setCov(data, cov);
        }

        // Calculates the 'mahalanobis' distance between v1 and v2
        // This is not the standard mahalanobis distance but the distance
        // d = (v1-v2) @ M @ (v1-v2)
        // Where M is the non-centered covariance matrix and @ symbolizes matrix multiplication
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) const
        {
            size_t dim = v1.size();
            // Calculates difference (v1 - v2)
            std::vector<float> delta(dim);
            for (unsigned int d = 0; d < dim; d++)
            {
                delta[d] = v1[d] - v2[d];
            }

            // Multiplies difference with matrix ((v1-v2) @ M)
            std::vector<float> temp(dim, 0);
            for (unsigned int d1 = 0; d1 < dim; d1++)
            {
                for (unsigned int d2 = 0; d2 < dim; d2++)
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

    /*

    class WeightedIPLoss: LossBase {
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) = null;
        std::vector<float> getCentroid(const data_t &data, std::unsigned<int> &members) = null;
    };

    */

#endif

} // namespace PQ