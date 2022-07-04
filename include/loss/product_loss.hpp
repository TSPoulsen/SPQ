#pragma once
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <vector>

#include "math_utils.hpp"
#include "loss/loss_base.hpp"

namespace PQ 
{
#ifdef __AVX2__

    class ProductLoss : public LossAVX
    {
    public:
        // Non-centered covariance matrix of data
        data_t cov;

        // Initializes the covariance matrix
        void init(data_t &data)
        {
            LossAVX::init(data);
            Math::setCov(*data_, cov);
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

#else

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
#endif
} // namespace PQ