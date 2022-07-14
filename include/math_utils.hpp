#pragma once

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <cstddef>

#include <vector>

namespace PQ 
{
    using data_t = std::vector<std::vector<float>>;

    inline const float* PTR_START(const data_t& data_ref, const size_t idx)
    {
        return &(data_ref[idx][0]);
    }

namespace Math
{
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

#ifdef __AVX2__
    float innerProduct(const float *v1, const float *v2, const size_t size)
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
#else
    // calculates the inner product between v1 and v2
    float innerProduct(const float *v1, const float *v2, const size_t size)
    {
        float sum = 0;
        for (size_t i = 0; i < size; i++)
        {
            sum += *(v1++) * *(v2++);
        }
        return sum;
    }

#endif

}
}