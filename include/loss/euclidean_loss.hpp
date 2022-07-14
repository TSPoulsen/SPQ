#pragma once
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <vector>

#include "math_utils.hpp"
#include "loss/loss_base.hpp"

namespace PQ
{
    class EuclideanLoss : public LossDefault
    {
    public:
        // Calculates the sum of squares of the difference between two vectors
        // This is analagous to euclidean distance
        double distance(const size_t idx, const std::vector<float> &v2) const;
    };


#ifdef __AVX2__

double EuclideanLoss::distance(const size_t idx, const std::vector<float> &v2) const
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

#else

double EuclideanLoss::distance(const size_t idx, const std::vector<float> &v2) const
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

#endif

} // namespace PQ