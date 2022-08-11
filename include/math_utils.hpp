#pragma once

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <cstddef>

#include <vector>
#include <random>

namespace PQ::util
{
    using data_t = std::vector<std::vector<float>>;


    inline const float* PTR_START(const data_t& data_ref, const size_t idx)
    {
        return &(data_ref[idx][0]);
    }

    // Fisher–Yates_shuffle
    // If sample_size == data_size, it returns a random permutation of indicies
    std::vector<size_t> randomSample(size_t sample_size, size_t data_size)
    {
        assert(sample_size <= data_size);
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        assert(sample_size <= data_size);
        std::vector<size_t> res(sample_size);
        for (unsigned int i = 0; i != data_size; ++i) {
            std::uniform_int_distribution<unsigned int> dis(0, i); // define the range
            std::size_t j = dis(gen);
            if (j < res.size()) {
                if (i < res.size()) {
                    res[i] = res[j];
                }
                res[j] = i;
            }
        }
        for (size_t idx : res) {
            assert(idx < data_size);
        }
        return res;
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

namespace internal
{
    // Scales all values in row by a factor k
    inline void scaleRow(std::vector<float> &row, const float k)
    {
        for (size_t c = 0; c < row.size(); c++) {
            row[c] = k* row[c];
        }
    }

    // row2 += k * row1
    inline void combineRows(const std::vector<float> &row1, std::vector<float> &row2, const float k)
    {
        for (std::size_t c = 0; c < row2.size(); c++)
        {
            row2[c] += row1[c] * k;
        }
    }
}

    // Input argument matrix is altered throughout the process
    data_t inverse(data_t &matrix)
    {
        size_t n = matrix.size();
        assert(n == matrix[0].size());

        // set up mean as identity
        data_t result(n, std::vector<float>(n, 0));
        for (size_t i = 0; i < n; i++) 
        {
            result[i][i] = 1.0;
        }
        
        // Eliminate L
        for (size_t col = 0; col < n; col++) 
        {
            double diag = matrix[col][col];
            for (std::size_t row = col + 1; row < n; row++) 
            {
                float target = matrix[row][col];
                float a = -target / diag;
                internal::combineRows(matrix[col], matrix[row], a);
                internal::combineRows(result[col], result[row], a);
            }
            internal::scaleRow(matrix[col], 1.0 / diag);
            internal::scaleRow(result[col], 1.0 / diag);
        }
        // Eliminate U
        for (size_t col = n - 1; col >= 1; col--) 
        {
            double diag = matrix[col][col];
            for (size_t row_plus_one = col; row_plus_one > 0; row_plus_one--)
            {
                size_t row = row_plus_one - 1;
                double target = matrix[row][col];
                double a = -target / diag;
                internal::combineRows(matrix[col], matrix[row], a);
                internal::combineRows(result[col], result[row], a);
            }
        }
        return result;
    }



}