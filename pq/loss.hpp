#pragma once
#include <vector>
#include <cassert>


namespace PQ {

    using dataT = std::vector<std::vector<float>>;

    enum LossType {
        euclidean,
        inner_product,
        weighted_inner_product // Will become anisotropic loss functions
    }; 


    class ILoss {
    public:
        virtual void padData(dataT&);
        virtual std::vector<float> getCentroid(const dataT&, const std::vector<unsigned int>&);
        virtual double distance(const std::vector<float>&, const std::vector<float>&);
    };

#ifdef __AVX2__
    #include <immintrin.h>

    class ILossAVX: ILoss {
        // Adds padding such that each vector is a multiple of 8
        void padData(dataT &data) {
            padding = (8 - (data[0].size() % 8))%8;
            for (std::vector<float> &vec : data) {
                for (unsigned int p = 0; p < padding; p++) {
                    vec.push_back(0);
                }
            }
        }
    }
    class EuclideanLoss: ILossAVX {
        // Calculates the sum of squares between two vectors using avx2
        double distance(const std::vector<float> &v1, const std::vector<float> &v2)
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

        // Assign a centroid `c` to the euclidean mean of all the points assigned to it using avx2
        std::vector<float> getCentroid(const dataT &data, std::unsigned<int> &members)
        {
            std::vector<float> centroid(0,data[0].size());

            unsigned int n256 = data[0].size()/8;
            __m256 sum[n256];
            for (unsigned int n = 0; n < n256; n++) {
                sum[n] = _mm256_setzero_ps();
            }
            for (unsigned int idx : members) {
                float *a = &data[idx][0];
                for (unsigned int n = 0; n < n256; n++, a+=8) {
                    sum[n] = _mm256_add_ps(sum[n], _mm256_loadu_ps(a));
                }
            }
            assert(members.size() != 0);
            float div = 1.0/members.size();
            alignas(32) float div_a[8] = {div, div, div, div, div, div, div, div};
            __m256 div_v = _mm256_load_ps(div_a);
            float *cen_s = &centroid[0];
            for (unsigned int n = 0; n < n256; n++, cen_s += 8) {
                _mm256_storeu_ps(cen_s,
                    _mm256_mul_ps(sum[n], div_v));
            }
            return centroid;
        }
    };

/*
    class IPLoss: ILoss {
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) = null;
        std::vector<float> getCentroid(const dataT &data, std::unsigned<int> &members) = null;
    };

    class WeightedIPLoss: ILoss {
        double distance(const std::vector<float> &v1, const std::vector<float> &v2) = null;
        std::vector<float> getCentroid(const dataT &data, std::unsigned<int> &members) = null;

    };
*/

#else
    class EuclideanLoss: ILoss {

        // Calculates the sum of squares of the difference between two vectors
        // This is analagous to euclidean distance
        double distance(const std::vector<float> &v1, const std::vector<float> &v2)
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
        // Sets a cluster centroid to the mean of all points inside the cluster
        // This is independent of the distance type
        std::vector<float> getCentroid(const dataT &data, const std::vector<unsigned int> &members)
        {
            std::vector<float> centroid;
            fill(centroid.begin(), centroid.end(), 0.0f);
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

#endif



} // namespace PQ