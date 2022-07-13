#pragma once
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <cassert>
#include <vector>
#include <iostream>

#include "math_utils.hpp"
#include "loss/loss_base.hpp"

namespace PQ
{

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

        // This is an approximation of the true scaling, see equation 3 (Theorem 3.4 in original paper)
        double scalingApproximation(const std::vector<float> &v1)
        {
            if (weight == 0.0) return 1.0;
            float v1_norm = Math::innerProduct(&v1[0], &v1[0], v1.size());
            double frac = (weight * weight) / v1_norm;
            return (frac / (1.0 - frac)) * (dim_ - padding_);
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
            assert(w >= 0.0);
            weight = w;
            for (const std::vector<float> &v1 : *data_)
            {
                parallel_scalings.push_back(scalingApproximation(v1));
            }
        }

        // This is purely used for testing purposes
        // Any code inside this loss class should just access m_parallel_scalings
        double getScaling(const size_t idx) const
        {
            assert(idx <= parallel_scalings.size());
            return parallel_scalings[idx];
        }

        double distance(const size_t idx, const std::vector<float> &v2) const
        {
            std::pair<double, double> errors = calculateErrors((*data_)[idx], v2);
            return parallel_scalings[idx] * errors.first + errors.second;
        }

        std::vector<float> getCentroid(const std::vector<unsigned int> &members) const 
        {
            return std::vector<float>();
        }
    };
} // namespace PQ