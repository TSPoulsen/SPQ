#pragma once
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <cassert>
#include <vector>
#include <iostream>

#include "math_utils.hpp"
#include "loss/loss_base.hpp"
#include "kmeans.hpp"

namespace PQ
{
    using namespace util;

class WeightedProductLoss : public ILoss
{
private:
    // Refferred to as `T` in paper for I(t>T) weight function
    // A weight of 0 weights parallel and orthogonal errors equally
    double weight_;
    // Same size as dataset (N_), which contains the scaling of parallel error for each data point
    std::vector<double> parallel_scalings_;


    // This is an approximation of the true scaling, see equation 3 (Theorem 3.4 in original paper)
    double scalingApproximation(const std::vector<float> &v1);

    // calculates the parallel and the orthogonal error and returns in a pair
    // The first element of the returned pair is the parallel error and the second is therefore the orthogonal
    std::pair<double, double> calculateErrors(const std::vector<float> &v1, const std::vector<float> &v2) const;

public:
    // Would very nice with an init which does not take any weight and instead bootstraps a weight
    void init(data_t &data, const  double w);
    // This is purely used for testing purposes
    // Any code inside this loss class should just access parallel_scalings_
    double getScaling(const size_t idx) const;
    double distance(const size_t idx, const std::vector<float> &v2) const;
    data_t initCentroids(const size_t K);
    std::vector<float> getCentroid(const std::vector<unsigned int> &members) const ;

};


double WeightedProductLoss::scalingApproximation(const std::vector<float> &v1)
{
    if (weight_ == 0.0) return 1.0;
    float v1_norm = innerProduct(&v1[0], &v1[0], v1.size());
    double frac = (weight_ * weight_) / v1_norm;
    return (frac / (1.0 - frac)) * (dim_ - padding_);
}


std::pair<double, double> WeightedProductLoss::calculateErrors(const std::vector<float> &v1, const std::vector<float> &v2) const
{
    const float *v1p = &v1[0];
    const float *v2p = &v2[0];

    float ip = innerProduct(v1p, v2p, dim_);
    float v1_norm = innerProduct(v1p, v1p, dim_);
    float v2_norm = innerProduct(v2p, v2p, dim_);
    float frac = (ip * ip) / (v1_norm);

    double para_err = v1_norm + frac - (2 * ip);
    double ortho_err = v2_norm - frac;
    return std::make_pair(para_err, ortho_err);
}


void WeightedProductLoss::init(data_t &data, const double w = 0.0)
{
    ILoss::init(data);
    assert(w >= 0.0);
    weight_ = w;
    for (const std::vector<float> &v1 : data_)
    {
        parallel_scalings_.push_back(scalingApproximation(v1));
    }
}


double WeightedProductLoss::getScaling(const size_t idx) const
{
    assert(idx <= parallel_scalings_.size());
    return parallel_scalings_[idx];
}


double WeightedProductLoss::distance(const size_t idx, const std::vector<float> &v2) const
{
    std::pair<double, double> errors = calculateErrors(data_[idx], v2);
    return parallel_scalings_[idx] * errors.first + errors.second;
}


data_t WeightedProductLoss::initCentroids(const size_t K)
{
    KMeans<EuclideanLoss> km(K);
    km.fit(data_);
    data_t centroids;
    for (const Cluster &c : km.clusters_)
    {
        centroids.push_back(c.centroid);
    }
    return centroids;
}


// TODO
std::vector<float> WeightedProductLoss::getCentroid(const std::vector<unsigned int> &members) const 
{
    return std::vector<float>();
}

} // namespace PQ