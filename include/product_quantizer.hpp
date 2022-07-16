#pragma once

#include "kmeans.hpp"
#include "math_utils.hpp"

#include "math.h"
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cfloat>
#include <iostream>
#include <inttypes.h>


namespace PQ{

struct Estimator
{

private:
    const size_t m_;
    const size_t k_;

public:
    // Vector which distance estimation is done relative to 
    const std::vector<float> query;
    // pointer to array with pq codes for all points (must be N x M array)
    const uint8_t * const codes;
    // Distances between query and codewords in the codebook (M x K array)
    double *distances;

    Estimator(const std::vector<float> vec, const uint8_t * const code_ptr, const size_t M, const size_t K):
        m_(M),
        k_(K),
        query(vec), 
        codes(code_ptr)
    {
        distances = new double[M*K];
    }

    ~Estimator()
    {
        delete distances;
    }

    double estimate(const size_t idx);

};

template <class TLoss>
class ProductQuantizer
{
private:
    const size_t kSampleSize = 100000u; // This still used?

    const size_t m_;
    const size_t k_;
    const size_t n_;
    const size_t dims_;

    data_t &data_;
    //codebook that contains m*k centroids
    std::vector<data_t> codebook_;
    std::vector<size_t> subspace_sizes_;
    // Stores product quantization codes for all points, (N x M array)
    uint8_t* codes_;
    bool is_build_ = false;

public:
    ///Builds short codes for vectors using Product Quantization by projecting every subspace down to neigherst kmeans cluster
    ///Uses these shortcodes for fast estimation of inner product between vectors  
    ///
    ///@param TLoss Type of loss function used for optimizing the clusters (default: euclidean)
    ///@param dataset Reference to the dataset. This enables instantiation of class before data is known
    ///@param m Number of subspaces to split the vectors in, distributes sizes as uniformely as possible
    ///@param k Number of clusters for each subspace, build using kmeans

    ProductQuantizer(data_t &data, size_t m_subspaces=8, size_t k_clusters=256);

    ~ProductQuantizer();

    // Returns memory usage in bytes
    uint64_t memory_usage();

    Estimator getEstimator(const std::vector<float> &query) const;

private:
    // builds the codebook (is called by constructor)
    void build();

    //builds a dataset where each vector only contains the mth chunk
    data_t getSubspace(const size_t m_index, const bool sample = true);

};


template<class TLoss>
ProductQuantizer<TLoss>::ProductQuantizer(data_t &data, size_t m_subspaces, size_t k_cluster=256)
:m_(m_subspaces),
dims_(data[0].size()),
k_(k_clusters),
n_(data.size()),
data_(data),
codebook_(m_),
subspace_sizes(m_, dims_/m_)
{
    assert(k_ <= 256);
    assert(k_ <= n_);
    codes_ = new uint8_t[n_ * m_];
    std::cout << "constructing PQFilter with k=" << k_ << " m=" << m_ << std::endl;

    //add leftover to subspaces
    size_t leftover = dims_ % m_;
    for (size_t i = 0u; i < leftover; i++)
    {
        subspace_sizes_[i] += 1;
    }
    build();
}

template<class TLoss>
data_t ProductQuantizer<TLoss>::getSubspace(const size_t m_index, const bool sample)
{
    assert(m_index < m_);
    // If dataset is large use at most `kSampleSize` points for quantization
    size_t sample_size = n_;
    if (sample)
        sample_size = std::min(n_, kSampleSize);

    std::vector<size_t> s_idcs = util::randomSample(sample_size, n_);
    std::sort(s_idcs.begin(), s_idcs.end()); // Such that order is maintained when the sample is the whole dataset

    data_t subspace;
    for (size_t sample_index : s_idcs)
    {
        subspace.push_back(data_[sample_index]);
    }
    return subspace;
}

template<class TLoss>
void ProductQuantizer<TLoss>::build()
{
    KMeans<TLoss> kmeans(k_); 
    for(size_t m_index = 0; m_index < M ; m_index++)
    {
        //std::cout << "creating codebook for subspace " << m_index << std::endl;

        //RunKmeans for the given subspace
        //gb_labels for this subspace will be the mth index of the PQcodes
        
        // Store centroids to codebook
        data_t subspace = getSubspace(m_index, true);
        kmeans.fit(subspace);
        data_t centroids;
        for (Cluster &c : kmeans.clusters_)
        {
            codebook[m_index].push_back(c.centroid);
        }
        assert(centroids.size() == k_);

        // Stores the assignemnt of points for the PQ code
        subspace = getSubspace(m_index, false);
        std::vector<uin8_t> assignments = kmeans.getAssignment(subspace);
        for (size_t i = 0u; i < n_; i++)
        {
            codes_[m_ * i + m_index] = assignments[i]; // TODO: make this write be sequential
        }
    }
    is_build_ = true;
}


template<class TLoss>
Estimator ProductQuantizer<TLoss>::getEstimator(const std::vector<float> &query) const
{
    assert(query.size() == dims_);
    Estimator e(query, codes_, m_, k_, n_);

    // No vector will be larger than this
    float padded_query[codebook_[0][0].size()];

    float *q_start = &query[0];
    for(size_t m_index = 0u; m_index < m_; m_index++)
    {
        // No need to fill in pad, as the other vectors should have padding which is enough
        std::copy(q_start, q_start + subspace_sizes_[m_index], padded_query);
        q_start += subspace_sizes_[m_index];

        size_t pad_dims = codebook_[m_index][0].size();
        for (size_t k_index = 0u; k_index < k_; k_index++)
        {
            e.distances[m_index * k_ + k_index] = Math::innerProduct(PTR_START(codebook_[m_index], k_index), padded_query, pad_dims);
        }
    }
    return e;
}

double Estimator::estimate(const size_t idx)
{
    double inner_product = 0.0;
    for (size_t m_index = 0u; m_index < m_; m_index++)
    {
        inner_product += distances[k_ * m_index + static_cast<size_t>(codes[idx * m_index + m_index])];
    }
    return inner_product;
}

} // namespace PQ
