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


namespace spq{

using namespace util;

struct Estimator
{
    // Cant get this one to work - i.e. only this function has access to constructor
    //template<class TL> friend Estimator ProductQuantizer<TL>::getEstimator(const std::vector<float> &query) const;
    template<class TLoss> friend class ProductQuantizer;

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
private:

    Estimator(const std::vector<float> vec, const uint8_t * const code_ptr, const size_t M, const size_t K):
        m_(M),
        k_(K),
        query(vec), 
        codes(code_ptr)
    {
        distances = new double[M*K];
    }

public:
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

    const data_t &data_;
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

    ProductQuantizer(const data_t &data, size_t m_subspaces=8, size_t k_clusters=256);

    ~ProductQuantizer();

    // Returns memory usage in bytes
    uint64_t memory_usage();

    // Returns a copy of the codebook without padding
    // Returned vector contains M elements each of which is a data_t
    // with size K x D/M (some are 1 larger to accomodate when D % M != 0)
    std::vector<data_t> getCodebook();

    Estimator getEstimator(const std::vector<float> &query) const;

    // returns product quantization code of the vector at index `idx` in the dataset
    // the returned vector will always be of length m (number of subspaces)
    std::vector<uint8_t> getCode(size_t idx);

private:
    // builds the codebook (is called by constructor)
    void build();

    //builds a dataset where each vector only contains the mth chunk
    data_t getSubspace(const size_t m_index, const bool sample = true);

};


template<class TLoss>
ProductQuantizer<TLoss>::ProductQuantizer(const data_t &data, size_t m_subspaces, size_t k_clusters)
:m_(m_subspaces),
dims_(data[0].size()),
k_(k_clusters),
n_(data.size()),
data_(data),
codebook_(m_),
subspace_sizes_(m_, dims_/m_)
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
ProductQuantizer<TLoss>::~ProductQuantizer() {}

template<class TLoss>
data_t ProductQuantizer<TLoss>::getSubspace(const size_t m_index, const bool sample)
{
    assert(m_index < m_);
    // If dataset is large use at most `kSampleSize` points for quantization
    size_t sample_size = n_;
    if (sample)
        sample_size = std::min(n_, kSampleSize);

    std::vector<size_t> s_idcs = randomSample(sample_size, n_);
    std::sort(s_idcs.begin(), s_idcs.end()); // Such that order is maintained when the sample is the whole dataset

    data_t subspace;
    size_t start_d = std::accumulate(&subspace_sizes_[0],&subspace_sizes_[m_index], 0uL);
    for (size_t sample_index : s_idcs)
    {
        const float * start = &data_[sample_index][start_d];
        const float * end   = &data_[sample_index][start_d + subspace_sizes_[m_index]];
        subspace.emplace_back(start, end);
    }
    return subspace;
}

template<class TLoss>
void ProductQuantizer<TLoss>::build()
{
    for(size_t m_index = 0; m_index < m_ ; m_index++)
    {
        //std::cout << "creating codebook for subspace " << m_index << std::endl;

        //RunKmeans for the given subspace
        //gb_labels for this subspace will be the mth index of the PQcodes
        
        // Store centroids to codebook
        data_t subspace = getSubspace(m_index, true);

        KMeans<TLoss> kmeans(k_); 
        kmeans.fit(subspace);
        for (Cluster &c : kmeans.clusters_)
        {
            codebook_[m_index].push_back(c.centroid);
        }
        assert(codebook_[m_index].size() == k_);

        // Stores the assignemnt of points for the PQ code
        subspace = getSubspace(m_index, false);
        std::vector<uint8_t> assignments = kmeans.getAssignment(subspace);
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
    Estimator est(query, codes_, m_, k_);

    // No vector will be larger than this
    float padded_query[codebook_[0][0].size()];

    const float *q_start = &query[0];
    for(size_t m_index = 0u; m_index < m_; m_index++)
    {
        // No need to fill in pad, as the other vectors should have padding which is enough
        std::copy(q_start, q_start + subspace_sizes_[m_index], padded_query);
        q_start += subspace_sizes_[m_index];

        size_t pad_dims = codebook_[m_index][0].size();
        for (size_t k_index = 0u; k_index < k_; k_index++)
        {
            est.distances[m_index * k_ + k_index] = innerProduct(PTR_START(codebook_[m_index], k_index), padded_query, pad_dims);
        }
    }
    return est;
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

template<class TLoss>
std::vector<data_t> ProductQuantizer<TLoss>::getCodebook()
{
    std::vector<data_t> clean_cb(m_);
    for (size_t m_index = 0; m_index < m_; m_index++)
    {
        for (const std::vector<float> &codeword : codebook_[m_index])
        {
            const float * start  = &codeword[0];
            const float * end    = &codeword[subspace_sizes_[m_index]];
            clean_cb[m_index].emplace_back(start, end);
        }
    }
    return clean_cb;
}

template<class TLoss>
std::vector<uint8_t> ProductQuantizer<TLoss>::getCode(size_t idx)
{
    return std::vector<uint8_t>(codes_ + (idx*m_), codes_ + ( (idx+1) * m_ ) );

}


} // namespace spq

