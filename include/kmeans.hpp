#pragma once

#include "loss/loss_base.hpp"
#include "loss/euclidean_loss.hpp"

#include <cassert>
#include <utility>
#include <vector>
#include <cfloat>
#include <iostream>

namespace PQ
{

struct Cluster
{
    std::vector<float> centroid;
    std::vector<unsigned int> members;
    Cluster(const std::vector<float> &cen) : centroid(cen) {}
    Cluster() {}
    Cluster(const Cluster &c1) : centroid(c1.centroid), members(c1.members) {} // TODO: Is this still needed
};

// The implementation of this class has taken parts from: https://github.com/yahoojapan/NGT/blob/master/lib/NGT/Clustering.h
/// Class for performing k-means clustering on a given dataset
template <class TLoss>
class KMeans
{
    //TODO: change names
public:
    std::vector<Cluster> clusters_;
private:
    TLoss loss_; // This should perhaps be private
    // Clustering configuration
    const unsigned int k_;
    const double tol_;            // Minium relative change between two iteration, otherwise stop training
    const unsigned int max_iter_; // Maximum number of iterations

    bool is_fitted_ = false;
    double inertia_ = DBL_MAX;

public:
    /// Constructs an KMeans instance with chosen parameters
    /// @param TLoss Type of loss function used for optimizing the clusters (default: euclidean)
    /// @param k_clusters Number of clusters to fit the data to (default: 256)
    /// @param max_iter The number of llyod iteration to at most perform before terminating (default: 100)
    /// @param tol The minimum relative change between two iterations. If this change is not achieved then training terminates (default: 0.001)

    // TODO: make data part of constructor and perhaps move configuration to fit
    // This will enable data to be a class member
    KMeans(unsigned int k_clusters = 256, unsigned int max_iter = 100, double tol = 0.001);

    // Finds the best set of clusters that minimizes the inertia
    void fit(data_t &data);

    // returns vector of size `points.size()` which contains the indicies of the clusters which each point lies closest to
    std::vector<uint8_t> getAssignment(data_t points) const;

private:
    // finds the closes cluster centroid to the given vector 
    // and returns the index of the cluster along with the distance to that cluster centroid
    std::pair<size_t,double> findClosestCluster(const size_t idx, ILoss &loss) const;
    // Assign all points to their cluster with the nearest centroid according to distance type
    // Currently does not handle if any clusters are empty
    double assignToClusters(const data_t &data);
};

template<class TLoss>
KMeans<TLoss>::KMeans(unsigned int k_clusters, unsigned int max_iter, double tol):
    k_(k_clusters),
    tol_(tol),
    max_iter_(max_iter),
    loss_(TLoss())
{
    std::cout << "Creating Kmeans with K=" << k_ << std::endl; // debug
    assert(k_ <= 256);
    assert(k_ > 0);
}

template<class TLoss>
void KMeans<TLoss>::fit(data_t &data)
{
    if (is_fitted_)
    {
        return;
    }
    loss_.init(data);
    assert(k_ <= data.size());

    unsigned int iteration = 0;
    double inertia_delta = 1.0;

    // Initalize clusters
    // std::cout << "Init centroids K = " << K << std::endl;
    // std::cout << "data size = " << data.size() << "," << data[0].size() << std::endl;
    data_t centroids = loss_.initCentroids(k_);
    // std::cout << "Init centroids DONE" << std::endl;
    for (std::vector<float> cen : centroids)
    {
        // std::cout << "pushing back" << std::endl;
        clusters_.push_back(Cluster(cen));
    }

    // std::cout << "Kmeans total iterations " << MAX_ITER << " : ";
    while (inertia_delta >= tol_ && iteration < max_iter_)
    {
        if (iteration % 10 == 0)
            std::cout << iteration << "-" << std::flush; // debug
        // Step 1 in llyod: assign points to clusters
        double current_inertia = assignToClusters(data);

        // Step 2 in llyod: Set clusters to be center of points in cluster
        // Assigns all centriods
        for (Cluster &c : clusters_)
        {
            c.centroid = loss_.getCentroid(c.members);
        }

        // Calculate inertia difference
        inertia_delta = (inertia_ - current_inertia) / inertia_;
        inertia_ = current_inertia;
        iteration++;
    }
    std::cout << std::endl; // debug
    is_fitted_ = true;
}


template<class TLoss>
std::pair<size_t,double> KMeans<TLoss>::findClosestCluster(const size_t idx, ILoss &loss) const
{
    double min_dist = DBL_MAX;
    size_t min_label;
    for (size_t cluster_index = 0; cluster_index < k_; cluster_index++)
    {
        double d = loss.distance(idx, clusters_[cluster_index].centroid);
        if (d < min_dist)
        {
            min_label = cluster_index;
            min_dist = d;
        }
    }
    return std::make_pair(min_label, min_dist);
}


template<class TLoss>
double KMeans<TLoss>::assignToClusters(const data_t &data)
{
    // Clear member variable for each cluster
    for (Cluster &c : clusters_)
    {
        c.members.clear();
    }

    double current_inertia = 0;

    for (size_t i = 0; i < data.size(); i++)
    {
        std::pair<size_t, double> p = findClosestCluster(i, loss_);
        current_inertia += p.second;
        clusters_[p.first].members.push_back(i);
    }
    return current_inertia;
}


template<class TLoss>
std::vector<uint8_t> KMeans<TLoss>::getAssignment(const data_t points) const
{
    // TODO: Fix when loss needs to have init done with new data
    TLoss tmp_loss;
    tmp_loss.init(points);
    std::vector<uint8_t> indicies;
    for (size_t idx = 0u; idx < points.size(); idx++)
    {
        std::pair<size_t, double> p = findClosestCluster(idx, tmp_loss);
        indicies.push_back( static_cast<uint8_t>(p.first) );
    }
    return indicies;
}
} // PQ