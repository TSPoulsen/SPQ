#pragma once

#include "loss/loss_base.hpp"
#include "loss/euclidean_loss.hpp"

#include <cassert>
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
    TLoss loss_; // This should perhaps be private
private:
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
    KMeans(unsigned int k_clusters = 256, unsigned int max_iter = 100, double tol = 0.001);

    // Finds the best set of clusters that minimizes the inertia
    void fit(data_t &data);

private:
    // Assign all points to their cluster with the nearest centroid according to distance type
    // Currently does not handle if any clusters are empty
    double assignToClusters(const data_t &data);
};

template<class TLoss>
KMeans<TLoss>::KMeans(unsigned int k_clusters, unsigned int max_iter, double tol):
    k_(k_clusters),
    tol_(tol),
    max_iter_(max_iter)
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
double KMeans<TLoss>::assignToClusters(const data_t &data)
{
    // Clear member variable for each cluster
    for (Cluster &c : clusters_)
    {
        c.members.clear();
    }

    double current_inertia = 0;

    for (unsigned int i = 0; i < data.size(); i++)
    {
        double min_dist = DBL_MAX;
        unsigned int min_label = data.size() + 1;
        for (unsigned int c_i = 0; c_i < k_; c_i++)
        {
            double d = loss_.distance(i, clusters_[c_i].centroid);
            if (d < min_dist)
            {
                min_label = c_i;
                min_dist = d;
            }
        }
        clusters_[min_label].members.push_back(i);
        current_inertia += min_dist;
    }
    return current_inertia;
}

} // PQ