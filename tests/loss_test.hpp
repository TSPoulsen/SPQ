#pragma once

#include "catch.hpp"
#include "loss/loss_base.hpp"
#include "loss/euclidean_loss.hpp"
#include "loss/product_loss.hpp"
#include "loss/weighted_loss.hpp"

#include <iostream>
#include <set>

namespace loss
{
    TEST_CASE("EuclideanLoss distance Test")
    {
        PQ::data_t vectors = {{0, 0, 0, 0},
                              {-1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};

        float ans[5][5] = {{0, 2, 2, 1, 3},
                           {2, 0, 4, 5, 3},
                           {2, 4, 0, 3, 1},
                           {1, 5, 3, 0, 4},
                           {3, 3, 1, 4, 0}};

        PQ::EuclideanLoss loss;
        loss.init(vectors);
        std::cout << vectors[0].size() << std::endl;
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                CHECK(loss.distance(i, vectors[j]) == ans[i][j]);
            }
        }
    }

    TEST_CASE("EuclideanLoss/IPLoss getCentroid Test")
    {
        PQ::data_t vectors = {{0, 0, 0, 0},
                              {-1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};
        std::vector<std::vector<unsigned int>> all_members = {{1}, {0, 1, 2, 3, 4}, {0, 1}, {3, 4}};
        PQ::data_t ans = {{-1, 0, 0, 1},
                          {0, 2.0 / 5.0, 2.0 / 5.0, 2.0 / 5.0},
                          {-0.5, 0, 0, 0.5},
                          {0.5, 0.5, 0.5, 0.5}};
        PQ::EuclideanLoss loss;
        loss.init(vectors);
        size_t ans_p = loss.padData(ans);
        for (size_t i = 0; i < ans.size(); i++)
        {
            std::vector<float> guess = loss.getCentroid(all_members[i]);
            CHECK(guess == ans[i]);
        }
    }

    TEST_CASE("initCentroids Test")
    {
        PQ::data_t vectors = {{0, 0, 0, 0},
                              {1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};

        PQ::EuclideanLoss loss;
        loss.init(vectors);
        for (size_t K = 0; K < vectors.size(); K++)
        {
            PQ::data_t centroids = loss.initCentroids(K);
            CHECK(centroids.size() == K);
            std::set<std::vector<float>> uniq(centroids.begin(), centroids.end());
            CHECK(uniq.size() == K);
        }
    }

    TEST_CASE("ProductLoss covariance creation Test")
    {
        PQ::data_t vectors = {{0, 0, 0, 0},
                              {-1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};
        PQ::data_t ans = { {0.4, 0, 0, -0.2},
                           {0, 0.4, 0.4, 0.2},
                           {0, 0.4, 0.4, 0.2},
                           {-0.2, 0.2, 0.2, 0.4}};
        PQ::ProductLoss loss;
        loss.init(vectors);
        loss.padData(ans);
        for (size_t i = 0; i < ans.size(); i++)
        {
            CHECK(loss.cov[i] == ans[i]);
        }
    }

    TEST_CASE("ProductLoss distance Test")
    {
        PQ::data_t vectors = {{0, 0, 0, 0},
                              {-1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};

        float ans[5][5] = {{0, 1.2, 1.6, 0.4, 2.8},
                           {1.2, 0, 2, 2.8, 2},
                           {1.6, 2, 0, 2, 0.4},
                           {0.4, 2.8, 2.0, 0, 3.6},
                           {2.8, 2, 0.4, 3.6, 0}};

        PQ::ProductLoss loss;
        loss.init(vectors);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                CHECK(loss.distance(i, vectors[j]) == Approx(ans[i][j]));
            }
        }
    }

    TEST_CASE("WeightedProductLoss scalings Test")
    {
        // This proxy only works for large dimensionality
        PQ::data_t vectors = {std::vector<float>(100,0),
                              std::vector<float>(100,1),
                              std::vector<float>(100,2)};
        PQ::WeightedProductLoss loss;

        SECTION ("T=0.0")
        {
            loss.init(vectors);
            for (size_t i = 0; i < vectors.size(); i++)
            {
                CHECK(loss.getScaling(i) == 1.0);
            }
        }
        SECTION ("T=0.2")
        {
            loss.init(vectors, 0.2);
            CHECK(loss.getScaling(0) != loss.getScaling(0));
            double ans[2] = {0.040,0.010};
            for (size_t i = 1; i < vectors.size(); i++)
            {
                CHECK(loss.getScaling(i) == Approx(ans[i-1]).epsilon(0.001));
            }
            loss.init(vectors, 0.2);

        }
        SECTION ("T=1.0")
        {
            loss.init(vectors, 1.0);
            double ans[2] = {1.010, 0.2506};
            for (size_t i = 1; i < vectors.size(); i++)
            {
                CHECK(loss.getScaling(i) == Approx(ans[i-1]).epsilon(0.001));
            }
        }

        SECTION ("T=9.0")
        {
            loss.init(vectors, 9.0);
            double ans[2] = {426.326, 25.39};
            for (size_t i = 1; i < vectors.size(); i++)
            {
                CHECK(loss.getScaling(i) == Approx(ans[i-1]).epsilon(0.001));
            }
        }

    }

} // namespace loss
