#pragma once

#include "catch.hpp"
#include "loss.hpp"

#include <iostream>
#include <set>

namespace loss
{
    TEST_CASE("inner_product Test")
    {
        PQ::data_t vectors = {{0, 0, 0, 0, 0, 0, 0, 0},
                              {1, 0, 0, 1, 0, 0, 0, 0},
                              {0, 1, 1, 0, 0, 0, 0, 0},
                              {1, 0, 0, 0, 0, 0, 0, 0},
                              {0, 1, 1, 1, 0, 0, 0, 0}};

        float ans[5][5] = {{0, 0, 0, 0, 0},
                           {0, 2, 0, 1, 1},
                           {0, 0, 2, 0, 2},
                           {0, 1, 0, 1, 0},
                           {0, 1, 2, 0, 3}};
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                CHECK(PQ::inner_product(&(vectors[i][0]), &(vectors[j][0]), 8) == ans[i][j]);
            }
        }
    }

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
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                CHECK(loss.distance(vectors[i], vectors[j]) == ans[i][j]);
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
            std::vector<float> guess = loss.getCentroid(vectors, all_members[i]);
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
            PQ::data_t centroids = loss.initCentroids(vectors, K);
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
        for (int i = 0; i < ans.size(); i++)
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
                CHECK(loss.distance(vectors[i], vectors[j]) == Approx(ans[i][j]));
            }
        }
    }

} // namespace loss
