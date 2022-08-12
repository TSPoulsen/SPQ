#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#ifdef __AVX2__
    #define AVX_TEST_CASE(name) DOCTEST_TEST_CASE(name " AVX")
#else
    #define AVX_TEST_CASE(name) DOCTEST_TEST_CASE(name)
#endif

#include "spq/loss/loss_base.hpp"
#include "spq/loss/euclidean_loss.hpp"
#include "spq/loss/product_loss.hpp"
#include "spq/loss/weighted_loss.hpp"

#include <iostream>
#include <set>

namespace loss
{
    using namespace spq;
    using namespace spq::util;

    AVX_TEST_CASE("EuclideanLoss distance Test")
    {
        data_t vectors = {{0, 0, 0, 0},
                              {-1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};

        float ans[5][5] = {{0, 2, 2, 1, 3},
                           {2, 0, 4, 5, 3},
                           {2, 4, 0, 3, 1},
                           {1, 5, 3, 0, 4},
                           {3, 3, 1, 4, 0}};

        EuclideanLoss loss;
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

    AVX_TEST_CASE("EuclideanLoss/ProductLoss getCentroid Test")
    {
        data_t vectors = {{0, 0, 0, 0},
                              {-1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};
        std::vector<std::vector<unsigned int>> all_members = {{1}, {0, 1, 2, 3, 4}, {0, 1}, {3, 4}};
        data_t ans = {{-1, 0, 0, 1},
                          {0, 2.0 / 5.0, 2.0 / 5.0, 2.0 / 5.0},
                          {-0.5, 0, 0, 0.5},
                          {0.5, 0.5, 0.5, 0.5}};
        EuclideanLoss loss;
        loss.init(vectors);
        loss.padData(ans);
        for (size_t i = 0; i < ans.size(); i++)
        {
            std::vector<float> guess = loss.getCentroid(all_members[i]);
            CHECK(guess == ans[i]);
        }
    }

    AVX_TEST_CASE("LossDefault initCentroids Test")
    {
        data_t vectors = {{0, 0, 0, 0},
                              {1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};

        EuclideanLoss loss;
        loss.init(vectors);
        for (size_t K = 0; K < vectors.size(); K++)
        {
            data_t centroids = loss.initCentroids(K);
            CHECK(centroids.size() == K);
            std::set<std::vector<float>> uniq(centroids.begin(), centroids.end());
            CHECK(uniq.size() == K);
        }
    }

    AVX_TEST_CASE("ProductLoss covariance creation Test")
    {
        data_t vectors = {{0, 0, 0, 0},
                              {-1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};
        data_t ans = { {0.4, 0, 0, -0.2},
                           {0, 0.4, 0.4, 0.2},
                           {0, 0.4, 0.4, 0.2},
                           {-0.2, 0.2, 0.2, 0.4}};
        ProductLoss loss;
        loss.init(vectors);
        loss.padData(ans);
        for (size_t i = 0; i < ans.size(); i++)
        {
            CHECK(loss.cov_[i] == ans[i]);
        }
    }

    AVX_TEST_CASE("ProductLoss distance Test")
    {
        data_t vectors = {{0, 0, 0, 0},
                              {-1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};

        float ans[5][5] = {{0, 1.2, 1.6, 0.4, 2.8},
                           {1.2, 0, 2, 2.8, 2},
                           {1.6, 2, 0, 2, 0.4},
                           {0.4, 2.8, 2.0, 0, 3.6},
                           {2.8, 2, 0.4, 3.6, 0}};

        ProductLoss loss;
        loss.init(vectors);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                CHECK(loss.distance(i, vectors[j]) == doctest::Approx(ans[i][j]));
            }
        }
    }

    AVX_TEST_CASE("WeightedProductLoss scalings Test")
    {
        // This proxy only works for large dimensionality

        data_t vectors = {std::vector<float>(100,0),
                              std::vector<float>(100,1),
                              std::vector<float>(100,2),
                              std::vector<float>(100,-1)};
        WeightedProductLoss loss;

        SUBCASE("T=0.0")
        {
            loss.init(vectors);
            for (size_t i = 0; i < vectors.size(); i++)
            {
                CHECK(loss.getScaling(i) == 1.0);
            }
        }
        SUBCASE("T=0.2")
        {
            loss.init(vectors, 0.2);
            CHECK(loss.getScaling(0) != loss.getScaling(0));
            double ans[3] = {0.040,0.010, 0.040};
            for (size_t i = 1; i < vectors.size(); i++)
            {
                CHECK(loss.getScaling(i) == doctest::Approx(ans[i-1]).epsilon(0.001));
            }

        }
        SUBCASE("T=1.0")
        {
            loss.init(vectors, 1.0);
            double ans[3] = {1.010, 0.2506, 1.010};
            for (size_t i = 1; i < vectors.size(); i++)
            {
                CHECK(loss.getScaling(i) == doctest::Approx(ans[i-1]).epsilon(0.001));
            }
        }

        SUBCASE("T=9.0")
        {
            loss.init(vectors, 9.0);
            double ans[3] = {426.326, 25.39, 426.326};
            for (size_t i = 1; i < vectors.size(); i++)
            {
                CHECK(loss.getScaling(i) == doctest::Approx(ans[i-1]).epsilon(0.001));
            }
        }

    }

    AVX_TEST_CASE("WeightedProductLoss distance test")
    {
        data_t vectors = {{0, 0, 0, 0},
                              {-1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};


        WeightedProductLoss loss;
        SUBCASE("T=0.0")
        {
            loss.init(vectors, 0.0);
            // dist(i,j)
            // where ans[i][j]
            float ans[5][5] = { {-1,   -1,  -1,  -1,  -1},
                                { 2.0,  0.0, 4.0, 5.0, 3.0},
                                { 2.0,  4.0, 0.0, 3.0, 1.0},
                                { 1.0,  5.0, 3.0, 0.0, 4.0},
                                { 3.0,  3.0, 1.0, 4.0, 0.0}};
            for (size_t i = 1; i < vectors.size(); i++)
            {
                for (size_t j = 0; j < vectors.size(); j++)
                {
                    CHECK(loss.distance(i,vectors[j]) ==  doctest::Approx(ans[i][j]).epsilon(0.001));
                }
            }
        }
        SUBCASE("T=0.2")
        {
            loss.init(vectors, 0.2);
            // dist(i,j)
            // where ans[i][j]
            float ans[5][5] = { {-1,   -1,     -1,     -1,     -1},
                                { 0.1633,    0.0,    2.1633, 0.8673, 2.5408},
                                { 0.1633,    2.1633, 0.0,    1.1633, 1.0},
                                { 0.1666,    1.6667, 2.1667, 0.0,    3.1667},
                                { 0.1621,    1.7387, 0.6847, 1.1622, 0.0}};
            for (size_t i = 1; i < vectors.size(); i++)
            {
                for (size_t j = 0; j < vectors.size(); j++)
                {
                    CHECK(loss.distance(i,vectors[j]) ==  doctest::Approx(ans[i][j]).epsilon(0.001));
                }
            }
        }
    }

    AVX_TEST_CASE("WeigtedProductLoss initCentroids Test")
    {
        data_t vectors = {{0, 0, 0, 0},
                              {1, 0, 0, 1},
                              {0, 1, 1, 0},
                              {1, 0, 0, 0},
                              {0, 1, 1, 1}};

        size_t K = 3;
        WeightedProductLoss loss;
        loss.init(vectors, 0.0);
        data_t centroids = loss.initCentroids(K);
        CHECK(centroids.size() == K);
        for (size_t ki = 0; ki < centroids.size(); ki++)
        {
            CHECK(centroids[ki].size() == vectors[0].size());
        }
    }

} // namespace loss
