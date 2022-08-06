#pragma once

#include "catch.hpp"
#include "math_utils.hpp"

#include <iostream>
#include <set>


namespace math_utils
{
    using namespace PQ;
    using namespace PQ::util;

    TEST_CASE("innerProduct Test")
    {
        data_t vectors = {{0, 0, 0, 0, 0, 0, 0, 0},
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
                CHECK(util::innerProduct(PTR_START(vectors,i), PTR_START(vectors,j), 8) == ans[i][j]);
            }
        }
    }
    TEST_CASE("innerProduct Test (large)")
    {
        data_t vectors = {std::vector<float>(128,0),
                              std::vector<float>(128,1),
                              std::vector<float>(128,2),
                              std::vector<float>(128,3)};

        float ans[4][4] = {{0, 0, 0, 0},
                           {0, 128, 256, 384},
                           {0, 256, 512, 768},
                           {0, 384, 768, 1152}};
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                CHECK(util::innerProduct(util::PTR_START(vectors,i), util::PTR_START(vectors,j), 128) == ans[i][j]);
            }
        }
    }

    TEST_CASE("randomSample Test")
    {
        SECTION("Actual Sample")
        {
            size_t data_size = 1e4;
            size_t sample_size = 1e3;
            std::vector<size_t> sample = randomSample(sample_size, data_size);
            std::set<size_t> unique(sample.begin(), sample.end());
            CHECK(unique.size() == sample_size);
        }
        SECTION("Full Sample")
        {
            size_t data_size = 1e4;
            std::vector<size_t> sample = randomSample(data_size, data_size);
            std::set<size_t> unique(sample.begin(), sample.end());
            CHECK(unique.size() == data_size);
        }
        std::make_pair(0,0);

    }
}