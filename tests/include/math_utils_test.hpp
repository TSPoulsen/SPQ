#pragma once

#include "catch.hpp"
#include "math_utils.hpp"

#include <iostream>
#include <set>


namespace math_utils
{
    TEST_CASE("innerProduct Test")
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

        PQ::data_t *p = &vectors;
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                CHECK(PQ::Math::innerProduct(PQ::PTR_START(p,i), PQ::PTR_START(p,j), 8) == ans[i][j]);
            }
        }
    }
    TEST_CASE("innerProduct Test (large)")
    {
        PQ::data_t vectors = {std::vector<float>(128,0),
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
                CHECK(PQ::Math::innerProduct(PQ::PTR_START(&vectors,i), PQ::PTR_START(&vectors,j), 128) == ans[i][j]);
            }
        }
    }
}