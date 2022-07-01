#pragma once

#include "catch.hpp"
#include "loss.hpp"

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
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                CHECK(PQ::Math::innerProduct(&(vectors[i][0]), &(vectors[j][0]), 8) == ans[i][j]);
            }
        }
    }
}