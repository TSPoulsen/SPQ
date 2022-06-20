#pragma once

#include "catch.hpp"
#include "loss.hpp"

#include <iostream>



namespace loss {
    TEST_CASE("inner_product test")
    {
        std::vector<std::vector<float>> vectors = { {0,0,0,0,  0,0,0,0},
                                                    {1,0,0,1,  0,0,0,0},
                                                    {0,1,1,0,  0,0,0,0},
                                                    {1,0,0,0,  0,0,0,0},
                                                    {0,1,1,1,  0,0,0,0},
                                                    {0,1,1,1,  0,0,0,0}};
                        
        float ans[5][5] = { {0,0,0,0,0},
                            {0,2,0,1,1},
                            {0,0,2,0,2},
                            {0,1,0,1,0},
                            {0,1,2,0,3}};
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                std::cout << vectors[i][0] << " " << vectors[j][0] << std::endl;
                CHECK(PQ::inner_product(&(vectors[i][0]), &(vectors[j][0]), 8) == ans[i][j]);
                if (!(PQ::inner_product(&(vectors[i][0]), &(vectors[j][0]), 8) == ans[i][j])) {
                    printf("%d %d\n", i, j);
                }
            }
        }
    }
} // namespace loss

