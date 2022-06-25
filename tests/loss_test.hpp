#pragma once

#include "catch.hpp"
#include "loss.hpp"

#include <iostream>



namespace loss {
    TEST_CASE("inner_product Test")
    {
        PQ::data_t vectors = { {0,0,0,0,  0,0,0,0},
                                                    {1,0,0,1,  0,0,0,0},
                                                    {0,1,1,0,  0,0,0,0},
                                                    {1,0,0,0,  0,0,0,0},
                                                    {0,1,1,1,  0,0,0,0}};
                        
        float ans[5][5] = { {0,0,0,0,0},
                            {0,2,0,1,1},
                            {0,0,2,0,2},
                            {0,1,0,1,0},
                            {0,1,2,0,3}};
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                CHECK(PQ::inner_product(&(vectors[i][0]), &(vectors[j][0]), 8) == ans[i][j]);
            }
        }
    }

    TEST_CASE("EuclideanLoss distance Test")
    {
        PQ::data_t vectors = {  {0,0,0,0},
                            {-1,0,0,1},
                            {0,1,1,0},
                            {1,0,0,0},
                            {0,1,1,1}};
                        
        float ans[5][5] = { {0,2,2,1,3},
                            {2,0,4,5,3},
                            {2,4,0,3,1},
                            {1,5,3,0,4},
                            {3,3,1,4,0}};

        PQ::EuclideanLoss loss;
        size_t padding = loss.padData(vectors);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                CHECK(loss.distance(vectors[i], vectors[j]) == ans[i][j]);
            }
        }

    }

    TEST_CASE("EuclideanLoss getCentroid Test")
    {
        PQ::data_t vectors = {  {0,0,0,0},
                                {-1,0,0,1},
                                {0,1,1,0},
                                {1,0,0,0},
                                {0,1,1,1}};
        std::vector<std::vector<unsigned int>> all_members = { {1}, {0,1,2,3,4}, {0,1}, {3,4} };
        PQ::data_t ans = { {-1,0,0,1},
                           {0, 2.0/5.0, 2.0/5.0, 2.0/5.0},
                           {-0.5, 0, 0, 0.5},
                           {0.5, 0.5, 0.5, 0.5}};
        PQ::EuclideanLoss loss;
        size_t vec_p = loss.padData(vectors);
        size_t ans_p = loss.padData(ans);
        for (size_t i = 0; i < ans.size(); i++) {
            std::vector<float> guess = loss.getCentroid(vectors, all_members[i]);
            CHECK(guess == ans[i]);
        }
    }

} // namespace loss

