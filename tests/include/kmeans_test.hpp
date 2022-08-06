#pragma once

#include "kmeans.hpp"

#include "catch.hpp"
#include <vector>
#include <iostream>
#include <set>

namespace kmeans
{

    using namespace PQ;
    using namespace PQ::util;

    struct TestData
    {
        unsigned int N, dims, K;
        data_t input,
            answer;
    };

    bool are_approx_equal(std::set<std::vector<float>> run_answer, std::set<std::vector<float>> real_answer)
    {
        // std::cout << "run_anwsers size: " << (run_answer).size() << " actual anwsers size: " << (real_answer).size() << std::endl;
        // std::cout << "run_anwsers dim: " << (*(run_answer.begin())).size() << " actual anwsers dim: " << (*(real_answer.begin())).size() << std::endl;
        if (run_answer.size() != real_answer.size())
            return false;
        unsigned int K = run_answer.size();
        unsigned int dims = (*run_answer.begin()).size();
        // std::cout << "what the fuck is the set size " << dims << std::endl;
        auto it_run = run_answer.begin(), it_real = real_answer.begin();
        bool are_equal = true;
        for (size_t i = 0; i < K; i++)
        {
            std::vector<float> run_i = *it_run, real_i = *it_real;
            if (run_i.size() != dims || real_i.size() != dims)
                return false;
            // std::cout << "should make it to heare" << std::endl;
            for (size_t d = 0; d < dims; d++)
            {
                are_equal = (are_equal && (run_i[d] == Approx(real_i[d])));
                // std::cout << "ans: " << run_i[d] << " correct: " << real_i[d] << " what it evaluates to: " << (run_i[d] == (Approx(real_i[d]))) <<  " are equal: " << are_equal << std::endl;
            }
            it_run++;
            it_real++;
        }
        // std::cout << "we are returning -> " <<are_equal << std::endl;
        return are_equal;
    }

    template <typename TLoss>
    void kmeans_correctness_test(struct TestData td)
    {
        // std::cout << "Start of test" << std::endl;
        KMeans<TLoss> kmeans(td.K);
        kmeans.fit(td.input);
        std::set<std::vector<float>> run_ans;
        for (unsigned int i = 0; i < td.K; i++)
        {
            const std::vector<float> &cen = kmeans.clusters_[i].centroid;
            run_ans.insert(cen);
            // std::cout << "inserted another item" << std::endl;
            //  for(const float &val: cen){
            // std::cout << val << " ";
            //}
            // std::cout << std::endl;
        }
        TLoss tmp_loss;
        tmp_loss.padData(td.answer);
        std::set<std::vector<float>> ans(td.answer.begin(), td.answer.end());
        bool are_equal = are_approx_equal(run_ans, ans);
        // std::cout << "end of test: " << are_equal << std::endl;
        REQUIRE(are_equal);
    }

    TEST_CASE("basic euclidean clustering 1")
    {
        struct TestData td;
        td.N = 4;
        td.dims = 2;
        td.K = 2;
        td.input = {
            {-1.0, 0.01},
            {-1.0, -0.01},
            {1.0, 0.01},
            {1.0, -0.01}};
        td.answer = {
            {-1.0, 0.0},
            {1.0, 0.0}};
        kmeans_correctness_test<EuclideanLoss>(td);
        kmeans_correctness_test<ProductLoss>(td);
        return;
    }

    TEST_CASE("basic euclidean clustering 2")
    {
        struct TestData td;
        td.N = 8;
        td.dims = 2;
        td.K = 4;
        td.input = {
            {-1.0, 0.01},
            {-1.0, -0.01},
            {1.0, 0.01},
            {1.0, -0.01},

            {0.01, -1.00},
            {-0.01, -1.00},
            {0.01, 1.00},
            {-0.01, 1.00},
        };
        td.answer = {
            {-1.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {0.0, -1.0}};
        kmeans_correctness_test<EuclideanLoss>(td);
        kmeans_correctness_test<ProductLoss>(td);
        return;
    }

    TEST_CASE("basic euclidean clustering 3")
    {
        struct TestData td;
        td.N = 8;
        td.dims = 2;
        td.K = 4;
        td.input = {
            {-1.0, 0.01},
            {-1.0, -0.01},
            {1.0, 0.01},
            {1.0, -0.01},

            {0.01, -1.00},
            {-0.01, -1.00},
            {0.01, 1.00},
            {-0.01, 1.00},

        };
        td.answer = {
            {-1.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {0.0, -1.0}};
        kmeans_correctness_test<EuclideanLoss>(td);
        kmeans_correctness_test<ProductLoss>(td);
    }

    TEST_CASE("basic euclidean clustering 4")
    {
        struct TestData td;
        td.N = 8;
        td.dims = 2;
        td.K = 3;
        td.input = {
            {0.0719421, 0.2894158},
            {0.09693447, 0.28799435},
            {0.14203255, 0.16500843},
            {0.2048598, 0.10420303},
            {0.89096942, 0.85557702},
            {0.61737082, 0.0460024},
            {0.31298214, 0.85340625},
            {0.69373669, 0.21694309}};
        td.answer = {
            {0.60197578, 0.85449163},
            {0.12894223, 0.2116554},
            {0.65555375, 0.13147274}};
        kmeans_correctness_test<EuclideanLoss>(td);
        kmeans_correctness_test<ProductLoss>(td);
    }
} // namespace kmeans