#pragma once

#include "catch.hpp"
/*
#include <vector>
#include <iostream>
#include <set>

using namespace puffinn;
namespace kmeans {

    struct TestData {
        unsigned int N, dims, K;
        std::vector<std::vector<float>> input;
        std::set<std::vector<float>> answer;
        KMeans::distanceType DT = KMeans::euclidean;
    };

    bool are_approx_equal(std::set<std::vector<float>> run_answer, std::set<std::vector<float>> real_answer)
    {
        //std::cout << "run_anwsers size: " << (run_answer).size() << " actual anwsers size: " << (real_answer).size() << std::endl;
        //std::cout << "run_anwsers dim: " << (*(run_answer.begin())).size() << " actual anwsers dim: " << (*(real_answer.begin())).size() << std::endl;
        if (run_answer.size() != real_answer.size()) return false;
        unsigned int K  = run_answer.size();
        unsigned int dims  = (*run_answer.begin()).size();
        //std::cout << "what the fuck is the set size " << dims << std::endl;
        auto it_run = run_answer.begin(), it_real = real_answer.begin();
        bool are_equal = true;
        for (size_t i = 0; i < K; i++) {
            std::vector<float> run_i = *it_run, real_i = *it_real;
            if (run_i.size() != dims || real_i.size() != dims) return false;
            //std::cout << "should make it to heare" << std::endl;
            for (size_t d = 0; d < dims; d++) {
                are_equal = (are_equal && (run_i[d] == Approx(real_i[d])));
                //std::cout << "ans: " << run_i[d] << " correct: " << real_i[d] << " what it evaluates to: " << (run_i[d] == (Approx(real_i[d]))) <<  " are equal: " << are_equal << std::endl;
            }
            it_run++;
            it_real++;
        }
        //std::cout << "we are returning -> " <<are_equal << std::endl;
        return are_equal;

    }

    void kmeans_correctness_test(struct TestData td)
    {
        //std::cout << "Start of test" << std::endl;
        PQ::KMeans kmeans(td.K, td.DT);
        kmeans.fit(td.input);
        std::set<std::vector<float>> run_ans;
        for (unsigned int i = 0; i < td.K; i++) {
            run_ans.insert(kmeans.getCentroid(i));
            //std::cout << "inserted another item" << std::endl;
            for(float val: kmeans.getCentroid(i)){
                //std::cout << val << " ";
            } 
            //std::cout << std::endl;
        }
        bool are_equal = are_approx_equal(run_ans, td.answer);
        //std::cout << "end of test: " << are_equal << std::endl;
        REQUIRE(are_equal);

    }

    TEST_CASE("basic kmeans clustering 1") {

        struct TestData td;
        td.N    = 4;
        td.dims = 2;
        td.K    = 2;
        td.input = {
            {-1.0,0.01},
            {-1.0,-0.01},
            {1.0,0.01},
            {1.0,-0.01}};
        td.answer = {
            {-1.0,0.0},
            {1.0,0.0}
        };
        kmeans_correctness_test(td);
        return;
        
    }

    TEST_CASE("basic kmeans clustering 2") {

        struct TestData td;
        td.N    = 8;
        td.dims = 2;
        td.K    = 4;
        td.input = {
            {-1.0,0.01},
            {-1.0,-0.01},
            {1.0,0.01},
            {1.0,-0.01},

            {0.01,-1.00},
            {-0.01,-1.00},
            {0.01,1.00},
            {-0.01,1.00},
            
            
            
            };
        td.answer = {
            {-1.0,0.0},
            {1.0,0.0},
            {0.0,1.0},
            {0.0,-1.0}
        };
        kmeans_correctness_test(td);
        return;
        
    }
    
    TEST_CASE("sumOfSquares test")
    {
        std::vector<std::vector<float>> data = {
                                    {-1, 1, 0, 0, 0},
                                    {1, 1, 0, 0, 0},
                                    {2.3, 7.4, 4.4, 0.1, 6.2},
                                    {0, 0, 0, 0, 0}
        };
       puffinn::KMeans clustering;
       clustering.padData(data);
       double ans1 = clustering.sumOfSquares(data[0], data[1]);
       double ans2 = clustering.sumOfSquares(data[1], data[0]);
       REQUIRE(ans1 == ans2);
       REQUIRE(ans1 == 4.0);
       double ans3 = clustering.sumOfSquares(data[2], data[3]);
       double ans4 = clustering.sumOfSquares(data[3], data[2]);
       REQUIRE(ans3 == ans4);
       REQUIRE(Approx(ans3).margin(0.001) == 117.86);
    }

    TEST_CASE("setCentroidMean test")
    {
        std::vector<std::vector<float>> data = {
                                    {-1, 1, 0, 0, 0},
                                    {1, 1, 0, 0, 0},
                                    {-1, -1, 0, 0, 0},
                                    {1, -1, 0, 0, 0}
        };
       puffinn::KMeans clustering(2);
       clustering.padData(data);
       std::vector<KMeans::Cluster> clusters = clustering.init_centroids_random(data);
       clusters[0].members = {0, 1};
       clusters[1].members = {2, 3};

       clustering.setCentroidMean(data, clusters[0]);
       std::vector<std::vector<float>> answers = {
                            {0, 1, 0, 0, 0},
                            {0, -1, 0, 0, 0}
       };
       clustering.padData(answers);
       clustering.setCentroidMean(data, clusters[1]);
       REQUIRE(clusters[0].centroid == answers[0]);
       REQUIRE(clusters[1].centroid == answers[1]);
        
    }
    
    TEST_CASE("basic kmeans clustering 3") {

        struct TestData td;
        td.N    = 8;
        td.dims = 2;
        td.K    = 4;
        td.input = {
            {-1.0,0.01},
            {-1.0,-0.01},
            {1.0,0.01},
            {1.0,-0.01},

            {0.01,-1.00},
            {-0.01,-1.00},
            {0.01,1.00},
            {-0.01,1.00},
            
            
            
            };
        td.answer = {
            {-1.0,0.0},
            {1.0,0.0},
            {0.0,1.0},
            {0.0,-1.0}
        };
        kmeans_correctness_test(td);
        return;
        
    }
    
    TEST_CASE("basic kmeans clustering 4") {

        struct TestData td;
        td.N    = 8;
        td.dims = 2;
        td.K    = 3;
        td.input = {
            {0.0719421 , 0.2894158 },
            {0.09693447, 0.28799435},
            {0.14203255, 0.16500843},
            {0.2048598 , 0.10420303},
            {0.89096942, 0.85557702},
            {0.61737082, 0.0460024 },
            {0.31298214, 0.85340625},
            {0.69373669, 0.21694309}
            };
        td.answer = {
            {0.60197578, 0.85449163},
            {0.12894223, 0.2116554 },
            {0.65555375, 0.13147274}
        };
        kmeans_correctness_test(td);
        return;
        
    }

    TEST_CASE("Basic covariance matrix Test 1") {

        struct TestData td;
        td.input = {
            {1,2},
            {2,4},
            {3,6},
            {4,8},
            {5,10}            
            };
        KMeans kmeans(2, KMeans::euclidean, 2);
        kmeans.padData(td.input);
        kmeans.createCovarianceMatrix(td.input);
        std::vector<float> cov = kmeans.getCovarianceMatrix(); 
        std::vector<float> answer = {11, 22, 22, 44}, run_answer = {cov[0], cov[1], cov[8], cov[9]};
        REQUIRE(run_answer ==  answer);
    }
    TEST_CASE("Basic covariance matrix Test 2") {

        struct TestData td;
        td.input = {
            {1,1},
            {2,2},
            {3,3},
            {4,4},
            {5,5}            
            };
        KMeans kmeans(2, KMeans::euclidean, 2);
        kmeans.padData(td.input);
        kmeans.createCovarianceMatrix(td.input);
        std::vector<float> cov = kmeans.getCovarianceMatrix(); 
        std::vector<float> answer = {11, 11, 11, 11}, run_answer = {cov[0], cov[1], cov[8], cov[9]};
        REQUIRE(run_answer ==  answer);
    }

    TEST_CASE("distanceType test"){
        unsigned int K = 3;
        std::vector<std::vector<float>> input = {
            {-1.82445727,  0.04013046},
            { 0.28121734, -0.25688856},
            {-1.82976968,  0.02095528},
            { 0.83968045,  0.30968837},
            { 0.91359481, -0.30797411},
            {-0.81865681,  0.4669873 },
            { 0.07045335, -0.48246012},
            { 1.97495704, -0.00925621},
            {-0.64461239, -0.1055485 },
            { 1.67525065,  0.01462244},
            { 1.17554273,  1.04013046},
            { 3.28121734,  0.74311144},
            { 1.17023032,  1.02095528},
            { 3.83968045,  1.30968837},
            { 3.91359481,  0.69202589},
            { 2.18134319,  1.4669873 },
            { 3.07045335,  0.51753988},
            { 4.97495704,  0.99074379},
            { 2.35538761,  0.8944515 },
            { 4.67525065,  1.01462244},
            { 4.17554273,  0.04013046},
            { 6.28121734, -0.25688856},
            { 4.17023032,  0.02095528},
            { 6.83968045,  0.30968837},
            { 6.91359481, -0.30797411},
            { 5.18134319,  0.4669873 },
            { 6.07045335, -0.48246012},
            { 7.97495704, -0.00925621},
            { 5.35538761, -0.1055485 },
            { 7.67525065,  0.01462244}};

        // Needs more runs and iterations for the result to be stable
        KMeans km_euc(K, KMeans::euclidean, 10, 300);
        KMeans km_maha(K, KMeans::mahalanobis, 10, 300);
        km_euc.fit(input);
        km_euc.createCovarianceMatrix(input);
        km_maha.fit(input);
        double  euc1 = km_euc.totalError(input, KMeans::euclidean),
                euc2 = km_maha.totalError(input, KMeans::euclidean),
                maha1 = km_maha.totalError(input, KMeans::mahalanobis),
                maha2 = km_euc.totalError(input, KMeans::mahalanobis);
        //std::cout << "euclidean: " << euc1 << " vs " << euc2 << std::endl;
        //std::cout << "mahalanobis: " << maha1 << " vs " << maha2 << std::endl;
        REQUIRE(maha1 <= maha2);
        REQUIRE(euc1 <= euc2);
    }
#if __AVX__
    TEST_CASE("Mahadist test")
    {
        std::vector<std::vector<float>> input = {
                {-2, 3 ,4},
                {-2, 3 ,4},
                {0, 0 ,0}};


        KMeans km_maha(1, KMeans::mahalanobis, 10, 300);
        km_maha.padData(input);
        km_maha.covarianceMatrix = std::vector<float>(8*8, 1);
        double d1 = km_maha.mahaDistance(input[0], input[1]);
        double d2 = km_maha.mahaDistance_avx(input[0], input[1]);

        double d3 = km_maha.mahaDistance(input[0], input[2]);
        double d4 = km_maha.mahaDistance_avx(input[0], input[2]);
        //std::cout << d1 << ", " << d2 << "\t" << d3 << ", " << d4 << std::endl;
        REQUIRE(d1 == d2);
        REQUIRE(d3 == d4);




    }
    TEST_CASE("MahaDist avx vs no avx")
    {
        std::vector<std::vector<float>> input = {
            {-1.82445727,  0.04013046},
            { 0.28121734, -0.25688856},
            {-1.82976968,  0.02095528},
            { 0.83968045,  0.30968837},
            { 0.91359481, -0.30797411},
            {-0.81865681,  0.4669873 },
            { 0.07045335, -0.48246012},
            { 1.97495704, -0.00925621},
            {-0.64461239, -0.1055485 },
            { 1.67525065,  0.01462244},
            { 1.17554273,  1.04013046},
            { 3.28121734,  0.74311144},
            { 1.17023032,  1.02095528},
            { 3.83968045,  1.30968837},
            { 3.91359481,  0.69202589},
            { 2.18134319,  1.4669873 },
            { 3.07045335,  0.51753988},
            { 4.97495704,  0.99074379},
            { 2.35538761,  0.8944515 },
            { 4.67525065,  1.01462244},
            { 4.17554273,  0.04013046},
            { 6.28121734, -0.25688856},
            { 4.17023032,  0.02095528},
            { 6.83968045,  0.30968837},
            { 6.91359481, -0.30797411},
            { 5.18134319,  0.4669873 },
            { 6.07045335, -0.48246012},
            { 7.97495704, -0.00925621},
            { 5.35538761, -0.1055485 },
            { 7.67525065,  0.01462244}};

        KMeans km_maha(1, KMeans::mahalanobis, 10, 300);
        km_maha.padData(input);
        km_maha.createCovarianceMatrix(input);
        double total = 0;
        double total_avx = 0;
        auto cov = km_maha.getCovarianceMatrix();

        for (unsigned int i = 0; i < input.size(); i++) {
            for (unsigned int j = 0; j < input.size(); j++) {
                total_avx += km_maha.mahaDistance_avx(input[i], input[j]);
                total += km_maha.mahaDistance(input[i], input[j]);
            }
        }
        REQUIRE(Approx(total).margin(0.0001) == total_avx);

    }
#endif
}
*/