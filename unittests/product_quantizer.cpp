#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#ifdef __AVX2__
    #define AVX_TEST_CASE(name) DOCTEST_TEST_CASE(name " AVX")
#else
    #define AVX_TEST_CASE(name) DOCTEST_TEST_CASE(name)
#endif

#include "spq/product_quantizer.hpp"
#include "spq/loss/euclidean_loss.hpp"
#include "spq/loss/product_loss.hpp"
#include "spq/loss/weighted_loss.hpp"

#include <set>
#include <functional>

namespace product_quantizer{

    using namespace spq;
    using namespace spq::util;

    void show(const data_t &d)
    {
        std::cout << "Showing" << std::endl;
        for (const auto &vec : d)
        {
            for (const float &val : vec)
            {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }


    // runs in the length of the smallest vector
    double l2Distance(const std::vector<float> &vec1, const std::vector<float> &vec2)
    {
        assert(vec1.size() == vec2.size());
        double distance = 0.0;
        for (size_t i = 0; i < vec1.size(); i++)
        {
            distance += std::pow(vec1[i] - vec2[i], 2);
        }
        return distance;
    }

    template<class TLoss>
    //Distance from PQCode to actual vector using precomputed PQCodes
    double calcQauntizationError(const data_t &data, ProductQuantizer<TLoss> &quantizer)
    {
        std::vector<data_t> codebook = quantizer.getCodebook();

        float sum = 0.0;

        for (size_t i = 0; i < data.size(); i++)
        {
            std::vector<uint8_t> code = quantizer.getCode(i);
            REQUIRE(code.size() == codebook.size());

            size_t acc_size = 0;
            for(size_t m = 0; m < code.size(); m++)
            {
                const std::vector<float> &cen = codebook[m][code[m]];
                size_t subspace_size = cen.size();

                const float* start = &data[i][acc_size];
                const float* end =   start + subspace_size;
                sum += l2Distance(std::vector<float>(start,end), cen);

                acc_size += subspace_size;
            }
            CHECK(acc_size == data[i].size());
        }
        return sum;

    }

    template<class TLoss>
    void errorTest(const data_t &data, const size_t m, const size_t k, std::function<bool(double)> condition)
    {
        ProductQuantizer<TLoss> quantizer(data, m, k);
        double q_err = calcQauntizationError<TLoss>(data, quantizer);
        REQUIRE(condition(q_err));
    }

    template<class TLoss>
    void getCodebookTest()
    {
        unsigned int N = 8, dims = 4, m = 4, k  = 8;
        data_t data = {     {-4.0, 1.0, -8.0, 1.0},
                            {-3.0, 2.0, -7.0, 2.0},
                            {-2.0, 3.0, -6.0, 3.0},
                            {-1.0, 4.0, -5.0, 4.0},
                            {1.0 , 5.0, -4.0, 5.0},
                            {2.0 , 6.0, -3.0, 6.0},
                            {3.0 , 7.0, -2.0, 7.0},
                            {4.0 , 8.0, -1.0, 8.0}};
        SUBCASE("M=1,K=1")
        {
            size_t  m = 1,
                    k = 1;
            ProductQuantizer<TLoss> quantizer(data, m, k);
            std::vector<data_t> cb = quantizer.getCodebook();
            REQUIRE(cb.size() == m);
            for (const data_t &sub_cb : cb)
            {
                REQUIRE(sub_cb.size() == k);
                for (const auto &vec : sub_cb)
                {
                    REQUIRE(vec.size() == 4);
                }
            }
        }

        SUBCASE("M=4,K=1")
        {
            size_t  m = 4,
                    k = 1;
            ProductQuantizer<TLoss> quantizer(data, m, k);
            std::vector<data_t> cb = quantizer.getCodebook();
            REQUIRE(cb.size() == m);
            for (const data_t &sub_cb : cb)
            {
                REQUIRE(sub_cb.size() == k);
                for (const auto &vec : sub_cb)
                {
                    REQUIRE(vec.size() == 1);
                }
            }
        }

        SUBCASE("M=1,K=8")
        {
            size_t  m = 1,
                    k = 1;
            ProductQuantizer<TLoss> quantizer(data, m, k);
            std::vector<data_t> cb = quantizer.getCodebook();
            REQUIRE(cb.size() == m);
            for (const data_t &sub_cb : cb)
            {
                REQUIRE(sub_cb.size() == k);
                for (const auto &vec : sub_cb)
                {
                    REQUIRE(vec.size() == 4);
                }
            }
        }

    }

    AVX_TEST_CASE("ProductQuantizer getCodebook EuclideanLoss")
    {
        getCodebookTest<EuclideanLoss>();
    }

    AVX_TEST_CASE("ProductQuantizer getCodebook ProductLoss")
    {
        getCodebookTest<ProductLoss>();
    }

    AVX_TEST_CASE("ProductQuantizer No error")
    {
        data_t data = {     {-4.0, 1.0, -8.0, 1.0},
                            {-3.0, 2.0, -7.0, 2.0},
                            {-2.0, 3.0, -6.0, 3.0},
                            {-1.0, 4.0, -5.0, 4.0},
                            {1.0 , 5.0, -4.0, 5.0},
                            {2.0 , 6.0, -3.0, 6.0},
                            {3.0 , 7.0, -2.0, -7.0},
                            {4.0 , 8.0, -1.0, -8.0}};
        size_t m = 4, k = 8;
        std::function<bool(double)> is_zero = [](double err) {return err == 0;};
        SUBCASE("EuclideanLoss")
        {
            errorTest<EuclideanLoss>(data, m, k, is_zero);
        }
        SUBCASE("ProductLoss")
        {
            errorTest<ProductLoss>(data, m, k, is_zero);
        }
        //SUBCASE("WeightedProductLoss")
        //{
            //errorTest<WeightedProductLoss>();
        //}
    }

    AVX_TEST_CASE("ProductQuantizer No error 2") 
    {
        data_t  data = {{-4.0, 1.0, -8.0, 1.0},
                        {-4.0, 1.0,  8.0,-1.0},
                        { 4.0,-1.0, -8.0, 1.0},
                        { 4.0,-1.0,  8.0,-1.0}};
        size_t m = 2, k = 2;
        std::function<bool(double)> is_zero = [](double err) {return err == 0;};
        SUBCASE("EuclideanLoss")
        {
            errorTest<EuclideanLoss>(data, m, k, is_zero);
        }
        SUBCASE("ProductLoss")
        {
            errorTest<ProductLoss>(data, m, k, is_zero);
        }
    }

    AVX_TEST_CASE("ProductQuantizer Some error") 
    {
        data_t data = { {-4.0, 1.0, -8.0, 1.0},
                        {-4.0, 1.0,  8.0,-1.0},
                        { 4.0,-1.0, -8.0, 1.0},
                        { 4.0,-1.0,  8.0,-1.0},
                        { 1.0, 0.0,  1.0, 0.0},
                        { 0.0,-1.0,  0.0,-1.0}};
        size_t m = 4, k = 2;

        std::function<bool(double)> gt_zero = [](double err) {return err > 0;};
        SUBCASE("EuclideanLoss")
        {
            errorTest<EuclideanLoss>(data, m, k, gt_zero);
        }
        SUBCASE("ProductLoss")
        {
            errorTest<ProductLoss>(data, m, k, gt_zero);
        }
    }
    
    AVX_TEST_CASE("ProductQuantizer getEstimator")
    {
        std::cout << "Enter TEST" << std::endl;
        data_t data = {     {-4.0, 1.0, -8.0, 1.0},
                            {-3.0, 2.0, -7.0, 2.0},
                            {-2.0, 3.0, -6.0, 3.0},
                            {-1.0, 4.0, -5.0, 4.0},
                            {1.0 , 5.0, -4.0, 5.0},
                            {2.0 , 6.0, -3.0, 6.0},
                            {3.0 , 7.0, -2.0, -7.0},
                            {4.0 , 8.0, -1.0, -8.0}};
        size_t m = 4, k = 8;
        std::vector<float> query = {1.0, 1.0, 1.0, 1.0};
        ProductQuantizer<EuclideanLoss> quantizer(data, m, k);
        Estimator est = quantizer.getEstimator(query);
        // The following code should not cause a segmentation fault
        double sum_dist = 0;

        SUBCASE("M=0")
        {
            for (size_t k_i = 0; k_i < k; k_i++)
            {
                sum_dist += est.distances[k_i];
            }
            CHECK(sum_dist == 0);
        }
        SUBCASE("M=1")
        {
            for (size_t k_i = 0; k_i < k; k_i++)
            {
                sum_dist += est.distances[k + k_i];
            }
            CHECK(sum_dist == 36.0);
        }
        SUBCASE("M=2")
        {
            for (size_t k_i = 0; k_i < k; k_i++)
            {
                sum_dist += est.distances[2*k + k_i];
            }
            CHECK(sum_dist == -36.0);
        }
        SUBCASE("M=3")
        {
            for (size_t k_i = 0; k_i < k; k_i++)
            {
                sum_dist += est.distances[3*k + k_i];
            }
            CHECK(sum_dist == 6.0);
        }

        SUBCASE("ESTIMATIONS")
        {
            CHECK(est.estimate(0) == -10.0);
            CHECK(est.estimate(1) == -6.0);
            CHECK(est.estimate(2) == -2.0);
            CHECK(est.estimate(3) == 2.0);
            CHECK(est.estimate(4) == 7.0);
            CHECK(est.estimate(5) == 11.0);
            CHECK(est.estimate(6) == 1.0);
            CHECK(est.estimate(7) == 3.0);
        }

    }
/*
    AVX_TEST_CASE("precomp test") {
        unsigned int N = 8, dims = 4, m = 4, k  = 4;
        std::vector<float>  data[N] = {
                                        {-4.0, 1.0, -8.0, 1.0},
                                        {-3.0, 2.0, -7.0, 2.0},
                                        {-2.0, 3.0, -6.0, 3.0},
                                        {-1.0, 4.0, -5.0, 4.0},
                                        {1.0 , 5.0, -4.0, 5.0},
                                        {2.0 , 6.0, -3.0, 6.0},
                                        {3.0 , 7.0, -2.0, 7.0},
                                        {4.0 , 8.0, -1.0, 8.0}};

        Dataset<UnitVectorFormat> dataset(dims, N);
        for (auto entry: data){
            dataset.insert(entry);
        }
        PQFilter pq1(dataset, m,k);
        pq1.rebuild();
        REQUIRE(pq1.totalQuantizationError() == pq1.totalQuantizationError_simple());
        
        alignas(32) int16_t tmp[pq1.getPadSize()];
        pq1.createPaddedQueryPoint(dataset[1], tmp);

        REQUIRE(pq1.asymmetricDistanceComputation_simple(dataset[0], dataset[1]) == pq1.asymmetricDistanceComputation(0, dataset[1]));
        REQUIRE(pq1.symmetricDistanceComputation_simple(dataset[2], dataset[3]) == pq1.symmetricDistanceComputation(2, dataset[3]));
    }
    
    /*
    AVX_TEST_CASE("AsymmetricFast Test"){
        std::vector<float> input[30] = {
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
        
        unsigned int N = 30, dims = 2, m = 1, k  = 4;
        Dataset<UnitVectorFormat> dataset(dims, N);
        for (auto entry: input){
            dataset.insert(entry);
        }
        PQFilter pq1(dataset, m,k);
        pq1.rebuild();
        bool isSame = true;
        for(unsigned int i = 0; i < N; i++){
            for( unsigned int j = 0; j < N; j++){
                alignas(32) int16_t tmp[pq1.getPadSize()];
                pq1.createPaddedQueryPoint(dataset[j], tmp);
                float hue = pq1.asymmetricDistanceComputation_simple(dataset[i], dataset[j]);
                float hue1 = pq1.asymmetricDistanceComputation_avx(i, tmp);
                isSame = (hue == hue1);
            }          
        }
        REQUIRE(isSame);
        
    }
    */
    


}