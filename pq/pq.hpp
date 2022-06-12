#pragma once

#include "kmeans.hpp"
#include "loss.hpp"

#include <vector>
#include <assert.h>
#include <cfloat>
#include <iostream>
#include <inttypes.h>


namespace PQ{

    template< typename TLoss = EuclideanLoss>
    class PQFilter{
    private:
        unsigned int M, K, dims;

        const unsigned int SAMPLE_SIZE = 100000u;

        // codebook that m*k centriods (each centroid of size dims)
        data_t *codebook;

        // lookup table of distances between a query and all codewords in the codebook 
        float *queryDistances;

        // Product Quantization codes of all points in the input data (N x M array)
        uint8_t *pqCodes;

        data_t dataset;
        //meta information about the subspaces to avoid recomputation 
        std::vector<unsigned int> subspaceSizes, offsets = {0}, subspaceSizesStored;
        bool is_build = false;
    public:

        ///Builds short codes for vectors using Product Quantization by projecting every subspace down to neigherst kmeans cluster
        ///Uses these shortcodes for fast estimation of inner product between vectors  
        ///
        ///@param dataset Reference to the dataset. This enables instantiation of class before data is known
        ///@param m Number of subspaces to split the vectors in, distributes sizes as uniformely as possible
        ///@param k Number of clusters for each subspace, build using kmeans
        ///@param mode The distance meassure that kmeans will use. Currently supportes -> ("euclidian, "mahalanobis")   
        PQFilter(data_t dataset, unsigned int m = 8, unsigned int k = 256)
        :M(m),
        K(k),
        dims(dataset[0].size()),
        dataset(dataset)
        {
            codebook = data_t[M];
            std::cout << "constructing PQFilter with k=" << K << " m=" << M << std::endl;
            assert(M%2 == 0);
        }

        ~PQFilter(){
            delete[] queryDistances;
        }

        //Builds the codebook and computes the inter-centroid-distances
        //Should be called every time the dataset significantly changes a d atleast once before querying
        void build()
        {

            if (dataset.get_size() == 0){
                is_build = false;
                std::fill_n(queryDistances, K*M, 0.0f);
                return;
            } 

            std::cout << "rebuilding pq_filter" << std::endl;
            pqCodes.resize(dataset.get_size());
            createCodebook();
            // createDistanceTable();
            //bootThreshold = bootStrapThreshold(100u, 5000u, 10u);
            //std::cout << "this is the boot threshold: " << bootThreshold << std::endl;
        }

        void createDistanceTable() {

        }


    #if __AVX2__        
        //Precompute all distance between centroids using AVX2 instructions
        void createDistanceTable(){
            for(unsigned int m = 0; m < M; m++){
                std::vector<std::vector<int16_t>> subspaceDists;
                for(unsigned int k1 = 0; k1 < K; k1++){
                    std::vector<int16_t> dists;
                    for(int k2 = 0; k2 < K; k2++){
                        dists.push_back(dot_product_i16_avx2(codebook[m][k1], codebook[m][k2], subspaceSizes[m]));
                    }
                    subspaceDists.push_back(dists);
                }
                centroidDistances.push_back(subspaceDists);
            }

        }

    #else        
        //Precompute all distance between centroids 
        void createDistanceTable(){
            for(unsigned int m = 0; m < M; m++){
                std::vector<std::vector<int16_t>> subspaceDists;
                for(int k1 = 0; k1 < K; k1++){
                    std::vector<int16_t> dists;
                    for(int k2 = 0; k2 < K; k2++){
                        dists.push_back(dot_product_i16_simple(codebook[m][k1], codebook[m][k2], subspaceSizes[m]));
                    }
                    subspaceDists.push_back(dists);
                }
                centroidDistances.push_back(subspaceDists);
            }

        }
    #endif

        //builds a dataset where each vector only contains the mth chunk
        std::vector<std::vector<float>> getSubspace(unsigned int m, const bool sample = true) {
            // If dataset is large use at most 100.000 points for PQ
            unsigned int N = dataset.get_size();
            if (sample)
                N = std::min(N, SAMPLE_SIZE);

            std::vector<unsigned int> s_idcs = random_sample(N, dataset.get_size());
            std::sort(s_idcs.begin(), s_idcs.end()); // Such that order is maintained when the sample is the whole dataset
            //std::cout << "first sampel idx: " << s_idcs[0] << std::endl;

            std::vector<std::vector<float>> subspace(N, std::vector<float>(subspaceSizes[m]));
            unsigned int subspace_idx = 0;
            for (unsigned int idx : s_idcs) {
                UnitVectorFormat::Type *start = dataset[idx] + offsets[m];
                for (unsigned int d = 0; d < subspaceSizes[m]; d++) {
                    subspace[subspace_idx][d] = UnitVectorFormat::from_16bit_fixed_point(*(start + d));
                }
                subspace_idx++;
            }
            return subspace;
        }


        //Runs kmeans for all m subspaces and stores the centroids in codebooks
        void createCodebook()
        {
            unsigned int k = std::min(K, dataset.get_size());
            //used to keep track of where subspace begins
            KMeans kmeans(k, MODE); 
            for(unsigned int m = 0; m < M ; m++)
            {
                std::cout << "creating codebook for subspace " << m << std::endl;
                //RunKmeans for the given subspace
                //gb_labels for this subspace will be the mth index of the PQcodes
                
                std::vector<std::vector<float>> subspace = getSubspace(m);
                kmeans.fit(subspace);
                std::vector<std::vector<float>> centroids  = kmeans.getAllCentroids();
                subspace = getSubspace(m,false);
            #ifdef __AVX2__
                kmeans.padData(subspace);
            #endif
                kmeans.assignToClusters(subspace, kmeans.gb_clusters);
                
                for (unsigned int ci = 0; ci < k; ci++ ) {
                    std::vector<unsigned int> cm = kmeans.getGBMembers(ci);
                    for (auto &mem : cm) {
                        pqCodes[mem].push_back((uint8_t)ci);
                    }
                }

                // Convert back to UnitVectorFormat and store in codebook
                codebook.push_back(Dataset<UnitVectorFormat>(subspaceSizes[m], dataset.get_size()));
                for (unsigned int i = 0; i < k; i++) {
                    UnitVectorFormat::Type *c_p = codebook[m][i];
                    float *vec_p = &centroids[i][0];
                    for (unsigned int d = 0; d < subspaceSizes[m]; d++) {
                        *c_p++ = UnitVectorFormat::to_16bit_fixed_point(*vec_p++);
                    }
                }
                //Sizes of the padded subspaces r
                subspaceSizesStored.push_back(codebook[m].get_description().storage_len);
            }
            is_build = true;

        }



        //Naive way of getting PQCode will be usefull if we decide to use samples to construct centroids
        std::vector<uint8_t> getPQCode(typename UnitVectorFormat::Type* vec) const {
            std::vector<uint8_t> pqCode;
            for(unsigned int m = 0; m < M; m++){
                float minDistance = FLT_MAX;
                uint8_t quantization = 0u;
                for(int k = 0; k < K; k++){
                    float d = UnitVectorFormat::distance(vec+offsets[m], codebook[m][k], subspaceSizes[m]);
                    if(d < minDistance){
                        minDistance = d;
                        quantization = k;    
                    }
                }
                pqCode.push_back(quantization);
            }
            return pqCode;
        }

        void precomp_query_to_centroids(typename UnitVectorFormat::Type* y) const {
            if (!is_build) return;
            alignas(32) int16_t paddedY[getPadSize()];
            createPaddedQueryPoint(y, paddedY);
            int16_t *a_p = &paddedY[0];
            float * p = queryDistances;
            const unsigned int *size_p = &subspaceSizesStored[0];
            for(unsigned int m = 0; m < M; m++){
                for(unsigned int k = 0; k < K; k++){
                    *p++ = UnitVectorFormat::from_16bit_fixed_point(dot_product_i16_avx2(codebook[m][k], a_p, *size_p));
                }
                a_p += *size_p++;
            }
        }

        float estimatedInnerProduct(unsigned int xi) const {
            float sum = 0.0f; // About -0.05 in fixpoint16 format

            const uint8_t *p = &pqCodes[xi][0];
            for(unsigned int var = 0; var < LIM; var += 4*K, p+=4){
                sum += queryDistances[var + *p];
                sum += queryDistances[var +   K + *(p+1)];
                sum += queryDistances[var + 2*K + *(p+2)];
                sum += queryDistances[var + 3*K + *(p+3)];
            }
            return sum; 
        }
        

        //Distance from PQCode to actual vector
        float quantizationError_simple(typename UnitVectorFormat::Type* vec) const {
            float sum = 0;
            std::vector<uint8_t> pqCode = getPQCode(vec);
            int centroidID;
            for(unsigned int m = 0; m < M; m++){
                centroidID = pqCode[m];
                sum += UnitVectorFormat::distance(vec + offsets[m], codebook[m][centroidID], subspaceSizes[m]);
            }
            
            return sum;
        }
        //Function overload to allow idx calls if vector is in the dataset
        float quantizationError_simple(unsigned int idx) const {
            return quantizationError_simple(dataset[idx]);
        }
        
        //Distance from PQCode to actual vector using precomputed PQCodes
        float quantizationError(unsigned int vec_i) const {
            float sum = 0;
            std::vector<uint8_t> pqCode = pqCodes[vec_i];
            int centroidID;
            for(unsigned int m = 0; m < M; m++){
                centroidID = pqCode[m];
                sum += UnitVectorFormat::distance(dataset[vec_i] + offsets[m], codebook[m][centroidID], subspaceSizes[m]);
            }
            return sum;
        }

        float totalQuantizationError_simple() const {
            float sum = 0;
            for(unsigned int i  = 0; i < dataset.get_size(); i++){
                sum += quantizationError_simple(dataset[i]);
            }
            return sum;
        }

        float totalQuantizationError() const {
            float sum = 0;
            for(unsigned int i  = 0; i < dataset.get_size(); i++){
                sum += quantizationError(i);
            }
            return sum;
        }

        //symmetric distance estimation, PQcodes computed at runtime
        float symmetricDistanceComputation_simple(typename UnitVectorFormat::Type* x, typename UnitVectorFormat::Type* y)const {
            float sum = 0;
            //quantize x and y
            std::vector<uint8_t> px = getPQCode(x), py = getPQCode(y);
            //approximate distance by product quantization (precomputed centroid distances required)
            for(unsigned int m = 0; m < M; m++){
                sum += centroidDistances[m][px[m]][py[m]];
            }
            return sum;
        }

        //symmetric distance estimation with precomputed PQcode for x
        int16_t symmetricDistanceComputation(unsigned int xi, typename UnitVectorFormat::Type* y) const {
            int16_t sum = 0;
            //quantize x and y
            std::vector<uint8_t> px = pqCodes[xi], py = getPQCode(y);
            //approximate distance by product quantization (precomputed centroid distances required)
            for(unsigned int m = 0; m < M; m++){
                sum += centroidDistances[m][px[m]][py[m]];
            }
            return sum;
        }

        //used to allocate enough memory to pad query point
        unsigned int getPadSize() const {
            unsigned int ans = 0;
            for(unsigned int m = 0; m < M; m++){
                ans += codebook[m].get_description().storage_len;
            }
            return ans;
        }        

        //builds a vector padded to align with each subspace at memory pointed to by "a"
        void createPaddedQueryPoint(typename UnitVectorFormat::Type* y, int16_t *a) const {
            for(unsigned int m = 0; m < M; m++){
                for(unsigned int i = 0; i < subspaceSizes[m]; i++){
                    *a++ = *y++;
                }
                unsigned int padd = (16 - (subspaceSizes[m] % 16))%16;
                a = std::fill_n(a, padd, 0);
            }
        }  
        
        //asymmetric distance estimation, PQCode computed at runtime
        int16_t asymmetricDistanceComputation_simple(typename UnitVectorFormat::Type* x, typename UnitVectorFormat::Type* y) const {
            int16_t sum = 0;
            std::vector<uint8_t> px = getPQCode(x);
            for(unsigned int m = 0; m <M; m++){
                sum += dot_product_i16_simple(y + offsets[m], codebook[m][px[m]], subspaceSizes[m]);
            }
            return sum;
        }

        #if __AVX2__
        // Fastest version of asymmetric distance computation using avx2
        ///@param xi index of vector in the dataset
        ///@param y pointer to start of UnitVector (padded for each subspace)
        
        int16_t asymmetricDistanceComputation_avx(unsigned int xi, typename UnitVectorFormat::Type* y) const {
            int16_t sum = 0;
            const uint8_t *px_p = &pqCodes[xi][0];
            const unsigned int *size_p = &subspaceSizesStored[0];
            const Dataset<UnitVectorFormat> *cb_p = &codebook[0];
            for(unsigned int m = 0; m <M; m++){
                sum += dot_product_i16_avx2(y, (*cb_p++)[*px_p++], *size_p);
                y+= *size_p++;
            }
            return sum;
        }
        
        //ovearlead function so you can query with pointer to vector instead of idx in dataset
        int16_t asymmetricDistanceComputation_avx(typename UnitVectorFormat::Type* x, typename UnitVectorFormat::Type* y) const {
            int16_t sum = 0;
            const unsigned int *size_p = &subspaceSizesStored[0];
            for(unsigned int m = 0; m <M; m++){
                sum += dot_product_i16_avx2(y, x, *size_p);
                x+= *size_p;
                y+= *size_p++;   
            }
            return sum;
        }

        #else
        int16_t asymmetricDistanceComputation_avx(unsigned int xi, typename UnitVectorFormat::Type* y) const {
            std::cerr << "assymetric avx failed -> no AVX2 found" << std::endl;
            return asymmetricDistanceComputation(xi, y);
        }

        #endif

        //asymmetric distance estimation using precomputed PQcodes
        ///@param xi index of vector in the dataset
        ///@param y pointer to start of UnitVector (not padded for each subspace)
        int16_t asymmetricDistanceComputation(unsigned int xi, typename UnitVectorFormat::Type* y) const{
            int16_t sum = 0;
            const uint8_t *px_p = &pqCodes[xi][0];
            const unsigned int *size_p = &subspaceSizes[0];
            const Dataset<UnitVectorFormat> *cb_p = &codebook[0];
            for(unsigned int m = 0; m <M; m++){
                sum += dot_product_i16_simple(y, (*cb_p++)[*px_p++], *size_p);
                y += *size_p++;
            }
            return sum;
        }

        float bootStrapThreshold(unsigned int nruns = 150, unsigned int sizeOfRun = 5000, unsigned int topK = 25){
            std::vector<unsigned int> q_idcs = random_sample(std::min(nruns,dataset.get_size()), dataset.get_size());
            float sumOfThresholds = 0.0;
            for (unsigned int bootQuery : q_idcs)
            {
                MaxBuffer maxbuffer(topK);
                std::vector<unsigned int> d_idcs = random_sample(std::min(sizeOfRun, dataset.get_size()), dataset.get_size());
                for (unsigned int idx : d_idcs) {
                    int16_t distance = dot_product_i16(dataset[bootQuery], dataset[idx], dataset.get_description().storage_len);
                    maxbuffer.insert(idx, UnitVectorFormat::from_16bit_fixed_point(distance));
                }
                auto best = maxbuffer.best_indices();
                precomp_query_to_centroids(dataset[bootQuery]);
                float min_est = 1.0f; // maximum value of inner product
                //std::cout << "init min_est" << UnitVectorFormat::from_16bit_fixed_point(min_est) << std::endl;
                for (auto &idx : best) {
                    min_est = std::min(min_est, estimatedInnerProduct(idx));
            }
                sumOfThresholds += min_est;
            }
            return sumOfThresholds/(nruns);            
        }

        //Functions below are just debugging tools and old code that might be useful down the road
        /*
        std::vector<float> getCentroid(unsigned int mID, unsigned int kID){
            return std::vector<float>(codebook[mID][kID], codebook[mID][kID]+ subspaceSizes[mID]);
        }

        void showCodebook(){
            for(unsigned int m = 0; m < M; m++){
                std::cout << "subspace: " << m << std::endl;
                for(int k = 0; k < K; k++){
                    std::cout << "cluster: "<< k << std::endl;
                    for(unsigned int l = 0; l < codebook[m].get_description().storage_len; l++){
                        std::cout << "\t" <<  UnitVectorFormat::from_16bit_fixed_point(*(codebook[m][k] + l)) << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            } 
        }

        void showPQCodes(){
            std::cout << "PQCODOES1: " << std::endl;
            for(unsigned int i = 0; i < dataset.get_size(); i++){
                for(uint8_t val: getPQCode(dataset[i])){
                    std::cout << (unsigned int) val << " ";
                }
            std::cout << std::endl;
            }
        }

        

        void showSubSizes(){
            for(auto a: subspaceSizes){
                std::cout << a << " ";
            }
            std::cout << std::endl;
        }

        //constructor is depricated
        
        PQFilter(Dataset<UnitVectorFormat> &dataset, std::vector<unsigned int> subs, unsigned int k = 256, KMeans::distanceType mode = KMeans::euclidean)
        :M(subs.size()),
        dims(dims),
        K(k),
        MODE(mode),
        dataset(dataset)
        {   

            pqCodes.resize(dataset.get_size());
            setSubspaceSizes(subs);
            createCodebook();
            createDistanceTable();
        }
        //helper function for depricated constructor
        void setSubspaceSizes(std::vector<unsigned int> subs){
            subspaceSizes = subs;
            offsets.clear();
            for(unsigned int i = 1; i < M; i++) offsets[i] = subs[i-1] + offsets[i-1]; 
        }
        */
    };
}
