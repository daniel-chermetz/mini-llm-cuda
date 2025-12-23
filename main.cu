// nvcc inference.cu -o inference -lcublas
// nvcc inference.cu -o inference -lcublas -gencode arch=compute_75,code=sm_75
/*
nvcc \
  main.cu \
  inference.cu \
  network_globals.cu \
  -o inference \
  -lcublas \
  -gencode arch=compute_75,code=sm_75
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "network_meta.h"
#include "network_globals.h"
#include "inference.h"

int main() {
	cublasCreate(&handle);

    /* ---------------- Device info ---------------- */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\n===== GPU INFO =====\n");
    printf("Device: \"%s\"\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    // 1. VRAM (Crucial for storing weights)
    // Convert bytes to Gigabytes (GB) for readability
    double totalMemGB = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    printf("Total VRAM: %.2f GB\n", totalMemGB);

    // 2. Compute Power
    printf("Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    printf("Clock Rate: %.0f MHz\n", (double)prop.clockRate / 1000.0);

    // 3. Kernel Launch Constraints
    // This tells you the maximum size of the arrays you can process in parallel blocks
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp Size: %d\n", prop.warpSize); // Almost always 32

    // 4. Memory Speed
    printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("Memory Clock: %.0f MHz\n", (double)prop.memoryClockRate / 1000.0);
    
    printf("====================\n\n");    

    srand(0);

    size_t seqTokenIndices_size = L * sizeof(int); 
    seqTokenIndices = (int*)malloc(seqTokenIndices_size);
    for (int i = 0; i < L; i++) {
        seqTokenIndices[i] = rand() % vocabSize;
    }
    cudaMalloc((void**)&seqTokenIndices_DEVICE, seqTokenIndices_size);    
    cudaMemcpy(seqTokenIndices_DEVICE, seqTokenIndices, seqTokenIndices_size, cudaMemcpyHostToDevice);

    size_t embedding_weights_size = dim * vocabSize * sizeof(float); 
    embedding_weights = (float*)malloc(embedding_weights_size);
    for (int i = 0; i < dim * vocabSize; i++) {
        embedding_weights[i] = ((float)rand() / (float)RAND_MAX);
    }
    cudaMalloc((void**)&embedding_weights_DEVICE, embedding_weights_size);
    cudaMemcpy(embedding_weights_DEVICE, embedding_weights, embedding_weights_size, cudaMemcpyHostToDevice);

    size_t final_rms_size = dim * sizeof(float); 
    final_rms_weights = (float*)malloc(final_rms_size);
    for (int i = 0; i < dim; i++) {
        final_rms_weights[i] = ((float)rand() / (float)RAND_MAX);
    }
    cudaMalloc((void**)&final_rms_weights_DEVICE, final_rms_size);
    cudaMemcpy(final_rms_weights_DEVICE, final_rms_weights, final_rms_size, cudaMemcpyHostToDevice);

    size_t preComputedRopeTheta_size = headDim * L * sizeof(float);
    preComputedRopeTheta = (float*)malloc(preComputedRopeTheta_size);
    getPreComputedRopeTheta(preComputedRopeTheta);
    cudaMalloc((void**)&preComputedRopeTheta_DEVICE, preComputedRopeTheta_size);
    cudaMemcpy(preComputedRopeTheta_DEVICE, preComputedRopeTheta, preComputedRopeTheta_size, cudaMemcpyHostToDevice);

    size_t x_size = dim * L * sizeof(float); 
    cudaMalloc((void**)&x_DEVICE, x_size);

    for (int transformerIndex = 0; transformerIndex < transformers; transformerIndex++) { 
        TransformerWeights *currentTransformerWeights = &transformerWeights[transformerIndex];
        TransformerWeights *currentTransformerWeights_DEVICE = &transformerWeights_DEVICE[transformerIndex];

        size_t rms1_weights_size = dim * sizeof(float); 
        currentTransformerWeights->rms1_weights = (float*)malloc(rms1_weights_size); 
        for (int i = 0; i < dim; i++) {
            currentTransformerWeights->rms1_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->rms1_weights, rms1_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->rms1_weights, currentTransformerWeights->rms1_weights, rms1_weights_size, cudaMemcpyHostToDevice);

        size_t query_weights_size = dim * dim * sizeof(float); 
        currentTransformerWeights->query_weights = (float*)malloc(query_weights_size); 
        for (int i = 0; i < dim * dim; i++) {
            currentTransformerWeights->query_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->query_weights, query_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->query_weights, currentTransformerWeights->query_weights, query_weights_size, cudaMemcpyHostToDevice);

        size_t key_weights_size = dim * dim * sizeof(float);
        currentTransformerWeights->key_weights = (float*)malloc(key_weights_size);
        for (int i = 0; i < dim * dim; i++) {
            currentTransformerWeights->key_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->key_weights, key_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->key_weights, currentTransformerWeights->key_weights, key_weights_size, cudaMemcpyHostToDevice);

        size_t value_weights_size = dim * dim * sizeof(float);
        currentTransformerWeights->value_weights = (float*)malloc(value_weights_size);
        for (int i = 0; i < dim * dim; i++) {
            currentTransformerWeights->value_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->value_weights, value_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->value_weights, currentTransformerWeights->value_weights, value_weights_size, cudaMemcpyHostToDevice);

        size_t output_proj_weights_size = dim * dim * sizeof(float);
        currentTransformerWeights->output_proj_weights = (float*)malloc(output_proj_weights_size);
        for (int i = 0; i < dim * dim; i++) {
            currentTransformerWeights->output_proj_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->output_proj_weights, output_proj_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->output_proj_weights, currentTransformerWeights->output_proj_weights, output_proj_weights_size, cudaMemcpyHostToDevice);

        size_t rms2_weights_size = dim * sizeof(float);
        currentTransformerWeights->rms2_weights = (float*)malloc(rms2_weights_size);
        for (int i = 0; i < dim; i++) {
            currentTransformerWeights->rms2_weights[i] = ((float)rand() / (float)RAND_MAX);
        }        
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->rms2_weights, rms2_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->rms2_weights, currentTransformerWeights->rms2_weights, rms2_weights_size, cudaMemcpyHostToDevice);

        size_t ffn_left_weights_size = dim * dim * ffnDimMultiplier * sizeof(float);
        currentTransformerWeights->ffn_left_weights = (float*)malloc(ffn_left_weights_size);
        for (int i = 0; i < dim * dim * ffnDimMultiplier; i++) {
            currentTransformerWeights->ffn_left_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_left_weights, ffn_left_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_left_weights, currentTransformerWeights->ffn_left_weights, ffn_left_weights_size, cudaMemcpyHostToDevice);

        size_t ffn_right_1_weights_size = dim * ffnDimMultiplier * dim * sizeof(float);
        currentTransformerWeights->ffn_right_1_weights = (float*)malloc(ffn_right_1_weights_size);
        for (int i = 0; i < dim * dim * ffnDimMultiplier; i++) {
            currentTransformerWeights->ffn_right_1_weights[i] = ((float)rand() / (float)RAND_MAX);
        }   
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_right_1_weights, ffn_right_1_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_right_1_weights, currentTransformerWeights->ffn_right_1_weights, ffn_right_1_weights_size, cudaMemcpyHostToDevice);

        size_t ffn_right_2_weights_size = dim * ffnDimMultiplier * dim * sizeof(float);
        currentTransformerWeights->ffn_right_2_weights = (float*)malloc(ffn_right_2_weights_size); 
        for (int i = 0; i < dim * dim * ffnDimMultiplier; i++) {
            currentTransformerWeights->ffn_right_2_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_right_2_weights, ffn_right_2_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_right_2_weights, currentTransformerWeights->ffn_right_2_weights, ffn_right_2_weights_size, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].x_sumByCol_RMS1, L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].x_postRMS1, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].queries, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].keys, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].values, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].queriesPostRoPE, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].keysPostRoPE, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnKtQByHead, attnHeads * L * L * sizeof(float));        
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnKtQByHeadScaledMasked, attnHeads * L * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnByHead_maxByCol_softmax, attnHeads * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnByHead_sumByCol_softmax, attnHeads * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnByHead_postSoftmax, attnHeads * L * L * sizeof(float));        
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].valueScaledSoftmaxAttn, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProj, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProjPlusResidual, dim * L * sizeof(float));                
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProjPlusResidual_sumByCol_RMS2, L * sizeof(float));        
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProjPlusResidual_postRMS2, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_1_preSilu, dim * ffnDimMultiplier * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_1_postSilu, dim * ffnDimMultiplier * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_2, dim * ffnDimMultiplier * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_postHadamard, dim * ffnDimMultiplier * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_final, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffnPlusResidual, dim * L * sizeof(float));       
    }

    cudaMalloc((void**)&vocabScores_DEVICE, vocabSize * L * sizeof(float));
    cudaMalloc((void**)&vocabScores_maxByCol_softmax_DEVICE, L * sizeof(float));
    cudaMalloc((void**)&vocabScores_sumByCol_softmax_DEVICE, L * sizeof(float));
    cudaMalloc((void**)&vocabScores_postSoftmax_DEVICE, vocabSize * L * sizeof(float)); 

    runInference();
    return 0;
}