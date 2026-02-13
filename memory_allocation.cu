#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "network_meta.h"
#include "network_globals.h"
#include "inference.h"
#include "training.h"
#include "optimizer.h"
#include "training_orchestrator.h"

// ============================================================================
// BATCH GRADIENT ACCUMULATION MEMORY
// ============================================================================

static void allocateBatchGradAccumulationMemory() {
    cudaMalloc((void**)&gradientAccumulation_embedding_weights, dim * vocabSize * sizeof(float));
    cudaMalloc((void**)&gradientAccumulation_final_RMS_gamma_weights, dim * sizeof(float));

    for (int transformerIndex = 0; transformerIndex < transformers; transformerIndex++) {
        cudaMalloc((void**)&gradientAccumulation[transformerIndex].ffn_left_weights, dim * ffnDim * sizeof(float));
        cudaMalloc((void**)&gradientAccumulation[transformerIndex].ffn_right_1_weights, ffnDim * dim * sizeof(float));
        cudaMalloc((void**)&gradientAccumulation[transformerIndex].ffn_right_2_weights, ffnDim * dim * sizeof(float));
        cudaMalloc((void**)&gradientAccumulation[transformerIndex].rms2_gamma_weights, dim * sizeof(float));
        cudaMalloc((void**)&gradientAccumulation[transformerIndex].output_proj_weights, dim * dim * sizeof(float));
        cudaMalloc((void**)&gradientAccumulation[transformerIndex].value_weights, dim * dim * sizeof(float));
        cudaMalloc((void**)&gradientAccumulation[transformerIndex].query_weights, dim * dim * sizeof(float));
        cudaMalloc((void**)&gradientAccumulation[transformerIndex].key_weights, dim * dim * sizeof(float));
        cudaMalloc((void**)&gradientAccumulation[transformerIndex].rms1_gamma_weights, dim * sizeof(float));
    }
}

// ============================================================================
// FAST EMA (FIRST MOMENT) MEMORY
// ============================================================================

static void allocateFastMomentumMemory() {
    cudaMalloc((void**)&fastEMA_embedding_weights, dim * vocabSize * sizeof(float));
    cudaMemset(fastEMA_embedding_weights, 0, dim * vocabSize * sizeof(float));

    cudaMalloc((void**)&fastEMA_final_RMS_gamma_weights, dim * sizeof(float));
    cudaMemset(fastEMA_final_RMS_gamma_weights, 0, dim * sizeof(float));

    for (int transformerIndex = 0; transformerIndex < transformers; transformerIndex++) {
        cudaMalloc((void**)&fastEMA[transformerIndex].ffn_left_weights, dim * ffnDim * sizeof(float));
        cudaMemset(fastEMA[transformerIndex].ffn_left_weights, 0, dim * ffnDim * sizeof(float));

        cudaMalloc((void**)&fastEMA[transformerIndex].ffn_right_1_weights, ffnDim * dim * sizeof(float));
        cudaMemset(fastEMA[transformerIndex].ffn_right_1_weights, 0, ffnDim * dim * sizeof(float));
        
        cudaMalloc((void**)&fastEMA[transformerIndex].ffn_right_2_weights, ffnDim * dim * sizeof(float));
        cudaMemset(fastEMA[transformerIndex].ffn_right_2_weights, 0, ffnDim * dim * sizeof(float));

        cudaMalloc((void**)&fastEMA[transformerIndex].rms2_gamma_weights, dim * sizeof(float));
        cudaMemset(fastEMA[transformerIndex].rms2_gamma_weights, 0, dim * sizeof(float));

        cudaMalloc((void**)&fastEMA[transformerIndex].output_proj_weights, dim * dim * sizeof(float));
        cudaMemset(fastEMA[transformerIndex].output_proj_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&fastEMA[transformerIndex].value_weights, dim * dim * sizeof(float));
        cudaMemset(fastEMA[transformerIndex].value_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&fastEMA[transformerIndex].query_weights, dim * dim * sizeof(float));
        cudaMemset(fastEMA[transformerIndex].query_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&fastEMA[transformerIndex].key_weights, dim * dim * sizeof(float));
        cudaMemset(fastEMA[transformerIndex].key_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&fastEMA[transformerIndex].rms1_gamma_weights, dim * sizeof(float));
        cudaMemset(fastEMA[transformerIndex].rms1_gamma_weights, 0, dim * sizeof(float));        
    }
}

// ============================================================================
// SLOW EMA MEMORY
// ============================================================================

static void allocateSlowMomentumMemory() {
    cudaMalloc((void**)&slowEMA_embedding_weights, dim * vocabSize * sizeof(float));
    cudaMemset(slowEMA_embedding_weights, 0, dim * vocabSize * sizeof(float));

    cudaMalloc((void**)&slowEMA_final_RMS_gamma_weights, dim * sizeof(float));
    cudaMemset(slowEMA_final_RMS_gamma_weights, 0, dim * sizeof(float));

    for (int transformerIndex = 0; transformerIndex < transformers; transformerIndex++) {
        cudaMalloc((void**)&slowEMA[transformerIndex].ffn_left_weights, dim * ffnDim * sizeof(float));
        cudaMemset(slowEMA[transformerIndex].ffn_left_weights, 0, dim * ffnDim * sizeof(float));

        cudaMalloc((void**)&slowEMA[transformerIndex].ffn_right_1_weights, ffnDim * dim * sizeof(float));
        cudaMemset(slowEMA[transformerIndex].ffn_right_1_weights, 0, ffnDim * dim * sizeof(float));
        
        cudaMalloc((void**)&slowEMA[transformerIndex].ffn_right_2_weights, ffnDim * dim * sizeof(float));
        cudaMemset(slowEMA[transformerIndex].ffn_right_2_weights, 0, ffnDim * dim * sizeof(float));

        cudaMalloc((void**)&slowEMA[transformerIndex].rms2_gamma_weights, dim * sizeof(float));
        cudaMemset(slowEMA[transformerIndex].rms2_gamma_weights, 0, dim * sizeof(float));

        cudaMalloc((void**)&slowEMA[transformerIndex].output_proj_weights, dim * dim * sizeof(float));
        cudaMemset(slowEMA[transformerIndex].output_proj_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&slowEMA[transformerIndex].value_weights, dim * dim * sizeof(float));
        cudaMemset(slowEMA[transformerIndex].value_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&slowEMA[transformerIndex].query_weights, dim * dim * sizeof(float));
        cudaMemset(slowEMA[transformerIndex].query_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&slowEMA[transformerIndex].key_weights, dim * dim * sizeof(float));
        cudaMemset(slowEMA[transformerIndex].key_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&slowEMA[transformerIndex].rms1_gamma_weights, dim * sizeof(float));
        cudaMemset(slowEMA[transformerIndex].rms1_gamma_weights, 0, dim * sizeof(float));        
    }
}

// ============================================================================
// VARIANCE (SECOND MOMENT) MEMORY
// ============================================================================

static void allocateVarianceMemory() {
    cudaMalloc((void**)&variance_embedding_weights, dim * vocabSize * sizeof(float));
    cudaMemset(variance_embedding_weights, 0, dim * vocabSize * sizeof(float));

    cudaMalloc((void**)&variance_final_RMS_gamma_weights, dim * sizeof(float));
    cudaMemset(variance_final_RMS_gamma_weights, 0, dim * sizeof(float));

    for (int transformerIndex = 0; transformerIndex < transformers; transformerIndex++) {
        cudaMalloc((void**)&variance[transformerIndex].ffn_left_weights, dim * ffnDim * sizeof(float));
        cudaMemset(variance[transformerIndex].ffn_left_weights, 0, dim * ffnDim * sizeof(float));

        cudaMalloc((void**)&variance[transformerIndex].ffn_right_1_weights, ffnDim * dim * sizeof(float));
        cudaMemset(variance[transformerIndex].ffn_right_1_weights, 0, ffnDim * dim * sizeof(float));
        
        cudaMalloc((void**)&variance[transformerIndex].ffn_right_2_weights, ffnDim * dim * sizeof(float));
        cudaMemset(variance[transformerIndex].ffn_right_2_weights, 0, ffnDim * dim * sizeof(float));

        cudaMalloc((void**)&variance[transformerIndex].rms2_gamma_weights, dim * sizeof(float));
        cudaMemset(variance[transformerIndex].rms2_gamma_weights, 0, dim * sizeof(float));

        cudaMalloc((void**)&variance[transformerIndex].output_proj_weights, dim * dim * sizeof(float));
        cudaMemset(variance[transformerIndex].output_proj_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&variance[transformerIndex].value_weights, dim * dim * sizeof(float));
        cudaMemset(variance[transformerIndex].value_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&variance[transformerIndex].query_weights, dim * dim * sizeof(float));
        cudaMemset(variance[transformerIndex].query_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&variance[transformerIndex].key_weights, dim * dim * sizeof(float));
        cudaMemset(variance[transformerIndex].key_weights, 0, dim * dim * sizeof(float));
        
        cudaMalloc((void**)&variance[transformerIndex].rms1_gamma_weights, dim * sizeof(float));
        cudaMemset(variance[transformerIndex].rms1_gamma_weights, 0, dim * sizeof(float));        
    }
}

// ============================================================================
// TRAINING MEMORY ALLOCATION
// ============================================================================

static void allocateTrainingMemory() {
    cudaMalloc((void**)&ropeThetaStore_DEVICE, dim * maxL * sizeof(float));

    cudaMalloc((void**)&dLoss_d_vocabScores, vocabSize * maxL * sizeof(float));
    cudaMalloc((void**)&dLoss_d_embedding_weights, dim * vocabSize * sizeof(float));

    cudaMalloc((void**)&dLoss_d_ffn_final_postRMS_postGamma, dim * maxL * sizeof(float));        
    cudaMalloc((void**)&dLoss_d_ffn_final_RMS_gamma_weights, dim * sizeof(float));

    cudaMalloc((void**)&ffn_final_sigma_scale_x_upGrad_byCol_RMS, maxL * sizeof(float));
    cudaMalloc((void**)&ffn_final_oneOverR_byCol_RMS, maxL * sizeof(float));
    cudaMalloc((void**)&ffn_final_oneOverColDimR3_byCol_RMS, maxL * sizeof(float));

    cudaMalloc((void**)&x_DEVICE_grad, dim * maxL * sizeof(float));

    // Backprop calculations (implicitly gradients)
    for (int transformerIndex = 0; transformerIndex < transformers; transformerIndex++) {
        cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_final_plus_residual, dim * maxL * sizeof(float));

        cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_left_weights, dim * ffnDim * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_postHadamard, ffnDim * maxL * sizeof(float));
        
        cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_1_postSilu, ffnDim * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_1_preSilu, ffnDim * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_1_weights, ffnDim * dim * sizeof(float));

        cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_2, ffnDim * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_2_weights, ffnDim * dim * sizeof(float));

        cudaMalloc((void**)&backpropCalculations[transformerIndex].outputProjPlusResidual_postRMS2_post_gamma, dim * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].rms2_gamma_weights, dim * sizeof(float));

        cudaMalloc((void**)&backpropCalculations[transformerIndex].rms2_sigma_scale_x_upGrad_byCol_RMS, maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].rms2_oneOverR_byCol_RMS, maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].rms2_oneOverColDimR3_byCol_RMS, maxL * sizeof(float));
        
        cudaMalloc((void**)&backpropCalculations[transformerIndex].outputProjPlusResidual, dim * maxL * sizeof(float));

        cudaMalloc((void**)&backpropCalculations[transformerIndex].valueScaledSoftmaxAttn, dim * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].output_proj_weights, dim * dim * sizeof(float));

        cudaMalloc((void**)&backpropCalculations[transformerIndex].values, dim * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].attnByHead_postSoftmax, attnHeads * maxL * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].attnSoftmaxGradSumByCol, attnHeads * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].attnKtQByHead, attnHeads * maxL * maxL * sizeof(float));

        cudaMalloc((void**)&backpropCalculations[transformerIndex].keysPostRoPE, dim * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].keysPreRoPE, dim * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].queriesPostRoPE, dim * maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].queriesPreRoPE, dim * maxL * sizeof(float));

        cudaMalloc((void**)&backpropCalculations[transformerIndex].value_weights, dim * dim * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].query_weights, dim * dim * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].key_weights, dim * dim * sizeof(float));

        cudaMalloc((void**)&backpropCalculations[transformerIndex].rms1_gamma_weights, dim * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].rms1_sigma_scale_x_upGrad_byCol_RMS, maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].rms1_oneOverR_byCol_RMS, maxL * sizeof(float));
        cudaMalloc((void**)&backpropCalculations[transformerIndex].rms1_oneOverColDimR3_byCol_RMS, maxL * sizeof(float));

        cudaMalloc((void**)&backpropCalculations[transformerIndex].x_postRMS1_post_gamma, dim * maxL * sizeof(float));
    }

    // Setup RoPE theta store for training
    setupRoPEThetaStore(maxL);

    // Allocate optimizer state memory
    allocateBatchGradAccumulationMemory();
    allocateFastMomentumMemory();
    allocateSlowMomentumMemory();
    allocateVarianceMemory();

    // Allocate beta power stores (10M iterations capacity)
    // Index 0 is placeholder (iteration starts at 1 to avoid division by zero)
    const int maxIterations = 10000000;  // 10M
    cudaMalloc((void**)&beta1_pow_store, (maxIterations + 1) * sizeof(float));
    cudaMalloc((void**)&beta2_pow_store, (maxIterations + 1) * sizeof(float));
    cudaMalloc((void**)&beta3_pow_store, (maxIterations + 1) * sizeof(float));

    // Set index 0 to 1.0f as placeholder (unused, iteration starts at 1)
    float one = 1.0f;
    cudaMemcpy(beta1_pow_store, &one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(beta2_pow_store, &one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(beta3_pow_store, &one, sizeof(float), cudaMemcpyHostToDevice);

    // Precompute beta powers for all iterations (indices 1 to maxIterations)
    int numBlocks = (maxIterations + threadsPerBlock - 1) / threadsPerBlock;
    preCalcPowBeta<<<numBlocks, threadsPerBlock>>>(beta1_pow_store, beta2_pow_store, beta3_pow_store, maxIterations, ADEAMIX_BETA1, ADEAMIX_BETA2, ADEAMIX_BETA3);
    cudaDeviceSynchronize(); // ask why we need it

    // Allocate training stories memory: MAX_TRAINING_STORIES stories x TOKENS_PER_STORY (maxL+1) tokens
    cudaMalloc((void**)&trainingStoryTokens_DEVICE, MAX_TRAINING_STORIES * TOKENS_PER_STORY * sizeof(int));
    cudaMalloc((void**)&trainingStoryRightEndIndices_DEVICE, MAX_TRAINING_STORIES * sizeof(int));
    printf("Allocated training stories memory: %d stories x %d tokens = %zu bytes\n",
           MAX_TRAINING_STORIES, TOKENS_PER_STORY, 
           (size_t)MAX_TRAINING_STORIES * TOKENS_PER_STORY * sizeof(int));
}

// ============================================================================
// MAIN MEMORY ALLOCATION FUNCTION
// ============================================================================

void allocateMemory(bool allocateTraining) {
    /* ---------------- Device info ---------------- */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\n===== GPU INFO =====\n");
    printf("Device: \"%s\"\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    // 1. VRAM (Crucial for storing weights)
    double totalMemGB = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    printf("Total VRAM: %.2f GB\n", totalMemGB);

    // 2. Compute Power
    printf("Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    printf("Clock Rate: %.0f MHz\n", (double)prop.clockRate / 1000.0);

    // 3. Kernel Launch Constraints
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp Size: %d\n", prop.warpSize);

    // 4. Memory Speed
    printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("Memory Clock: %.0f MHz\n", (double)prop.memoryClockRate / 1000.0);
    
    printf("====================\n\n");    

    srand(0);

    // Sequence token indices (allocated at maxL+1 capacity: training needs L+1 tokens)
    size_t seqTokenIndices_size = (maxL + 1) * sizeof(int); 
    seqTokenIndices = (int*)malloc(seqTokenIndices_size);
    for (int i = 0; i < maxL + 1; i++) {
        seqTokenIndices[i] = rand() % vocabSize;
    }
    cudaMalloc((void**)&seqTokenIndices_DEVICE, seqTokenIndices_size);    
    cudaMemcpy(seqTokenIndices_DEVICE, seqTokenIndices, seqTokenIndices_size, cudaMemcpyHostToDevice);

    // Embedding weights
    size_t embedding_weights_size = dim * vocabSize * sizeof(float); 
    embedding_weights = (float*)malloc(embedding_weights_size);
    for (int i = 0; i < dim * vocabSize; i++) {
        embedding_weights[i] = ((float)rand() / (float)RAND_MAX);
    }
    cudaMalloc((void**)&embedding_weights_DEVICE, embedding_weights_size);
    cudaMemcpy(embedding_weights_DEVICE, embedding_weights, embedding_weights_size, cudaMemcpyHostToDevice);

    // Final RMS weights
    size_t final_rms_size = dim * sizeof(float); 
    final_rms_weights = (float*)malloc(final_rms_size);
    for (int i = 0; i < dim; i++) {
        final_rms_weights[i] = ((float)rand() / (float)RAND_MAX);
    }
    cudaMalloc((void**)&final_rms_weights_DEVICE, final_rms_size);
    cudaMemcpy(final_rms_weights_DEVICE, final_rms_weights, final_rms_size, cudaMemcpyHostToDevice);

    // Precomputed RoPE theta (allocated at maxL capacity)
    size_t preComputedRopeTheta_size = headDim * maxL * sizeof(float);
    preComputedRopeTheta = (float*)malloc(preComputedRopeTheta_size);
    getPreComputedRopeTheta(preComputedRopeTheta, maxL);
    cudaMalloc((void**)&preComputedRopeTheta_DEVICE, preComputedRopeTheta_size);
    cudaMemcpy(preComputedRopeTheta_DEVICE, preComputedRopeTheta, preComputedRopeTheta_size, cudaMemcpyHostToDevice);

    // Input tensor (allocated at maxL capacity)
    size_t x_size = dim * maxL * sizeof(float); 
    cudaMalloc((void**)&x_DEVICE, x_size);

    // Transformer weights and calculations
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

        size_t ffn_left_weights_size = dim * ffnDim * sizeof(float);
        currentTransformerWeights->ffn_left_weights = (float*)malloc(ffn_left_weights_size);
        for (int i = 0; i < dim * ffnDim; i++) {
            currentTransformerWeights->ffn_left_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_left_weights, ffn_left_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_left_weights, currentTransformerWeights->ffn_left_weights, ffn_left_weights_size, cudaMemcpyHostToDevice);

        size_t ffn_right_1_weights_size = dim * ffnDim * sizeof(float);
        currentTransformerWeights->ffn_right_1_weights = (float*)malloc(ffn_right_1_weights_size);
        for (int i = 0; i < dim * ffnDim; i++) {
            currentTransformerWeights->ffn_right_1_weights[i] = ((float)rand() / (float)RAND_MAX);
        }   
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_right_1_weights, ffn_right_1_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_right_1_weights, currentTransformerWeights->ffn_right_1_weights, ffn_right_1_weights_size, cudaMemcpyHostToDevice);

        size_t ffn_right_2_weights_size = dim * ffnDim * sizeof(float);
        currentTransformerWeights->ffn_right_2_weights = (float*)malloc(ffn_right_2_weights_size); 
        for (int i = 0; i < dim * ffnDim; i++) {
            currentTransformerWeights->ffn_right_2_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_right_2_weights, ffn_right_2_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_right_2_weights, currentTransformerWeights->ffn_right_2_weights, ffn_right_2_weights_size, cudaMemcpyHostToDevice);

        // Transformer calculation buffers (allocated at maxL capacity)
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].x_sumByCol_RMS1, maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].x_postRMS1_pre_gamma, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].x_postRMS1_post_gamma, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].queries, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].keys, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].values, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].queriesPostRoPE, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].keysPostRoPE, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnKtQByHead, attnHeads * maxL * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnKtQByHeadScaledMasked, attnHeads * maxL * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnByHead_maxByCol_softmax, attnHeads * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnByHead_sumByCol_softmax, attnHeads * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnByHead_expfCache_softmax, attnHeads * maxL * maxL * sizeof(float));        
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnByHead_postSoftmax, attnHeads * maxL * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].valueScaledSoftmaxAttn, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProj, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProjPlusResidual, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProjPlusResidual_sumByCol_RMS2, maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProjPlusResidual_postRMS2_pre_gamma, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProjPlusResidual_postRMS2_post_gamma, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_1_preSilu, ffnDim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_1_postSilu, ffnDim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_2, ffnDim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_postHadamard, ffnDim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_final, dim * maxL * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffnPlusResidual, dim * maxL * sizeof(float));
    }

    // Final output buffers (allocated at maxL capacity)
    cudaMalloc((void**)&ffn_sumByCol_RMS_DEVICE, maxL * sizeof(float));
    cudaMalloc((void**)&ffn_postRMS_pre_gamma_DEVICE, dim * maxL * sizeof(float));
    cudaMalloc((void**)&ffn_postRMS_post_gamma_DEVICE, dim * maxL * sizeof(float));

    cudaMalloc((void**)&vocabScores_DEVICE, vocabSize * maxL * sizeof(float));
    cudaMalloc((void**)&vocabScores_maxByCol_softmax_DEVICE, maxL * sizeof(float));
    cudaMalloc((void**)&vocabScores_sumByCol_softmax_DEVICE, maxL * sizeof(float));
    cudaMalloc((void**)&vocabScores_expfCache_softmax_DEVICE, vocabSize * maxL * sizeof(float));
    cudaMalloc((void**)&vocabScores_postSoftmax_DEVICE, vocabSize * maxL * sizeof(float));

    if (allocateTraining) {
        allocateTrainingMemory();
    }
}
