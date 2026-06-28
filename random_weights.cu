/*
 * random_weights.cu
 * Populate model weights with random values for testing
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "network_meta.h"
#include "network_globals.h"

// CUDA kernel to initialize weights with random values in range [-range, +range]
__global__ void initRandomWeights(float* weights, int size, float range, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    // Generate random float in [-range, +range]
    float r = curand_uniform(&state);  // [0, 1)
    weights[idx] = (r * 2.0f - 1.0f) * range;  // [-range, +range)
}

// CUDA kernel to initialize weights with a constant value
__global__ void initConstantWeights(float* weights, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    weights[idx] = value;
}

void initializeRandomWeights(float range) {
    printf("\n=== Initializing Random Weights (range: [-%.4f, +%.4f]) ===\n", range, range);
    
    unsigned long seed = 42;  // Fixed seed for reproducibility
    int numBlocks;
    int size;
    
    // Embedding weights (dim x vocabSize)
    size = dim * vocabSize;
    numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    initRandomWeights<<<numBlocks, threadsPerBlock>>>(embedding_weights_DEVICE, size, range, seed++);
    printf("  Embedding weights: %d elements\n", size);
    
    // Final RMS weights (dim) - initialize to 1.0
    size = dim;
    numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    initConstantWeights<<<numBlocks, threadsPerBlock>>>(final_rms_weights_DEVICE, size, 1.0f);
    printf("  Final RMS weights: %d elements (initialized to 1.0)\n", size);
    
    // Transformer weights
    for (int t = 0; t < transformers; t++) {
        // RMS1 weights (dim) - initialize to 1.0
        size = dim;
        numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        initConstantWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].rms1_weights, size, 1.0f);
        
        // Query weights (dim x dim)
        size = dim * dim;
        numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        initRandomWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].query_weights, size, range, seed++);
        
        // Key weights (dim x dim)
        initRandomWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].key_weights, size, range, seed++);
        
        // Value weights (dim x dim)
        initRandomWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].value_weights, size, range, seed++);
        
        // Output projection weights (dim x dim)
        initRandomWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].output_proj_weights, size, range, seed++);
        
        // RMS2 weights (dim) - initialize to 1.0
        size = dim;
        numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        initConstantWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].rms2_weights, size, 1.0f);
        
        // FFN left weights (dim x ffnDim)
        size = dim * ffnDim;
        numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        initRandomWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].ffn_left_weights, size, range, seed++);
        
        // FFN right 1 weights (ffnDim x dim)
        size = ffnDim * dim;
        numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        initRandomWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].ffn_right_1_weights, size, range, seed++);
        
        // FFN right 2 weights (ffnDim x dim)
        initRandomWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].ffn_right_2_weights, size, range, seed++);
        
        // CONFIG_QK_RMS_NORM: query/key RMS gamma weights (dim) - initialize to 1.0
        if (CONFIG_QK_RMS_NORM) {
            size = dim;
            numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
            initConstantWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].query_RMS_weights, size, 1.0f);
            initConstantWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].key_RMS_weights, size, 1.0f);
        }
        
        // CONFIG_QUERY_GATING: gated query weights (dim x dim)
        if (CONFIG_QUERY_GATING) {
            size = dim * dim;
            numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
            initRandomWeights<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].gated_query_weights, size, range, seed++);
        }
        
        printf("  Transformer %d weights initialized\n", t);
    }
    
    cudaDeviceSynchronize();
    printf("=== Random Weight Initialization Complete ===\n\n");
}
