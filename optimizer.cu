#include <cuda_runtime.h>
#include <math.h>

#include "network_meta.h"
#include "network_globals.h"

__global__ void preCalcPowBeta(float* beta1_pow_store, float* beta2_pow_store, float* beta3_pow_store, int numIterationsToCalc, float BETA1_, float BETA2_, float BETA3_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numIterationsToCalc) {
    	return;
    }

    beta1_pow_store[index + 1] = (1.0f - powf(BETA1_, index + 1));
    beta2_pow_store[index + 1] = (1.0f - powf(BETA2_, index + 1));
    beta3_pow_store[index + 1] = (1.0f - powf(BETA3_, index + 1));
}

__global__ void update_fast_EMA(float* fastEMA, float* grad, float BETA1_, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
    	return;
    }

    fastEMA[index] = BETA1_ * fastEMA[index] + (1.0f - BETA1_) * grad[index];
}

__global__ void update_slow_EMA(float* slowEMA, float* grad, float BETA3_, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
    	return;
    }

    slowEMA[index] = BETA3_ * slowEMA[index] + (1.0f - BETA3_) * grad[index];
}

__global__ void update_variance(float* variance, float* grad, float BETA2_, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
    	return;
    }

    variance[index] = BETA2_ * variance[index] + (1.0f - BETA2_) * grad[index] * grad[index];
}

// iterationNum starts with 1
__global__ void update_weights_per_adeamix(float* weight, float* fastEMA, float* slowEMA, float* variance, float* beta1_pow_store, float* beta2_pow_store, float* beta3_pow_store, float learningRate, int iterationNum, float ALPHA_, float WEIGHT_DECAY_, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
    	return;
    }

    float fastEMABiased = fastEMA[index] / beta1_pow_store[iterationNum];
    float slowEMABiased = slowEMA[index] / beta3_pow_store[iterationNum];
    float varianceBiased = variance[index] / beta2_pow_store[iterationNum];

    float momentum_final = fastEMABiased + ALPHA_ * slowEMABiased;
    float momentum_variance_scaled_final = momentum_final / (sqrtf(varianceBiased) + 1e-8f);

	weight[index] = weight[index] - learningRate * momentum_variance_scaled_final - learningRate * WEIGHT_DECAY_ * weight[index];
}

void apply_adeamix_optimizer(int iterationNum, float learningRate) {
	int xTotalThreads;
	int numBlocks;

	// ============================================================================
	// EMBEDDING WEIGHTS
	// ============================================================================
	xTotalThreads = dim * vocabSize;
	numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA_embedding_weights, gradientAccumulation_embedding_weights, ADEAMIX_BETA1, xTotalThreads);
	update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA_embedding_weights, gradientAccumulation_embedding_weights, ADEAMIX_BETA3, xTotalThreads);
	update_variance<<<numBlocks, threadsPerBlock>>>(variance_embedding_weights, gradientAccumulation_embedding_weights, ADEAMIX_BETA2, xTotalThreads);
	update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(embedding_weights_DEVICE, fastEMA_embedding_weights, slowEMA_embedding_weights, variance_embedding_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);

	// ============================================================================
	// FINAL RMS GAMMA WEIGHTS
	// ============================================================================
	xTotalThreads = dim;
	numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA_final_RMS_gamma_weights, gradientAccumulation_final_RMS_gamma_weights, ADEAMIX_BETA1, xTotalThreads);
	update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA_final_RMS_gamma_weights, gradientAccumulation_final_RMS_gamma_weights, ADEAMIX_BETA3, xTotalThreads);
	update_variance<<<numBlocks, threadsPerBlock>>>(variance_final_RMS_gamma_weights, gradientAccumulation_final_RMS_gamma_weights, ADEAMIX_BETA2, xTotalThreads);
	update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(final_rms_weights_DEVICE, fastEMA_final_RMS_gamma_weights, slowEMA_final_RMS_gamma_weights, variance_final_RMS_gamma_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);

	// ============================================================================
	// TRANSFORMER LAYER WEIGHTS
	// ============================================================================
	for (int t = 0; t < transformers; t++) {
		// FFN Left Weights (dim x ffnDim)
		xTotalThreads = dim * ffnDim;
		numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA[t].ffn_left_weights, gradientAccumulation[t].ffn_left_weights, ADEAMIX_BETA1, xTotalThreads);
		update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA[t].ffn_left_weights, gradientAccumulation[t].ffn_left_weights, ADEAMIX_BETA3, xTotalThreads);
		update_variance<<<numBlocks, threadsPerBlock>>>(variance[t].ffn_left_weights, gradientAccumulation[t].ffn_left_weights, ADEAMIX_BETA2, xTotalThreads);
		update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].ffn_left_weights, fastEMA[t].ffn_left_weights, slowEMA[t].ffn_left_weights, variance[t].ffn_left_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);

		// FFN Right 1 Weights (ffnDim x dim)
		xTotalThreads = ffnDim * dim;
		numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA[t].ffn_right_1_weights, gradientAccumulation[t].ffn_right_1_weights, ADEAMIX_BETA1, xTotalThreads);
		update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA[t].ffn_right_1_weights, gradientAccumulation[t].ffn_right_1_weights, ADEAMIX_BETA3, xTotalThreads);
		update_variance<<<numBlocks, threadsPerBlock>>>(variance[t].ffn_right_1_weights, gradientAccumulation[t].ffn_right_1_weights, ADEAMIX_BETA2, xTotalThreads);
		update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].ffn_right_1_weights, fastEMA[t].ffn_right_1_weights, slowEMA[t].ffn_right_1_weights, variance[t].ffn_right_1_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);

		// FFN Right 2 Weights (ffnDim x dim)
		update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA[t].ffn_right_2_weights, gradientAccumulation[t].ffn_right_2_weights, ADEAMIX_BETA1, xTotalThreads);
		update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA[t].ffn_right_2_weights, gradientAccumulation[t].ffn_right_2_weights, ADEAMIX_BETA3, xTotalThreads);
		update_variance<<<numBlocks, threadsPerBlock>>>(variance[t].ffn_right_2_weights, gradientAccumulation[t].ffn_right_2_weights, ADEAMIX_BETA2, xTotalThreads);
		update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].ffn_right_2_weights, fastEMA[t].ffn_right_2_weights, slowEMA[t].ffn_right_2_weights, variance[t].ffn_right_2_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);

		// RMS2 Gamma Weights (dim)
		xTotalThreads = dim;
		numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA[t].rms2_gamma_weights, gradientAccumulation[t].rms2_gamma_weights, ADEAMIX_BETA1, xTotalThreads);
		update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA[t].rms2_gamma_weights, gradientAccumulation[t].rms2_gamma_weights, ADEAMIX_BETA3, xTotalThreads);
		update_variance<<<numBlocks, threadsPerBlock>>>(variance[t].rms2_gamma_weights, gradientAccumulation[t].rms2_gamma_weights, ADEAMIX_BETA2, xTotalThreads);
		update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].rms2_weights, fastEMA[t].rms2_gamma_weights, slowEMA[t].rms2_gamma_weights, variance[t].rms2_gamma_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);

		// Output Projection Weights (dim x dim)
		xTotalThreads = dim * dim;
		numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA[t].output_proj_weights, gradientAccumulation[t].output_proj_weights, ADEAMIX_BETA1, xTotalThreads);
		update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA[t].output_proj_weights, gradientAccumulation[t].output_proj_weights, ADEAMIX_BETA3, xTotalThreads);
		update_variance<<<numBlocks, threadsPerBlock>>>(variance[t].output_proj_weights, gradientAccumulation[t].output_proj_weights, ADEAMIX_BETA2, xTotalThreads);
		update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].output_proj_weights, fastEMA[t].output_proj_weights, slowEMA[t].output_proj_weights, variance[t].output_proj_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);

		// Value Weights (dim x dim)
		update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA[t].value_weights, gradientAccumulation[t].value_weights, ADEAMIX_BETA1, xTotalThreads);
		update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA[t].value_weights, gradientAccumulation[t].value_weights, ADEAMIX_BETA3, xTotalThreads);
		update_variance<<<numBlocks, threadsPerBlock>>>(variance[t].value_weights, gradientAccumulation[t].value_weights, ADEAMIX_BETA2, xTotalThreads);
		update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].value_weights, fastEMA[t].value_weights, slowEMA[t].value_weights, variance[t].value_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);

		// Query Weights (dim x dim)
		update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA[t].query_weights, gradientAccumulation[t].query_weights, ADEAMIX_BETA1, xTotalThreads);
		update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA[t].query_weights, gradientAccumulation[t].query_weights, ADEAMIX_BETA3, xTotalThreads);
		update_variance<<<numBlocks, threadsPerBlock>>>(variance[t].query_weights, gradientAccumulation[t].query_weights, ADEAMIX_BETA2, xTotalThreads);
		update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].query_weights, fastEMA[t].query_weights, slowEMA[t].query_weights, variance[t].query_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);

		// Key Weights (dim x dim)
		update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA[t].key_weights, gradientAccumulation[t].key_weights, ADEAMIX_BETA1, xTotalThreads);
		update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA[t].key_weights, gradientAccumulation[t].key_weights, ADEAMIX_BETA3, xTotalThreads);
		update_variance<<<numBlocks, threadsPerBlock>>>(variance[t].key_weights, gradientAccumulation[t].key_weights, ADEAMIX_BETA2, xTotalThreads);
		update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].key_weights, fastEMA[t].key_weights, slowEMA[t].key_weights, variance[t].key_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);

		// RMS1 Gamma Weights (dim)
		xTotalThreads = dim;
		numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		update_fast_EMA<<<numBlocks, threadsPerBlock>>>(fastEMA[t].rms1_gamma_weights, gradientAccumulation[t].rms1_gamma_weights, ADEAMIX_BETA1, xTotalThreads);
		update_slow_EMA<<<numBlocks, threadsPerBlock>>>(slowEMA[t].rms1_gamma_weights, gradientAccumulation[t].rms1_gamma_weights, ADEAMIX_BETA3, xTotalThreads);
		update_variance<<<numBlocks, threadsPerBlock>>>(variance[t].rms1_gamma_weights, gradientAccumulation[t].rms1_gamma_weights, ADEAMIX_BETA2, xTotalThreads);
		update_weights_per_adeamix<<<numBlocks, threadsPerBlock>>>(transformerWeights_DEVICE[t].rms1_weights, fastEMA[t].rms1_gamma_weights, slowEMA[t].rms1_gamma_weights, variance[t].rms1_gamma_weights, beta1_pow_store, beta2_pow_store, beta3_pow_store, learningRate, iterationNum, ADEAMIX_ALPHA, ADEAMIX_WEIGHT_DECAY, xTotalThreads);
	}
}