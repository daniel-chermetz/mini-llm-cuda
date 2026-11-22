#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "network_meta.h"
#include "network_globals.h"

__global__ void apply_1D_conv(
	float* x_post_conv_post_silu, 
	float* x_post_conv, 
	const float* x, 
	const float* conv_weights, 
	const int L
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= dim * L) {
		return;
	}

	int col = index / dim;
	int channel = index - col * dim;
	int channelOffset = channel * 4;

	float w0 = conv_weights[channelOffset];
	float w1 = conv_weights[channelOffset + 1];
	float w2 = conv_weights[channelOffset + 2];
	float w3 = conv_weights[channelOffset + 3];

	int x2_index = index - dim;
	int x1_index = x2_index - dim;
	int x0_index = x1_index - dim;

	float x0 = x0_index >= 0 ? x[x0_index] : 0;
	float x1 = x1_index >= 0 ? x[x1_index] : 0;
	float x2 = x2_index >= 0 ? x[x2_index] : 0;
	float x3 = x[index];

	float val = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
	x_post_conv[index] = val;
	x_post_conv_post_silu[index] = val / (1.0f + expf(-val));
}

__global__ void apply_l2_norm(
	float* x_post_l2_norm, 
	float* l2_inverse_col_sums_by_col_by_head,
	const float* x,
	const int L
) {
    int globalColIndex = blockIdx.x;
    int headIndex = globalColIndex / L;
    int headRelativeColIndex = globalColIndex - headIndex * L;
    int tIndex = threadIdx.x;

    int colOffset = headRelativeColIndex * dim + headIndex * headDim;

    extern __shared__ float sData[];

    float threadSum = 0.0f;
    for (int rowIndex = tIndex; rowIndex < headDim; rowIndex += blockDim.x) {
    	threadSum += (x[colOffset + rowIndex] * x[colOffset + rowIndex]);
    }
    sData[tIndex] = threadSum;
    __syncthreads();

    for (int reductionSize = blockDim.x / 2; reductionSize > 0; reductionSize /= 2) {
    	if (tIndex < reductionSize) {
    		sData[tIndex] = sData[tIndex] + sData[tIndex + reductionSize];
    	}
    	__syncthreads();
    }

    if (tIndex == 0) {
    	sData[0] = rsqrtf(sData[0] + 1e-6f);
    	l2_inverse_col_sums_by_col_by_head[globalColIndex] = sData[0];
    }
    __syncthreads();

 	for (int rowIndex = tIndex; rowIndex < headDim; rowIndex += blockDim.x) {
 		int index = colOffset + rowIndex;
 		x_post_l2_norm[index] = x[index] * sData[0];
 	}
}

__global__ void set_alpha_beta_gates(
	float* alpha_gate,
	float* alpha_softplus,
	float* beta_gate,
	const float* alpha, // [L, attnHeads]
	const float* beta, // [L, attnHeads]
	const float* alpha_to_gate_by_head_weights, // [attnHeads]
	const float* alpha_to_gate_bias_by_head_weights, // [attnHeads]
	const int L
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= attnHeads * L) {
		return;
	}

    int headIndex = index / L;

    float x = alpha[index] + alpha_to_gate_bias_by_head_weights[headIndex];
    float softplus_x = x > 20.0f ? x : log1pf(expf(x)); // ln(1 + e^(alpha + bias))
    alpha_softplus[index] = softplus_x;
    alpha_gate[index] = expf(-expf(alpha_to_gate_by_head_weights[headIndex]) * softplus_x);

    beta_gate[index] = 1.0f / (1.0f + expf(-beta[index]));
}

__global__ void compute_alpha_prefix_sum(
	float* alpha_prefix_sum,
	const float* alpha_gate,
	const int L
) {
	int headIndex = blockIdx.x;
	int gate_head_offset = headIndex * L;

	alpha_prefix_sum[gate_head_offset] = 0.0f; // the unused alpha[0]
	float running_sum = 0.0f;
	for (int alpha_index = gate_head_offset + 1; alpha_index < (gate_head_offset + L); alpha_index++) {
		running_sum += logf(fmaxf(alpha_gate[alpha_index], 1e-20f));
		alpha_prefix_sum[alpha_index] = running_sum;
	}
}

__global__ void scale_attention_by_head(
	float* scaled_attention,
	float* attn_alpha_beta_gate_scale,
	const float* attention,
	const float* alpha_prefix_sum,
	const float* beta_gate,
	const int L
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int headSize = L * L;
	int maxSize = headSize * attnHeads;

	if (index >= maxSize) {
		return;
	}

	int headIndex = index / headSize;
	int headRelativeIndex = index - headSize * headIndex;
	int headRelativeColIndex = headRelativeIndex / L;
	int headRelativeRowIndex = headRelativeIndex - headRelativeColIndex * L;

	if (headRelativeRowIndex > headRelativeColIndex) {
		scaled_attention[index] = 0.0f;
		return;
	}

	int gate_head_offset = headIndex * L;
	int gate_current_col_based_index = gate_head_offset + headRelativeColIndex;
	int gate_current_row_based_index = gate_head_offset + headRelativeRowIndex;
	
	float curr_attn_gate_scale = expf(alpha_prefix_sum[gate_current_col_based_index] - alpha_prefix_sum[gate_current_row_based_index]) * beta_gate[gate_current_row_based_index];
	attn_alpha_beta_gate_scale[index] = curr_attn_gate_scale;
	scaled_attention[index] = attention[index] * curr_attn_gate_scale;
}

void runLinearAttnPreparationInference(int tIndex, int linearAttnIndex, int L) {
	int blockSize = 256;
	int dim_L_numBlocks = (dim * L + blockSize - 1) / blockSize;
	int attnHeads_L_numBlocks = (attnHeads * L + blockSize - 1) / blockSize;	

	tWeights = transformerWeights_DEVICE[tIndex];
	laWeights = linearAttnWeights_DEVICE[linearAttnIndex];
	tCalcs = transformerCalculations_DEVICE[tIndex];
	laCalcs = linearAttnCalculations_DEVICE[linearAttnIndex];

	apply_1D_conv<<<dim_L_numBlocks, blockSize>>>(
		laCalcs.keys_post_conv_post_silu, 
		laCalcs.keys_post_conv, 
		tCalcs.keys, 
		laWeights.conv_key_weights, 
		L
	);
	apply_1D_conv<<<dim_L_numBlocks, blockSize>>>(
		laCalcs.queries_post_conv_post_silu, 
		laCalcs.queries_post_conv, 
		tCalcs.queries, 
		laWeights.conv_query_weights, 
		L
	);	
	apply_1D_conv<<<dim_L_numBlocks, blockSize>>>(
		laCalcs.values_post_conv_post_silu, 
		laCalcs.values_post_conv, 
		tCalcs.values, 
		laWeights.conv_value_weights, 
		L
	);

	int num_cols_scaled_by_attn_heads = L * attnHeads;

	apply_l2_norm<<<num_cols_scaled_by_attn_heads, 64, 64 * sizeof(float)>>>(
		laCalcs.keys_post_l2_norm, 
		laCalcs.keys_l2_inverse_col_sums_by_col_by_head,
		laCalcs.keys_post_conv_post_silu,
		L
	);
	apply_l2_norm<<<num_cols_scaled_by_attn_heads, 64, 64 * sizeof(float)>>>(
		laCalcs.queries_post_l2_norm,
		laCalcs.queries_l2_inverse_col_sums_by_col_by_head,		
		laCalcs.queries_post_conv_post_silu,
		L
	);

	// tCalcs.x_postRMS1_post_gamma: [dim, L]
	// laWeights.alpha_weights: [dim, attnHeads]
	// laWeights.beta_weights: [dim, attnHeads]
	// laCalcs.alpha_activation: [L, attnHeads]
	// laCalcs.beta_activation: [L, attnHeads]
	cublasGemmEx(
	    handle,
	    CUBLAS_OP_T,
	    CUBLAS_OP_N,
	    L, // row C
	    attnHeads, // cols C
	    dim, // contracting (shared) dim
	    &alpha,
	    tCalcs.x_postRMS1_post_gamma,
	    CUDA_R_32F,
	    dim, // lda, mem col size for col-major
	    laWeights.alpha_weights,
	    CUDA_R_32F,
	    dim, // ldb, mem col size for col-major      
	    &beta,
	    laCalcs.alpha_activation,
	    CUDA_R_32F,
	    L, // ldc, mem col size
	    CUBLAS_COMPUTE_32F,
	    CUBLAS_GEMM_DEFAULT             
	);

	cublasGemmEx(
	    handle,
	    CUBLAS_OP_T,
	    CUBLAS_OP_N,
	    L, // row C
	    attnHeads, // cols C
	    dim, // contracting (shared) dim
	    &alpha,
	    tCalcs.x_postRMS1_post_gamma,
	    CUDA_R_32F,
	    dim, // lda, mem col size for col-major
	    laWeights.beta_weights,
	    CUDA_R_32F,
	    dim, // ldb, mem col size for col-major      
	    &beta,
	    laCalcs.beta_activation,
	    CUDA_R_32F,
	    L, // ldc, mem col size
	    CUBLAS_COMPUTE_32F,
	    CUBLAS_GEMM_DEFAULT             
	);

	set_alpha_beta_gates<<<attnHeads_L_numBlocks, blockSize>>>(
		laCalcs.alpha_gate,
		laCalcs.alpha_softplus,
		laCalcs.beta_gate,
		laCalcs.alpha_activation, // [L, attnHeads]
		laCalcs.beta_activation, // [L, attnHeads]
		laWeights.alpha_to_gate_by_head_weights, // [attnHeads]
		laWeights.alpha_to_gate_bias_by_head_weights, // [attnHeads]
		L
	);	
}

// the "linear" part will apply only during post-training inference
void runLinearAttnGateScalingInference(int tIndex, int linearAttnIndex, int L) {
	int blockSize = 256;
	int attnHeads_L_L_numBlocks = (attnHeads * L * L + blockSize - 1) / blockSize;

	tWeights = transformerWeights_DEVICE[tIndex];
	laWeights = linearAttnWeights_DEVICE[linearAttnIndex];
	tCalcs = transformerCalculations_DEVICE[tIndex];
	laCalcs = linearAttnCalculations_DEVICE[linearAttnIndex];

	compute_alpha_prefix_sum<<<attnHeads, 1>>>(
		laCalcs.alpha_prefix_sum,
		laCalcs.alpha_gate,
		L		
	);

	scale_attention_by_head<<<attnHeads_L_L_numBlocks, blockSize>>>(
		laCalcs.scaled_attnKtQByHead,
		laCalcs.attn_alpha_beta_gate_scale,
		tCalcs.attnKtQByHead,
		laCalcs.alpha_prefix_sum,
		laCalcs.beta_gate,
		L
	);
}
