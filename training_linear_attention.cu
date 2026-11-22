#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "network_meta.h"
#include "network_globals.h"

__global__ void compute_scaled_attention_grad_sums_across_rows_and_cols(
	float* colGradSums,
	float* rowGradSums,
	float* beta_grad,
	float* unscaled_attn_grad,	
	const float* upgrad_scaled_attention,
	const float* scaled_attention,
	const float* attn_alpha_beta_gate_scale,
	const float* beta_gate,
	const int L
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int maxSize = L * attnHeads;
	if (index >= maxSize) {
		return;
	}

	int headIndex = index / L;
	int headRelativeTokenIndex = index - L * headIndex;

	int attn_head_offset = headIndex * L * L;

	// col > index for rows
	float row_sum = 0.0f;
	int diagonal_row_index = attn_head_offset + headRelativeTokenIndex + (headRelativeTokenIndex) * L;
	for (int rowCellIndex = diagonal_row_index + L; rowCellIndex < attn_head_offset + headRelativeTokenIndex + (L * L); rowCellIndex += L) {
		row_sum += (scaled_attention[rowCellIndex] * upgrad_scaled_attention[rowCellIndex]);
	}
	rowGradSums[index] = row_sum;
	float row_beta = beta_gate[index];
	beta_grad[index] = row_beta > 1e-20f ? ((row_sum + scaled_attention[diagonal_row_index] * upgrad_scaled_attention[diagonal_row_index]) / row_beta) : 0.0f;

	// row < index for cols
	float col_sum = 0.0f;
	int col_end_index = diagonal_row_index - headRelativeTokenIndex + L;
	for (int colCellIndex = attn_head_offset + headRelativeTokenIndex * L; colCellIndex < col_end_index; colCellIndex += 1) {
		float current_upgrad = colCellIndex <= diagonal_row_index ? upgrad_scaled_attention[colCellIndex] : 0.0f;
		col_sum += ((colCellIndex < diagonal_row_index) ? (scaled_attention[colCellIndex] * current_upgrad) : 0.0f);
		unscaled_attn_grad[colCellIndex] = colCellIndex <= diagonal_row_index ? (attn_alpha_beta_gate_scale[colCellIndex] * current_upgrad) : 0.0f;
	}
	colGradSums[index] = col_sum;
}

__global__ void unwind_alpha_beta_gates(
	float* alpha_grad,
	const float* alpha_gate,
	const float* colGradSums,
	const float* rowGradSums,
	const int L
) {
	int headIndex = blockIdx.x;
	int headOffset = headIndex * L;

	alpha_grad[headOffset] = 0.0f;

	float active_sum = 0.0f;
	for (int globalTokenIndex = headOffset + 1; globalTokenIndex < headOffset + L; globalTokenIndex++) {
		active_sum += (rowGradSums[globalTokenIndex - 1] - colGradSums[globalTokenIndex - 1]);
		float current_alpha_val = alpha_gate[globalTokenIndex];
		alpha_grad[globalTokenIndex] = current_alpha_val > 1e-20f ? (active_sum / current_alpha_val) : 0.0f;
	}
}

__global__ void compute_alpha_beta_gate_weights_grads(
	float* alpha_intermediate_activation_grad,
	float* beta_intermediate_activation_grad,
	float* alpha_to_gate_by_head_weight_grad, // must be initialized to zero
	float* alpha_to_gate_bias_by_head_weight_grad, // must be initialized to zero
	const float* alpha_gate_upgrad,
	const float* beta_gate_upgrad,
	const float* alpha_gate,
	const float* alpha_softplus,
	const float* beta_gate,
	const float* alpha_intermdiate_activation,	
	const float* alpha_to_gate_by_head_weights, // alpha intermediate activation to alpha gate
	const float* alpha_to_gate_bias_by_head_weights, // alpha intermediate activation to alpha gate
	const int L
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= attnHeads * L) {
		return;
	}

    int headIndex = index / L;

	// from Forward:
	// float x = alpha[index] + alpha_to_gate_bias_by_head_weights[headIndex];
    // float softplus_x = x > 20.0f ? x : log1pf(expf(x)); // ln(1 + e^(alpha + bias))
    // alpha_gate[index] = expf(-expf(alpha_to_gate_by_head_weights[headIndex]) * softplus_x);
    // beta_gate[index] = 1.0f / (1.0f + expf(-beta[index]));

    float current_beta_gate = beta_gate[index];
    beta_intermediate_activation_grad[index] = beta_gate_upgrad[index] * current_beta_gate * (1.0f - current_beta_gate);

    float current_alpha_gate_upgrad = alpha_gate_upgrad[index];    
    float current_alpha_gate = alpha_gate[index];
    float current_alpha_softplus = alpha_softplus[index];
    float current_neg_exp_alpha_to_gate_weight = -expf(alpha_to_gate_by_head_weights[headIndex]);    

    // dY * y * (−exp(A) * s)
    atomicAdd(
    	&alpha_to_gate_by_head_weight_grad[headIndex], 
    	current_alpha_gate_upgrad * current_alpha_gate * current_neg_exp_alpha_to_gate_weight * current_alpha_softplus
    );

    // dY⋅y⋅(−exp(A))
    float softplus_grad = current_alpha_gate_upgrad * current_alpha_gate * current_neg_exp_alpha_to_gate_weight; // move two lines up and softplus_grad * current_alpha_softplus equals to what's currently in atomicAdd 
	// when current_alpha_softplus > 20.0f, this expression becomes ~1.0f, remaining accurate even as an actual softplus was circumvented
	float dSoftplus = 1.0f / (1.0f + expf(-(alpha_intermdiate_activation[index] + alpha_to_gate_bias_by_head_weights[headIndex])));
	alpha_intermediate_activation_grad[index] = dSoftplus * softplus_grad;
	atomicAdd(
		&alpha_to_gate_bias_by_head_weight_grad[headIndex], 
		dSoftplus * softplus_grad
	);
}

__global__ void get_L2_col_features_x_grad_sums_across_cols(
	float* features_times_grad_sums_by_col_by_head,
	const float* x_post_l2_upgrad,
	const float* x_post_l2,
	const int L
) {
	int index = blockIdx.x;
	int tIndex = threadIdx.x;

    int headIndex = index / L;
    int headRelativeColIndex = index - headIndex * L;

    int colOffset = headRelativeColIndex * dim + headIndex * headDim;

    extern __shared__ float sData[];

    float threadSum = 0.0f;
    for (int rowIndex = tIndex; rowIndex < headDim; rowIndex += blockDim.x) {
    	int feature_index = colOffset + rowIndex;
    	threadSum += (x_post_l2[feature_index] * x_post_l2_upgrad[feature_index]);
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
    	features_times_grad_sums_by_col_by_head[index] = sData[0];
    } 
}

__global__ void compute_x_pre_l2_grads(
	float* x_pre_l2_grad,
	const float* x_post_l2_upgrad,
	const float* x_post_l2,	
	const float* l2_inverse_col_sums_by_col_by_head,
	const float* features_times_grad_sums_by_col_by_head,
	const int L	
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int maxCount = dim * L;
	if (index >= maxCount) {
		return;
	}

	int colIndex = index / dim;
	int rowIndex = index - colIndex * dim;
	int headIndex = rowIndex / headDim;

	int colOffsetWithLColsByHead = headIndex * L + colIndex;

	x_pre_l2_grad[index] = l2_inverse_col_sums_by_col_by_head[colOffsetWithLColsByHead] * (x_post_l2_upgrad[index] - x_post_l2[index] * features_times_grad_sums_by_col_by_head[colOffsetWithLColsByHead]);
}

__global__ void compute_conv_weight_grads(
	float* conv_weights_grad,
	float* x_post_conv_pre_silu_grad,
	const float* x_pre_l2_upgrad,
	const float* x_post_conv_pre_silu,
	const float* x_pre_conv,
	int L
) {
	int channelIndex = blockIdx.x;
	int conv_weight_channel_offset = (channelIndex + 1) * 4 - 1;
	int threadsPerBlock = blockDim.x;
	int tIndex = threadIdx.x; // 128
	int shared_data_thread_offset = (tIndex + 1) * 4 - 1; // 3, 7, 11, 15

	extern __shared__ float sData[]; // 128 * 4
	sData[shared_data_thread_offset] = 0.0f;
	sData[shared_data_thread_offset - 1] = 0.0f;
	sData[shared_data_thread_offset - 2] = 0.0f;
	sData[shared_data_thread_offset - 3] = 0.0f;

	for (int tokenIndex = tIndex; tokenIndex < L; tokenIndex += threadsPerBlock) {
		int post_conv_index = channelIndex + tokenIndex * dim;
		
		float token_post_conv_pre_silu = x_post_conv_pre_silu[post_conv_index];
		float sig = 1.0f / (1.0f + expf(-token_post_conv_pre_silu));
		float token_post_conv_pre_silu_upgrad = sig * (1.0f + token_post_conv_pre_silu * (1.0f - sig));

		float token_x_pre_l2_upgrad = x_pre_l2_upgrad[post_conv_index];

		float total_upgrad = token_post_conv_pre_silu_upgrad * token_x_pre_l2_upgrad;
		x_post_conv_pre_silu_grad[post_conv_index] = total_upgrad;

		int x3_index = post_conv_index;
		sData[shared_data_thread_offset] += (x_pre_conv[x3_index] * total_upgrad);

		int x2_index = x3_index - dim;
		sData[shared_data_thread_offset - 1] += (tokenIndex >= 1 ? x_pre_conv[x2_index] * total_upgrad : 0.0f);

		int x1_index = x2_index - dim;	
		sData[shared_data_thread_offset - 2] += (tokenIndex >= 2 ? (x_pre_conv[x1_index] * total_upgrad) : 0.0f);

		int x0_index = x1_index - dim;
		sData[shared_data_thread_offset - 3] += (tokenIndex >= 3 ? (x_pre_conv[x0_index] * total_upgrad) : 0.0f);
	}

	__syncthreads();

	for (int reductionSize = blockDim.x / 2; reductionSize > 0; reductionSize /= 2) {
		if (tIndex < reductionSize) {
			int current_shared_data_thread_offset = (tIndex + 1) * 4 - 1;
			int reduction_size_offset = reductionSize * 4;

			sData[current_shared_data_thread_offset] = sData[current_shared_data_thread_offset] + sData[current_shared_data_thread_offset + reduction_size_offset];
			sData[current_shared_data_thread_offset - 1] = sData[current_shared_data_thread_offset - 1] + sData[current_shared_data_thread_offset + reduction_size_offset - 1];
			sData[current_shared_data_thread_offset - 2] = sData[current_shared_data_thread_offset - 2] + sData[current_shared_data_thread_offset + reduction_size_offset - 2];
			sData[current_shared_data_thread_offset - 3] = sData[current_shared_data_thread_offset - 3] + sData[current_shared_data_thread_offset + reduction_size_offset - 3];
		}

		__syncthreads();
	}

	if (tIndex == 0) {
		conv_weights_grad[conv_weight_channel_offset] += sData[3];
		conv_weights_grad[conv_weight_channel_offset - 1] += sData[2];
		conv_weights_grad[conv_weight_channel_offset - 2] += sData[1];
		conv_weights_grad[conv_weight_channel_offset - 3] += sData[0];		
	}
}

__global__ void compute_x_pre_conv_grads(
	float* x_pre_conv_pre_silu_grad,
	const float* x_post_conv_pre_silu_upgrad,
	const float* conv_weights,
	const int L
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int maxCount = dim * L;
	if (index >= maxCount) {
		return;
	}

	int tokenIndex = index / dim;
	int channelIndex = index - tokenIndex * dim;

	int conv_weight_channel_offset = (channelIndex + 1) * 4 - 1;

	// w3 * token 0 --> token 0 upgrad
	// w2 * token 0 --> token 1 upgrad
	// w1 * token 0 --> token 2 upgrad
	// w0 * token 0 --> token 3 upgrad

	float current_x_pre_conv_pre_silu_grad = conv_weights[conv_weight_channel_offset] * x_post_conv_pre_silu_upgrad[index];
	current_x_pre_conv_pre_silu_grad += (tokenIndex < L - 1 ? (conv_weights[conv_weight_channel_offset - 1] * x_post_conv_pre_silu_upgrad[index + dim]) : 0.0f);
	current_x_pre_conv_pre_silu_grad += (tokenIndex < L - 2 ? (conv_weights[conv_weight_channel_offset - 2] * x_post_conv_pre_silu_upgrad[index + 2 * dim]) : 0.0f);
	current_x_pre_conv_pre_silu_grad += (tokenIndex < L - 3 ? (conv_weights[conv_weight_channel_offset - 3] * x_post_conv_pre_silu_upgrad[index + 3 * dim]) : 0.0f);

	x_pre_conv_pre_silu_grad[index] = current_x_pre_conv_pre_silu_grad;
}

void run_linear_attn_gate_unwinding_for_training(int tIndex, int linearAttnIndex, int L) {
	// IMPORTANT: right side (attnKtQByHead) by head used to have gradient zeroed for masked features (the features are masked but the gradient looks to unmasked V),
	// hence we need to repeat part of what the softmax kernel used to do in manually zeroing the masked upstream gradients.
	int blockSize = 256;
	int attnHeads_L_numBlocks = (attnHeads * L + blockSize - 1) / blockSize;	

	tCalcs = transformerCalculations_DEVICE[tIndex];
	laCalcs = linearAttnCalculations_DEVICE[linearAttnIndex];

	laWeights = linearAttnWeights_DEVICE[linearAttnIndex];

	tBackpropCalculations = backpropCalculations[tIndex];
	laBackpropCalculations = linearAttnBackpropCalculations[linearAttnIndex];

	compute_scaled_attention_grad_sums_across_rows_and_cols<<<attnHeads_L_numBlocks, blockSize>>>(
		laBackpropCalculations.colGradSums,
		laBackpropCalculations.rowGradSums,
		laBackpropCalculations.beta_gate,
		tBackpropCalculations.attnKtQByHead,		
		laBackpropCalculations.scaled_attnKtQByHead, // const upgrad
		laCalcs.scaled_attnKtQByHead,
		laCalcs.attn_alpha_beta_gate_scale,
		laCalcs.beta_gate,
		L
	);

	unwind_alpha_beta_gates<<<attnHeads, 1>>>(
		laBackpropCalculations.alpha_beta,
		laCalcs.alpha_gate,
		laBackpropCalculations.colGradSums,
		laBackpropCalculations.rowGradSums,
		L
	);

	compute_alpha_beta_gate_weights_grads<<<attnHeads_L_numBlocks, blockSize>>>(
		laBackpropCalculations.alpha_activation,
		laBackpropCalculations.beta_activation,
		laBackpropCalculations.alpha_to_gate_by_head_weights, // must be pre-training initialized to zero
		laBackpropCalculations.alpha_to_gate_bias_by_head_weights, // must be pre-training initialized to zero
		laBackpropCalculations.alpha_gate,
		laBackpropCalculations.beta_gate,
		laCalcs.alpha_gate,
		laCalcs.alpha_softplus,
		laCalcs.beta_gate,
		laCalcs.alpha_activation,	
		laWeights.alpha_to_gate_by_head_weights, // alpha intermediate activation to alpha gate
		laWeights.alpha_to_gate_bias_by_head_weights, // alpha intermediate activation to alpha gate
		L
	);

	// tCalcs.x_postRMS1_post_gamma: [dim, L]
	// laWeights.alpha_weights: [dim, attnHeads]
	// laWeights.beta_weights: [dim, attnHeads]
	// laCalcs.alpha_activation: [L, attnHeads]
	// laCalcs.beta_activation: [L, attnHeads]

	/** Alpha weights and x_postRMS1_post_gamma grad **/	

	// dL/dA = G @ B.t
	// forward: (tCalcs.x_postRMS1_post_gamma)^T * laWeights.alpha_weights = laCalcs.alpha_activation
	// grad((tCalcs.x_postRMS1_post_gamma)^T) = alpha_activation_upgrad * laWeights.alpha_weights^T
	// grad(tCalcs.x_postRMS1_post_gamma) = (alpha_activation_upgrad * laWeights.alpha_weights^T)^T =
	// laWeights.alpha_weights [dim, attnHeads] * alpha_activation_upgrad^T [attnHeads, L]
	cublasGemmEx(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		dim, // rows C
		L, // cols C
		attnHeads, // contracting (shared) dim
		&alpha, // always 1
		laWeights.alpha_weights,
		CUDA_R_32F,
		dim, // lda, mem col size for col-major
		laBackpropCalculations.alpha_activation,
		CUDA_R_32F,
		L, // ldb, mem col size for col-major
		&beta_one, // 1 to accumulate with the existing x_postRMS1_post_gamma
		tBackpropCalculations.x_postRMS1_post_gamma, // must be initialized to zero at the beginning of every sequence!
		CUDA_R_32F,
		dim, // ldc, mem col size
		CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT           
	);

	// dL/dB = A.t @ G
	// laBackpropCalculations.alpha_weights [dim, attnHeads] = ((tCalcs.x_postRMS1_post_gamma)^T)^T [dim, L] @ alpha_activation_upgrad [L, attnHeads]
	cublasGemmEx(
	    handle,
	    CUBLAS_OP_N,
	    CUBLAS_OP_N,
	    dim, // rows C
	    attnHeads, // cols C
	    L, // contracting (shared) dim
	    &alpha,
	    tCalcs.x_postRMS1_post_gamma,
	    CUDA_R_32F,
	    dim, // lda, mem col size for col-major
	    laBackpropCalculations.alpha_activation,
	    CUDA_R_32F,
	    L, // ldb, mem col size for col-major
	    &beta_one, // 1 to accumulate with the existing alpha_weights
	    laBackpropCalculations.alpha_weights, // must be initialized to zero pre-training!
	    CUDA_R_32F,
	    dim, // ldc, mem col size
	    CUBLAS_COMPUTE_32F,
	    CUBLAS_GEMM_DEFAULT
	);


	/** Beta weights and x_postRMS1_post_gamma grad **/	

	// dL/dA = G @ B.t
	cublasGemmEx(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		dim, // rows C
		L, // cols C
		attnHeads, // contracting (shared) dim
		&alpha, // always 1
		laWeights.beta_weights,
		CUDA_R_32F,
		dim, // lda, mem col size for col-major
		laBackpropCalculations.beta_activation,
		CUDA_R_32F,
		L, // ldb, mem col size for col-major
		&beta_one, // 1 to accumulate with the existing x_postRMS1_post_gamma
		tBackpropCalculations.x_postRMS1_post_gamma, // must be initialized to zero at the beginning of every sequence!
		CUDA_R_32F,
		dim, // ldc, mem col size
		CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT           
	);

	// dL/dB = A.t @ G
	cublasGemmEx(
	    handle,
	    CUBLAS_OP_N,
	    CUBLAS_OP_N,
	    dim, // rows C
	    attnHeads, // cols C
	    L, // contracting (shared) dim
	    &alpha,
	    tCalcs.x_postRMS1_post_gamma,
	    CUDA_R_32F,
	    dim, // lda, mem col size for col-major
	    laBackpropCalculations.beta_activation,
	    CUDA_R_32F,
	    L, // ldb, mem col size for col-major
	    &beta_one, // 1 to accumulate with the existing alpha_weights
	    laBackpropCalculations.beta_weights, // must be initialized to zero pre-training!
	    CUDA_R_32F,
	    dim, // ldc, mem col size
	    CUBLAS_COMPUTE_32F,
	    CUBLAS_GEMM_DEFAULT
	);
}

void run_linear_attn_V_K_Q_unwinding_for_training(int tIndex, int linearAttnIndex, int L) {
	// no implemented yet - that's the L2 and conv backpropagation
}
