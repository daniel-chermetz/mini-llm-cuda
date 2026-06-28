#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "network_meta.h"
#include "network_globals.h"

__global__ void dLoss_dVocabScores_backprop(float* dLoss_d_vocabScores, float* vocabScores_postSoftmax, int* seqTokenIndices, int vocabSize_, int L_, int leftStartIndex, int rightEndIndex) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = vocabSize_ * L_ - 1;
    if (index > maxIndex) {
    	return;
    }

    int tokenIndex = index / vocabSize_;
    if (tokenIndex < leftStartIndex || tokenIndex > rightEndIndex) {
    	dLoss_d_vocabScores[index] = 0.0f;
    	return;
    }

    int vocabIndex = index - tokenIndex * vocabSize_;
    if (vocabIndex == seqTokenIndices[tokenIndex + 1]) {
    	dLoss_d_vocabScores[index] = vocabScores_postSoftmax[index] - 1.0f;
    } else {
    	dLoss_d_vocabScores[index] = vocabScores_postSoftmax[index];
    }
}

// transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1,
// transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_sumByCol_RMS2,
// ffn_sumByCol_RMS_DEVICE
__global__ void preCalcRMSColWideVals(
	float* sigma_scale_x_upGrad_byCol_RMS, 
	float* oneOverR_byCol_RMS, 
	float* oneOverColDimR3_byCol_RMS, 
	float* x_preRMS, 
	float* x_sumByCol_RMS,
	float* gamma_weights_RMS, 
	float* upstream_grad, 
	int colDim, 
	int L_
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = L_ - 1;
    if (index > maxIndex) {
    	return;
    }

    float x_sum_currentCol = x_sumByCol_RMS[index];
    oneOverR_byCol_RMS[index] = 1.0f / x_sum_currentCol;
    oneOverColDimR3_byCol_RMS[index] = 1.0f / ((float)colDim * x_sum_currentCol * x_sum_currentCol * x_sum_currentCol);

    int x_preRMS_coloffset = colDim * index;
    sigma_scale_x_upGrad_byCol_RMS[index] = 0;
	for (int i = 0; i < colDim; i++) {
		sigma_scale_x_upGrad_byCol_RMS[index] += (gamma_weights_RMS[i] * x_preRMS[x_preRMS_coloffset + i] * upstream_grad[x_preRMS_coloffset + i]);
	}    
}

__global__ void dLoss_dPreRMSNorm(
	float* x_pre_RMS_grad,
	float* x_pre_RMS,
	float* sigma_scale_x_upGrad_byCol_RMS, 
	float* oneOverR_byCol_RMS, 
	float* oneOverColDimR3_byCol_RMS,
	float* gamma_weights_RMS,
	float* upstream_grad,
	int colDim, 
	int L_
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (colDim * L_)) {
    	return;
    }

    int colIndex = index / colDim;
    int rowIndex = index - colIndex * colDim;

    x_pre_RMS_grad[index] = 
    	upstream_grad[index] * 
    	gamma_weights_RMS[rowIndex] * 
    	oneOverR_byCol_RMS[colIndex]
    	-
    	x_pre_RMS[index] *
    	sigma_scale_x_upGrad_byCol_RMS[colIndex] *
    	oneOverColDimR3_byCol_RMS[colIndex];
}

__global__ void dLoss_d_RMS_gamma_weights(
	float* x_postRMS_pre_gamma_grad,
	float* x_postRMS_pre_gamma,
	int rightEndIndex,
	float* upstream_grad,
	int colDim
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= colDim) {
    	return;
    }

    x_postRMS_pre_gamma_grad[index] = 0.0f;
    for (int colIndex = 0; colIndex <= rightEndIndex; colIndex++) {
    	int colOffset = colIndex * colDim;
		x_postRMS_pre_gamma_grad[index] += (x_postRMS_pre_gamma[colOffset + index] * upstream_grad[colOffset + index]);
    }
}

__global__ void preCalcRMSColWideVals_acrossHeads(
	float* sigma_scale_x_upGrad_byCol_RMS_byHead,
	float* oneOverR_byCol_RMS_byHead,
	float* oneOverHeadDimR3_byCol_RMS_byHead,
	float* x_preRMS,
	float* x_sumByCol_RMS_byHead,
	float* gamma_weights_RMS,
	float* upstream_grad,
	int L_
) {
    int colIndex = blockIdx.x;
    int headIndex = blockIdx.y;
    int colHeadIndex = attnHeads * colIndex + headIndex;
    int headOffset = headIndex * headDim;

    float x_sum_currentCol = x_sumByCol_RMS_byHead[colHeadIndex];
    oneOverR_byCol_RMS_byHead[colHeadIndex] = 1.0f / x_sum_currentCol;
    oneOverHeadDimR3_byCol_RMS_byHead[colHeadIndex] = 1.0f / ((float)headDim * x_sum_currentCol * x_sum_currentCol * x_sum_currentCol);

    int x_preRMS_coloffset = dim * colIndex + headOffset;
    sigma_scale_x_upGrad_byCol_RMS_byHead[colHeadIndex] = 0;
	for (int i = 0; i < headDim; i++) {
		sigma_scale_x_upGrad_byCol_RMS_byHead[colHeadIndex] += (gamma_weights_RMS[headOffset + i] * x_preRMS[x_preRMS_coloffset + i] * upstream_grad[x_preRMS_coloffset + i]);
	}
}

__global__ void dLoss_dPreRMSNorm_acrossHeads(
	float* x_pre_RMS_grad,
	float* x_pre_RMS,
	float* sigma_scale_x_upGrad_byCol_RMS_byHead,
	float* oneOverR_byCol_RMS_byHead,
	float* oneOverHeadDimR3_byCol_RMS_byHead,
	float* gamma_weights_RMS,
	float* upstream_grad
) {
	int colIndex = blockIdx.x; // L: 1280 at the most
	int headIndex = blockIdx.y; // attnHeads: 16 at the most
	int withinHeadRowIndex = threadIdx.x; // 256 at the most

	int colHeadIndex = colIndex * attnHeads + headIndex;
	int rowIndex = headIndex * headDim + withinHeadRowIndex;

	int globalIndex = colIndex * dim + rowIndex;

    x_pre_RMS_grad[globalIndex] =
    	upstream_grad[globalIndex] *
    	gamma_weights_RMS[rowIndex] *
    	oneOverR_byCol_RMS_byHead[colHeadIndex]
    	-
    	x_pre_RMS[globalIndex] *
    	sigma_scale_x_upGrad_byCol_RMS_byHead[colHeadIndex] *
    	oneOverHeadDimR3_byCol_RMS_byHead[colHeadIndex];
}

__global__ void dLoss_d_ffn_right_pre_hadamard_backprop(float* dLoss_d_ffn_right_1_postSilu, float* dLoss_d_ffn_right_2, float* ffn_right_1_postSilu, float* ffn_right_2, float* dLoss_d_ffn_right_postHadamard, int ffnDim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = ffnDim_ * L_ - 1;
    if (index > maxIndex) {
    	return;
    }

    dLoss_d_ffn_right_1_postSilu[index] = ffn_right_2[index] * dLoss_d_ffn_right_postHadamard[index];
    dLoss_d_ffn_right_2[index] = ffn_right_1_postSilu[index] * dLoss_d_ffn_right_postHadamard[index];
}

__global__ void dLoss_d_ffn_right_1_preSilu(float* dLoss_d_ffn_right_1_preSilu, float* ffn_right_1_postSilu, float* ffn_right_1_preSilu, float* dLoss_d_ffn_right_1_postSilu, int ffnDim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max = ffnDim_ * L_;
    if (index >= max) {
    	return;
    }

    float x = ffn_right_1_preSilu[index];
    float y = ffn_right_1_postSilu[index];
    float sigma = 1.0f / (1.0f + expf(-x));
    dLoss_d_ffn_right_1_preSilu[index] = dLoss_d_ffn_right_1_postSilu[index] * (y + sigma - y * sigma);
}

__global__ void dLoss_d_output_projection_residual_path(float* outputProjPlusResidual_backprop, float* dLoss_d_ffn_final_plus_residual, int dim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max = dim_ * L_;
    if (index >= max) {
    	return;
    }

    outputProjPlusResidual_backprop[index] += dLoss_d_ffn_final_plus_residual[index];
}

__global__ void preCalcSoftmaxGradSumByCol(float* softmaxGradSumByCol, float* attnByHead_postSoftmax, float* attnByHead_postSoftmax_upGrad, int attnHeads_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max = attnHeads_ * L_;
    if (index >= max) {
    	return;
    }

    int attnHeadIndex = index / L_;
    int headRelativeColIndex = index - attnHeadIndex * L_;
    int attnHeadOffset = attnHeadIndex * (L_ * L_);
    int globalColOffset = attnHeadOffset + headRelativeColIndex * L_;

    float sum = 0.0f;
    for (int rowIndex = 0; rowIndex <= headRelativeColIndex; rowIndex++) {
    	sum += attnByHead_postSoftmax[globalColOffset + rowIndex] * attnByHead_postSoftmax_upGrad[globalColOffset + rowIndex];
    }
    softmaxGradSumByCol[index] = sum;
}

__global__ void dLoss_d_pre_softmax_pre_div_by_sqrt_head_dim(float* preSoftmax_preDivSqrtHeadDim_grad, float* attnByHead_postSoftmax, float* softmaxGradSumByCol, float* attnByHead_postSoftmax_upGrad, int attnHeads_, int headDim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max = attnHeads_ * L_ * L_;
    if (index >= max) {
    	return;
    }

    int globalColIndex = index / L_;
    int attnHeadIndex = globalColIndex / L_;
    int headRelativeColIndex = globalColIndex - attnHeadIndex * L_;
    int rowIndex = index - globalColIndex * L_;

    if (rowIndex > headRelativeColIndex) {
    	// causal masking
    	preSoftmax_preDivSqrtHeadDim_grad[index] = 0;
    	return;
    }

    float invSqrtHeadDim = 1.0f / sqrtf((float)headDim_);
    preSoftmax_preDivSqrtHeadDim_grad[index] = invSqrtHeadDim * (attnByHead_postSoftmax[index] * attnByHead_postSoftmax_upGrad[index] - attnByHead_postSoftmax[index] * softmaxGradSumByCol[globalColIndex]);
}

// run only once
__global__ void preCalcRoPETheta(float* ropeThetaStore, int ropeBase, int dimPairs_, int headDim_, int maxL_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max = dimPairs_ * maxL_;
    if (index >= max) {
    	return;
    }

    int colIndex = index / dimPairs_;
    int pairIndex = index - colIndex * dimPairs_;
    int pairsPerHead = headDim_ / 2;
    int headIndex = pairIndex / pairsPerHead;    
    int headRelativePairIndex = pairIndex - headIndex * pairsPerHead;

    float wi = powf((float)ropeBase, (-2.0f * (float)headRelativePairIndex / (float)headDim_));
    ropeThetaStore[index * 2] = cosf((float)colIndex * wi);
    ropeThetaStore[index * 2 + 1] = sinf((float)colIndex * wi);
}

__global__ void dLoss_d_pre_rope(float* preRoPE_grad, float* ropeThetaStore, float* postRoPE_upGrad, int dimPairs_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max = dimPairs_ * L_;
    if (index >= max) {
    	return;
    }

    float cosTheta = ropeThetaStore[index * 2];
    float sinTheta = ropeThetaStore[index * 2 + 1];

    float evenOfPairUpGrad = postRoPE_upGrad[index * 2];
    float oddOfPairUpGrad = postRoPE_upGrad[index * 2 + 1];

    // column major, [attnHeads * pairsPerHead * 2, L] : [dim, L], index: 0 to attnHeads * pairsPerHead * L
    preRoPE_grad[index * 2] = evenOfPairUpGrad * cosTheta + oddOfPairUpGrad * sinTheta;
    preRoPE_grad[index * 2 + 1] = -evenOfPairUpGrad * sinTheta + oddOfPairUpGrad * cosTheta;
}

__global__ void add_residual_path_upGrad_to_x_gradient(float* x_grad, float* outputProjPlusResidual_upGrad, int dim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max = dim_ * L_;
    if (index >= max) {
    	return;
    }

    x_grad[index] += outputProjPlusResidual_upGrad[index];
}

__global__ void add_x_grad_to_embeddings_grad(float* embedding_weights_grad, float* x_grad, int* seqTokenIndicesInFullEmbeddings, int dim_, int vocabSize_, int rightEndIndex) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max = dim_ * (rightEndIndex + 1);
    if (index >= max) {
    	return;
    }

    int seqTokenIndex = index / dim_;
    int featureIndex = index - seqTokenIndex * dim_;
    int embeddingIndex = seqTokenIndicesInFullEmbeddings[seqTokenIndex];

    // OLD: embedding_weights_grad[embeddingIndex * dim_ + featureIndex] += x_grad[leftOffset + index];
    atomicAdd(&embedding_weights_grad[embeddingIndex * dim_ + featureIndex], x_grad[index]);
}

__global__ void dLoss_d_hadamard_valueScaledSoftmaxAttn_gatedQueries(float* gatedQueriesPostSigmoid_grad, float* valueScaledSoftmaxAttn_grad, float* gatedValueScaledSoftmaxAttn_grad, float* valueScaledSoftmaxAttn, float* gatedQueriesPostSigmoid, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxCount = dim * L_;
    if (index >= maxCount) {
    	return;
    }

    valueScaledSoftmaxAttn_grad[index] = gatedQueriesPostSigmoid[index] * gatedValueScaledSoftmaxAttn_grad[index];
    gatedQueriesPostSigmoid_grad[index] = valueScaledSoftmaxAttn[index] * gatedValueScaledSoftmaxAttn_grad[index];
}

__global__ void dLoss_d_gatedQueriesPreSigmoid(float* gatedQueriesPreSigmoid_grad, float* gatedQueriesPostSigmoid_grad, float* gatedQueriesPostSigmoid, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxCount = dim * L_;
    if (index >= maxCount) {
    	return;
    }

    float s = gatedQueriesPostSigmoid[index];
    gatedQueriesPreSigmoid_grad[index] = (s * (1.0f - s) * gatedQueriesPostSigmoid_grad[index]);
}

void setupRoPEThetaStore(int maxL_) {
	int xTotalThreads = dim / 2 * maxL_;
    int numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;	
	preCalcRoPETheta<<<numBlocks, threadsPerBlock>>>(
		ropeThetaStore_DEVICE, ropeDenomBase, dimPairs, headDim, maxL_
	);
}

void getGradientsForTraining(int leftStartIndex, int rightEndIndex, int L) {
	int xTotalThreads = vocabSize * L;
    int numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	dLoss_dVocabScores_backprop<<<numBlocks, threadsPerBlock>>>(
		dLoss_d_vocabScores, 
		vocabScores_postSoftmax_DEVICE, 
		seqTokenIndices_DEVICE, 
		vocabSize, 
		L, 
		leftStartIndex, 
		rightEndIndex
	);

	// (dLoss/dVocabScores) * (dVocabScores/d_ffn_final_postRMS_DEVICE)
	// embedding_weights: [vocabSize, dim]; ffn_postRMS_DEVICE: [dim, L]; dLoss_dVocabScores: [vocabSize, L]
	// vocab.T @ (dLoss/dVocabScores)
	// i.e. changing B[3][5] updates the 5th column of the result by the 3rd col of A
	// vocab.T[3rd row - has been 3rd col] @ dLoss/dVocabScores[5th col] --> [3, 5] of the result
	// changing B[3][5] --> found in [3, 5] of backprop matrix
	cublasGemmEx(
	    handle,
	    CUBLAS_OP_N,
	    CUBLAS_OP_N,
	    dim, // rows C
	    L, // cols C
	    vocabSize, // contracting (shared) dim
	    &alpha,
	    embedding_weights_DEVICE, // [dim, vocabSize]
	    CUDA_R_32F,
	    dim, // lda, mem col size for col-major
	    dLoss_d_vocabScores, // [vocabSize, L]
	    CUDA_R_32F,
	    vocabSize, // ldb, mem col size for col-major      
	    &beta,
	    dLoss_d_ffn_final_postRMS_postGamma,
	    CUDA_R_32F,
	    dim, // ldc, mem col size
	    CUBLAS_COMPUTE_32F,
	    CUBLAS_GEMM_DEFAULT             
	);

	// (dLoss/dVocabScores) * (dVocabScores/dEmbedding_weights)
	// embedding_weights: [vocabSize, dim]; ffn_postRMS_DEVICE: [dim, L]; dLoss_dVocabScores: [vocabSize, L]
	// (dLoss/dVocabScores) @ ffn_postRMS_DEVICE.T
	// i.e. changing A[3][5] updates the 3th row of the result by the 5th row of B
	// (dLoss/dVocabScores)[3rd row] @ ffn_postRMS_DEVICE.T[5th col - has been been 5th row] of the result
	// changing A[3][5] --> found in [3, 5] of backprop matrix

	// dL/d(A_forward_pass) = dL/dC * dC / d(A_forward_pass) = G * B.t,
	// dL/d(A_forward_pass_transpose) = dL/dC * dC / d(A_forward_pass_transpose) = B [dim, L] * G.t [L, vocabSize]
	cublasGemmEx(
	    handle,
	    CUBLAS_OP_N,
	    CUBLAS_OP_T,
	    dim, // rows C
	    vocabSize, // cols C
	    L, // contracting (shared) dim
	    &alpha,
	    ffn_postRMS_post_gamma_DEVICE,
	    CUDA_R_32F,
	    dim, // lda, mem col size for col-major
	    dLoss_d_vocabScores, // [vocanSize, L].t
	    CUDA_R_32F,
	    vocabSize, // ldb, mem col size for col-major      
	    &beta,
	    dLoss_d_embedding_weights, // [dim, vocabSize]
	    CUDA_R_32F,
	    dim, // ldc, mem col size
	    CUBLAS_COMPUTE_32F,
	    CUBLAS_GEMM_DEFAULT             
	);

	xTotalThreads =  L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	preCalcRMSColWideVals<<<numBlocks, threadsPerBlock>>>(
		ffn_final_sigma_scale_x_upGrad_byCol_RMS,
		ffn_final_oneOverR_byCol_RMS,
		ffn_final_oneOverColDimR3_byCol_RMS,
		transformerCalculations_DEVICE[0].ffnPlusResidual, // zero is actually the highest transformer (legacy from old gpu.js code, kept for compataility previously trained models)
		ffn_sumByCol_RMS_DEVICE,
		final_rms_weights_DEVICE,
		dLoss_d_ffn_final_postRMS_postGamma,
		dim,
		L
	);
	xTotalThreads = dim * L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	dLoss_dPreRMSNorm<<<numBlocks, threadsPerBlock>>>(
		backpropCalculations[0].ffn_final_plus_residual, // pre-RMS
		transformerCalculations_DEVICE[0].ffnPlusResidual, // zero is actually the highest transformer (legacy from old gpu.js code, kept for compataility previously trained models)
		ffn_final_sigma_scale_x_upGrad_byCol_RMS, 
		ffn_final_oneOverR_byCol_RMS, 
		ffn_final_oneOverColDimR3_byCol_RMS,
		final_rms_weights_DEVICE,
		dLoss_d_ffn_final_postRMS_postGamma,
		dim, 
		L
	);
	xTotalThreads = dim;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	dLoss_d_RMS_gamma_weights<<<numBlocks, threadsPerBlock>>>(
		dLoss_d_ffn_final_RMS_gamma_weights,
		ffn_postRMS_pre_gamma_DEVICE,
		rightEndIndex,
		dLoss_d_ffn_final_postRMS_postGamma,
		dim
	);	

	for (int tIndex = 0; tIndex < transformers; tIndex++) {
		// (dLoss/d_ffn_final) * (d_ffn_final/d_ffn_right_postHadamard)
		// transformerWeights_DEVICE[tIndex].ffn_left_weights: [dim, ffnDim],
		// transformerCalculations_DEVICE[tIndex].ffn_right_postHadamard: [ffnDim, L],
		// transformerCalculations_DEVICE[tIndex].ffn_final [dim L],
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_T,
		    CUBLAS_OP_N,
		    ffnDim, // rows C
		    L, // cols C
		    dim, // contracting (shared) dim
		    &alpha,
		    transformerWeights_DEVICE[tIndex].ffn_left_weights,
		    CUDA_R_32F,
		    dim, // lda, mem col size for col-major
		    backpropCalculations[tIndex].ffn_final_plus_residual, // implicit dLoss/d_ffn_final_plus_residual
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major      
		    &beta,
		    backpropCalculations[tIndex].ffn_right_postHadamard, // implicit dLoss/d_ffn_right_postHadamard
		    CUDA_R_32F,
		    ffnDim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);

		// (dLoss/d_ffn_final) * (d_ffn_final/ffn_left_weights)
		// transformerWeights_DEVICE[tIndex].ffn_left_weights: [dim, ffnDim],
		// transformerCalculations_DEVICE[tIndex].ffn_right_postHadamard: [ffnDim, L],
		// transformerCalculations_DEVICE[tIndex].ffn_final [dim, L],
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_N,
		    CUBLAS_OP_T,
		    dim, // rows C
		    ffnDim, // cols C
		    L, // contracting (shared) dim
		    &alpha,
		    backpropCalculations[tIndex].ffn_final_plus_residual,
		    CUDA_R_32F,
		    dim, // lda, mem col size for col-major
		    transformerCalculations_DEVICE[tIndex].ffn_right_postHadamard,
		    CUDA_R_32F,
		    ffnDim, // ldb, mem col size for col-major      
		    &beta,
		    backpropCalculations[tIndex].ffn_left_weights,
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);

		xTotalThreads = ffnDim * L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		dLoss_d_ffn_right_pre_hadamard_backprop<<<numBlocks, threadsPerBlock>>>(
			backpropCalculations[tIndex].ffn_right_1_postSilu, 
			backpropCalculations[tIndex].ffn_right_2, 
			transformerCalculations_DEVICE[tIndex].ffn_right_1_postSilu, 
			transformerCalculations_DEVICE[tIndex].ffn_right_2, 
			backpropCalculations[tIndex].ffn_right_postHadamard, 
			ffnDim, 
			L
		);

		xTotalThreads = ffnDim * L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		dLoss_d_ffn_right_1_preSilu<<<numBlocks, threadsPerBlock>>>(
			backpropCalculations[tIndex].ffn_right_1_preSilu,
			transformerCalculations_DEVICE[tIndex].ffn_right_1_postSilu, 
			transformerCalculations_DEVICE[tIndex].ffn_right_1_preSilu,
			backpropCalculations[tIndex].ffn_right_1_postSilu, 
			ffnDim,
			L
		);

		// dL/dB = A.t @ G, dL/dA = G @ B.t

		// dL/dB = A.t @ G
		// (dLoss / d_ffn_right_1_preSilu) * (d_ffn_right_1_preSilu / d_ffn_right_1_x)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_T,
		    CUBLAS_OP_N,
		    dim, // rows C
		    L, // cols C
		    ffnDim, // contracting (shared) dim
		    &alpha,
		    transformerWeights_DEVICE[tIndex].ffn_right_1_weights,
		    CUDA_R_32F,
		    ffnDim, // lda, mem col size for col-major
		    backpropCalculations[tIndex].ffn_right_1_preSilu,
		    CUDA_R_32F,
		    ffnDim, // ldb, mem col size for col-major      
		    &beta,
		    backpropCalculations[tIndex].outputProjPlusResidual_postRMS2_post_gamma,
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);

		// dL/dA = G [ffnDim, L] @ B.t [L, dim]
		// (dLoss / d_ffn_right_1_preSilu) * (d_ffn_right_1_preSilu / d_ffn_right_1_weights)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_N,
		    CUBLAS_OP_T,
		    ffnDim, // rows C
		    dim, // cols C
		    L, // contracting (shared) dim
		    &alpha,
		    backpropCalculations[tIndex].ffn_right_1_preSilu,
		    CUDA_R_32F,
		    ffnDim, // lda, mem col size for col-major
		    transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2_post_gamma,
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major      
		    &beta,
		    backpropCalculations[tIndex].ffn_right_1_weights,
		    CUDA_R_32F,
		    ffnDim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);

		// dL/dB [dim, L] = A.t [dim, ffnDim] @ G [ffnDim, L]
		// (dLoss / d_ffn_right_2) * (d_ffn_right_2 / d_ffn_right_2_x)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_T,
		    CUBLAS_OP_N,
		    dim, // rows C
		    L, // cols C
		    ffnDim, // contracting (shared) dim
		    &alpha,
		    transformerWeights_DEVICE[tIndex].ffn_right_2_weights,
		    CUDA_R_32F,
		    ffnDim, // lda, mem col size for col-major
		    backpropCalculations[tIndex].ffn_right_2,
		    CUDA_R_32F,
		    ffnDim, // ldb, mem col size for col-major      
		    &beta_one,
		    backpropCalculations[tIndex].outputProjPlusResidual_postRMS2_post_gamma,
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);

		// dL/dA = G [ffnDim, L] @ B.t [L, dim]
		// (dLoss / d_ffn_right_2) * (d_ffn_right_2 / d_ffn_right_2_weights)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_N,
		    CUBLAS_OP_T,
		    ffnDim, // rows C
		    dim, // cols C
		    L, // contracting (shared) dim
		    &alpha,
		    backpropCalculations[tIndex].ffn_right_2,
		    CUDA_R_32F,
		    ffnDim, // lda, mem col size for col-major
		    transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2_post_gamma,
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major      
		    &beta,
		    backpropCalculations[tIndex].ffn_right_2_weights,
		    CUDA_R_32F,
		    ffnDim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);

		xTotalThreads = L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		preCalcRMSColWideVals<<<numBlocks, threadsPerBlock>>>(
			backpropCalculations[tIndex].rms2_sigma_scale_x_upGrad_byCol_RMS, 
			backpropCalculations[tIndex].rms2_oneOverR_byCol_RMS, 
			backpropCalculations[tIndex].rms2_oneOverColDimR3_byCol_RMS, 
			transformerCalculations_DEVICE[tIndex].outputProjPlusResidual, 
			transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_sumByCol_RMS2,
			transformerWeights_DEVICE[tIndex].rms2_weights, 
			backpropCalculations[tIndex].outputProjPlusResidual_postRMS2_post_gamma, 
			dim, L
		);
		xTotalThreads = dim * L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		dLoss_dPreRMSNorm<<<numBlocks, threadsPerBlock>>>(
			backpropCalculations[tIndex].outputProjPlusResidual,
			transformerCalculations_DEVICE[tIndex].outputProjPlusResidual,
			backpropCalculations[tIndex].rms2_sigma_scale_x_upGrad_byCol_RMS, 
			backpropCalculations[tIndex].rms2_oneOverR_byCol_RMS, 
			backpropCalculations[tIndex].rms2_oneOverColDimR3_byCol_RMS, 
			transformerWeights_DEVICE[tIndex].rms2_weights, 
			backpropCalculations[tIndex].outputProjPlusResidual_postRMS2_post_gamma,
			dim, L
		);
		xTotalThreads = dim;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		dLoss_d_RMS_gamma_weights<<<numBlocks, threadsPerBlock>>>(
			backpropCalculations[tIndex].rms2_gamma_weights,
			transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2_pre_gamma,
			rightEndIndex,
			backpropCalculations[tIndex].outputProjPlusResidual_postRMS2_post_gamma,
			dim
		);

		xTotalThreads = dim * L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		dLoss_d_output_projection_residual_path<<<numBlocks, threadsPerBlock>>>(
			backpropCalculations[tIndex].outputProjPlusResidual, 
			backpropCalculations[tIndex].ffn_final_plus_residual, 
			dim, L
		);

		// dL/dB = A.t [dim, dim] @ G [dim, L]
		// (dLoss / d_outputProjPlusResidual) / (d_outputProjPlusResidual / d_valueScaledSoftmaxAttn)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_T,
		    CUBLAS_OP_N,
		    dim, // rows C
		    L, // cols C
		    dim, // contracting (shared) dim
		    &alpha,
		    transformerWeights_DEVICE[tIndex].output_proj_weights,
		    CUDA_R_32F,
		    dim, // lda, mem col size for col-major
		    backpropCalculations[tIndex].outputProjPlusResidual,
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major
		    &beta,
		    (CONFIG_QUERY_GATING ? backpropCalculations[tIndex].gatedValueScaledSoftmaxAttn : backpropCalculations[tIndex].valueScaledSoftmaxAttn),
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);

		// dL/dA [dim, dim] = G [dim, L] @ B.t [L, dim]
		// (dLoss / d_outputProjPlusResidual) / (d_outputProjPlusResidual / d_output_proj_weights)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_N,
		    CUBLAS_OP_T,
		    dim, // rows C
		    dim, // cols C
		    L, // contracting (shared) dim
		    &alpha,
		    backpropCalculations[tIndex].outputProjPlusResidual,
		    CUDA_R_32F,
		    dim, // lda, mem col size for col-major
		    (CONFIG_QUERY_GATING ? transformerCalculations_DEVICE[tIndex].gatedValueScaledSoftmaxAttn : transformerCalculations_DEVICE[tIndex].valueScaledSoftmaxAttn),
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major
		    &beta,
		    backpropCalculations[tIndex].output_proj_weights,
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);

		if (CONFIG_QUERY_GATING) {
			dLoss_d_hadamard_valueScaledSoftmaxAttn_gatedQueries<<<numBlocks, threadsPerBlock>>>(
				backpropCalculations[tIndex].gatedQueriesPostSigmoid,
				backpropCalculations[tIndex].valueScaledSoftmaxAttn,
				backpropCalculations[tIndex].gatedValueScaledSoftmaxAttn,
				transformerCalculations_DEVICE[tIndex].valueScaledSoftmaxAttn,
				transformerCalculations_DEVICE[tIndex].gatedQueriesPostSigmoid,
				L
			);

			dLoss_d_gatedQueriesPreSigmoid<<<numBlocks, threadsPerBlock>>>(
				backpropCalculations[tIndex].gatedQueriesPreSigmoid,
				backpropCalculations[tIndex].gatedQueriesPostSigmoid,
				transformerCalculations_DEVICE[tIndex].gatedQueriesPostSigmoid,
				L
			);
		}

		// dL/dA [dim, L] = G [dim, L] @ B.t [L, L]
		// (dLoss / d_valueScaledSoftmaxAttn) / (d_valueScaledSoftmaxAttn / d_values)	
	    cublasGemmStridedBatchedEx(
	        handle,
	        CUBLAS_OP_N,
	        CUBLAS_OP_T,
	        headDim, // rows C 
	        L, // cols C
	        L, // contracting (shared) dim
	        &alpha,
	        backpropCalculations[tIndex].valueScaledSoftmaxAttn,
	        CUDA_R_32F,
	        dim, // lda, mem col size
	        headDim, // mem stride to reach next head
	        transformerCalculations_DEVICE[tIndex].attnByHead_postSoftmax,
	        CUDA_R_32F,
	        L, // ldb, col size in mem for col-major
	        (L * L), // mem stride to reach next head
	        &beta,
	        backpropCalculations[tIndex].values,
	        CUDA_R_32F,
	        dim, // ldc, col size in mem
	        headDim, // mem stride to reach next head
	        attnHeads,
	        CUBLAS_COMPUTE_32F,
	        CUBLAS_GEMM_DEFAULT
	    );

	    // CRITICAL: must mask to zero where row > col (in a later kernel, together with pre-softmax)
		// dL/dB [L, L] = A.t [L, headDim] @ G [headDim, L]
		// (dLoss / d_valueScaledSoftmaxAttn) / (d_valueScaledSoftmaxAttn / d_attnByHead_postSoftmax)	
	    cublasGemmStridedBatchedEx(
	        handle,
	        CUBLAS_OP_T,
	        CUBLAS_OP_N,
	        L, // rows C 
	        L, // cols C
	        headDim, // contracting (shared) dim
	        &alpha,
	        transformerCalculations_DEVICE[tIndex].values,
	        CUDA_R_32F,
	        dim, // lda, mem col size
	        headDim, // mem stride to reach next head
	        backpropCalculations[tIndex].valueScaledSoftmaxAttn,
	        CUDA_R_32F,
	        dim, // ldb, col size in mem for col-major
	        headDim, // mem stride to reach next head
	        &beta,
	        backpropCalculations[tIndex].attnByHead_postSoftmax,
	        CUDA_R_32F,
	        L, // ldc, col size in mem
	        L * L, // mem stride to reach next head
	        attnHeads,
	        CUBLAS_COMPUTE_32F,
	        CUBLAS_GEMM_DEFAULT
	    );

		xTotalThreads = attnHeads * L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	    preCalcSoftmaxGradSumByCol<<<numBlocks, threadsPerBlock>>>(
	    	backpropCalculations[tIndex].attnSoftmaxGradSumByCol, 
	    	transformerCalculations_DEVICE[tIndex].attnByHead_postSoftmax, 
	    	backpropCalculations[tIndex].attnByHead_postSoftmax, 
	    	attnHeads, L
	    );
		xTotalThreads = attnHeads * L * L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		dLoss_d_pre_softmax_pre_div_by_sqrt_head_dim<<<numBlocks, threadsPerBlock>>>(
			backpropCalculations[tIndex].attnKtQByHead, 
			transformerCalculations_DEVICE[tIndex].attnByHead_postSoftmax, 
			backpropCalculations[tIndex].attnSoftmaxGradSumByCol, 
			backpropCalculations[tIndex].attnByHead_postSoftmax, 
			attnHeads, headDim, L
		);

		// dL/d(K.t) [L, headDim] = G [L, L] @ Q.t [L, headDim]
		// dL/dK [headDim, L] = Q [headDim, L] * G.t [L, L]
		// (dLoss / d_attnKtQByHead) / (d_attnKtQByHead / d_K)	
	    cublasGemmStridedBatchedEx(
	        handle,
	        CUBLAS_OP_N,
	        CUBLAS_OP_T,
	        headDim, // rows C 
	        L, // cols C
	        L, // contracting (shared) dim
	        &alpha,
	        transformerCalculations_DEVICE[tIndex].queriesPostRoPE,
	        CUDA_R_32F,
	        dim, // lda, mem col size
	        headDim, // mem stride to reach next head
	        backpropCalculations[tIndex].attnKtQByHead,
	        CUDA_R_32F,
	        L, // ldb, col size in mem for col-major
	        (L * L), // mem stride to reach next head
	        &beta,
	        backpropCalculations[tIndex].keysPostRoPE,
	        CUDA_R_32F,
	        dim, // ldc, col size in mem
	        headDim, // mem stride to reach next head
	        attnHeads,
	        CUBLAS_COMPUTE_32F,
	        CUBLAS_GEMM_DEFAULT
	    );

	    // dL/dQ [headDim, L] = (K.t).t [headDim, L] @ G [L, L] 
		// (dLoss / d_attnKtQByHead) / (d_attnKtQByHead / d_Q)	
	    cublasGemmStridedBatchedEx(
	        handle,
	        CUBLAS_OP_N,
	        CUBLAS_OP_N,
	        headDim, // rows C 
	        L, // cols C
	        L, // contracting (shared) dim
	        &alpha,
	        transformerCalculations_DEVICE[tIndex].keysPostRoPE,
	        CUDA_R_32F,
	        dim, // lda, mem col size
	        headDim, // mem stride to reach next head
	        backpropCalculations[tIndex].attnKtQByHead,
	        CUDA_R_32F,
	        L, // ldb, col size in mem for col-major
	        (L * L), // mem stride to reach next head
	        &beta,
	        backpropCalculations[tIndex].queriesPostRoPE,
	        CUDA_R_32F,
	        dim, // ldc, col size in mem
	        headDim, // mem stride to reach next head
	        attnHeads,
	        CUBLAS_COMPUTE_32F,
	        CUBLAS_GEMM_DEFAULT
	    );

		xTotalThreads = dimPairs * L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	    dLoss_d_pre_rope<<<numBlocks, threadsPerBlock>>>(
	    	backpropCalculations[tIndex].keysPreRoPE, 
	    	ropeThetaStore_DEVICE, 
	    	backpropCalculations[tIndex].keysPostRoPE, 
	    	dimPairs, 
	    	L
	    );
	    dLoss_d_pre_rope<<<numBlocks, threadsPerBlock>>>(
	    	backpropCalculations[tIndex].queriesPreRoPE, 
	    	ropeThetaStore_DEVICE, 
	    	backpropCalculations[tIndex].queriesPostRoPE, 
	    	dimPairs, 
	    	L
	    );

	    if (CONFIG_QK_RMS_NORM) {
            dim3 qkRMSGridDim(L, attnHeads);

			preCalcRMSColWideVals_acrossHeads<<<qkRMSGridDim, 1>>>(
				backpropCalculations[tIndex].rms_queries_sigma_scale_x_upGrad_byCol_RMS_byHead,
				backpropCalculations[tIndex].rms_queries_oneOverR_byCol_RMS_byHead,
				backpropCalculations[tIndex].rms_queries_oneOverHeadDimR3_byCol_RMS_byHead,
				transformerCalculations_DEVICE[tIndex].queries,
				transformerCalculations_DEVICE[tIndex].queries_RMS_sumByColByHead,
				transformerWeights_DEVICE[tIndex].query_RMS_weights,
				backpropCalculations[tIndex].queriesPreRoPE,
				L
			);

			preCalcRMSColWideVals_acrossHeads<<<qkRMSGridDim, 1>>>(
				backpropCalculations[tIndex].rms_keys_sigma_scale_x_upGrad_byCol_RMS_byHead,
				backpropCalculations[tIndex].rms_keys_oneOverR_byCol_RMS_byHead,
				backpropCalculations[tIndex].rms_keys_oneOverHeadDimR3_byCol_RMS_byHead,
				transformerCalculations_DEVICE[tIndex].keys,
				transformerCalculations_DEVICE[tIndex].keys_RMS_sumByColByHead,
				transformerWeights_DEVICE[tIndex].key_RMS_weights,
				backpropCalculations[tIndex].keysPreRoPE,
				L
			);

			dLoss_dPreRMSNorm_acrossHeads<<<qkRMSGridDim, headDim>>>(
				backpropCalculations[tIndex].queriesPreRoPE_preRMS,
				transformerCalculations_DEVICE[tIndex].queries,
				backpropCalculations[tIndex].rms_queries_sigma_scale_x_upGrad_byCol_RMS_byHead,
				backpropCalculations[tIndex].rms_queries_oneOverR_byCol_RMS_byHead,
				backpropCalculations[tIndex].rms_queries_oneOverHeadDimR3_byCol_RMS_byHead,
				transformerWeights_DEVICE[tIndex].query_RMS_weights,
				backpropCalculations[tIndex].queriesPreRoPE
			);

			dLoss_dPreRMSNorm_acrossHeads<<<qkRMSGridDim, headDim>>>(
				backpropCalculations[tIndex].keysPreRoPE_preRMS,
				transformerCalculations_DEVICE[tIndex].keys,
				backpropCalculations[tIndex].rms_keys_sigma_scale_x_upGrad_byCol_RMS_byHead,
				backpropCalculations[tIndex].rms_keys_oneOverR_byCol_RMS_byHead,
				backpropCalculations[tIndex].rms_keys_oneOverHeadDimR3_byCol_RMS_byHead,
				transformerWeights_DEVICE[tIndex].key_RMS_weights,
				backpropCalculations[tIndex].keysPreRoPE
			);

			xTotalThreads = dim;
	    	numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;

			dLoss_d_RMS_gamma_weights<<<numBlocks, threadsPerBlock>>>(
				backpropCalculations[tIndex].query_gamma_weights,
				transformerCalculations_DEVICE[tIndex].queries_post_RMS_pre_gamma,
				rightEndIndex,
				backpropCalculations[tIndex].queriesPreRoPE,
				dim
	    	);

			dLoss_d_RMS_gamma_weights<<<numBlocks, threadsPerBlock>>>(
				backpropCalculations[tIndex].key_gamma_weights,
				transformerCalculations_DEVICE[tIndex].keys_post_RMS_pre_gamma,
				rightEndIndex,
				backpropCalculations[tIndex].keysPreRoPE,
				dim
	    	);
	    }

		// dL/dA [dim, dim] = G [dim, L] @ B.t [L, dim]
		// (dLoss / d_values) / (d_values / d_value_weights)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_N,
		    CUBLAS_OP_T,
		    dim, // rows C
		    dim, // cols C
		    L, // contracting (shared) dim
		    &alpha,
		    backpropCalculations[tIndex].values,
		    CUDA_R_32F,
		    dim, // lda, mem col size for col-major
		    transformerCalculations_DEVICE[tIndex].x_postRMS1_post_gamma,
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major
		    &beta,
		    backpropCalculations[tIndex].value_weights,
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);
		// dL/dA [dim, dim] = G [dim, L] @ B.t [L, dim]
		// (dLoss / d_keysPreRoPE) / (d_keysPreRoPE / d_key_weights)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_N,
		    CUBLAS_OP_T,
		    dim, // rows C
		    dim, // cols C
		    L, // contracting (shared) dim
		    &alpha,
		    (CONFIG_QK_RMS_NORM ? backpropCalculations[tIndex].keysPreRoPE_preRMS : backpropCalculations[tIndex].keysPreRoPE),
		    CUDA_R_32F,
		    dim, // lda, mem col size for col-major
		    transformerCalculations_DEVICE[tIndex].x_postRMS1_post_gamma,
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major
		    &beta,
		    backpropCalculations[tIndex].key_weights,
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);
		// dL/dA [dim, dim] = G [dim, L] @ B.t [L, dim]
		// (dLoss / d_queriesPreRoPE) / (d_queriesPreRoPE / d_queries_weights)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_N,
		    CUBLAS_OP_T,
		    dim, // rows C
		    dim, // cols C
		    L, // contracting (shared) dim
		    &alpha,
		    (CONFIG_QK_RMS_NORM ? backpropCalculations[tIndex].queriesPreRoPE_preRMS : backpropCalculations[tIndex].queriesPreRoPE),
		    CUDA_R_32F,
		    dim, // lda, mem col size for col-major
		    transformerCalculations_DEVICE[tIndex].x_postRMS1_post_gamma,
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major
		    &beta,
		    backpropCalculations[tIndex].query_weights,
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);
		if (CONFIG_QUERY_GATING) {
			// dL/dA [dim, dim] = G [dim, L] @ B.t [L, dim]
			// (dLoss / d_gatedQueriesPreSigmoid) / (d_gatedQueriesPreSigmoid / d_gated_query_weights)
			cublasGemmEx(
			    handle,
			    CUBLAS_OP_N,
			    CUBLAS_OP_T,
			    dim, // rows C
			    dim, // cols C
			    L, // contracting (shared) dim
			    &alpha,
			    backpropCalculations[tIndex].gatedQueriesPreSigmoid,
			    CUDA_R_32F,
			    dim, // lda, mem col size for col-major
			    transformerCalculations_DEVICE[tIndex].x_postRMS1_post_gamma,
			    CUDA_R_32F,
			    dim, // ldb, mem col size for col-major
			    &beta,
			    backpropCalculations[tIndex].gated_query_weights,
			    CUDA_R_32F,
			    dim, // ldc, mem col size
			    CUBLAS_COMPUTE_32F,
			    CUBLAS_GEMM_DEFAULT
			);
		}

		// dL/dB [dim, L] = A.t [dim, dim] @ G [dim, L]
		// (dLoss / d_values) / (d_values / d_x)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_T,
		    CUBLAS_OP_N,
		    dim, // rows C
		    L, // cols C
		    dim, // contracting (shared) dim
		    &alpha,
		    transformerWeights_DEVICE[tIndex].value_weights,
		    CUDA_R_32F,
		    dim, // lda, mem col size for col-major
		    backpropCalculations[tIndex].values,
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major
		    &beta,
		    backpropCalculations[tIndex].x_postRMS1_post_gamma,
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT
		);
		// dL/dB [dim, L] = A.t [dim, dim] @ G [dim, L]
		// (dLoss / d_keysPreRoPE) / (d_keysPreRoPE / d_x)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_T,
		    CUBLAS_OP_N,
		    dim, // rows C
		    L, // cols C
		    dim, // contracting (shared) dim
		    &alpha,
		    transformerWeights_DEVICE[tIndex].key_weights,
		    CUDA_R_32F,
		    dim, // lda, mem col size for col-major
		    (CONFIG_QK_RMS_NORM ? backpropCalculations[tIndex].keysPreRoPE_preRMS : backpropCalculations[tIndex].keysPreRoPE),
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major
		    &beta_one,
		    backpropCalculations[tIndex].x_postRMS1_post_gamma,
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);
		// dL/dB [dim, L] = A.t [dim, dim] @ G [dim, L]
		// (dLoss / d_queriesPreRoPE) / (d_queriesPreRoPE / d_x)
		cublasGemmEx(
		    handle,
		    CUBLAS_OP_T,
		    CUBLAS_OP_N,
		    dim, // rows C
		    L, // cols C
		    dim, // contracting (shared) dim
		    &alpha,
		    transformerWeights_DEVICE[tIndex].query_weights,
		    CUDA_R_32F,
		    dim, // lda, mem col size for col-major
		    (CONFIG_QK_RMS_NORM ? backpropCalculations[tIndex].queriesPreRoPE_preRMS : backpropCalculations[tIndex].queriesPreRoPE),
		    CUDA_R_32F,
		    dim, // ldb, mem col size for col-major
		    &beta_one,
		    backpropCalculations[tIndex].x_postRMS1_post_gamma,
		    CUDA_R_32F,
		    dim, // ldc, mem col size
		    CUBLAS_COMPUTE_32F,
		    CUBLAS_GEMM_DEFAULT             
		);

		if (CONFIG_QUERY_GATING) {
			// dL/dB [dim, L] = A.t [dim, dim] @ G [dim, L]
			// (dLoss / d_gatedQueriesPreSigmoid) / (d_gatedQueriesPreSigmoid / d_x)
			cublasGemmEx(
			    handle,
			    CUBLAS_OP_T,
			    CUBLAS_OP_N,
			    dim, // rows C
			    L, // cols C
			    dim, // contracting (shared) dim
			    &alpha,
			    transformerWeights_DEVICE[tIndex].gated_query_weights,
			    CUDA_R_32F,
			    dim, // lda, mem col size for col-major
			    backpropCalculations[tIndex].gatedQueriesPreSigmoid,
			    CUDA_R_32F,
			    dim, // ldb, mem col size for col-major
			    &beta_one,
			    backpropCalculations[tIndex].x_postRMS1_post_gamma,
			    CUDA_R_32F,
			    dim, // ldc, mem col size
			    CUBLAS_COMPUTE_32F,
			    CUBLAS_GEMM_DEFAULT
			);
		}

		xTotalThreads = L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		preCalcRMSColWideVals<<<numBlocks, threadsPerBlock>>>(
			backpropCalculations[tIndex].rms1_sigma_scale_x_upGrad_byCol_RMS, 
			backpropCalculations[tIndex].rms1_oneOverR_byCol_RMS, 
			backpropCalculations[tIndex].rms1_oneOverColDimR3_byCol_RMS, 
			(tIndex == (transformers - 1) ?
				x_DEVICE :
				transformerCalculations_DEVICE[tIndex + 1].ffnPlusResidual
			),
			transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1,
			transformerWeights_DEVICE[tIndex].rms1_weights, 
			backpropCalculations[tIndex].x_postRMS1_post_gamma, 
			dim, L
		);
		xTotalThreads = dim * L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		dLoss_dPreRMSNorm<<<numBlocks, threadsPerBlock>>>(
			(tIndex == (transformers - 1) ?
				x_DEVICE_grad :
				backpropCalculations[tIndex + 1].ffn_final_plus_residual
			),
			(tIndex == (transformers - 1) ?
				x_DEVICE :
				transformerCalculations_DEVICE[tIndex + 1].ffnPlusResidual
			),
			backpropCalculations[tIndex].rms1_sigma_scale_x_upGrad_byCol_RMS, 
			backpropCalculations[tIndex].rms1_oneOverR_byCol_RMS, 
			backpropCalculations[tIndex].rms1_oneOverColDimR3_byCol_RMS, 
			transformerWeights_DEVICE[tIndex].rms1_weights, 
			backpropCalculations[tIndex].x_postRMS1_post_gamma,
			dim, L
		);
		xTotalThreads = dim;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		dLoss_d_RMS_gamma_weights<<<numBlocks, threadsPerBlock>>>(
			backpropCalculations[tIndex].rms1_gamma_weights,
			transformerCalculations_DEVICE[tIndex].x_postRMS1_pre_gamma,
			rightEndIndex,
			backpropCalculations[tIndex].x_postRMS1_post_gamma,
			dim
		);

		xTotalThreads = dim * L;
	    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
		add_residual_path_upGrad_to_x_gradient<<<numBlocks, threadsPerBlock>>>(
			(tIndex == (transformers - 1) ?
				x_DEVICE_grad :
				backpropCalculations[tIndex + 1].ffn_final_plus_residual
			),
			backpropCalculations[tIndex].outputProjPlusResidual,
			dim, L
		);
	}

	// add x_DEVICE_grad to embedding_weights
	xTotalThreads = dim * (rightEndIndex + 1);
	numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	add_x_grad_to_embeddings_grad<<<numBlocks, threadsPerBlock>>>(dLoss_d_embedding_weights, x_DEVICE_grad, seqTokenIndices_DEVICE, dim, vocabSize, rightEndIndex);
}

__global__ void add_step_grads_to_batch_accumulation(float* gradAccumulationTensor, float* stepGradTensor, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
    	return;
    }

    gradAccumulationTensor[index] += stepGradTensor[index];
}

void accumulateGradientsFromLastTrainingStep(bool resetGradAccumulation) {
	if (resetGradAccumulation) {
		cudaMemcpy(gradientAccumulation_embedding_weights, dLoss_d_embedding_weights, dim * vocabSize * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(gradientAccumulation_final_RMS_gamma_weights, dLoss_d_ffn_final_RMS_gamma_weights, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    	for (int transformerIndex = 0; transformerIndex < transformers; transformerIndex++) {
			cudaMemcpy(gradientAccumulation[transformerIndex].ffn_left_weights, backpropCalculations[transformerIndex].ffn_left_weights, dim * ffnDim * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(gradientAccumulation[transformerIndex].ffn_right_1_weights, backpropCalculations[transformerIndex].ffn_right_1_weights, dim * ffnDim * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(gradientAccumulation[transformerIndex].ffn_right_2_weights, backpropCalculations[transformerIndex].ffn_right_2_weights, dim * ffnDim * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(gradientAccumulation[transformerIndex].rms2_gamma_weights, backpropCalculations[transformerIndex].rms2_gamma_weights, dim * sizeof(float), cudaMemcpyDeviceToDevice);

			cudaMemcpy(gradientAccumulation[transformerIndex].output_proj_weights, backpropCalculations[transformerIndex].output_proj_weights, dim * dim * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(gradientAccumulation[transformerIndex].value_weights, backpropCalculations[transformerIndex].value_weights, dim * dim * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(gradientAccumulation[transformerIndex].query_weights, backpropCalculations[transformerIndex].query_weights, dim * dim * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(gradientAccumulation[transformerIndex].key_weights, backpropCalculations[transformerIndex].key_weights, dim * dim * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(gradientAccumulation[transformerIndex].rms1_gamma_weights, backpropCalculations[transformerIndex].rms1_gamma_weights, dim * sizeof(float), cudaMemcpyDeviceToDevice);

			if (CONFIG_QK_RMS_NORM) {
				cudaMemcpy(gradientAccumulation[transformerIndex].query_gamma_weights, backpropCalculations[transformerIndex].query_gamma_weights, dim * sizeof(float), cudaMemcpyDeviceToDevice);
				cudaMemcpy(gradientAccumulation[transformerIndex].key_gamma_weights, backpropCalculations[transformerIndex].key_gamma_weights, dim * sizeof(float), cudaMemcpyDeviceToDevice);
			}
			if (CONFIG_QUERY_GATING) {
				cudaMemcpy(gradientAccumulation[transformerIndex].gated_query_weights, backpropCalculations[transformerIndex].gated_query_weights, dim * dim * sizeof(float), cudaMemcpyDeviceToDevice);
			}
    	}
	} else {
		int xTotalThreads = dim * vocabSize;
		int numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;		
		add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation_embedding_weights, dLoss_d_embedding_weights, dim * vocabSize);

		xTotalThreads = dim;
		numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;		
		add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation_final_RMS_gamma_weights, dLoss_d_ffn_final_RMS_gamma_weights, dim);

    	for (int transformerIndex = 0; transformerIndex < transformers; transformerIndex++) {
			xTotalThreads = ffnDim * dim;
			numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;		
			add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].ffn_left_weights, backpropCalculations[transformerIndex].ffn_left_weights, ffnDim * dim);
			add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].ffn_right_1_weights, backpropCalculations[transformerIndex].ffn_right_1_weights, ffnDim * dim);
			add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].ffn_right_2_weights, backpropCalculations[transformerIndex].ffn_right_2_weights, ffnDim * dim);

			xTotalThreads = dim * dim;
			numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
			add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].output_proj_weights, backpropCalculations[transformerIndex].output_proj_weights, dim * dim);
			add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].value_weights, backpropCalculations[transformerIndex].value_weights, dim * dim);
			add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].query_weights, backpropCalculations[transformerIndex].query_weights, dim * dim);
			add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].key_weights, backpropCalculations[transformerIndex].key_weights, dim * dim);
			if (CONFIG_QUERY_GATING) {
				add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].gated_query_weights, backpropCalculations[transformerIndex].gated_query_weights, dim * dim);
			}

			xTotalThreads = dim;
			numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
			add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].rms1_gamma_weights, backpropCalculations[transformerIndex].rms1_gamma_weights, dim);
			add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].rms2_gamma_weights, backpropCalculations[transformerIndex].rms2_gamma_weights, dim);
			if (CONFIG_QK_RMS_NORM) {
				add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].key_gamma_weights, backpropCalculations[transformerIndex].key_gamma_weights, dim);
				add_step_grads_to_batch_accumulation<<<numBlocks, threadsPerBlock>>>(gradientAccumulation[transformerIndex].query_gamma_weights, backpropCalculations[transformerIndex].query_gamma_weights, dim);
			}
		}
	}
}
