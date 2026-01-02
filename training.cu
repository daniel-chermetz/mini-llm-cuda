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
    	dLoss_d_vocabScores[index] = 0;
    	return;
    }

    int vocabIndex = index - tokenIndex * vocabSize_;
    if (vocabIndex == seqTokenIndices[tokenIndex + 1]) {
    	dLoss_d_vocabScores[index] = vocabScores_postSoftmax[index] - 1;
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
    oneOverR_byCol_RMS[index] = 1 / x_sum_currentCol;
    oneOverColDimR3_byCol_RMS[index] = 1 / (colDim * x_sum_currentCol * x_sum_currentCol * x_sum_currentCol);

    int x_preRMS_coloffset = colDim * index;
    sigma_scale_x_upGrad_byCol_RMS[index] = 0;
	for (int i = 0; i < colDim; i++) {
		sigma_scale_x_upGrad_byCol_RMS[index] += (gamma_weights_RMS[i] * x_preRMS[x_preRMS_coloffset + i] * upstream_grad[x_preRMS_coloffset + i]);
	}    
}

__global__ void dLoss_dPreRMSNorm(
	float* x_post_RMS_pre_gamma_grad,
	float* x_pre_RMS_grad,
	float* x_post_RMS_pre_gamma,
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

    x_post_RMS_pre_gamma_grad[index] = upstream_grad[index] * x_post_RMS_pre_gamma[index];
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

void getGradientsForTraining(int leftStartIndex, int rightEndIndex) {
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
	    CUBLAS_OP_T,
	    CUBLAS_OP_N,
	    dim, // rows C
	    L, // cols C
	    vocabSize, // contracting (shared) dim
	    &alpha,
	    embedding_weights_DEVICE,
	    CUDA_R_32F,
	    vocabSize, // lda, mem col size for col-major
	    dLoss_d_vocabScores,
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
	cublasGemmEx(
	    handle,
	    CUBLAS_OP_N,
	    CUBLAS_OP_T,
	    vocabSize, // rows C
	    dim, // cols C
	    L, // contracting (shared) dim
	    &alpha,
	    dLoss_d_vocabScores,
	    CUDA_R_32F,
	    vocabSize, // lda, mem col size for col-major
	    ffn_postRMS_gamma_scaled_DEVICE,
	    CUDA_R_32F,
	    dim, // ldb, mem col size for col-major      
	    &beta,
	    dLoss_d_embedding_weights,
	    CUDA_R_32F,
	    vocabSize, // ldc, mem col size
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
		dLoss_d_ffn_final_RMS_gamma_weights,
		backpropCalculations[0].ffn_final_plus_residual, // pre-RMS
		ffn_postRMS_pre_gamma_DEVICE,
		transformerCalculations_DEVICE[0].ffnPlusResidual, // zero is actually the highest transformer (legacy from old gpu.js code, kept for compataility previously trained models)
		ffn_final_sigma_scale_x_upGrad_byCol_RMS, 
		ffn_final_oneOverR_byCol_RMS, 
		ffn_final_oneOverColDimR3_byCol_RMS,
		final_rms_weights_DEVICE,
		dLoss_d_ffn_final_postRMS_postGamma,
		dim, 
		L
	);

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
	    transformerWeights_DEVICE[0].ffn_left_weights,
	    CUDA_R_32F,
	    dim, // lda, mem col size for col-major
	    backpropCalculations[0].ffn_final_plus_residual, // implicit dLoss/d_ffn_final_plus_residual
	    CUDA_R_32F,
	    dim, // ldb, mem col size for col-major      
	    &beta,
	    backpropCalculations[0].ffn_right_postHadamard, // implicit dLoss/d_ffn_right_postHadamard
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
	    backpropCalculations[0].ffn_final_plus_residual,
	    CUDA_R_32F,
	    dim, // lda, mem col size for col-major
	    transformerCalculations_DEVICE[0].ffn_right_postHadamard,
	    CUDA_R_32F,
	    ffnDim, // ldb, mem col size for col-major      
	    &beta,
	    backpropCalculations[0].ffn_left_weights,
	    CUDA_R_32F,
	    dim, // ldc, mem col size
	    CUBLAS_COMPUTE_32F,
	    CUBLAS_GEMM_DEFAULT             
	);

	xTotalThreads = ffnDim * L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	dLoss_d_ffn_right_pre_hadamard_backprop<<<numBlocks, threadsPerBlock>>>(
		backpropCalculations[0].ffn_right_1_postSilu, 
		backpropCalculations[0].ffn_right_2, 
		transformerCalculations_DEVICE[0].ffn_right_1_postSilu, 
		transformerCalculations_DEVICE[0].ffn_right_2, 
		backpropCalculations[0].ffn_right_postHadamard, 
		ffnDim, 
		L
	);

	xTotalThreads = ffnDim * L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
	dLoss_d_ffn_right_1_preSilu<<<numBlocks, threadsPerBlock>>>(
		transformerCalculations_DEVICE[0].ffn_right_1_preSilu,
		transformerCalculations_DEVICE[0].ffn_right_1_postSilu, 
		transformerCalculations_DEVICE[0].ffn_right_1_preSilu,
		backpropCalculations[0].ffn_right_1_postSilu, 
		ffnDim_, 
		L_
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
	    transformerWeights_DEVICE[0].ffn_right_1_weights,
	    CUDA_R_32F,
	    ffnDim, // lda, mem col size for col-major
	    backpropCalculations[0].ffn_right_1_preSilu,
	    CUDA_R_32F,
	    ffnDim, // ldb, mem col size for col-major      
	    &beta,
	    backpropCalculations[0].outputProjPlusResidual_postRMS2,
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
	    backpropCalculations[0].ffn_right_1_preSilu,
	    CUDA_R_32F,
	    ffnDim, // lda, mem col size for col-major
	    transformerCalculations_DEVICE[0].outputProjPlusResidual_postRMS2,
	    CUDA_R_32F,
	    dim, // ldb, mem col size for col-major      
	    &beta,
	    backpropCalculations[0].ffn_right_1_weights,
	    CUDA_R_32F,
	    ffnDim, // ldc, mem col size
	    CUBLAS_COMPUTE_32F,
	    CUBLAS_GEMM_DEFAULT             
	);

	float beta_one = 1.0;
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
	    transformerWeights_DEVICE[0].ffn_right_2_weights,
	    CUDA_R_32F,
	    ffnDim, // lda, mem col size for col-major
	    backpropCalculations[0].ffn_right_2,
	    CUDA_R_32F,
	    ffnDim, // ldb, mem col size for col-major      
	    &beta_one,
	    backpropCalculations[0].outputProjPlusResidual_postRMS2,
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
	    backpropCalculations[0].ffn_right_2,
	    CUDA_R_32F,
	    ffnDim, // lda, mem col size for col-major
	    transformerCalculations_DEVICE[0].outputProjPlusResidual_postRMS2,
	    CUDA_R_32F,
	    dim, // ldb, mem col size for col-major      
	    &beta,
	    backpropCalculations[0].ffn_right_2_weights,
	    CUDA_R_32F,
	    ffnDim, // ldc, mem col size
	    CUBLAS_COMPUTE_32F,
	    CUBLAS_GEMM_DEFAULT             
	);
}