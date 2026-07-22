#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "network_meta.h"
#include "network_globals.h"

__global__ void grad_x_ple_and_grad_ple_gate(
	float* grad_x_ple, 
	float* grad_ple_gate_post_sigmoid_post_gamma,
	float* grad_ple_gate_post_sigmoid_pre_gamma,
	float* grad_ple_gate_pre_sigmoid_pre_gamma,		
	const float* upstream_grad_sum_ffnPlusResidual_x_ple_gated, 
	const float* ple_gate_post_sigmoid_post_gamma,
	const float* ple_gate_post_sigmoid_pre_gamma,
    const float* x_DEVICE,
	const float* x_ple,
	const float* gamma_weights,
	const int L_
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxCount = dim * L_;
    if (index >= maxCount) {
        return;
    }
    int rowIndex = index - (index / dim) * dim;

    grad_x_ple[index] = upstream_grad_sum_ffnPlusResidual_x_ple_gated[index] * ple_gate_post_sigmoid_post_gamma[index];
    
    grad_ple_gate_post_sigmoid_post_gamma[index] = upstream_grad_sum_ffnPlusResidual_x_ple_gated[index] * (x_DEVICE[index] + x_ple[index]);
    
    grad_ple_gate_post_sigmoid_pre_gamma[index] = grad_ple_gate_post_sigmoid_post_gamma[index] * gamma_weights[rowIndex];
    
    float x_sigmoid = ple_gate_post_sigmoid_pre_gamma[index];
	grad_ple_gate_pre_sigmoid_pre_gamma[index] = grad_ple_gate_post_sigmoid_pre_gamma[index] * (x_sigmoid - x_sigmoid * x_sigmoid);
}

__global__ void grad_ple_gamma_weights(
	float* grad_gamma_weights,
	const float* upstream_grad_ple_gate_post_sigmoid_post_gamma,
	const float* ple_gate_post_sigmoid_pre_gamma,
	const int L_
) {
	int gammaIndex = blockIdx.x;
	int blockDimX = blockDim.x;
    int tIndex = threadIdx.x; // horizontally across ple_gate_post_sigmoid_pre_gamma

    extern __shared__ float sData[]; // blockDim.x from host invocation; must be a power of 2

    float gradSum = 0.0f;
    for (int i = tIndex; i < L_; i += blockDimX) {
    	int gateIndex = gammaIndex + i * dim;
    	gradSum += (ple_gate_post_sigmoid_pre_gamma[gateIndex] * upstream_grad_ple_gate_post_sigmoid_post_gamma[gateIndex]);
    }
    sData[tIndex] = gradSum;
    __syncthreads();

    // blockDim.x = 256; reductionSize = 128; reductionSize > 0; reductionSize: 128, 64, 32, 16, etc.
    for (int reductionSize = blockDimX / 2; reductionSize > 0; reductionSize /= 2) {
    	if (tIndex < reductionSize) {
    		sData[tIndex] += sData[tIndex + reductionSize];
    	}
    	__syncthreads();
    }

    if (tIndex == 0) {
    	grad_gamma_weights[gammaIndex] = sData[0];
    }
}

__global__ void grad_ple_embeddings(
    float* ple_embedding_weights_grad, 
    const float* grad_x_ple, 
    const int* seqTokenIndicesInFullEmbeddings, 
    const int rightEndIndex
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max = dim * (rightEndIndex + 1);
    if (index >= max) {
    	return;
    }

    int seqTokenIndex = index / dim;
    int featureIndex = index - seqTokenIndex * dim;
    int embeddingIndex = seqTokenIndicesInFullEmbeddings[seqTokenIndex];

    // must intialize ple_embedding_weights_grad to 0.0f at the beginning of each batch, and let the gradients accumulate
    atomicAdd(&ple_embedding_weights_grad[embeddingIndex * dim + featureIndex], grad_x_ple[index]);
}

void getGradientsForPLELayerTraining(const int pleIndex, const int downstreamTransformerIndex, const int leftStartIndex, const int rightEndIndex, const int L) {
    int xTotalThreads = dim * L;
    int numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;

    grad_x_ple_and_grad_ple_gate<<<numBlocks, threadsPerBlock>>>(
        ple_backpropCalculations[pleIndex].grad_x_ple, 
        ple_backpropCalculations[pleIndex].grad_ple_gate_post_sigmoid_post_gamma,
        ple_backpropCalculations[pleIndex].grad_ple_gate_post_sigmoid_pre_gamma,
        ple_backpropCalculations[pleIndex].grad_ple_gate_pre_sigmoid_pre_gamma,
        backpropCalculations[downstreamTransformerIndex].ffn_final_plus_residual, // at this point the upstream transformer updated this grad, treating it as its x-inputs
        pleCalculations_DEVICE[pleIndex].ple_gate_post_sigmoid_post_gamma,
        pleCalculations_DEVICE[pleIndex].ple_gate_post_sigmoid_pre_gamma,     
        x_DEVICE,
        pleCalculations_DEVICE[pleIndex].x_ple,
        pleWeights_DEVICE[pleIndex].gamma_weights,
        L
    );

    size_t sharedMemSize = 256 * sizeof(float);
    grad_ple_gamma_weights<<<dim, 256, sharedMemSize>>>(
        ple_backpropCalculations[pleIndex].gamma_weights,
        ple_backpropCalculations[pleIndex].grad_ple_gate_post_sigmoid_post_gamma,
        pleCalculations_DEVICE[pleIndex].ple_gate_post_sigmoid_pre_gamma,
        L
    );

    // dL / dA = G * B.t,
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        dim, // rows C
        dim, // cols C
        L, // contracting (shared) dim
        &alpha,
        ple_backpropCalculations[pleIndex].grad_ple_gate_pre_sigmoid_pre_gamma,
        CUDA_R_32F,
        dim, // lda, mem col size for col-major
        transformerCalculations_DEVICE[downstreamTransformerIndex].ffnPlusResidual, // [dim, L].t
        CUDA_R_32F,
        dim, // ldb, mem col size for col-major      
        &beta,
        ple_backpropCalculations[pleIndex].ple_weights, // [dim, dim]
        CUDA_R_32F,
        dim, // ldc, mem col size
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT             
    );

    grad_ple_embeddings<<<numBlocks, threadsPerBlock>>>(
        ple_backpropCalculations[pleIndex].embedding_weights, 
        ple_backpropCalculations[pleIndex].grad_x_ple, 
        seqTokenIndices_DEVICE, 
        rightEndIndex
    );

    // dL / dB = A.t * G,
    // this is tricky: 
    // calculate backpropCalculations[downstreamTransformerIndex].ffn_final_plus_residual based on the upstream layer (as though there was no PLE in between)
    // then add to it the gradient of the non-residual PLE path
    // when using upstream_grad_sum_ffnPlusResidual_x_ple_gated earlier, this would be backpropCalculations[downstreamTransformerIndex].ffn_final_plus_residual before this cuBLAS updated it
    // we temporarily identify the upstream grad of ffn_final_plus_residual (grad of x to the next transformer) with the downstream one (output of last transformer), 
    // calculate the PLE path grad with the upstream as the grad of x to the next transformer,
    // but then finally update the same tensor to add up the PLE path grad, making it the downstream transformer output grad, but no longer the same as the upstream x input to the next transformer grad
    cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        dim, // rows C
        L, // cols C
        dim, // contracting (shared) dim
        &alpha,
        pleWeights_DEVICE[pleIndex].ple_weights, // [dim, dim].t
        CUDA_R_32F,
        dim, // lda, mem col size for col-major
        ple_backpropCalculations[pleIndex].grad_ple_gate_pre_sigmoid_pre_gamma, // [dim, L]
        CUDA_R_32F,
        dim, // ldb, mem col size for col-major      
        &beta_one,
        backpropCalculations[downstreamTransformerIndex].ffn_final_plus_residual, // [dim, L]
        CUDA_R_32F,
        dim, // ldc, mem col size
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT             
    );    
}
