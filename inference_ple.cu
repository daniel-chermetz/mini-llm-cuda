#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "network_meta.h"
#include "network_globals.h"

__global__ void setPLEInputSeqEmbeddings(float* x_ple, const int* seqTokenIndices, const float* embedding_weights, const int L_) {
    int currentIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int maxCount = L_ * dim;
    if (currentIndex >= maxCount) {
        return;
    }

    int lIndex = currentIndex / dim;
    int tokenIndex = seqTokenIndices[lIndex];
    int rowIndex = currentIndex - lIndex * dim;

    x_ple[currentIndex] = embedding_weights[tokenIndex * dim + rowIndex];
}

__global__ void applyGatingToTransformerInputOntoPleLayer(float* sum_ffnPlusResidual_x_ple_gated, const float* ffnPlusResidual, const float* x_DEVICE, const float* x_ple, float* ple_gate_post_sigmoid_post_gamma, float* ple_gate_post_sigmoid_pre_gamma, const float* ple_gate_pre_sigmoid_pre_gamma, const float* gamma_weights, const int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxCount = dim * L_;
    if (index >= maxCount) {
        return;
    }
    int rowIndex = index - (index / dim) * dim;

    ple_gate_post_sigmoid_pre_gamma[index] = 1.0f / (1.0f + expf(-ple_gate_pre_sigmoid_pre_gamma[index]));
    ple_gate_post_sigmoid_post_gamma[index] = gamma_weights[rowIndex] * ple_gate_post_sigmoid_pre_gamma[index];
    sum_ffnPlusResidual_x_ple_gated[index] = ffnPlusResidual[index] + (x_DEVICE[index] + x_ple[index]) * ple_gate_post_sigmoid_post_gamma[index];
}

int runPLEInference(int pleIndex, int downstreamTransformerIndex, int L) {
	int xTotalThreads = dim * L;
    int numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    
    setPLEInputSeqEmbeddings<<<numBlocks, threadsPerBlock>>>(pleCalculations_DEVICE[pleIndex].x_ple, seqTokenIndices_DEVICE, pleWeights_DEVICE[pleIndex].embedding_weights, L);

	cublasGemmEx(
	    handle,
	    CUBLAS_OP_N,
	    CUBLAS_OP_N,
	    dim, // row C
	    L, // cols C
	    dim, // contracting (shared) dim
	    &alpha,
	    pleWeights_DEVICE[pleIndex].ple_weights,
	    CUDA_R_32F,
	    dim, // lda, mem col size for col-major
	    transformerCalculations_DEVICE[downstreamTransformerIndex].ffnPlusResidual,
	    CUDA_R_32F,
	    dim, // ldb, mem col size for col-major      
	    &beta,
	    pleCalculations_DEVICE[pleIndex].ple_gate_pre_sigmoid_pre_gamma,
	    CUDA_R_32F,
	    dim, // ldc, mem col size
	    CUBLAS_COMPUTE_32F,
	    CUBLAS_GEMM_DEFAULT             
	);

	applyGatingToTransformerInputOntoPleLayer<<<numBlocks, threadsPerBlock>>>(
		pleCalculations_DEVICE[pleIndex].sum_ffnPlusResidual_x_ple_gated,
		transformerCalculations_DEVICE[downstreamTransformerIndex].ffnPlusResidual,
		x_DEVICE,
		pleCalculations_DEVICE[pleIndex].x_ple,
		pleCalculations_DEVICE[pleIndex].ple_gate_post_sigmoid_post_gamma,
		pleCalculations_DEVICE[pleIndex].ple_gate_post_sigmoid_pre_gamma,
		pleCalculations_DEVICE[pleIndex].ple_gate_pre_sigmoid_pre_gamma,
		pleWeights_DEVICE[pleIndex].gamma_weights,
	    L
	);

	return 0;
}
