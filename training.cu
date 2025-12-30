__global__ void dLoss_dVocabScores_backprop(float* dLoss_dVocabScores, float* vocabScores_postSoftmax, int* seqTokenIndices, int vocabSize_, int L_, int leftStartIndex, int rightEndIndex) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = vocabSize_ * L_ - 1;
    if (index > maxIndex) {
    	return;
    }

    int tokenIndex = index / vocabSize_;
    if (tokenIndex < leftStartIndex || tokenIndex > rightEndIndex) {
    	dLoss_dVocabScores[index] = 0;
    	return;
    }

    int vocabIndex = index - tokenIndex * vocabSize_;
    if (vocabIndex == seqTokenIndices[tokenIndex]) {
    	dLoss_dVocabScores[index] = vocabScores_postSoftmax[index] - 1;
    } else {
    	dLoss_dVocabScores[index] = vocabScores_postSoftmax[index];
    }
}

// ffn_right_1_postSilu
// ffn_right_2
// ffn_right_postHadamard
__global__ void dLoss_d_ffn_right_pre_hadamard_backprop(float* dLoss_d_ffn_right_1_postSilu, float* dLoss_d_ffn_right_2, float* ffn_right_1_postSilu, float* ffn_right_2, float* dLoss_d_ffn_right_postHadamard, int ffnDim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = ffnDim_ * L_ - 1;
    if (index > maxIndex) {
    	return;
    }

    dLoss_d_ffn_right_1_postSilu[index] = ffn_right_2[index] * dLoss_d_ffn_right_postHadamard[index];
    dLoss_d_ffn_right_2[index] = ffn_right_1_postSilu[index] * dLoss_d_ffn_right_postHadamard[index];
}

// (dLoss/dVocabScores) * (dVocabScores/dFFN_postRMS_DEVICE)
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
    dLoss_dVocabScores,
    CUDA_R_32F,
    vocabSize, // ldb, mem col size for col-major      
    &beta,
    dLoss_dFFN_postRMS,
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
    dLoss_dVocabScores,
    CUDA_R_32F,
    vocabSize, // lda, mem col size for col-major
    ffn_postRMS_DEVICE,
    CUDA_R_32F,
    dim, // ldb, mem col size for col-major      
    &beta,
    dLoss_dEmbedding_weights,
    CUDA_R_32F,
    vocabSize, // ldc, mem col size
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT             
);

// skipping RMS for now

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
    backpropCalculations_DEVICE[0].ffn_final, // implicit dLoss/d_ffn_final
    CUDA_R_32F,
    dim, // ldb, mem col size for col-major      
    &beta,
    backpropCalculations_DEVICE[0].ffn_right_postHadamard, // implicit dLoss/d_ffn_right_postHadamard
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
    backpropCalculations_DEVICE[0].ffn_final,
    CUDA_R_32F,
    dim, // lda, mem col size for col-major
    transformerCalculations_DEVICE[0].ffn_right_postHadamard,
    CUDA_R_32F,
    ffnDim, // ldb, mem col size for col-major      
    &beta,
    backpropCalculations_DEVICE[0].ffn_left_weights,
    CUDA_R_32F,
    dim, // ldc, mem col size
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT             
);
