#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "network_meta.h"
#include "network_globals.h"

__global__ void setInputSeqEmbeddings(float* x, int* seqTokenIndices, float* embedding_weights, int dim_, int L_) {
    int currentIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = L_ * dim_ - 1;

    if (currentIndex > maxIndex) {
        return;
    }

    int lIndex = currentIndex / dim_;
    int tokenIndex = seqTokenIndices[lIndex];
    int rowIndex = currentIndex - lIndex * dim_;

    x[currentIndex] = embedding_weights[tokenIndex * dim_ + rowIndex];
}

__global__ void getRMSColSums(float* rmsSumByCol, float* x, int dim_, int L_) {
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int maxColIndex = L_ - 1;
    
    if (colIndex > maxColIndex) {
        return;
    }

    int colOffset = dim_ * colIndex;

    float sumSquared = 0;
    for (int i = 0; i < dim_; i++) {
        float val = x[colOffset + i];
        sumSquared += (val * val);
    }
    rmsSumByCol[colIndex] = sqrtf((sumSquared / dim_) + 1e-8);
}

__global__ void applyRMSNorm(float* postRMS_post_gamma, float* postRMS_pre_gamma, float* preRMS, float* rmsSumByCol, float* rms_weights, int dim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = dim_ * L_ - 1;
    
    if (index > maxIndex) {
        return;
    }

    int colIndex = index / dim_;
    int rowIndex = index - colIndex * dim_;
    postRMS_pre_gamma[index] = (preRMS[index] / rmsSumByCol[colIndex]);
    postRMS_post_gamma[index] = (rms_weights[rowIndex] * postRMS_pre_gamma[index]);
}

__global__ void applyRoPE(float* keysOrValuesPostRoPE, float* keysOrValues, float* preComputedRopeTheta, int headDim_, int dim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = dim_ * L_ - 1;
    if (index > maxIndex) {
        return;
    }

    int colIndex = index / dim_;
    int rowIndex = index - colIndex * dim_;
    int headIndex = rowIndex / headDim_;
    int headRelativeRowIndex = rowIndex - headIndex * headDim_;
    int headRelativeColOffset = colIndex * headDim_;

    if (headRelativeRowIndex % 2 == 0) {
        float cosTheta = preComputedRopeTheta[headRelativeColOffset + headRelativeRowIndex];
        float sinTheta = preComputedRopeTheta[headRelativeColOffset + headRelativeRowIndex + 1];

        keysOrValuesPostRoPE[index] = cosTheta * keysOrValues[index] - sinTheta * keysOrValues[index + 1];
    } else {
        float cosTheta = preComputedRopeTheta[headRelativeColOffset + headRelativeRowIndex - 1];
        float sinTheta = preComputedRopeTheta[headRelativeColOffset + headRelativeRowIndex];

        keysOrValuesPostRoPE[index] = sinTheta * keysOrValues[index - 1] + cosTheta * keysOrValues[index];
    }
}

__global__ void getHeadDimScaledMaskedAttn(float* attnKtQByHeadScaledMasked, float* attnKtQByHead, int attnHeads_, int headDim_, int L_) {
    int L2 = L_ * L_;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = attnHeads_ * L2 - 1;
    if (index > maxIndex) {
        return;
    }

    int headIndex = index / L2;
    int colIndex = (index - headIndex * L2) / L_;
    int rowIndex = index - headIndex * L2 - colIndex * L_;

    if (rowIndex > colIndex) {
        attnKtQByHeadScaledMasked[index] = 0;
        return;
    }

    attnKtQByHeadScaledMasked[index] = attnKtQByHead[index] / sqrtf(headDim_);
}

__global__ void getAttnHeadsMaxByCol_softmax(float* attnByHead_maxByCol_softmax, float* attnHeadDimScaledMaskedKtQByHead, int attnHeads_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = attnHeads_ * L_ - 1;
    if (index > maxIndex) {
        return;
    }

    float colMax = -1.0e20f;
    int headIndex = index / L_;
    int colIndex = index - headIndex * L_;
    int colOffset = index * L_;
    for (int rowIndex = 0; rowIndex <= colIndex; rowIndex++) {
        float val = attnHeadDimScaledMaskedKtQByHead[colOffset + rowIndex];
        if (val > colMax) {
            colMax = val;
        }
    }

    attnByHead_maxByCol_softmax[index] = colMax;
}

__global__ void getAttnHeadsSumByCol_softmax(float* attnByHead_sumByCol_softmax, float* attnHeadDimScaledMaskedKtQByHead, float* attnByHead_maxByCol_softmax, int attnHeads_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = attnHeads_ * L_ - 1;
    if (index > maxIndex) {
        return;
    }

    float sum = 0.0f;
    int headIndex = index / L_;
    int colIndex = index - headIndex * L_;    
    int colOffset = index * L_;
    for (int rowIndex = 0; rowIndex <= colIndex; rowIndex++) {
        sum += expf(attnHeadDimScaledMaskedKtQByHead[colOffset + rowIndex] - attnByHead_maxByCol_softmax[index]);
    }
    attnByHead_sumByCol_softmax[index] = sum;
}

__global__ void applySoftmaxToAttnHeads(float* attnByHead_postSoftmax, float* attnHeadDimScaledMaskedKtQByHead, float* sumByCol, float* maxByCol, int attnHeads_, int L_) {
    int L2 = L_ * L_;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = attnHeads_ * L2 - 1;
    if (index > maxIndex) {
        return;
    }

    int headIndex = index / L2;
    int headRelativeColIndex = (index - headIndex * L2) / L_;
    int globalColIndex = headIndex * L_ + headRelativeColIndex;
    int rowIndex = index - globalColIndex * L_;

    if (rowIndex <= headRelativeColIndex) {
        attnByHead_postSoftmax[index] = (expf(attnHeadDimScaledMaskedKtQByHead[index] - maxByCol[globalColIndex]) / sumByCol[globalColIndex]);
        return;
    }

    attnByHead_postSoftmax[index] = 0;    
}

__global__ void addResidualToOutputProj(float* outputProjPlusResidual, float* outputProj, float* xFromTransformerStart, int dim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = dim_ * L_ - 1;
    if (index > maxIndex) {
        return;
    }

    outputProjPlusResidual[index] = outputProj[index] + xFromTransformerStart[index];
}

__global__ void applySiluToFFN(float* ffn_right_1_postSilu, float* ffn_right_1_preSilu, int ffnDim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = ffnDim_ * L_ - 1;
    if (index > maxIndex) {
        return;
    }
    // SiLU: x / (1 + exp(-x))
    float val = ffn_right_1_preSilu[index];
    ffn_right_1_postSilu[index] = val / (1.0f + expf(-val));
}

__global__ void hadamardMultiplyFFN(float* ffn_right_postHadamard, float* ffn_right_1_postSilu, float* ffn_right_2, int ffnDim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = ffnDim_ * L_ - 1;
    if (index > maxIndex) {
        return;
    }

    ffn_right_postHadamard[index] = ffn_right_1_postSilu[index] * ffn_right_2[index];
}

__global__ void addResidualToFFN(float* ffnPlusResidual, float* ffn_final, float* xFromTransformerStart, int dim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = dim_ * L_ - 1;
    if (index > maxIndex) {
        return;
    }

    ffnPlusResidual[index] = ffn_final[index] + xFromTransformerStart[index];
}

__global__ void getVocabMaxByCol_softmax(float* vocab_maxByCol_softmax, float* vocabScores, int vocabSize_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = L_ - 1;
    if (index > maxIndex) {
        return;
    }

    float colMax = -1.0e20f;
    int colOffset = index * vocabSize_;
    for (int rowIndex = 0; rowIndex < vocabSize_; rowIndex++) {
        float val = vocabScores[colOffset + rowIndex];
        if (val > colMax) {
            colMax = val;
        }
    }

    vocab_maxByCol_softmax[index] = colMax;
}

__global__ void getVocabSumByCol_softmax(float* vocab_sumByCol_softmax, float* vocabScores, float* vocab_maxByCol_softmax, int vocabSize_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = L_ - 1;
    if (index > maxIndex) {
        return;
    }

    float sum = 0.0f;
    int colOffset = index * vocabSize_;
    for (int rowIndex = 0; rowIndex < vocabSize_; rowIndex++) {
        sum += expf(vocabScores[colOffset + rowIndex] - vocab_maxByCol_softmax[index]);
    }
    vocab_sumByCol_softmax[index] = sum;
}

__global__ void applySoftmaxToVocab(float* vocabScores_postSoftmax, float* vocabScores, float* vocab_sumByCol_softmax, float* vocab_maxByCol_softmax, int vocabSize_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = vocabSize_ * L_ - 1;
    if (index > maxIndex) {
        return;
    }

    int colIndex = index / vocabSize_;
    vocabScores_postSoftmax[index] = (expf(vocabScores[index] - vocab_maxByCol_softmax[colIndex]) / vocab_sumByCol_softmax[colIndex]);
}

void getPreComputedRopeTheta(float* preComputedRopeTheta) {
    int numPairs = headDim / 2;
    for (int colIndex = 0; colIndex < L; colIndex++) {
        int colOffset = colIndex * headDim;
        for (int pairIndex = 0; pairIndex < numPairs; pairIndex++) {
            float theta = colIndex * powf(ropeDenomBase, (-2.0f * pairIndex / headDim));
            preComputedRopeTheta[colOffset + pairIndex * 2] = cosf(theta);
            preComputedRopeTheta[colOffset + pairIndex * 2 + 1] = sinf(theta);            
        }
    }
}

int runInference() {
    int xTotalThreads = L * dim;
    int numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    setInputSeqEmbeddings<<<numBlocks, threadsPerBlock>>>(x_DEVICE, seqTokenIndices_DEVICE, embedding_weights_DEVICE, dim, L);

    for (int tIndexCountUp = 0; tIndexCountUp < transformers; tIndexCountUp++) {
        int tIndex = transformers - 1 - tIndexCountUp;

        xTotalThreads = L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        if (tIndex == transformers - 1) {
            getRMSColSums<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1, x_DEVICE, dim, L);
        } else {
            getRMSColSums<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1, transformerCalculations_DEVICE[tIndex + 1].ffnPlusResidual, dim, L);
        }

        xTotalThreads = dim * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        if (tIndex == transformers - 1) {
            applyRMSNorm<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].x_postRMS1_post_gamma, transformerCalculations_DEVICE[tIndex].x_postRMS1_pre_gamma, x_DEVICE, transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1, transformerWeights_DEVICE[tIndex].rms1_weights, dim, L);
        } else {
            applyRMSNorm<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].x_postRMS1_post_gamma, transformerCalculations_DEVICE[tIndex].x_postRMS1_pre_gamma, transformerCalculations_DEVICE[tIndex + 1].ffnPlusResidual, transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1, transformerWeights_DEVICE[tIndex].rms1_weights, dim, L);
        }

        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            dim, // row C
            L, // cols C
            dim, // contracting (shared) dim
            &alpha,
            transformerWeights_DEVICE[tIndex].query_weights,
            CUDA_R_32F,
            dim, // lda, mem col size for col-major
            transformerCalculations_DEVICE[tIndex].x_postRMS1_post_gamma,
            CUDA_R_32F,
            dim, // ldb, mem col size for col-major      
            &beta,
            transformerCalculations_DEVICE[tIndex].queries,
            CUDA_R_32F,
            dim, // ldc, mem col size
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT             
        );

        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            dim, // rows C
            L, // cols C
            dim, // contracting (shared) dim
            &alpha,
            transformerWeights_DEVICE[tIndex].key_weights,
            CUDA_R_32F,
            dim, // lda, mem col size for col-major
            transformerCalculations_DEVICE[tIndex].x_postRMS1_post_gamma,
            CUDA_R_32F,
            dim, // ldb, mem col size for col-major      
            &beta,
            transformerCalculations_DEVICE[tIndex].keys,
            CUDA_R_32F,
            dim, // ldc, mem col size
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT             
        );

        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            dim, // rows C
            L, // cols C
            dim, // contracting (shared) dim
            &alpha,
            transformerWeights_DEVICE[tIndex].value_weights,
            CUDA_R_32F,
            dim, // lda, mem col size for col-major
            transformerCalculations_DEVICE[tIndex].x_postRMS1_post_gamma,
            CUDA_R_32F,
            dim, // ldb, mem col size for col-major        
            &beta,
            transformerCalculations_DEVICE[tIndex].values,
            CUDA_R_32F,
            dim, // ldc, mem col size
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT             
        );

        xTotalThreads = dim * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;   
        applyRoPE<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].queriesPostRoPE, transformerCalculations_DEVICE[tIndex].queries, preComputedRopeTheta_DEVICE, headDim, dim, L);
        applyRoPE<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].keysPostRoPE, transformerCalculations_DEVICE[tIndex].keys, preComputedRopeTheta_DEVICE, headDim, dim, L);

        // K.t @ Q
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            L, // m (K.t@Q one attn head rows)
            L, // n (K.t@Q one attn head cols)
            headDim, // k
            &alpha,
            transformerCalculations_DEVICE[tIndex].keysPostRoPE,
            CUDA_R_32F,
            dim, // lda, col size in mem (but with CUBLAS_OP_T is logical row)
            headDim, // mem stride in K.t to reach next head (with CUBLAS_OP_T moves along a logical row)
            transformerCalculations_DEVICE[tIndex].queriesPostRoPE,
            CUDA_R_32F,
            dim, // ldb, col size in mem for col-major
            headDim, // mem stride in Q to reach next head
            &beta,
            transformerCalculations_DEVICE[tIndex].attnKtQByHead,
            CUDA_R_32F,
            L, // ldc, col size in mem
            L * L, // mem stride in K.t@Q to reach next head
            attnHeads,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );   
        
        xTotalThreads = attnHeads * L * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        getHeadDimScaledMaskedAttn<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].attnKtQByHeadScaledMasked, transformerCalculations_DEVICE[tIndex].attnKtQByHead, attnHeads, headDim, L);

        xTotalThreads = attnHeads * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        getAttnHeadsMaxByCol_softmax<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].attnByHead_maxByCol_softmax, transformerCalculations_DEVICE[tIndex].attnKtQByHeadScaledMasked, attnHeads, L);
        getAttnHeadsSumByCol_softmax<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].attnByHead_sumByCol_softmax, transformerCalculations_DEVICE[tIndex].attnKtQByHeadScaledMasked, transformerCalculations_DEVICE[tIndex].attnByHead_maxByCol_softmax, attnHeads, L);
        xTotalThreads = attnHeads * L * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;    
        applySoftmaxToAttnHeads<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].attnByHead_postSoftmax, transformerCalculations_DEVICE[tIndex].attnKtQByHeadScaledMasked, transformerCalculations_DEVICE[tIndex].attnByHead_sumByCol_softmax, transformerCalculations_DEVICE[tIndex].attnByHead_maxByCol_softmax, attnHeads, L);

        // V @ attnByHead_postSoftmax
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            headDim, // m (C rows)
            L, // n (C cols)
            L, // k (shared dim)
            &alpha,
            transformerCalculations_DEVICE[tIndex].values,
            CUDA_R_32F,
            dim, // lda, col size in mem for col-major
            headDim, // mem stride in values to reach next head
            transformerCalculations_DEVICE[tIndex].attnByHead_postSoftmax,
            CUDA_R_32F,
            L, // ldb, col size in mem for col-major
            (L * L), // mem stride in softmaxAttn to reach next head
            &beta,
            transformerCalculations_DEVICE[tIndex].valueScaledSoftmaxAttn,
            CUDA_R_32F,
            dim, // ldc, fused attn heads col size in mem; would have headDim if mult. by head
            headDim, // mem stride in valueScaledSoftmaxAttn with fused attn heads to reach next head; would have been (headDim * L) if mult.
            attnHeads,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

        // output_proj_weights @ valueScaledSoftmaxAttn (heads already fused to dim sized cols)
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            dim, // m (C rows)
            L, // n (C cols)
            dim, // k (shared dim)
            &alpha,
            transformerWeights_DEVICE[tIndex].output_proj_weights,
            CUDA_R_32F,
            dim, // lda, col size in mem for col-major
            transformerCalculations_DEVICE[tIndex].valueScaledSoftmaxAttn,
            CUDA_R_32F,
            dim, // ldb, col size in mem for col-major
            &beta,
            transformerCalculations_DEVICE[tIndex].outputProj,
            CUDA_R_32F,
            dim, // ldc, col size in mem
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

        xTotalThreads = dim * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        if (tIndex == transformers - 1) {
            addResidualToOutputProj<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].outputProjPlusResidual, transformerCalculations_DEVICE[tIndex].outputProj, x_DEVICE, dim, L);
        } else {
            addResidualToOutputProj<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].outputProjPlusResidual, transformerCalculations_DEVICE[tIndex].outputProj, transformerCalculations_DEVICE[tIndex + 1].ffnPlusResidual, dim, L);
        }      

        xTotalThreads = L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        getRMSColSums<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_sumByCol_RMS2, transformerCalculations_DEVICE[tIndex].outputProjPlusResidual, dim, L);
        xTotalThreads = dim * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;    
        applyRMSNorm<<<numBlocks, threadsPerBlock>>>(
            transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2_post_gamma,
            transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2_pre_gamma,
            transformerCalculations_DEVICE[tIndex].outputProjPlusResidual,
            transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_sumByCol_RMS2,
            transformerWeights_DEVICE[tIndex].rms2_weights, dim, L
        );

        // ffn_right_1_weights @ outputProjPlusResidual_postRMS2
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            ffnDim, // m (C rows)
            L, // n (C cols)
            dim, // k (shared dim)
            &alpha,
            transformerWeights_DEVICE[tIndex].ffn_right_1_weights,
            CUDA_R_32F,
            ffnDim, // lda, col size in mem for col-major
            transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2_post_gamma,
            CUDA_R_32F,
            dim, // ldb, col size in mem for col-major
            &beta,
            transformerCalculations_DEVICE[tIndex].ffn_right_1_preSilu,
            CUDA_R_32F,
            ffnDim, // ldc, col size in mem
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

        // ffn_right_2_weights @ outputProjPlusResidual_postRMS2
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            ffnDim, // m (C rows)
            L, // n (C cols)
            dim, // k (shared dim)
            &alpha,
            transformerWeights_DEVICE[tIndex].ffn_right_2_weights,
            CUDA_R_32F,
            ffnDim, // lda, col size in mem for col-major
            transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2_post_gamma,
            CUDA_R_32F,
            dim, // ldb, col size in mem for col-major
            &beta,
            transformerCalculations_DEVICE[tIndex].ffn_right_2,
            CUDA_R_32F,
            ffnDim, // ldc, col size in mem
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

        xTotalThreads = ffnDim * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;    
        applySiluToFFN<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].ffn_right_1_postSilu, transformerCalculations_DEVICE[tIndex].ffn_right_1_preSilu, ffnDim, L);
        hadamardMultiplyFFN<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].ffn_right_postHadamard, transformerCalculations_DEVICE[tIndex].ffn_right_1_postSilu, transformerCalculations_DEVICE[tIndex].ffn_right_2, ffnDim, L);

        // ffn_left_weights @ ffn_right_postHadamard
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            dim, // m (C rows)
            L, // n (C cols)
            ffnDim, // k (shared dim)
            &alpha,
            transformerWeights_DEVICE[tIndex].ffn_left_weights,
            CUDA_R_32F,
            dim, // lda, col size in mem for col-major
            transformerCalculations_DEVICE[tIndex].ffn_right_postHadamard,
            CUDA_R_32F,
            ffnDim, // ldb, col size in mem for col-major
            &beta,
            transformerCalculations_DEVICE[tIndex].ffn_final,
            CUDA_R_32F,
            dim, // ldc, col size in mem
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

        xTotalThreads = dim * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        addResidualToFFN<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].ffnPlusResidual, transformerCalculations_DEVICE[tIndex].ffn_final, transformerCalculations_DEVICE[tIndex].outputProjPlusResidual, dim, L);
    }

    // char* filename_final_ffn_plus_residual = "final_FFN_plus_residual";
    // saveTensorToJSON_WebGPULayout(transformerCalculations_DEVICE[0].ffnPlusResidual, dim, L, filename_final_ffn_plus_residual);

    xTotalThreads = L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    getRMSColSums<<<numBlocks, threadsPerBlock>>>(ffn_sumByCol_RMS_DEVICE, transformerCalculations_DEVICE[0].ffnPlusResidual, dim, L);
    xTotalThreads = dim * L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    applyRMSNorm<<<numBlocks, threadsPerBlock>>>(
        ffn_postRMS_post_gamma_DEVICE,
        ffn_postRMS_pre_gamma_DEVICE,
        transformerCalculations_DEVICE[0].ffnPlusResidual,
        ffn_sumByCol_RMS_DEVICE,
        final_rms_weights_DEVICE,
        dim, L
    );

    // char* filename_ffn_post_rms = "final_FFN_postRMS_DEVICE";
    // saveTensorToJSON_WebGPULayout(ffn_postRMS_DEVICE, dim, L, filename_ffn_post_rms);

    // embedding_weights.T @ ffnPlusResidual (last transformer)
    // embedding_weights.T: [vocabSize, dim]
    // ffnPlusResidual: [dim, L]
    cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        vocabSize, // m (C rows)
        L, // n (C cols)
        dim, // k (shared dim)
        &alpha,
        embedding_weights_DEVICE,
        CUDA_R_32F,
        dim, // lda, col size in mem for col-major
        ffn_postRMS_DEVICE,
        CUDA_R_32F,
        dim, // ldb, col size in mem for col-major
        &beta,
        vocabScores_DEVICE,
        CUDA_R_32F,
        vocabSize, // ldc, col size in mem
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    xTotalThreads = L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    getVocabMaxByCol_softmax<<<numBlocks, threadsPerBlock>>>(vocabScores_maxByCol_softmax_DEVICE, vocabScores_DEVICE, vocabSize, L);
    getVocabSumByCol_softmax<<<numBlocks, threadsPerBlock>>>(vocabScores_sumByCol_softmax_DEVICE, vocabScores_DEVICE, vocabScores_maxByCol_softmax_DEVICE, vocabSize, L);
    xTotalThreads = vocabSize * L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    applySoftmaxToVocab<<<numBlocks, threadsPerBlock>>>(vocabScores_postSoftmax_DEVICE, vocabScores_DEVICE, vocabScores_sumByCol_softmax_DEVICE, vocabScores_maxByCol_softmax_DEVICE, vocabSize, L);

    return 0;
}