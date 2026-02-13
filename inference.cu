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

__global__ void getRMSColSums(float* rmsSumByCol, float* x, int dim_) {
    int colIndex = blockIdx.x;
    int tIndex = threadIdx.x;

    int colOffset = colIndex * dim_;

    extern __shared__ float sData[]; // blockDim.x from host invocation; must be a power of 2    

    float threadSum = 0.0f;
    for (int rowIndex = tIndex; rowIndex < dim_; rowIndex += blockDim.x) {
        float val = x[colOffset + rowIndex];
        threadSum += (val * val);
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
        rmsSumByCol[colIndex] = sqrtf((sData[0] / dim_) + 1e-8);
    }
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
    int maxCount = dim_ / 2 * L_;
    if (index >= maxCount) {
        return;
    }

    int effectiveIndex = index * 2;
    int colIndex = effectiveIndex / dim_;
    int rowIndex = effectiveIndex - colIndex * dim_;
    int headIndex = rowIndex / headDim_;
    int headRelativeRowIndex = rowIndex - headIndex * headDim_;
    int headRelativeColOffset = colIndex * headDim_;

    float cosTheta = preComputedRopeTheta[headRelativeColOffset + headRelativeRowIndex];
    float sinTheta = preComputedRopeTheta[headRelativeColOffset + headRelativeRowIndex + 1];
    float firstValOfPair = keysOrValues[effectiveIndex];
    float secondValOfPair = keysOrValues[effectiveIndex + 1];

    keysOrValuesPostRoPE[effectiveIndex] = cosTheta * firstValOfPair - sinTheta * secondValOfPair;
    keysOrValuesPostRoPE[effectiveIndex + 1] = sinTheta * firstValOfPair + cosTheta * secondValOfPair;
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
    int attnHeadIndex = blockIdx.x;
    int headRelativeColIndex = blockIdx.y;
    int tIndex = threadIdx.x; // rowIndex

    int colIndex = attnHeadIndex * L_ + headRelativeColIndex;
    int colOffset = colIndex * L_;

    extern __shared__ float sData[]; // blockDim.x from host invocation; must be a power of 2    

    float threadColMax = -1.0e20f;
    for (int rowIndex = tIndex; rowIndex <= headRelativeColIndex; rowIndex += blockDim.x) {
        if (attnHeadDimScaledMaskedKtQByHead[colOffset + rowIndex] > threadColMax) {
            threadColMax = attnHeadDimScaledMaskedKtQByHead[colOffset + rowIndex];
        }
    }
    sData[tIndex] = threadColMax;
    __syncthreads();

    for (int reductionSize = blockDim.x / 2; reductionSize > 0; reductionSize /= 2) {
        if (tIndex < reductionSize) {
            sData[tIndex] = sData[tIndex] > sData[tIndex + reductionSize] ? sData[tIndex] : sData[tIndex + reductionSize];
        }
        __syncthreads();
    }

    if (tIndex == 0) {
        attnByHead_maxByCol_softmax[colIndex] = sData[0];
    }
}

__global__ void getAttnHeadsSumByCol_softmax(float* attnByHead_sumByCol_softmax, float* attnByHead_expfCache_softmax, float* attnHeadDimScaledMaskedKtQByHead, float* attnByHead_maxByCol_softmax, int attnHeads_, int L_) {
    int attnHeadIndex = blockIdx.x;
    int headRelativeColIndex = blockIdx.y;
    int tIndex = threadIdx.x; // rowIndex

    int colIndex = attnHeadIndex * L_ + headRelativeColIndex;
    int colOffset = colIndex * L_;

    extern __shared__ float sData[]; // blockDim.x from host invocation; must be a power of 2

    float threadSum = 0.0f;
    for (int rowIndex = tIndex; rowIndex <= headRelativeColIndex; rowIndex += blockDim.x /* numThreads per block */) {
        float expVal = expf(attnHeadDimScaledMaskedKtQByHead[colOffset + rowIndex] - attnByHead_maxByCol_softmax[colIndex]);
        attnByHead_expfCache_softmax[colOffset + rowIndex] = expVal;
        threadSum += expVal;
    }
    sData[tIndex] = threadSum;
    __syncthreads();

    // blockDim.x = 256; reductionSize = 128; reductionSize > 0; reductionSize: 128, 64, 32, 16, etc.
    for (int reductionSize = (blockDim.x / 2); reductionSize > 0; reductionSize /= 2) {
        if (tIndex < reductionSize) {
            sData[tIndex] = sData[tIndex] + sData[tIndex + reductionSize];
        }
        __syncthreads();
    }

    if (tIndex == 0) {
        attnByHead_sumByCol_softmax[colIndex] = sData[0];
    }
}

__global__ void applySoftmaxToAttnHeads(float* attnByHead_postSoftmax, float* expfCache, float* sumByCol, int attnHeads_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int L2 = L_ * L_;
    int maxCount = attnHeads_ * L2;
    if (index >= maxCount) {
        return;
    }

    int headIndex = index / L2;
    int headRelativeColIndex = (index - headIndex * L2) / L_;
    int globalColIndex = headIndex * L_ + headRelativeColIndex;
    int rowIndex = index - globalColIndex * L_;

    if (rowIndex <= headRelativeColIndex) {
        attnByHead_postSoftmax[index] = (expfCache[index] / sumByCol[globalColIndex]);
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

__global__ void getVocabMaxByCol_softmax(float* vocab_maxByCol_softmax, float* vocabScores, int vocabSize_) {
    int colIndex = blockIdx.x;
    int tIndex = threadIdx.x; // rowIndex

    int colOffset = colIndex * vocabSize_;

    extern __shared__ float sData[]; // blockDim.x from host invocation; must be a power of 2    

    float threadColMax = -1.0e20f;
    for (int rowIndex = tIndex; rowIndex < vocabSize_; rowIndex += blockDim.x) {
        if (vocabScores[colOffset + rowIndex] > threadColMax) {
            threadColMax = vocabScores[colOffset + rowIndex];
        }
    }
    sData[tIndex] = threadColMax;
    __syncthreads();

    for (int reductionSize = blockDim.x / 2; reductionSize > 0; reductionSize /= 2) {
        if (tIndex < reductionSize) {
            sData[tIndex] = sData[tIndex] > sData[tIndex + reductionSize] ? sData[tIndex] : sData[tIndex + reductionSize];
        }
        __syncthreads();
    }

    if (tIndex == 0) {
        vocab_maxByCol_softmax[colIndex] = sData[0];
    }
}

__global__ void getVocabSumByCol_softmax(float* vocab_sumByCol_softmax, float* vocab_expfCache_softmax, float* vocabScores, float* vocab_maxByCol_softmax, int vocabSize_) {
    int colIndex = blockIdx.x;
    int tIndex = threadIdx.x; // rowIndex

    int colOffset = colIndex * vocabSize_;

    extern __shared__ float sData[]; // blockDim.x from host invocation; must be a power of 2

    float threadSum = 0.0f;
    float colMax = vocab_maxByCol_softmax[colIndex];
    for (int rowIndex = tIndex; rowIndex < vocabSize_; rowIndex += blockDim.x /* numThreads per block */) {
        float expVal = expf(vocabScores[colOffset + rowIndex] - colMax);
        vocab_expfCache_softmax[colOffset + rowIndex] = expVal;
        threadSum += expVal;
    }
    sData[tIndex] = threadSum;
    __syncthreads();

    // blockDim.x = 256; reductionSize = 128; reductionSize > 0; reductionSize: 128, 64, 32, 16, etc.
    for (int reductionSize = (blockDim.x / 2); reductionSize > 0; reductionSize /= 2) {
        if (tIndex < reductionSize) {
            sData[tIndex] = sData[tIndex] + sData[tIndex + reductionSize];
        }
        __syncthreads();
    }

    if (tIndex == 0) {
        vocab_sumByCol_softmax[colIndex] = sData[0];
    }
}

__global__ void applySoftmaxToVocab(float* vocabScores_postSoftmax, float* vocab_expfCache, float* vocab_sumByCol_softmax, int vocabSize_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxCount = vocabSize_ * L_;
    if (index >= maxCount) {
        return;
    }

    int colIndex = index / vocabSize_;
    vocabScores_postSoftmax[index] = vocab_expfCache[index] / vocab_sumByCol_softmax[colIndex];
}

void getPreComputedRopeTheta(float* preComputedRopeTheta, int maxL_) {
    int numPairs = headDim / 2;
    for (int colIndex = 0; colIndex < maxL_; colIndex++) {
        int colOffset = colIndex * headDim;
        for (int pairIndex = 0; pairIndex < numPairs; pairIndex++) {
            float theta = colIndex * powf(ropeDenomBase, (-2.0f * pairIndex / headDim));
            preComputedRopeTheta[colOffset + pairIndex * 2] = cosf(theta);
            preComputedRopeTheta[colOffset + pairIndex * 2 + 1] = sinf(theta);            
        }
    }
}

int runInference(int L) {
    int xTotalThreads = L * dim;
    int numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = 256 * sizeof(float);

    setInputSeqEmbeddings<<<numBlocks, threadsPerBlock>>>(x_DEVICE, seqTokenIndices_DEVICE, embedding_weights_DEVICE, dim, L);

    for (int tIndexCountUp = 0; tIndexCountUp < transformers; tIndexCountUp++) {
        int tIndex = transformers - 1 - tIndexCountUp;

        if (tIndex == transformers - 1) {
            getRMSColSums<<<L, 256, sharedMemSize>>>(transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1, x_DEVICE, dim);
        } else {
            getRMSColSums<<<L, 256, sharedMemSize>>>(transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1, transformerCalculations_DEVICE[tIndex + 1].ffnPlusResidual, dim);
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

        xTotalThreads = dimPairs * L;
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

        dim3 attnSoftmaxGridDim(attnHeads, L);
        getAttnHeadsMaxByCol_softmax<<<attnSoftmaxGridDim, 256, sharedMemSize>>>(transformerCalculations_DEVICE[tIndex].attnByHead_maxByCol_softmax, transformerCalculations_DEVICE[tIndex].attnKtQByHeadScaledMasked, attnHeads, L);
        getAttnHeadsSumByCol_softmax<<<attnSoftmaxGridDim, 256, sharedMemSize>>>(transformerCalculations_DEVICE[tIndex].attnByHead_sumByCol_softmax, transformerCalculations_DEVICE[tIndex].attnByHead_expfCache_softmax, transformerCalculations_DEVICE[tIndex].attnKtQByHeadScaledMasked, transformerCalculations_DEVICE[tIndex].attnByHead_maxByCol_softmax, attnHeads, L);
        xTotalThreads = attnHeads * L * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;    
        applySoftmaxToAttnHeads<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].attnByHead_postSoftmax, transformerCalculations_DEVICE[tIndex].attnByHead_expfCache_softmax, transformerCalculations_DEVICE[tIndex].attnByHead_sumByCol_softmax, attnHeads, L);

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

        getRMSColSums<<<L, 256, sharedMemSize>>>(transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_sumByCol_RMS2, transformerCalculations_DEVICE[tIndex].outputProjPlusResidual, dim);
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

    getRMSColSums<<<L, 256, sharedMemSize>>>(ffn_sumByCol_RMS_DEVICE, transformerCalculations_DEVICE[0].ffnPlusResidual, dim);
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
        ffn_postRMS_post_gamma_DEVICE,
        CUDA_R_32F,
        dim, // ldb, col size in mem for col-major
        &beta,
        vocabScores_DEVICE,
        CUDA_R_32F,
        vocabSize, // ldc, col size in mem
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    getVocabMaxByCol_softmax<<<L, 256, sharedMemSize>>>(vocabScores_maxByCol_softmax_DEVICE, vocabScores_DEVICE, vocabSize);
    getVocabSumByCol_softmax<<<L, 256, sharedMemSize>>>(vocabScores_sumByCol_softmax_DEVICE, vocabScores_expfCache_softmax_DEVICE, vocabScores_DEVICE, vocabScores_maxByCol_softmax_DEVICE, vocabSize);
    xTotalThreads = vocabSize * L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    applySoftmaxToVocab<<<numBlocks, threadsPerBlock>>>(vocabScores_postSoftmax_DEVICE, vocabScores_expfCache_softmax_DEVICE, vocabScores_sumByCol_softmax_DEVICE, vocabSize, L);

    return 0;
}