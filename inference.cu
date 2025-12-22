// nvcc inference.cu -o inference -lcublas
// nvcc inference.cu -o inference -lcublas -gencode arch=compute_75,code=sm_75

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define dim 8
#define attnHeads 2 
#define headDim 4
#define ropeDenomBase 10000
#define ffnDimMultiplier 4
#define ffnDim 32
#define transformers 4 
#define L 3
#define vocabSize 10

int* seqTokenIndices;
int* seqTokenIndices_DEVICE;

float* embedding_weights;
float* embedding_weights_DEVICE;

float* final_rms_weights;
float* final_rms_weights_DEVICE;

typedef struct { 
    float* rms1_weights; 
    float* query_weights; 
    float* key_weights; 
    float* value_weights; 
    float* output_proj_weights; 
    float* rms2_weights; 
    float* ffn_left_weights; 
    float* ffn_right_1_weights; 
    float* ffn_right_2_weights; 
} TransformerWeights;

TransformerWeights transformerWeights[transformers];
TransformerWeights transformerWeights_DEVICE[transformers];

float* preComputedRopeTheta;
float* preComputedRopeTheta_DEVICE;

float* x_DEVICE;
typedef struct {
    float* x_sumByCol_RMS1;
    float* x_postRMS1; 
    float* queries; 
    float* keys;
    float* values;
    float* queriesPostRoPE;
    float* keysPostRoPE;
    float* attnKtQByHead;
    float* attnKtQByHeadScaledMasked;
    float* attnByHead_maxByCol_softmax;
    float* attnByHead_sumByCol_softmax;
    float* attnByHead_postSoftmax;
    float* valueScaledSoftmaxAttn;
    float* outputProj;
    float* outputProjPlusResidual;
    float* outputProjPlusResidual_sumByCol_RMS2;
    float* outputProjPlusResidual_postRMS2;
    float* ffn_right_1_preSilu;
    float* ffn_right_1_postSilu;
    float* ffn_right_2;
    float* ffn_right_postHadamard;
    float* ffn_final;
    float* ffnPlusResidual;
} TransformerCalculations_DEVICE;
TransformerCalculations_DEVICE transformerCalculations_DEVICE[transformers];

float* vocabScores_DEVICE;
float* vocabScores_maxByCol_softmax_DEVICE;
float* vocabScores_sumByCol_softmax_DEVICE;
float* vocabScores_postSoftmax_DEVICE;

// __global__?
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

__global__ void setInputSeqEmbeddings(float* x, int* seqTokenIndices, float* embedding_weights, int dim_, int L_) {
    int currentIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = L_ * dim_ - 1;

    if (currentIndex > maxIndex) {
        return;
    }

    int lIndex = currentIndex / dim_;
    int coordIndex = currentIndex - lIndex * dim_;
    int tokenIndex = seqTokenIndices[lIndex];

    x[currentIndex] = embedding_weights[tokenIndex * dim_ + coordIndex];
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

__global__ void applyRMSNorm(float* postRMSMat, float* preRMSMat, float* rmsSumByCol, float* rms_weights, int dim_, int L_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = dim_ * L_ - 1;
    
    if (index > maxIndex) {
        return;
    }

    int colIndex = index / dim_;
    int rowIndex = index - colIndex * dim_;
    postRMSMat[index] = (rms_weights[rowIndex] * (preRMSMat[index] / rmsSumByCol[colIndex]));
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

void printFloatMatrixToDebug(float* deviceMatrix, int matrixSize, int numValsToPrint, int numValsInPrintRow) {
    float* debug_cpu_matix = (float*)malloc(matrixSize * sizeof(float));
    cudaMemcpy(debug_cpu_matix, deviceMatrix, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < numValsToPrint; i++) {
        if (i == 0) {
            printf("[");
        }
        printf("%f", debug_cpu_matix[i]);
        if (i < numValsToPrint - 1) {
            if ((i + 1) % numValsInPrintRow == 0) {
                printf(",\n ");
            } else {
                printf(", ");
            }
        }
    }
    printf("]\n\n");
}

int runInference() {
    int threadsPerBlock = 256;
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    int xTotalThreads = L * dim;
    int numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    setInputSeqEmbeddings<<<numBlocks, threadsPerBlock>>>(x_DEVICE, seqTokenIndices_DEVICE, embedding_weights_DEVICE, dim, L);

    for (int tIndex = 0; tIndex < transformers; tIndex++) {
        xTotalThreads = L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        if (tIndex == 0) {
            getRMSColSums<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1, x_DEVICE, dim, L);
        } else {
            getRMSColSums<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1, transformerCalculations_DEVICE[tIndex - 1].ffn_final, dim, L);
        }

        xTotalThreads = dim * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        if (tIndex == 0) {
            applyRMSNorm<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].x_postRMS1, x_DEVICE, transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1, transformerWeights_DEVICE[tIndex].rms1_weights, dim, L);
        } else {
            applyRMSNorm<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].x_postRMS1, transformerCalculations_DEVICE[tIndex - 1].ffn_final, transformerCalculations_DEVICE[tIndex].x_sumByCol_RMS1, transformerWeights_DEVICE[tIndex].rms1_weights, dim, L);
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
            transformerCalculations_DEVICE[tIndex].x_postRMS1,
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
            transformerCalculations_DEVICE[tIndex].x_postRMS1,
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
            transformerCalculations_DEVICE[tIndex].x_postRMS1,
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
        if (tIndex == 0) {
            addResidualToOutputProj<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].outputProjPlusResidual, transformerCalculations_DEVICE[tIndex].outputProj, x_DEVICE, dim, L);
        } else {
            addResidualToOutputProj<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].outputProjPlusResidual, transformerCalculations_DEVICE[tIndex].outputProj, transformerCalculations_DEVICE[tIndex - 1].ffn_final, dim, L);
        }

        xTotalThreads = L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        getRMSColSums<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_sumByCol_RMS2, transformerCalculations_DEVICE[tIndex].outputProjPlusResidual, dim, L);
        xTotalThreads = dim * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;    
        applyRMSNorm<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2, transformerCalculations_DEVICE[tIndex].outputProjPlusResidual, transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_sumByCol_RMS2, transformerWeights_DEVICE[tIndex].rms2_weights, dim, L);
        //printf("outputProjPlusResidual_postRMS2\n");
        //printFloatMatrixToDebug(transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2, dim * L, dim * L, L);         

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
            transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2,
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
            transformerCalculations_DEVICE[tIndex].outputProjPlusResidual_postRMS2,
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
        //printf("Silu\n");
        //printFloatMatrixToDebug(transformerCalculations_DEVICE[tIndex].ffn_right_1_postSilu, ffnDim * L, ffnDim * L, L);         
        hadamardMultiplyFFN<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].ffn_right_postHadamard, transformerCalculations_DEVICE[tIndex].ffn_right_1_postSilu, transformerCalculations_DEVICE[tIndex].ffn_right_2, ffnDim, L);
        //printf("Hadamard\n");        
        //printFloatMatrixToDebug(transformerCalculations_DEVICE[tIndex].ffn_right_postHadamard, ffnDim * L, ffnDim * L, L);         

        // ffn_left_weights @ ffn_right_weights
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
        // printf("ffn_final\n");
        // printFloatMatrixToDebug(transformerCalculations_DEVICE[tIndex].ffn_final, dim * L, dim * L, L);

        xTotalThreads = dim * L;
        numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        if (tIndex == 0) {
            addResidualToFFN<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].ffnPlusResidual, transformerCalculations_DEVICE[tIndex].ffn_final, x_DEVICE, dim, L);
        } else {
            addResidualToFFN<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[tIndex].ffnPlusResidual, transformerCalculations_DEVICE[tIndex].ffn_final, transformerCalculations_DEVICE[tIndex - 1].ffn_final, dim, L);            
        }
    }

    // embedding_weights.T @ ffnPlusResidual (last transformer)
    // embedding_weights:[dim, vocabSize] --> embedding_weights.T:[vocabSize, dim]
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
        transformerCalculations_DEVICE[transformers - 1].ffnPlusResidual,
        CUDA_R_32F,
        dim, // ldb, col size in mem for col-major
        &beta,
        vocabScores_DEVICE,
        CUDA_R_32F,
        vocabSize, // ldc, col size in mem
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    printf("vocabScores_DEVICE\n");
    printFloatMatrixToDebug(vocabScores_DEVICE, vocabSize * L, vocabSize * L, L);

    xTotalThreads = L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    getVocabMaxByCol_softmax<<<numBlocks, threadsPerBlock>>>(vocabScores_maxByCol_softmax_DEVICE, vocabScores_DEVICE, vocabSize, L);
    printf("vocabScores_maxByCol_softmax_DEVICE\n");
    printFloatMatrixToDebug(vocabScores_maxByCol_softmax_DEVICE, L, L, 1);
    getVocabSumByCol_softmax<<<numBlocks, threadsPerBlock>>>(vocabScores_sumByCol_softmax_DEVICE, vocabScores_DEVICE, vocabScores_maxByCol_softmax_DEVICE, vocabSize, L);
    printf("vocabScores_sumByCol_softmax_DEVICE\n");
    printFloatMatrixToDebug(vocabScores_sumByCol_softmax_DEVICE, L, L, 1);
    xTotalThreads = vocabSize * L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    applySoftmaxToVocab<<<numBlocks, threadsPerBlock>>>(vocabScores_postSoftmax_DEVICE, vocabScores_DEVICE, vocabScores_sumByCol_softmax_DEVICE, vocabScores_maxByCol_softmax_DEVICE, vocabSize, L);
    printf("vocabScores_postSoftmax_DEVICE\n");
    printFloatMatrixToDebug(vocabScores_postSoftmax_DEVICE, vocabSize * L, vocabSize * L, L);
    
    cublasDestroy(handle);
    return 0;
}

int main() {
    /* ---------------- Device info ---------------- */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\n===== GPU INFO =====\n");
    printf("Device: \"%s\"\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    // 1. VRAM (Crucial for storing weights)
    // Convert bytes to Gigabytes (GB) for readability
    double totalMemGB = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    printf("Total VRAM: %.2f GB\n", totalMemGB);

    // 2. Compute Power
    printf("Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    printf("Clock Rate: %.0f MHz\n", (double)prop.clockRate / 1000.0);

    // 3. Kernel Launch Constraints
    // This tells you the maximum size of the arrays you can process in parallel blocks
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp Size: %d\n", prop.warpSize); // Almost always 32

    // 4. Memory Speed
    printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("Memory Clock: %.0f MHz\n", (double)prop.memoryClockRate / 1000.0);
    
    printf("====================\n\n");    

    srand(0);

    size_t seqTokenIndices_size = L * sizeof(int); 
    seqTokenIndices = (int*)malloc(seqTokenIndices_size);
    for (int i = 0; i < L; i++) {
        seqTokenIndices[i] = rand() % vocabSize;
    }
    cudaMalloc((void**)&seqTokenIndices_DEVICE, seqTokenIndices_size);    
    cudaMemcpy(seqTokenIndices_DEVICE, seqTokenIndices, seqTokenIndices_size, cudaMemcpyHostToDevice);

    size_t embedding_weights_size = dim * vocabSize * sizeof(float); 
    embedding_weights = (float*)malloc(embedding_weights_size);
    for (int i = 0; i < dim * vocabSize; i++) {
        embedding_weights[i] = ((float)rand() / (float)RAND_MAX);
    }
    cudaMalloc((void**)&embedding_weights_DEVICE, embedding_weights_size);
    cudaMemcpy(embedding_weights_DEVICE, embedding_weights, embedding_weights_size, cudaMemcpyHostToDevice);

    size_t final_rms_size = dim * sizeof(float); 
    final_rms_weights = (float*)malloc(final_rms_size);
    for (int i = 0; i < dim; i++) {
        final_rms_weights[i] = ((float)rand() / (float)RAND_MAX);
    }
    cudaMalloc((void**)&final_rms_weights_DEVICE, final_rms_size);
    cudaMemcpy(final_rms_weights_DEVICE, final_rms_weights, final_rms_size, cudaMemcpyHostToDevice);

    size_t preComputedRopeTheta_size = headDim * L * sizeof(float);
    preComputedRopeTheta = (float*)malloc(preComputedRopeTheta_size);
    getPreComputedRopeTheta(preComputedRopeTheta);
    cudaMalloc((void**)&preComputedRopeTheta_DEVICE, preComputedRopeTheta_size);
    cudaMemcpy(preComputedRopeTheta_DEVICE, preComputedRopeTheta, preComputedRopeTheta_size, cudaMemcpyHostToDevice);

    size_t x_size = dim * L * sizeof(float); 
    cudaMalloc((void**)&x_DEVICE, x_size);

    for (int transformerIndex = 0; transformerIndex < transformers; transformerIndex++) { 
        TransformerWeights *currentTransformerWeights = &transformerWeights[transformerIndex];
        TransformerWeights *currentTransformerWeights_DEVICE = &transformerWeights_DEVICE[transformerIndex];

        size_t rms1_weights_size = dim * sizeof(float); 
        currentTransformerWeights->rms1_weights = (float*)malloc(rms1_weights_size); 
        for (int i = 0; i < dim; i++) {
            currentTransformerWeights->rms1_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->rms1_weights, rms1_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->rms1_weights, currentTransformerWeights->rms1_weights, rms1_weights_size, cudaMemcpyHostToDevice);

        size_t query_weights_size = dim * dim * sizeof(float); 
        currentTransformerWeights->query_weights = (float*)malloc(query_weights_size); 
        for (int i = 0; i < dim * dim; i++) {
            currentTransformerWeights->query_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->query_weights, query_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->query_weights, currentTransformerWeights->query_weights, query_weights_size, cudaMemcpyHostToDevice);

        size_t key_weights_size = dim * dim * sizeof(float);
        currentTransformerWeights->key_weights = (float*)malloc(key_weights_size);
        for (int i = 0; i < dim * dim; i++) {
            currentTransformerWeights->key_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->key_weights, key_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->key_weights, currentTransformerWeights->key_weights, key_weights_size, cudaMemcpyHostToDevice);

        size_t value_weights_size = dim * dim * sizeof(float);
        currentTransformerWeights->value_weights = (float*)malloc(value_weights_size);
        for (int i = 0; i < dim * dim; i++) {
            currentTransformerWeights->value_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->value_weights, value_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->value_weights, currentTransformerWeights->value_weights, value_weights_size, cudaMemcpyHostToDevice);

        size_t output_proj_weights_size = dim * dim * sizeof(float);
        currentTransformerWeights->output_proj_weights = (float*)malloc(output_proj_weights_size);
        for (int i = 0; i < dim * dim; i++) {
            currentTransformerWeights->output_proj_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->output_proj_weights, output_proj_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->output_proj_weights, currentTransformerWeights->output_proj_weights, output_proj_weights_size, cudaMemcpyHostToDevice);

        size_t rms2_weights_size = dim * sizeof(float);
        currentTransformerWeights->rms2_weights = (float*)malloc(rms2_weights_size);
        for (int i = 0; i < dim; i++) {
            currentTransformerWeights->rms2_weights[i] = ((float)rand() / (float)RAND_MAX);
        }        
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->rms2_weights, rms2_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->rms2_weights, currentTransformerWeights->rms2_weights, rms2_weights_size, cudaMemcpyHostToDevice);

        size_t ffn_left_weights_size = dim * dim * ffnDimMultiplier * sizeof(float);
        currentTransformerWeights->ffn_left_weights = (float*)malloc(ffn_left_weights_size);
        for (int i = 0; i < dim * dim * ffnDimMultiplier; i++) {
            currentTransformerWeights->ffn_left_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_left_weights, ffn_left_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_left_weights, currentTransformerWeights->ffn_left_weights, ffn_left_weights_size, cudaMemcpyHostToDevice);

        size_t ffn_right_1_weights_size = dim * ffnDimMultiplier * dim * sizeof(float);
        currentTransformerWeights->ffn_right_1_weights = (float*)malloc(ffn_right_1_weights_size);
        for (int i = 0; i < dim * dim * ffnDimMultiplier; i++) {
            currentTransformerWeights->ffn_right_1_weights[i] = ((float)rand() / (float)RAND_MAX);
        }   
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_right_1_weights, ffn_right_1_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_right_1_weights, currentTransformerWeights->ffn_right_1_weights, ffn_right_1_weights_size, cudaMemcpyHostToDevice);

        size_t ffn_right_2_weights_size = dim * ffnDimMultiplier * dim * sizeof(float);
        currentTransformerWeights->ffn_right_2_weights = (float*)malloc(ffn_right_2_weights_size); 
        for (int i = 0; i < dim * dim * ffnDimMultiplier; i++) {
            currentTransformerWeights->ffn_right_2_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_right_2_weights, ffn_right_2_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_right_2_weights, currentTransformerWeights->ffn_right_2_weights, ffn_right_2_weights_size, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].x_sumByCol_RMS1, L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].x_postRMS1, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].queries, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].keys, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].values, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].queriesPostRoPE, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].keysPostRoPE, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnKtQByHead, attnHeads * L * L * sizeof(float));        
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnKtQByHeadScaledMasked, attnHeads * L * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnByHead_maxByCol_softmax, attnHeads * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnByHead_sumByCol_softmax, attnHeads * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].attnByHead_postSoftmax, attnHeads * L * L * sizeof(float));        
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].valueScaledSoftmaxAttn, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProj, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProjPlusResidual, dim * L * sizeof(float));                
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProjPlusResidual_sumByCol_RMS2, L * sizeof(float));        
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].outputProjPlusResidual_postRMS2, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_1_preSilu, dim * ffnDimMultiplier * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_1_postSilu, dim * ffnDimMultiplier * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_2, dim * ffnDimMultiplier * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_right_postHadamard, dim * ffnDimMultiplier * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffn_final, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].ffnPlusResidual, dim * L * sizeof(float));       
    }

    cudaMalloc((void**)&vocabScores_DEVICE, vocabSize * L * sizeof(float));
    cudaMalloc((void**)&vocabScores_maxByCol_softmax_DEVICE, L * sizeof(float));
    cudaMalloc((void**)&vocabScores_sumByCol_softmax_DEVICE, L * sizeof(float));
    cudaMalloc((void**)&vocabScores_postSoftmax_DEVICE, vocabSize * L * sizeof(float)); 

    runInference();
    return 0;
}
