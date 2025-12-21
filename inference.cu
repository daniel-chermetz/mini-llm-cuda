// nvcc inference.cu -o inference -lcublas
// nvcc inference.cu -o inference -lcublas -gencode arch=compute_75,code=sm_75

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define dim 32 
#define attnHeads 4 
#define headDim 8
#define ropeDenomBase 10000 
#define ffnDimMultiplier 4 
#define transformers 2 
#define L 3
#define vocabSize 10097

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
    float* ffn_right1_weights; 
    float* ffn_right2_weights; 
} TransformerWeights;

TransformerWeights transformerWeights[transformers];
TransformerWeights transformerWeights_DEVICE[transformers];

float* preComputedRopeTheta;
float* preComputedRopeTheta_DEVICE;

float* x_DEVICE;
typedef struct {
    float* rmsNormCols;
    float* xPostRMS1; 
    float* queries; 
    float* keys;
    float* values;
    float* queriesPostRoPE;
    float* keysPostRoPE;
    float* KtQByHead;
    float* sqrtMaskedKtQByHead;
    float* maxByCol;
    float* sumByCol;
    float* softmaxAttn;
} TransformerCalculations_DEVICE;

TransformerCalculations_DEVICE transformerCalculations_DEVICE[transformers];

void getPreComputedRopeTheta(float* preComputedRopeTheta) {
    // Math.cos(colIndex / Math.pow(NetworkMeta.ropeDenomBase, (2 * pairIndex / headDim)));
    // (Math.sin(colIndex / Math.pow(NetworkMeta.ropeDenomBase, (2 * pairIndex / headDim))));
    int numPairs = headDim / 2;
    for (int colIndex = 0; colIndex < L; colIndex++) {
        int colOffset = colIndex * headDim;
        for (int pairIndex = 0; pairIndex < numPairs; pairIndex++) {
            float theta = colIndex * powf(ropeDenomBase, (-2.0f * pairIndex / headDim));
            preComputedRopeTheta[colOffset + pairIndex*2] = cosf(theta);
            preComputedRopeTheta[colOffset + pairIndex*2 + 1] = sinf(theta);            
        }
    }
}

// blockIdx.x, blockDim.x, threadIdx.x
__global__ void setInputSeqEmbeddings(float* x, int* seqTokenIndices, float* embedding_weights, int L_, int dim_) {
    int currentIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = L_ * dim_ - 1;

    if (currentIndex > maxIndex) {
        return;
    }

    int lIndex = currentIndex / dim_;
    int coordIndex = currentIndex - lIndex * dim_;
    
    int tokenIndex = seqTokenIndices[lIndex];

    // col-major
    x[currentIndex] = embedding_weights[tokenIndex * dim_ + coordIndex];
}

// blockIdx.x, blockDim.x, threadIdx.x
__global__ void getRMSNormCols(float* rmsNormCols, float* x, int L_, int dim_) {
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
    rmsNormCols[colIndex] = sqrtf((sumSquared / dim_) + 1e-8);
}

// blockIdx.x, blockDim.x, threadIdx.x
__global__ void RMSNorm(float* xPostRMS1, float* x, float* rmsNormCols, float* rms1_weights, int L_, int dim_) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = dim_ * L_ - 1;
    
    if (index > maxIndex) {
        return;
    }

    int colIndex = index / dim_;
    int rowIndex = index - colIndex * dim_;
    xPostRMS1[index] = (rms1_weights[rowIndex] * (x[index] / rmsNormCols[colIndex]));
}

// blockIdx.x, blockDim.x, threadIdx.x
__global__ void RoPE(float* keysOrValuesPostRoPE, float* keysOrValues, float* preComputedRopeTheta, int headDim_, int dim_, int L_) {
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

// blockIdx.x, blockDim.x, threadIdx.x
__global__ void sqrtMaskAttn(float* sqrtMaskedKtQByHead, float* KtQByHead, int L_, int attnHeads_, int headDim_) {
    int L2 = (L_ * L_);

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = attnHeads_ * L2 - 1;
    if (index > maxIndex) {
        return;
    }

    int headIndex = index / L2;
    int colIndex = (index - headIndex * L2) / L;
    int rowIndex = index - headIndex * L2 - colIndex * L;

    if (rowIndex > colIndex) {
        sqrtMaskedKtQByHead[index] = 0;
        return;
    }

    sqrtMaskedKtQByHead[index] = KtQByHead[index] / sqrtf(headDim_);
}

// blockIdx.x, blockDim.x, threadIdx.x
__global__ void getColMax(float* maxByCol, float* sqrtMaskedKtQByHead, int L_, int attnHeads_) {
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
        float val = sqrtMaskedKtQByHead[colOffset + rowIndex];
        if (val > colMax) {
            colMax = val;
        }
    }

    maxByCol[index] = colMax;
}

// blockIdx.x, blockDim.x, threadIdx.x
__global__ void getColSum(float* sumByCol, float* maxByCol, float* sqrtMaskedKtQByHead, int L_, int attnHeads_) {
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
        sum += expf(sqrtMaskedKtQByHead[colOffset + rowIndex] - maxByCol[index]);
    }
    sumByCol[index] = sum;
}

// blockIdx.x, blockDim.x, threadIdx.x
__global__ void softmaxCols(float* softmaxAttn, float* sqrtMaskedKtQByHead, float* sumByCol, float* maxByCol, int L_, int attnHeads_) {
    int L2 = (L_ * L_);

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = attnHeads_ * L2 - 1;
    if (index > maxIndex) {
        return;
    }

    int headIndex = index / L2;
    int headRelativeColIndex = (index - headIndex * L2) / L_;
    int globalColIndex = headIndex * L_ + headRelativeColIndex;
    // int rowIndex = index - headIndex * L2 - headRelativeColIndex * L_;
    int rowIndex = index - globalColIndex * L_;


    if (rowIndex <= headRelativeColIndex) {
        softmaxAttn[index] = (expf(sqrtMaskedKtQByHead[index] - maxByCol[globalColIndex]) / sumByCol[globalColIndex]);
        return;
    }

    softmaxAttn[index] = 0;    
}

int runInference() {
    int xTotalThreads = L * dim;
    int threadsPerBlock = 256;
    int numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // __global__ void setInputSeqEmbeddings(float* x, float* seqTokenIndices, float* embedding_weights, int L, int dim) {
    setInputSeqEmbeddings<<<numBlocks, threadsPerBlock>>>(x_DEVICE, seqTokenIndices_DEVICE, embedding_weights_DEVICE, L, dim);

    float* debug_cpu_x = (float*)malloc(L * dim * sizeof(float));
    cudaMemcpy(debug_cpu_x, x_DEVICE, L * dim * sizeof(float), cudaMemcpyDeviceToHost);
    //printf("First 10 values of the first token embedding:\n");
    for(int i=0; i < 10; i++) {
    //    printf("%f ", debug_cpu_x[i]);
    }
    //printf("\n");

    xTotalThreads = L;
    threadsPerBlock = 256;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    // __global__ void getRMSNormCols(float* rmsNormCols, float* x, int L_, int dim_) {
    getRMSNormCols<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[0].rmsNormCols, x_DEVICE, L, dim);

    float* debug_cpu_rmsNormCols = (float*)malloc(L * sizeof(float));
    cudaMemcpy(debug_cpu_rmsNormCols, transformerCalculations_DEVICE[0].rmsNormCols, L * sizeof(float), cudaMemcpyDeviceToHost);
    //printf("First 10 values of RMS Norm of cols:\n");
    for(int i=0; i < 10; i++) {
    //    printf("%f ", debug_cpu_rmsNormCols[i]);
    }
    //printf("\n");

    xTotalThreads = dim * L;
    threadsPerBlock = 256;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    // __global__ void RMSNorm(float* xPostRMS1, float* x, float* rmsNormCols, float* rms1_weights, int L_, int dim_) {
    RMSNorm<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[0].xPostRMS1, x_DEVICE, transformerCalculations_DEVICE[0].rmsNormCols, transformerWeights_DEVICE[0].rms1_weights, L, dim);

    float* debug_cpu_xPostRMS1 = (float*)malloc(dim * L * sizeof(float));
    cudaMemcpy(debug_cpu_xPostRMS1, transformerCalculations_DEVICE[0].xPostRMS1, dim * L * sizeof(float), cudaMemcpyDeviceToHost);
    //printf("First 10 values of xPostRMS1:\n");
    for(int i=0; i < 100; i++) {
    //    printf("%f ", debug_cpu_xPostRMS1[i]);
    }
    //printf("\n");

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // C = A @ B
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        dim, // rows A & C
        L, // cols B & C
        dim, // cols A, rows B
        &alpha,
        transformerWeights_DEVICE[0].query_weights,
        CUDA_R_32F,
        dim, // rows A
        transformerCalculations_DEVICE[0].xPostRMS1,
        CUDA_R_32F,
        dim, // rows B        
        &beta,
        transformerCalculations_DEVICE[0].queries,
        CUDA_R_32F,
        dim, // rows C
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT             
    );

    float* debug_cpu_queries = (float*)malloc(dim * L * sizeof(float));
    cudaMemcpy(debug_cpu_queries, transformerCalculations_DEVICE[0].queries, dim * L * sizeof(float), cudaMemcpyDeviceToHost);
    printf("First 100 values of queries:\n");
    for(int i=0; i < 100; i++) {
        printf("%f ", debug_cpu_queries[i]);
    }
    printf("\n");

    // C = A @ B
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        dim, // rows A & C
        L, // cols B & C
        dim, // cols A, rows B
        &alpha,
        transformerWeights_DEVICE[0].key_weights,
        CUDA_R_32F,
        dim, // rows A
        transformerCalculations_DEVICE[0].xPostRMS1,
        CUDA_R_32F,
        dim, // rows B        
        &beta,
        transformerCalculations_DEVICE[0].keys,
        CUDA_R_32F,
        dim, // rows C
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT             
    );

    // C = A @ B
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        dim, // rows A & C
        L, // cols B & C
        dim, // cols A, rows B
        &alpha,
        transformerWeights_DEVICE[0].value_weights,
        CUDA_R_32F,
        dim, // rows A
        transformerCalculations_DEVICE[0].xPostRMS1,
        CUDA_R_32F,
        dim, // rows B        
        &beta,
        transformerCalculations_DEVICE[0].values,
        CUDA_R_32F,
        dim, // rows C
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT             
    );

    xTotalThreads = dim * L;
    threadsPerBlock = 256;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;   
    RoPE<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[0].queriesPostRoPE, transformerCalculations_DEVICE[0].queries, preComputedRopeTheta_DEVICE, headDim, dim, L);
    RoPE<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[0].keysPostRoPE, transformerCalculations_DEVICE[0].keys, preComputedRopeTheta_DEVICE, headDim, dim, L);
    float* debug_cpu_queriesPostRope = (float*)malloc(dim * L * sizeof(float));
    cudaMemcpy(debug_cpu_queriesPostRope, transformerCalculations_DEVICE[0].queriesPostRoPE, dim * L * sizeof(float), cudaMemcpyDeviceToHost);
    printf("First 50 values of queries post RoPE:\n");
    for(int i=0; i < 50; i++) {
        printf("%f ", debug_cpu_queriesPostRope[i]);
    }
    printf("\n");
    float* debug_cpu_keysPostRope = (float*)malloc(dim * L * sizeof(float));
    cudaMemcpy(debug_cpu_keysPostRope, transformerCalculations_DEVICE[0].keysPostRoPE, dim * L * sizeof(float), cudaMemcpyDeviceToHost);
    printf("First 50 values of keys post RoPE:\n");
    for(int i=0; i < 50; i++) {
        printf("%f ", debug_cpu_keysPostRope[i]);
    }
    printf("\n");

    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        L, // m (K.t@Q rows)
        L, // n (K.t@Q cols)
        headDim, // k
        &alpha,
        transformerCalculations_DEVICE[0].keysPostRoPE,
        CUDA_R_32F,
        dim, // lda, col size in mem for col-major
        headDim, // mem stride in K.t to reach next head
        transformerCalculations_DEVICE[0].queriesPostRoPE,
        CUDA_R_32F,
        dim, // ldb, col size in mem for col-major
        headDim, // mem stride in Q to reach next head
        &beta,
        transformerCalculations_DEVICE[0].KtQByHead,
        CUDA_R_32F,
        L, // ldc, col size in mem for col-major
        L * L, // mem stride in K.t@Q to reach next head
        attnHeads,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    float* debug_cpu_KtQByHead = (float*)malloc(attnHeads * L * L * sizeof(float));
    cudaMemcpy(debug_cpu_KtQByHead, transformerCalculations_DEVICE[0].KtQByHead, attnHeads * L * L * sizeof(float), cudaMemcpyDeviceToHost);
    printf("First 36 values of KtQByHead:\n");
    for(int i=0; i < 36; i++) {
        printf("%f ", debug_cpu_KtQByHead[i]);
    }
    printf("\n");
    
    xTotalThreads = attnHeads * L * L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    // float* sqrtMaskedKtQByHead, float* KtQByHead, int L_, int attnHeads_, int headDim_  
    sqrtMaskAttn<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[0].sqrtMaskedKtQByHead, transformerCalculations_DEVICE[0].KtQByHead, L, attnHeads, headDim);

    xTotalThreads = attnHeads * L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
    // __global__ void getColMax(float* maxByCol, float* sqrtMaskedKtQByHead, int L_, int attnHeads_) {
    getColMax<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[0].maxByCol, transformerCalculations_DEVICE[0].sqrtMaskedKtQByHead, L, attnHeads);
    // __global__ void getColSum(float* sumByCol, float* maxByCol, float* sqrtMaskedKtQByHead, int L_, int attnHeads_) {    
    getColSum<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[0].sumByCol, transformerCalculations_DEVICE[0].maxByCol, transformerCalculations_DEVICE[0].sqrtMaskedKtQByHead, L, attnHeads);

    xTotalThreads = attnHeads * L * L;
    numBlocks = (xTotalThreads + threadsPerBlock - 1) / threadsPerBlock;    
    //__global__ void softmaxCols(float* softmaxAttn, float* sqrtMaskedKtQByHead, float* sumByCol, float* maxByCol, int L_, int attnHeads_) {
    softmaxCols<<<numBlocks, threadsPerBlock>>>(transformerCalculations_DEVICE[0].softmaxAttn, transformerCalculations_DEVICE[0].sqrtMaskedKtQByHead, transformerCalculations_DEVICE[0].sumByCol, transformerCalculations_DEVICE[0].maxByCol, L, attnHeads);

    // V @ softmax_attn
    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        headDim, // m (C rows)
        L, // n (C cols)
        L, // k (shared dim)
        &alpha,
        transformerCalculations_DEVICE[0].values,
        CUDA_R_32F,
        dim, // lda, col size in mem for col-major
        headDim, // mem stride in values to reach next head
        transformerCalculations_DEVICE[0].softmaxAttn,
        CUDA_R_32F,
        L, // ldb, col size in mem for col-major
        (L * L), // mem stride in softmaxAttn to reach next head
        &beta,
        transformerCalculations_DEVICE[0].valueScaledSoftmaxAttn,
        CUDA_R_32F,
        headDim, // ldc, col size in mem for col-major
        (headDim * L), // mem stride in valueScaledSoftmaxAttn to reach next head
        attnHeads,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );  

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
    /*float* debug_cpu_preComputedRopeTheta = (float*)malloc(preComputedRopeTheta_size);
    cudaMemcpy(debug_cpu_preComputedRopeTheta, preComputedRopeTheta_DEVICE, preComputedRopeTheta_size, cudaMemcpyDeviceToHost);
    printf("First 10 values of preComputedRopeTheta:\n");
    for(int i=0; i < 10; i++) {
        printf("%f ", debug_cpu_preComputedRopeTheta[i]);
    }
    printf("\n");*/

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

        size_t ffn_right1_weights_size = dim * ffnDimMultiplier * dim * sizeof(float);
        currentTransformerWeights->ffn_right1_weights = (float*)malloc(ffn_right1_weights_size);
        for (int i = 0; i < dim * dim * ffnDimMultiplier; i++) {
            currentTransformerWeights->ffn_right1_weights[i] = ((float)rand() / (float)RAND_MAX);
        }   
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_right1_weights, ffn_right1_weights_size);        
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_right1_weights, currentTransformerWeights->ffn_right1_weights, ffn_right1_weights_size, cudaMemcpyHostToDevice);

        size_t ffn_right2_weights_size = dim * ffnDimMultiplier * dim * sizeof(float);
        currentTransformerWeights->ffn_right2_weights = (float*)malloc(ffn_right2_weights_size); 
        for (int i = 0; i < dim * dim * ffnDimMultiplier; i++) {
            currentTransformerWeights->ffn_right2_weights[i] = ((float)rand() / (float)RAND_MAX);
        }
        cudaMalloc((void**)&currentTransformerWeights_DEVICE->ffn_right2_weights, ffn_right2_weights_size);
        cudaMemcpy(currentTransformerWeights_DEVICE->ffn_right2_weights, currentTransformerWeights->ffn_right2_weights, ffn_right2_weights_size, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].rmsNormCols, L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].xPostRMS1, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].queries, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].keys, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].values, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].queriesPostRoPE, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].keysPostRoPE, dim * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].KtQByHead, attnHeads * L * L * sizeof(float));        
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].sqrtMaskedKtQByHead, attnHeads * L * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].maxByCol, attnHeads * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].sumByCol, attnHeads * L * sizeof(float));
        cudaMalloc((void**)&transformerCalculations_DEVICE[transformerIndex].softmaxAttn, attnHeads * L * L * sizeof(float));
    }

    /*printf("Random sequence token indices:\n");
    for (int i = 0; i < L; i++) {
        printf("%d, ", seqTokenIndices[i]);
    }*/

    runInference();

    return 0;
}
