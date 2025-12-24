// nvcc inference.cu -o inference -lcublas
// nvcc inference.cu -o inference -lcublas -gencode arch=compute_75,code=sm_75
/*
nvcc \
  main.cu \
  cJSON.c \
  inference.cu \
  network_globals.cu \
  -o inference \
  -lcublas \
  -gencode arch=compute_75,code=sm_75
*/
/*
nvcc main.cu cJSON.c inference.cu network_globals.cu -o inference -lcublas -gencode arch=compute_75,code=sm_75
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cJSON.h"
#include "network_meta.h"
#include "network_globals.h"
#include "inference.h"

void allocateMemory() {
	cublasCreate(&handle);

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

}

// ============================================================================
// VOCABULARY STORAGE AND HASH TABLE FOR TOKEN LOOKUP
// ============================================================================

#define VOCAB_HASH_SIZE 20011  // Prime number larger than vocabSize for good distribution

// Store the actual token strings
char** vocabTokens = NULL;  // Array of vocabSize strings

// Hash table entry for token -> index lookup
typedef struct VocabHashEntry {
    char* token;
    int index;
    struct VocabHashEntry* next;  // For collision chaining
} VocabHashEntry;

VocabHashEntry* vocabHashTable[VOCAB_HASH_SIZE] = {NULL};

// DJB2 hash function - fast and good distribution for strings
unsigned int hashToken(const char* str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }
    return hash % VOCAB_HASH_SIZE;
}

// Insert a token into the hash table
void vocabHashInsert(const char* token, int index) {
    unsigned int hashIdx = hashToken(token);
    
    VocabHashEntry* entry = (VocabHashEntry*)malloc(sizeof(VocabHashEntry));
    entry->token = vocabTokens[index];  // Point to the stored string
    entry->index = index;
    entry->next = vocabHashTable[hashIdx];  // Chain at front
    vocabHashTable[hashIdx] = entry;
}

// Lookup a token and return its index (-1 if not found)
int vocabLookup(const char* token) {
    unsigned int hashIdx = hashToken(token);
    VocabHashEntry* entry = vocabHashTable[hashIdx];
    
    while (entry) {
        if (strcmp(entry->token, token) == 0) {
            return entry->index;
        }
        entry = entry->next;
    }
    return -1;  // Not found
}

// Get token string by index
const char* vocabGetToken(int index) {
    if (index >= 0 && index < vocabSize && vocabTokens) {
        return vocabTokens[index];
    }
    return NULL;
}

// Load vocabulary from JSON file
int loadVocab(const char* vocabPath) {
    printf("--- Attempting to load vocabulary from %s... ---\n", vocabPath);
    
    // Open and read file
    FILE* file = fopen(vocabPath, "rb");
    if (!file) {
        printf("Error: Vocabulary file not found at '%s'\n", vocabPath);
        return 0;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read file content
    char* jsonContent = (char*)malloc(fileSize + 1);
    if (!jsonContent) {
        printf("Error: Failed to allocate memory for vocab file.\n");
        fclose(file);
        return 0;
    }
    
    fread(jsonContent, 1, fileSize, file);
    jsonContent[fileSize] = '\0';
    fclose(file);
    
    // Parse JSON
    cJSON* root = cJSON_Parse(jsonContent);
    if (!root) {
        printf("Error: Failed to parse vocabulary JSON.\n");
        free(jsonContent);
        return 0;
    }
    
    if (!cJSON_IsArray(root)) {
        printf("Error: Vocabulary JSON is not an array.\n");
        cJSON_Delete(root);
        free(jsonContent);
        return 0;
    }
    
    int arraySize = cJSON_GetArraySize(root);
    if (arraySize != vocabSize) {
        printf("Warning: Vocabulary size (%d) differs from expected vocabSize (%d)\n", arraySize, vocabSize);
    }
    
    // Allocate token storage
    vocabTokens = (char**)malloc(vocabSize * sizeof(char*));
    if (!vocabTokens) {
        printf("Error: Failed to allocate vocabulary token array.\n");
        cJSON_Delete(root);
        free(jsonContent);
        return 0;
    }
    
    // Initialize to NULL
    for (int i = 0; i < vocabSize; i++) {
        vocabTokens[i] = NULL;
    }
    
    // Load tokens and build hash table
    int loadedCount = 0;
    for (int i = 0; i < arraySize && i < vocabSize; i++) {
        cJSON* item = cJSON_GetArrayItem(root, i);
        if (cJSON_IsString(item) && item->valuestring) {
            // Allocate and copy the token string
            size_t len = strlen(item->valuestring);
            vocabTokens[i] = (char*)malloc(len + 1);
            strcpy(vocabTokens[i], item->valuestring);
            
            // Insert into hash table
            vocabHashInsert(item->valuestring, i);
            loadedCount++;
        }
    }
    
    cJSON_Delete(root);
    free(jsonContent);
    
    printf("Loaded %d vocabulary tokens.\n", loadedCount);
    printf("--- Successfully loaded vocabulary. ---\n\n");
    
    return 1;
}

// ============================================================================
// TENSOR READING HELPERS
// ============================================================================

typedef struct {
    int shape[4];
    int shapeDims;
    char dtype[16];
    int numElements;
    int bytesPerElement;
} TensorMeta;

// Parse tensor metadata from cJSON object
int parseTensorMeta(cJSON* tensorObj, TensorMeta* meta) {
    if (!tensorObj) return 0;
    
    // Parse shape array
    cJSON* shapeArr = cJSON_GetObjectItem(tensorObj, "shape");
    if (!shapeArr || !cJSON_IsArray(shapeArr)) return 0;
    
    meta->shapeDims = cJSON_GetArraySize(shapeArr);
    if (meta->shapeDims > 4) meta->shapeDims = 4;
    
    for (int i = 0; i < meta->shapeDims; i++) {
        cJSON* dimItem = cJSON_GetArrayItem(shapeArr, i);
        meta->shape[i] = cJSON_IsNumber(dimItem) ? dimItem->valueint : 0;
    }
    
    // Parse dtype
    cJSON* dtypeObj = cJSON_GetObjectItem(tensorObj, "dtype");
    if (!dtypeObj || !cJSON_IsString(dtypeObj)) return 0;
    strncpy(meta->dtype, dtypeObj->valuestring, sizeof(meta->dtype) - 1);
    meta->dtype[sizeof(meta->dtype) - 1] = '\0';
    
    // Calculate elements and bytes per element
    meta->numElements = 1;
    for (int i = 0; i < meta->shapeDims; i++) {
        meta->numElements *= meta->shape[i];
    }
    
    if (strcmp(meta->dtype, "float32") == 0) {
        meta->bytesPerElement = 4;
    } else if (strcmp(meta->dtype, "float64") == 0) {
        meta->bytesPerElement = 8;
    } else {
        printf("Unsupported dtype: %s\n", meta->dtype);
        return 0;
    }
    
    return 1;
}

// Read tensor data from buffer and store in column-major format
// JS stores row-major (row0, row1, ...), cuBLAS wants column-major (col0, col1, ...)
int readTensorColumnMajor(const unsigned char* dataBuffer, size_t* dataOffset, size_t fileSize,
                          const TensorMeta* meta, float* dest) {
    size_t byteLength = meta->numElements * meta->bytesPerElement;
    
    if (*dataOffset + byteLength > fileSize) {
        printf("Error: Attempting to read past buffer end.\n");
        return 0;
    }
    
    const unsigned char* src = dataBuffer + *dataOffset;
    
    if (meta->shapeDims == 1) {
        // 1D tensor - just copy directly
        if (meta->bytesPerElement == 4) {
            memcpy(dest, src, byteLength);
        } else {
            // float64 -> float32 conversion
            const double* srcDouble = (const double*)src;
            for (int i = 0; i < meta->numElements; i++) {
                dest[i] = (float)srcDouble[i];
            }
        }
    } else if (meta->shapeDims == 2) {
        // 2D tensor - transpose from row-major to column-major
        int rows = meta->shape[0];
        int cols = meta->shape[1];
        
        if (meta->bytesPerElement == 4) {
            const float* srcFloat = (const float*)src;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    // Row-major index: r * cols + c
                    // Column-major index: c * rows + r
                    dest[c * rows + r] = srcFloat[r * cols + c];
                }
            }
        } else {
            const double* srcDouble = (const double*)src;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    dest[c * rows + r] = (float)srcDouble[r * cols + c];
                }
            }
        }
    } else {
        printf("Error: Unsupported tensor dimension: %d\n", meta->shapeDims);
        return 0;
    }
    
    *dataOffset += byteLength;
    return 1;
}

// ============================================================================
// MAIN MODEL LOADING FUNCTION
// ============================================================================

int loadModel(const char* modelName) {
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "./model/%s.bin", modelName);
    
    printf("\n--- Attempting to load model weights from %s... ---\n", filepath);
    
    // Open and read file
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        printf("Error: Model file not found at '%s'\n", filepath);
        return 0;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read entire file into buffer
    unsigned char* fileBuffer = (unsigned char*)malloc(fileSize);
    if (!fileBuffer) {
        printf("Error: Failed to allocate memory for file buffer.\n");
        fclose(file);
        return 0;
    }
    
    if (fread(fileBuffer, 1, fileSize, file) != fileSize) {
        printf("Error: Failed to read file.\n");
        free(fileBuffer);
        fclose(file);
        return 0;
    }
    fclose(file);
    
    // Read header length (8 bytes, little-endian uint64)
    uint64_t headerLength = 0;
    for (int i = 0; i < 8; i++) {
        headerLength |= ((uint64_t)fileBuffer[i]) << (i * 8);
    }
    
    size_t headerStart = 8;
    size_t headerEnd = headerStart + headerLength;
    
    if (headerEnd > fileSize) {
        printf("Error: Header length specified in file is larger than the file itself.\n");
        free(fileBuffer);
        return 0;
    }
    
    // Extract header JSON (null-terminate it)
    char* headerJson = (char*)malloc(headerLength + 1);
    memcpy(headerJson, fileBuffer + headerStart, headerLength);
    headerJson[headerLength] = '\0';
    
    // Calculate data offset with 8-byte alignment padding
    const size_t ALIGNMENT = 8;
    size_t paddingNeeded = (ALIGNMENT - (headerEnd % ALIGNMENT)) % ALIGNMENT;
    size_t dataOffset = headerEnd + paddingNeeded;
    
    printf("Header length: %llu bytes\n", (unsigned long long)headerLength);
    printf("Data starts at offset: %zu\n", dataOffset);
    
    // Parse JSON header
    cJSON* root = cJSON_Parse(headerJson);
    if (!root) {
        printf("Error parsing JSON header.\n");
        free(headerJson);
        free(fileBuffer);
        return 0;
    }
    
    TensorMeta tensorMeta;
    int success = 1;
    
    // ========== Load tokenEmbeddings ==========
    cJSON* tokenEmbObj = cJSON_GetObjectItem(root, "tokenEmbeddings");
    if (tokenEmbObj && success) {
        if (parseTensorMeta(tokenEmbObj, &tensorMeta)) {
            printf("Loading tokenEmbeddings: [%d x %d] %s\n", 
                   tensorMeta.shape[0], tensorMeta.shape[1], tensorMeta.dtype);
            if (!readTensorColumnMajor(fileBuffer, &dataOffset, fileSize, &tensorMeta, embedding_weights)) {
                success = 0;
            } else {
                cudaMemcpy(embedding_weights_DEVICE, embedding_weights, 
                          tensorMeta.numElements * sizeof(float), cudaMemcpyHostToDevice);
            }
        }
    }
    
    // ========== Load finalRMSNormGamma ==========
    cJSON* finalRmsObj = cJSON_GetObjectItem(root, "finalRMSNormGamma");
    if (finalRmsObj && success) {
        if (parseTensorMeta(finalRmsObj, &tensorMeta)) {
            printf("Loading finalRMSNormGamma: [%d] %s\n", 
                   tensorMeta.shape[0], tensorMeta.dtype);
            if (!readTensorColumnMajor(fileBuffer, &dataOffset, fileSize, &tensorMeta, final_rms_weights)) {
                success = 0;
            } else {
                cudaMemcpy(final_rms_weights_DEVICE, final_rms_weights, 
                          tensorMeta.numElements * sizeof(float), cudaMemcpyHostToDevice);
            }
        }
    }
    
    // ========== Load transformerBlocks ==========
    cJSON* blocksArr = cJSON_GetObjectItem(root, "transformerBlocks");
    if (blocksArr && cJSON_IsArray(blocksArr) && success) {
        int numBlocks = cJSON_GetArraySize(blocksArr);
        
        for (int blockIdx = 0; blockIdx < transformers && blockIdx < numBlocks && success; blockIdx++) {
            printf("Loading transformer block %d/%d...\n", blockIdx + 1, transformers);
            
            cJSON* blockObj = cJSON_GetArrayItem(blocksArr, blockIdx);
            if (!blockObj) {
                printf("Error: Expected transformer block object.\n");
                success = 0;
                break;
            }
            
            TransformerWeights* tw = &transformerWeights[blockIdx];
            TransformerWeights* tw_d = &transformerWeights_DEVICE[blockIdx];
            
            // Weight name mappings: JSON key -> C pointer pairs
            struct { const char* jsonKey; float* hostPtr; float** devicePtr; } weightMap[] = {
                {"rmsNormGamma1", tw->rms1_weights, &tw_d->rms1_weights},
                {"queryWeights", tw->query_weights, &tw_d->query_weights},
                {"keyWeights", tw->key_weights, &tw_d->key_weights},
                {"valueWeights", tw->value_weights, &tw_d->value_weights},
                {"outputProjWeights", tw->output_proj_weights, &tw_d->output_proj_weights},
                {"rmsNormGamma2", tw->rms2_weights, &tw_d->rms2_weights},
                {"ffnLeftWeights", tw->ffn_left_weights, &tw_d->ffn_left_weights},
                {"ffnRightWeights1", tw->ffn_right_1_weights, &tw_d->ffn_right_1_weights},
                {"ffnRightWeights2", tw->ffn_right_2_weights, &tw_d->ffn_right_2_weights},
            };
            
            int numWeights = sizeof(weightMap) / sizeof(weightMap[0]);
            
            for (int w = 0; w < numWeights && success; w++) {
                cJSON* weightObj = cJSON_GetObjectItem(blockObj, weightMap[w].jsonKey);
                if (weightObj) {
                    if (parseTensorMeta(weightObj, &tensorMeta)) {
                        if (!readTensorColumnMajor(fileBuffer, &dataOffset, fileSize, &tensorMeta, weightMap[w].hostPtr)) {
                            success = 0;
                        } else {
                            cudaMemcpy(*weightMap[w].devicePtr, weightMap[w].hostPtr, 
                                      tensorMeta.numElements * sizeof(float), cudaMemcpyHostToDevice);
                        }
                    }
                } else {
                    printf("Warning: Weight '%s' not found in block %d\n", weightMap[w].jsonKey, blockIdx);
                }
            }
        }
    }
    
    cJSON_Delete(root);
    free(headerJson);
    free(fileBuffer);
    
    if (success) {
        printf("--- Successfully loaded and assigned weights from %s. ---\n\n", filepath);
    } else {
        printf("!!! Error loading weights from %s. !!!\n\n", filepath);
    }
    
    return success;
}

// ============================================================================
// TOKENIZED STORY LOADING AND CONTEXT SETUP
// ============================================================================

// Load a story from tokenized stories JSON and set up inference context
// storyIndex: which story in the file (0-based)
// percentage: 0-100, how much of the story to use as context (minimum 1 token)
int loadStoryContext(const char* storiesPath, int storyIndex, int percentage) {
    printf("--- Loading story context from %s ---\n", storiesPath);
    printf("Story index: %d, Context percentage: %d%%\n", storyIndex, percentage);
    
    // Open and read file
    FILE* file = fopen(storiesPath, "rb");
    if (!file) {
        printf("Error: Stories file not found at '%s'\n", storiesPath);
        return 0;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read file content
    char* jsonContent = (char*)malloc(fileSize + 1);
    if (!jsonContent) {
        printf("Error: Failed to allocate memory for stories file.\n");
        fclose(file);
        return 0;
    }
    
    fread(jsonContent, 1, fileSize, file);
    jsonContent[fileSize] = '\0';
    fclose(file);
    
    // Parse JSON
    cJSON* root = cJSON_Parse(jsonContent);
    if (!root) {
        printf("Error: Failed to parse stories JSON.\n");
        free(jsonContent);
        return 0;
    }
    
    if (!cJSON_IsArray(root)) {
        printf("Error: Stories JSON is not an array.\n");
        cJSON_Delete(root);
        free(jsonContent);
        return 0;
    }
    
    int numStories = cJSON_GetArraySize(root);
    if (storyIndex < 0 || storyIndex >= numStories) {
        printf("Error: Story index %d out of range (0-%d).\n", storyIndex, numStories - 1);
        cJSON_Delete(root);
        free(jsonContent);
        return 0;
    }
    
    // Get the selected story
    cJSON* storyArr = cJSON_GetArrayItem(root, storyIndex);
    if (!storyArr || !cJSON_IsArray(storyArr)) {
        printf("Error: Story at index %d is not an array.\n", storyIndex);
        cJSON_Delete(root);
        free(jsonContent);
        return 0;
    }
    
    int totalTokens = cJSON_GetArraySize(storyArr);
    if (totalTokens == 0) {
        printf("Error: Story at index %d has no tokens.\n", storyIndex);
        cJSON_Delete(root);
        free(jsonContent);
        return 0;
    }
    
    // Calculate context length based on percentage (minimum 1 token)
    int contextLength = (totalTokens * percentage) / 100;
    if (contextLength < 1) contextLength = 1;
    if (contextLength > L) {
        printf("Warning: Context length %d exceeds max sequence length L=%d, truncating.\n", contextLength, L);
        contextLength = L;
    }
    
    printf("Story has %d tokens, using %d tokens as context.\n", totalTokens, contextLength);
    
    // Print the context tokens and convert to indices
    printf("\n=== Story Context ===\n");
    
    int success = 1;
    for (int i = 0; i < contextLength && success; i++) {
        cJSON* tokenItem = cJSON_GetArrayItem(storyArr, i);
        if (!tokenItem || !cJSON_IsString(tokenItem)) {
            printf("Error: Token at position %d is not a string.\n", i);
            success = 0;
            break;
        }
        
        const char* tokenStr = tokenItem->valuestring;
        
        // Lookup token index in vocabulary
        int tokenIndex = vocabLookup(tokenStr);
        if (tokenIndex < 0) {
            printf("Error: Token '%s' not found in vocabulary.\n", tokenStr);
            success = 0;
            break;
        }
        
        // Set the index in seqTokenIndices
        seqTokenIndices[i] = tokenIndex;
        
        // Print token (escape newlines for readability)
        if (strcmp(tokenStr, "\n") == 0) {
            printf("[\\n]");
        } else if (strcmp(tokenStr, "\t") == 0) {
            printf("[\\t]");
        } else {
            printf("%s", tokenStr);
        }
    }
    printf("\n=====================\n\n");
    
    if (!success) {
        cJSON_Delete(root);
        free(jsonContent);
        return 0;
    }
    
    // Pad remaining positions with token index 5 ("?") if context is shorter than L
    for (int i = contextLength; i < L; i++) {
        seqTokenIndices[i] = 5;  // Padding token
    }
    
    // Copy to device
    cudaMemcpy(seqTokenIndices_DEVICE, seqTokenIndices, L * sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Set %d context tokens in seqTokenIndices (padded to L=%d with token 5).\n", contextLength, L);
    printf("--- Story context loaded successfully. ---\n\n");
    
    cJSON_Delete(root);
    free(jsonContent);
    
    return contextLength - 1;  // Return rightSeqEndIndex (last real token index)
}

// ============================================================================
// TEXT GENERATION WITH TOP-K SAMPLING
// ============================================================================

// Structure to hold token index and probability for sorting
typedef struct {
    int tokenIdx;
    float prob;
} TokenProb;

// Comparison function for qsort (descending probability)
int compareTokenProb(const void* a, const void* b) {
    float diff = ((TokenProb*)b)->prob - ((TokenProb*)a)->prob;
    return (diff > 0) ? 1 : (diff < 0 ? -1 : 0);
}

int main(int argc, char* argv[]) {
    allocateMemory();
    
    const char* modelName = (argc > 1) ? argv[1] : "model";
    if (!loadModel(modelName)) {
        printf("Failed to load model '%s', using random weights.\n", modelName);
    }
    
    // Load vocabulary
    char vocabPath[512];
    snprintf(vocabPath, sizeof(vocabPath), "./model/vocab.json");
    if (!loadVocab(vocabPath)) {
        printf("Failed to load vocabulary.\n");
    }
    
    // Load story context
    // Default: story index 0, 50% of tokens as context
    int storyIndex = (argc > 2) ? atoi(argv[2]) : 0;
    int contextPercent = (argc > 3) ? atoi(argv[3]) : 50;
    
    const char* storiesPath = "./tokenizedStories/tokenizedStories_0001.json";
    int rightSeqEndIndex = loadStoryContext(storiesPath, storyIndex, contextPercent);
    if (rightSeqEndIndex < 0) {
        printf("Failed to load story context.\n");
        return 1;
    }
    
    // Check if sequence is already full (L tokens)
    if (rightSeqEndIndex >= L - 1) {
        printf("Sequence already contains L=%d tokens. No generation needed.\n", L);
        runInference();
        return 0;
    }
    
    // Allocate host memory for vocabulary scores
    float* vocabScores_postSoftmax = (float*)malloc(vocabSize * L * sizeof(float));
    if (!vocabScores_postSoftmax) {
        printf("Error: Failed to allocate memory for vocabulary scores.\n");
        return 1;
    }
    
    // Generation loop
    int tokensGenerated = 0;
    int maxTokensToGenerate = L - (rightSeqEndIndex + 1);  // How many tokens until we hit L total
    
    printf("\n========================================\n");
    printf("Starting text generation (max %d new tokens)\n", maxTokensToGenerate);
    printf("Press Enter to generate next token, or 'q' + Enter to quit\n");
    printf("========================================\n\n");
    
    while (rightSeqEndIndex < L - 1) {
        // Run inference
        runInference();
        
        // Copy vocabulary scores from device to host
        cudaMemcpy(vocabScores_postSoftmax, vocabScores_postSoftmax_DEVICE, 
                   vocabSize * L * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Calculate offset for the prediction position
        // vocabScores_postSoftmax is column-major: [vocabSize x L]
        // For position rightSeqEndIndex, we want column rightSeqEndIndex
        size_t offset = rightSeqEndIndex * vocabSize;
        
        // Create array of token probabilities for sorting
        TokenProb* tokenProbs = (TokenProb*)malloc(vocabSize * sizeof(TokenProb));
        for (int i = 0; i < vocabSize; i++) {
            tokenProbs[i].tokenIdx = i;
            tokenProbs[i].prob = vocabScores_postSoftmax[offset + i];
        }
        
        // Sort by probability (descending)
        qsort(tokenProbs, vocabSize, sizeof(TokenProb), compareTokenProb);
        
        // Display top-10 most probable tokens
        printf("\n--- Top 10 Most Probable Next Tokens (position %d) ---\n", rightSeqEndIndex + 1);
        for (int i = 0; i < 10 && i < vocabSize; i++) {
            const char* tokenStr = vocabGetToken(tokenProbs[i].tokenIdx);
            if (tokenStr) {
                // Escape special characters for display
                if (strcmp(tokenStr, "\n") == 0) {
                    printf("%2d. [\\n]         (idx: %5d, prob: %.6f)\n", 
                           i + 1, tokenProbs[i].tokenIdx, tokenProbs[i].prob);
                } else if (strcmp(tokenStr, "\t") == 0) {
                    printf("%2d. [\\t]         (idx: %5d, prob: %.6f)\n", 
                           i + 1, tokenProbs[i].tokenIdx, tokenProbs[i].prob);
                } else if (strlen(tokenStr) == 0) {
                    printf("%2d. [EMPTY]       (idx: %5d, prob: %.6f)\n", 
                           i + 1, tokenProbs[i].tokenIdx, tokenProbs[i].prob);
                } else {
                    printf("%2d. %-12s (idx: %5d, prob: %.6f)\n", 
                           i + 1, tokenStr, tokenProbs[i].tokenIdx, tokenProbs[i].prob);
                }
            }
        }
        printf("-------------------------------------------------------\n");
        
        // Get the most probable token
        int nextTokenIdx = tokenProbs[0].tokenIdx;
        const char* nextToken = vocabGetToken(nextTokenIdx);
        
        printf("\nMost probable token: ");
        if (nextToken) {
            if (strcmp(nextToken, "\n") == 0) {
                printf("[\\n]");
            } else if (strcmp(nextToken, "\t") == 0) {
                printf("[\\t]");
            } else {
                printf("%s", nextToken);
            }
        }
        printf(" (index: %d)\n", nextTokenIdx);
        
        free(tokenProbs);
        
        // Wait for user input
        printf("\nPress Enter to add this token and continue, or 'q' + Enter to quit: ");
        fflush(stdout);
        
        char input[10];
        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }
        
        // Check if user wants to quit
        if (input[0] == 'q' || input[0] == 'Q') {
            printf("Generation stopped by user.\n");
            break;
        }
        
        // Append the token to the sequence
        rightSeqEndIndex++;
        seqTokenIndices[rightSeqEndIndex] = nextTokenIdx;
        tokensGenerated++;
        
        // Update the device memory with the new sequence
        cudaMemcpy(seqTokenIndices_DEVICE, seqTokenIndices, L * sizeof(int), cudaMemcpyHostToDevice);
        
        printf("\n[Token added. Total context: %d tokens, generated: %d]\n", 
               rightSeqEndIndex + 1, tokensGenerated);
    }
    
    printf("\n========================================\n");
    printf("Generation complete. Generated %d new tokens.\n", tokensGenerated);
    printf("Final sequence length: %d tokens\n", rightSeqEndIndex + 1);
    printf("========================================\n\n");
    
    // Display final generated sequence
    printf("=== Final Generated Sequence ===\n");
    for (int i = 0; i <= rightSeqEndIndex; i++) {
        const char* token = vocabGetToken(seqTokenIndices[i]);
        if (token) {
            if (strcmp(token, "\n") == 0) {
                printf("[\\n]");
            } else if (strcmp(token, "\t") == 0) {
                printf("[\\t]");
            } else {
                printf("%s", token);
            }
        }
    }
    printf("\n================================\n");
    
    free(vocabScores_postSoftmax);
    return 0;
}