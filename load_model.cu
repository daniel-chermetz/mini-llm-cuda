#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "./cJSON/cJSON.h"

#include "network_meta.h"
#include "network_globals.h"

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

float* transposeTensorColumnMajor(float* preTransposeTensor, int preTransposeColSize, int preTransposeRowSize) {
    // preTransposeColSize :== number of rows
    // preTransposeRowSize :== number of columns
    int numElements = preTransposeRowSize * preTransposeColSize;
    float* transposedTensor = (float*)malloc(numElements * sizeof(float));
    for (int i = 0; i < numElements; i++) {
        int colIndex = i / preTransposeColSize;
        int rowIndex = i - colIndex * preTransposeColSize;

        int postTransposeIndex = rowIndex * preTransposeRowSize + colIndex;
        transposedTensor[postTransposeIndex] = preTransposeTensor[i];
    }

    return transposedTensor;
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
    
    // ========== Load token ==========
    cJSON* tokenEmbObj = cJSON_GetObjectItem(root, "tokenEmbeddings");
    if (tokenEmbObj && success) {
        if (parseTensorMeta(tokenEmbObj, &tensorMeta)) {
            printf("Loading tokenEmbeddings: [%d x %d] %s\n", 
                   tensorMeta.shape[0], tensorMeta.shape[1], tensorMeta.dtype);
            if (!readTensorColumnMajor(fileBuffer, &dataOffset, fileSize, &tensorMeta, embedding_weights)) {
                success = 0;
            } else {
                float* transposed_embedding_weights = transposeTensorColumnMajor(embedding_weights, vocabSize, dim);
                free(embedding_weights);
                cudaMemcpy(embedding_weights_DEVICE, transposed_embedding_weights, 
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
                {"rmsGamma", tw->rms1_weights, &tw_d->rms1_weights},
                {"queryWeights", tw->query_weights, &tw_d->query_weights},
                {"keyWeights", tw->key_weights, &tw_d->key_weights},
                {"valueWeights", tw->value_weights, &tw_d->value_weights},
                {"outputProjectionWeights", tw->output_proj_weights, &tw_d->output_proj_weights},
                {"rmsGamma2", tw->rms2_weights, &tw_d->rms2_weights},
                {"feedForwardWeights1A", tw->ffn_right_1_weights, &tw_d->ffn_right_1_weights},
                {"feedForwardWeights1B", tw->ffn_right_2_weights, &tw_d->ffn_right_2_weights},
                {"feedForwardWeights2", tw->ffn_left_weights, &tw_d->ffn_left_weights},
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