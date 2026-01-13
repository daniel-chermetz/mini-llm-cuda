/*
 * save_model.cu
 * Save model weights to binary file (compatible with JS loader format)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "network_meta.h"
#include "network_globals.h"
#include "./cJSON/cJSON.h"

// Helper to add tensor metadata to JSON and copy data to host buffer
static size_t addTensorToSave(cJSON* parent, const char* name, float* devicePtr, 
                               int* shape, int shapeDims, float* hostBuffer, size_t offset) {
    // Calculate total elements
    size_t numElements = 1;
    for (int i = 0; i < shapeDims; i++) {
        numElements *= shape[i];
    }
    
    // Create tensor metadata
    cJSON* tensorObj = cJSON_CreateObject();
    cJSON* shapeArr = cJSON_CreateIntArray(shape, shapeDims);
    cJSON_AddItemToObject(tensorObj, "shape", shapeArr);
    cJSON_AddStringToObject(tensorObj, "dtype", "float32");
    cJSON_AddItemToObject(parent, name, tensorObj);
    
    // Copy data from device to host buffer at offset
    cudaMemcpy(hostBuffer + offset, devicePtr, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    
    return numElements;
}

void saveModelWeights(const char* filename, int iterationNum) {
    printf("\n=== Saving model weights to %s (iteration %d) ===\n", filename, iterationNum);
    
    // Calculate total size needed for all weights
    size_t totalElements = 0;
    
    // Embedding weights: dim x vocabSize
    totalElements += dim * vocabSize;
    // Final RMS gamma: dim
    totalElements += dim;
    // Per transformer:
    for (int t = 0; t < transformers; t++) {
        totalElements += dim;           // rmsGamma (rms1)
        totalElements += dim * dim;     // queryWeights
        totalElements += dim * dim;     // keyWeights
        totalElements += dim * dim;     // valueWeights
        totalElements += dim * dim;     // outputProjectionWeights
        totalElements += dim;           // rmsGamma2 (rms2)
        totalElements += ffnDim * dim;  // feedForwardWeights1A (ffn_right_1)
        totalElements += ffnDim * dim;  // feedForwardWeights1B (ffn_right_2)
        totalElements += dim * ffnDim;  // feedForwardWeights2 (ffn_left)
    }
    
    // Allocate host buffer for all weights
    float* hostBuffer = (float*)malloc(totalElements * sizeof(float));
    if (!hostBuffer) {
        printf("Error: Failed to allocate host buffer for saving weights\n");
        return;
    }
    
    // Build JSON metadata
    cJSON* metadata = cJSON_CreateObject();
    cJSON* transformerBlocks = cJSON_CreateArray();
    
    size_t offset = 0;
    int shape2D[2];
    int shape1D[1];
    
    // Token embeddings (dim x vocabSize)
    shape2D[0] = dim; shape2D[1] = vocabSize;
    offset += addTensorToSave(metadata, "tokenEmbeddings", embedding_weights_DEVICE, shape2D, 2, hostBuffer, offset);
    
    // Final RMS norm gamma (dim)
    shape1D[0] = dim;
    offset += addTensorToSave(metadata, "finalRMSNormGamma", final_rms_weights_DEVICE, shape1D, 1, hostBuffer, offset);
    
    // Transformer blocks
    for (int t = 0; t < transformers; t++) {
        cJSON* blockMeta = cJSON_CreateObject();
        
        // rmsGamma (dim)
        shape1D[0] = dim;
        offset += addTensorToSave(blockMeta, "rmsGamma", transformerWeights_DEVICE[t].rms1_weights, shape1D, 1, hostBuffer, offset);
        
        // queryWeights (dim x dim)
        shape2D[0] = dim; shape2D[1] = dim;
        offset += addTensorToSave(blockMeta, "queryWeights", transformerWeights_DEVICE[t].query_weights, shape2D, 2, hostBuffer, offset);
        
        // keyWeights (dim x dim)
        offset += addTensorToSave(blockMeta, "keyWeights", transformerWeights_DEVICE[t].key_weights, shape2D, 2, hostBuffer, offset);
        
        // valueWeights (dim x dim)
        offset += addTensorToSave(blockMeta, "valueWeights", transformerWeights_DEVICE[t].value_weights, shape2D, 2, hostBuffer, offset);
        
        // outputProjectionWeights (dim x dim)
        offset += addTensorToSave(blockMeta, "outputProjectionWeights", transformerWeights_DEVICE[t].output_proj_weights, shape2D, 2, hostBuffer, offset);
        
        // rmsGamma2 (dim)
        shape1D[0] = dim;
        offset += addTensorToSave(blockMeta, "rmsGamma2", transformerWeights_DEVICE[t].rms2_weights, shape1D, 1, hostBuffer, offset);
        
        // feedForwardWeights1A = ffn_right_1 (ffnDim x dim)
        shape2D[0] = ffnDim; shape2D[1] = dim;
        offset += addTensorToSave(blockMeta, "feedForwardWeights1A", transformerWeights_DEVICE[t].ffn_right_1_weights, shape2D, 2, hostBuffer, offset);
        
        // feedForwardWeights1B = ffn_right_2 (ffnDim x dim)
        offset += addTensorToSave(blockMeta, "feedForwardWeights1B", transformerWeights_DEVICE[t].ffn_right_2_weights, shape2D, 2, hostBuffer, offset);
        
        // feedForwardWeights2 = ffn_left (dim x ffnDim)
        shape2D[0] = dim; shape2D[1] = ffnDim;
        offset += addTensorToSave(blockMeta, "feedForwardWeights2", transformerWeights_DEVICE[t].ffn_left_weights, shape2D, 2, hostBuffer, offset);
        
        cJSON_AddItemToArray(transformerBlocks, blockMeta);
    }
    
    cJSON_AddItemToObject(metadata, "transformerBlocks", transformerBlocks);
    
    // Convert metadata to string
    char* headerString = cJSON_PrintUnformatted(metadata);
    size_t headerLength = strlen(headerString);
    
    // Calculate padding for 8-byte alignment
    const size_t ALIGNMENT = 8;
    size_t paddingNeeded = (ALIGNMENT - (headerLength % ALIGNMENT)) % ALIGNMENT;
    
    // Open file for writing
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Failed to open %s for writing\n", filename);
        free(hostBuffer);
        free(headerString);
        cJSON_Delete(metadata);
        return;
    }
    
    // Write header length (8 bytes, little-endian)
    uint64_t headerLen64 = (uint64_t)headerLength;
    fwrite(&headerLen64, sizeof(uint64_t), 1, file);
    
    // Write header string
    fwrite(headerString, 1, headerLength, file);
    
    // Write padding
    if (paddingNeeded > 0) {
        char padding[8] = {0};
        fwrite(padding, 1, paddingNeeded, file);
    }
    
    // Write weight data
    fwrite(hostBuffer, sizeof(float), totalElements, file);
    
    fclose(file);
    
    size_t totalSize = sizeof(uint64_t) + headerLength + paddingNeeded + (totalElements * sizeof(float));
    printf("Successfully saved weights to %s. Total size: %zu bytes (%zu elements)\n", 
           filename, totalSize, totalElements);
    
    // Cleanup
    free(hostBuffer);
    free(headerString);
    cJSON_Delete(metadata);
}
