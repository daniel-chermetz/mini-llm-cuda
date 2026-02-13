/*
nvcc main.cu ./cJSON/cJSON.c inference.cu load_model.cu network_globals.cu training.cu \
     inference_orchestrator.cu gradient_testing.cu memory_allocation.cu \
     training_orchestrator.cu optimizer.cu random_weights.cu save_model.cu \
     -o inference \
     -lcublas -lcurand \
     -arch=sm_86
*/

#include <chrono>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cJSON.h"

#include "network_meta.h"
#include "network_globals.h"
#include "load_model.h"
#include "inference.h"
#include "training.h"
#include "inference_orchestrator.h"
#include "gradient_testing.h"
#include "memory_allocation.h"
#include "training_orchestrator.h"
#include "random_weights.h"

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
    if (contextLength > maxL + 1) {
        printf("Warning: Context length %d exceeds max capacity (maxL+1=%d), truncating.\n", contextLength, maxL + 1);
        contextLength = maxL + 1;
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
    
    // Copy to device (only the actual context tokens, no padding needed)
    cudaMemcpy(seqTokenIndices_DEVICE, seqTokenIndices, contextLength * sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Set %d context tokens in seqTokenIndices (story length L=%d).\n", contextLength, contextLength);
    printf("--- Story context loaded successfully. ---\n\n");
    
    cJSON_Delete(root);
    free(jsonContent);
    
    return contextLength;  // Return story length (L)
}

int main(int argc, char* argv[]) {
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    allocateMemory(true);
    
    const char* modelNameFromArg = (argc > 1) ? argv[1] : "model";
    //char* modelName = "model_10_lr_5e6";
    //char* modelName = "model_11_lr_1e5";
    //char* modelName = "model_12_lr_4e6";
    //char* modelName = "model_13_lr_3e6";
    //char* modelName = "model_14_lr_3e6";
    // char* modelName = "model_15_lr_3e6";
    // if (!loadModel(modelName)) {
    //     printf("Failed to load model '%s', using random weights.\n", modelName);
    // }
    
    // Use random weights for training from scratch
    initializeRandomWeights(0.02f);  // Range: [-0.02, +0.02]
    
    // Load vocabulary
    char vocabPath[512];
    snprintf(vocabPath, sizeof(vocabPath), "./model/vocab.json");
    if (!loadVocab(vocabPath)) {
        printf("Failed to load vocabulary.\n");
    }
    
    // ========================================================================
    // MODE SELECTION: Comment/uncomment to switch between modes
    // ========================================================================
    
    // --- INFERENCE MODE ---
    // Run text generation inference loop
    const char* storiesPath = "./tokenizedStories/tokenizedStories_0001.json";
    int storyIndex = (argc > 2) ? atoi(argv[2]) : 0;
    int contextPercent = (argc > 3) ? atoi(argv[3]) : 100;
    bool skipUserInput = true;
    bool verboseOutput = false;
    // runInferenceLoop(storiesPath, storyIndex, contextPercent, skipUserInput, verboseOutput);
    
    // --- GRADIENT TESTING MODE ---
    // Run gradient verification tests
    // runGradientTests();
    
    // --- TRAINING MODE ---
    // Run the training loop over all tokenized stories
    runTrainingLoop();
    
    return 0;
}
