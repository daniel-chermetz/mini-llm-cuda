/*
 * training_orchestrator.cu
 * Module for orchestrating the training loop over tokenized stories
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "cJSON.h"
#include "network_meta.h"
#include "network_globals.h"  // Must come before <termios.h> to avoid cuBLAS macro conflicts
#include "load_model.h"
#include "inference.h"
#include "training.h"
#include "training_orchestrator.h"

// System headers that may define conflicting macros (include after cuBLAS)
#include <sys/stat.h>
#include <dirent.h>
#include <termios.h>
#include <unistd.h>

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Base path for tokenized stories
#define TOKENIZED_STORIES_PATH "./tokenizedStories"

// ============================================================================
// MODULE-LEVEL STORAGE
// ============================================================================

// Host-side storage for rightEndIndices (kept on host for easy access in inner loop)
static int* hostRightEndIndices_storage = nullptr;
static int hostRightEndIndices_count = 0;

// Host-side buffer for softmax scores (for logging predictions)
static float* hostVocabSoftmax = nullptr;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Check if a file exists
static bool fileExists(const char* path) {
    struct stat buffer;
    return (stat(path, &buffer) == 0);
}

// Generate the path for a tokenized stories file given its index (1-based)
// Example: index 1 -> "./tokenizedStories/tokenizedStories_0001.json"
static void getStoriesFilePath(int fileIndex, char* outPath, size_t outSize) {
    snprintf(outPath, outSize, "%s/tokenizedStories_%04d.json", TOKENIZED_STORIES_PATH, fileIndex);
}

// Wait for user to press any key
static void waitForKeypress() {
    printf("\nPress any key to continue...\n");
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

// Get display-safe token string
static void getDisplayToken(int tokenIdx, char* outStr, size_t outSize) {
    const char* tokenStr = vocabGetToken(tokenIdx);
    if (tokenStr) {
        if (strcmp(tokenStr, "\n") == 0) {
            snprintf(outStr, outSize, "[\\n]");
        } else if (strcmp(tokenStr, "\t") == 0) {
            snprintf(outStr, outSize, "[\\t]");
        } else if (strcmp(tokenStr, " ") == 0) {
            snprintf(outStr, outSize, "[SP]");
        } else if (strlen(tokenStr) == 0) {
            snprintf(outStr, outSize, "[EMPTY]");
        } else {
            snprintf(outStr, outSize, "%s", tokenStr);
        }
    } else {
        snprintf(outStr, outSize, "[idx=%d]", tokenIdx);
    }
}

// ============================================================================
// STORY LOADING FUNCTIONS
// ============================================================================

// Load all stories from a single JSON file into device memory
// Returns: number of stories loaded, or -1 on error
static int loadStoriesFromFile(const char* filePath, int* numStoriesOut) {
    printf("--- Loading stories from %s ---\n", filePath);
    
    // Open and read file
    FILE* file = fopen(filePath, "rb");
    if (!file) {
        printf("Error: Stories file not found at '%s'\n", filePath);
        return -1;
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
        return -1;
    }
    
    fread(jsonContent, 1, fileSize, file);
    jsonContent[fileSize] = '\0';
    fclose(file);
    
    // Parse JSON
    cJSON* root = cJSON_Parse(jsonContent);
    if (!root) {
        printf("Error: Failed to parse stories JSON.\n");
        free(jsonContent);
        return -1;
    }
    
    if (!cJSON_IsArray(root)) {
        printf("Error: Stories JSON is not an array.\n");
        cJSON_Delete(root);
        free(jsonContent);
        return -1;
    }
    
    int numStories = cJSON_GetArraySize(root);
    if (numStories == 0) {
        printf("Warning: Stories file is empty.\n");
        cJSON_Delete(root);
        free(jsonContent);
        *numStoriesOut = 0;
        return 0;
    }
    
    if (numStories > MAX_TRAINING_STORIES) {
        printf("Warning: File contains %d stories, truncating to %d.\n", numStories, MAX_TRAINING_STORIES);
        numStories = MAX_TRAINING_STORIES;
    }
    
    printf("Processing %d stories...\n", numStories);
    
    // Allocate host buffer for all stories' token indices and rightEndIndices
    // Format: [story0_token0, story0_token1, ..., story0_token256, story1_token0, ...]
    int* hostTokenIndices = (int*)malloc(numStories * TOKENS_PER_STORY * sizeof(int));
    int* hostRightEndIndices = (int*)malloc(numStories * sizeof(int));
    
    if (!hostTokenIndices || !hostRightEndIndices) {
        printf("Error: Failed to allocate host memory for token indices.\n");
        if (hostTokenIndices) free(hostTokenIndices);
        if (hostRightEndIndices) free(hostRightEndIndices);
        cJSON_Delete(root);
        free(jsonContent);
        return -1;
    }
    
    // Process each story
    int storiesLoaded = 0;
    for (int storyIdx = 0; storyIdx < numStories; storyIdx++) {
        cJSON* storyArr = cJSON_GetArrayItem(root, storyIdx);
        if (!storyArr || !cJSON_IsArray(storyArr)) {
            printf("Warning: Story at index %d is not an array, skipping.\n", storyIdx);
            continue;
        }
        
        int storyTokenCount = cJSON_GetArraySize(storyArr);
        if (storyTokenCount == 0) {
            printf("Warning: Story at index %d has no tokens, skipping.\n", storyIdx);
            continue;
        }
        
        // Calculate rightEndIndex:
        // It's one index before the last true token (for prediction purposes)
        // If story has < 257 tokens, rightEndIndex is (lastTrueTokenIdx - 1)
        // rightEndIndex can never exceed 255 (L-1)
        int lastTrueTokenIdx = (storyTokenCount > TOKENS_PER_STORY) 
                              ? (TOKENS_PER_STORY - 1) 
                              : (storyTokenCount - 1);
        int rightEndIndex = lastTrueTokenIdx - 1;
        if (rightEndIndex < 0) rightEndIndex = 0;  // Edge case: single-token story
        if (rightEndIndex > L - 1) rightEndIndex = L - 1;  // Cannot exceed 255
        
        hostRightEndIndices[storiesLoaded] = rightEndIndex;
        
        // Calculate base offset for this story in the token array
        int baseOffset = storiesLoaded * TOKENS_PER_STORY;
        
        // Process tokens (first L+1 = 257 tokens)
        int tokensToProcess = (storyTokenCount > TOKENS_PER_STORY) 
                             ? TOKENS_PER_STORY 
                             : storyTokenCount;
        
        bool tokenError = false;
        for (int tokenPos = 0; tokenPos < tokensToProcess && !tokenError; tokenPos++) {
            cJSON* tokenItem = cJSON_GetArrayItem(storyArr, tokenPos);
            if (!tokenItem || !cJSON_IsString(tokenItem)) {
                printf("Warning: Token at position %d in story %d is not a string.\n", tokenPos, storyIdx);
                tokenError = true;
                break;
            }
            
            const char* tokenStr = tokenItem->valuestring;
            int tokenIndex = vocabLookup(tokenStr);
            if (tokenIndex < 0) {
                printf("Warning: Token '%s' not found in vocabulary (story %d, pos %d).\n", 
                       tokenStr, storyIdx, tokenPos);
                tokenError = true;
                break;
            }
            
            hostTokenIndices[baseOffset + tokenPos] = tokenIndex;
        }
        
        if (tokenError) {
            continue;  // Skip this story
        }
        
        // Pad remaining positions with padding token (~)
        for (int tokenPos = tokensToProcess; tokenPos < TOKENS_PER_STORY; tokenPos++) {
            hostTokenIndices[baseOffset + tokenPos] = PADDING_TOKEN_INDEX;
        }
        
        storiesLoaded++;
    }
    
    printf("Successfully processed %d stories (out of %d in file).\n", storiesLoaded, numStories);
    
    // Copy to device memory
    if (storiesLoaded > 0) {
        cudaMemcpy(trainingStoryTokens_DEVICE, hostTokenIndices, 
                   storiesLoaded * TOKENS_PER_STORY * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(trainingStoryRightEndIndices_DEVICE, hostRightEndIndices,
                   storiesLoaded * sizeof(int), cudaMemcpyHostToDevice);
        printf("Copied %d stories (%d tokens) to GPU memory.\n", 
               storiesLoaded, storiesLoaded * TOKENS_PER_STORY);
    }
    
    // Cleanup token indices (no longer needed on host)
    free(hostTokenIndices);
    
    // Keep rightEndIndices on host for easy access in inner loop
    // Free any previous allocation
    if (hostRightEndIndices_storage != nullptr) {
        free(hostRightEndIndices_storage);
    }
    hostRightEndIndices_storage = hostRightEndIndices;
    hostRightEndIndices_count = storiesLoaded;
    
    cJSON_Delete(root);
    free(jsonContent);
    
    *numStoriesOut = storiesLoaded;
    return 0;
}

// ============================================================================
// SEQUENCE PROCESSING AND LOGGING
// ============================================================================

// Process a single sequence: run inference, log predictions, compute gradients
// storyIndex: index into the loaded stories (0-based)
// Returns: sequence loss (average negative log likelihood)
static float processSequence(int storyIndex) {
    // Get the rightEndIndex for this story (from host storage)
    int rightEndIndex = hostRightEndIndices_storage[storyIndex];
    int leftStartIndex = 0;
    
    // Calculate offset into trainingStoryTokens_DEVICE for this story
    // Each story has TOKENS_PER_STORY (257) tokens
    int storyOffset = storyIndex * TOKENS_PER_STORY;
    
    // Copy this story's first L tokens to seqTokenIndices_DEVICE
    // (The inference uses L=256 positions, tokens 0..255)
    // Token at index 256 is only used as the target for position 255
    // check here about 257 <--- Daniel
    cudaMemcpy(seqTokenIndices_DEVICE, 
               trainingStoryTokens_DEVICE + storyOffset, 
               L * sizeof(int), 
               cudaMemcpyDeviceToDevice);
    
    // Also copy to host seqTokenIndices for token display
    cudaMemcpy(seqTokenIndices, 
               trainingStoryTokens_DEVICE + storyOffset, 
               TOKENS_PER_STORY * sizeof(int), 
               cudaMemcpyDeviceToHost);
    
    // Run forward pass (inference)
    runInference();
    
    // Copy softmax scores back to host for logging
    // vocabScores_postSoftmax_DEVICE is [vocabSize x L] column-major
    if (hostVocabSoftmax == nullptr) {
        hostVocabSoftmax = (float*)malloc(vocabSize * L * sizeof(float));
    }
    cudaMemcpy(hostVocabSoftmax, vocabScores_postSoftmax_DEVICE, 
               vocabSize * L * sizeof(float), cudaMemcpyDeviceToHost);
    
    // ========================================================================
    // Log predictions: token[i] --> token[i+1] (probability%)
    // ========================================================================
    printf("\n--- Story %d Predictions (rightEndIndex = %d) ---\n", storyIndex, rightEndIndex);
    
    float totalLoss = 0.0f;
    int numPredictions = 0;
    
    // For each position from 0 to rightEndIndex, we predict the next token
    for (int pos = 0; pos <= rightEndIndex; pos++) {
        int currentTokenIdx = seqTokenIndices[pos];
        int nextTokenIdx = seqTokenIndices[pos + 1];  // Target token
        
        // Skip padding predictions
        if (nextTokenIdx == PADDING_TOKEN_INDEX) {
            continue;
        }
        
        // Get softmax probability for the correct next token
        // vocabScores_postSoftmax is column-major: column 'pos' starts at pos * vocabSize
        float correctProb = hostVocabSoftmax[pos * vocabSize + nextTokenIdx];
        
        // Accumulate loss: -log(probability)
        if (correctProb > 0.0f) {
            totalLoss += -logf(correctProb);
        } else {
            totalLoss += 100.0f;  // Large penalty for zero probability
        }
        numPredictions++;
        
        // Get display tokens
        char currentTokenStr[64];
        char nextTokenStr[64];
        getDisplayToken(currentTokenIdx, currentTokenStr, sizeof(currentTokenStr));
        getDisplayToken(nextTokenIdx, nextTokenStr, sizeof(nextTokenStr));
        
        // Print: current --> next (probability%)
        printf("%s --> %s (%.2f%%)\n", currentTokenStr, nextTokenStr, correctProb * 100.0f);
    }
    
    // Calculate and print average loss
    float avgLoss = (numPredictions > 0) ? (totalLoss / numPredictions) : 0.0f;
    printf("\n--- Sequence Loss: %.4f (over %d predictions) ---\n", avgLoss, numPredictions);
    
    // ========================================================================
    // Compute gradients for backpropagation
    // ========================================================================
    getGradientsForTraining(leftStartIndex, rightEndIndex);
    
    return avgLoss;
}

// ============================================================================
// MAIN TRAINING LOOP
// ============================================================================

int runTrainingLoop(void) {
    printf("\n========================================\n");
    printf("Starting Training Loop\n");
    printf("========================================\n\n");
    
    int fileIndex = 1;
    int totalStoriesProcessed = 0;
    
    // Iterate through all tokenizedStories_XXXX.json files
    while (true) {
        char filePath[512];
        getStoriesFilePath(fileIndex, filePath, sizeof(filePath));
        
        // Check if file exists
        if (!fileExists(filePath)) {
            printf("No more story files found (checked %s).\n", filePath);
            break;
        }
        
        // Load stories from this file
        int numStoriesLoaded = 0;
        int result = loadStoriesFromFile(filePath, &numStoriesLoaded);
        
        if (result != 0) {
            printf("Error loading stories from %s, skipping to next file.\n", filePath);
            fileIndex++;
            continue;
        }
        
        if (numStoriesLoaded == 0) {
            printf("No valid stories in %s, skipping to next file.\n", filePath);
            fileIndex++;
            continue;
        }
        
        totalStoriesProcessed += numStoriesLoaded;
        printf("Loaded %d stories from file %d. Total processed so far: %d\n", 
               numStoriesLoaded, fileIndex, totalStoriesProcessed);
        
        // ====================================================================
        // TRAINING LOOP FOR THIS FILE'S STORIES
        // Process stories in batches of batchSize
        // ====================================================================
        
        int numBatches = (numStoriesLoaded + batchSize - 1) / batchSize;
        printf("Processing %d batches (batch size = %d)...\n", numBatches, batchSize);
        
        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            int batchStart = batchIdx * batchSize;
            int batchEnd = batchStart + batchSize;
            if (batchEnd > numStoriesLoaded) batchEnd = numStoriesLoaded;
            int currentBatchSize = batchEnd - batchStart;
            
            printf("\n==== Batch %d/%d (stories %d to %d) ====\n", 
                   batchIdx + 1, numBatches, batchStart, batchEnd - 1);
            
            float batchTotalLoss = 0.0f;
            
            // Inner loop: process each story in the batch
            for (int storyIdxInBatch = 0; storyIdxInBatch < currentBatchSize; storyIdxInBatch++) {
                int globalStoryIdx = batchStart + storyIdxInBatch;
                
                printf("\n>>> Processing story %d (batch item %d/%d) <<<\n", 
                       globalStoryIdx, storyIdxInBatch + 1, currentBatchSize);
                
                // Process sequence: inference + gradient computation + logging
                float seqLoss = processSequence(globalStoryIdx);
                batchTotalLoss += seqLoss;
                
                // Wait for user keypress before continuing
                waitForKeypress();
            }
            
            float batchAvgLoss = batchTotalLoss / currentBatchSize;
            printf("\n==== Batch %d complete. Average batch loss: %.4f ====\n", 
                   batchIdx + 1, batchAvgLoss);
            
            // TODO: After batch, update weights using optimizer
            // apply_adeamix_optimizer(iterationCount, learningRate);
        }
        
        printf("Finished processing file %d.\n\n", fileIndex);
        
        // Move to next file
        fileIndex++;
    }
    
    // Cleanup
    if (hostVocabSoftmax != nullptr) {
        free(hostVocabSoftmax);
        hostVocabSoftmax = nullptr;
    }
    if (hostRightEndIndices_storage != nullptr) {
        free(hostRightEndIndices_storage);
        hostRightEndIndices_storage = nullptr;
        hostRightEndIndices_count = 0;
    }
    
    printf("\n========================================\n");
    printf("Training Loop Complete\n");
    printf("Total files processed: %d\n", fileIndex - 1);
    printf("Total stories processed: %d\n", totalStoriesProcessed);
    printf("========================================\n\n");
    
    return 0;
}
