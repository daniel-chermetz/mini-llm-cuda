/*
 * training_orchestrator.cu
 * Module for orchestrating the training loop over tokenized stories
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

#include "cJSON/cJSON.h"
#include "network_meta.h"
#include "network_globals.h"
#include "load_model.h"
#include "inference.h"
#include "training.h"
#include "training_orchestrator.h"
#include "optimizer.h"
#include "save_model.h"

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Base path for tokenized stories
#define TOKENIZED_STORIES_PATH "./tokenizedStories"

// Printing frequency: print every N batches (set to 1 to print all)
#define PRINT_EVERY_N_BATCHES 20

// Save frequency: save model every N optimizer iterations (100k = 100000)
#define SAVE_EVERY_N_ITERATIONS 20000

// Model save path template (iteration number will be inserted)
#define MODEL_SAVE_PATH "./model/victorian_weights_iter_%d.bin"

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

// Wait for user to press Enter
static void waitForKeypress() {
    printf("\nPress Enter to continue...\n");
    // Clear any pending input
    int c;
    while ((c = getchar()) != '\n' && c != EOF) {}
    getchar();  // Wait for Enter
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
    // Format: [story0_token0, story0_token1, ..., story0_tokenN, story1_token0, ...]
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
        
        // Calculate rightEndIndex: the last position from which a prediction can be made.
        // We load up to TOKENS_PER_STORY (maxL+1) tokens. The token at rightEndIndex+1 is
        // the target for the prediction at rightEndIndex.
        // Example: 30-token story → load 30, rightEndIndex = 28, L = 29
        // Example: 1000-token story (maxL=256) → load 257, rightEndIndex = 255, L = 256
        int tokensToLoad = (storyTokenCount > TOKENS_PER_STORY) 
                          ? TOKENS_PER_STORY 
                          : storyTokenCount;
        int rightEndIndex = tokensToLoad - 2;
        if (rightEndIndex < 0) {
            printf("Warning: Story at index %d has fewer than 2 tokens, skipping.\n", storyIdx);
            continue;  // Need at least 2 tokens for one prediction
        }
        
        hostRightEndIndices[storiesLoaded] = rightEndIndex;
        
        // Calculate base offset for this story in the token array
        int baseOffset = storiesLoaded * TOKENS_PER_STORY;
        
        // Process tokens (up to tokensToLoad tokens)
        int tokensToProcess = tokensToLoad;
        
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
        printf("Copied %d stories to GPU memory (slot size = %d tokens).\n", 
               storiesLoaded, TOKENS_PER_STORY);
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

// Return type for processSequence: total loss and prediction count
typedef struct {
    float totalLoss;
    int numPredictions;
} SequenceLossResult;

// Process a single sequence: run inference, compute loss, compute gradients
// storyIndex: index into the loaded stories (0-based)
// shouldPrint: if true, print token-by-token predictions
// Returns: total loss and number of predictions for this sequence
static SequenceLossResult processSequence(int storyIndex, bool shouldPrint) {
    // Get the rightEndIndex for this story (from host storage)
    int rightEndIndex = hostRightEndIndices_storage[storyIndex];
    int L = rightEndIndex + 1;  // Number of positions for inference (0..rightEndIndex)
    int leftStartIndex = 0;
    
    // Calculate offset into trainingStoryTokens_DEVICE for this story
    // Each story slot has TOKENS_PER_STORY (maxL+1) entries
    int storyOffset = storyIndex * TOKENS_PER_STORY;
    
    // Copy this story's L+1 tokens to seqTokenIndices_DEVICE
    // Inference processes L positions (0..L-1 = 0..rightEndIndex)
    // Token at index L (= rightEndIndex+1) is the target for position rightEndIndex
    cudaMemcpy(seqTokenIndices_DEVICE, 
               trainingStoryTokens_DEVICE + storyOffset, 
               (L + 1) * sizeof(int), 
               cudaMemcpyDeviceToDevice);
    
    // Run forward pass (inference) over L positions
    runInference(L);
    
    float totalLoss = 0.0f;
    int numPredictions = 0;
    
    // Compute loss only when printing (avoids costly device-to-host copy every iteration)
    if (shouldPrint) {
        // Copy token indices to host (needed for target lookup)
        cudaMemcpy(seqTokenIndices, 
                   trainingStoryTokens_DEVICE + storyOffset, 
                   (L + 1) * sizeof(int), 
                   cudaMemcpyDeviceToHost);
        
        // Allocate softmax buffer if needed (allocated at maxL capacity)
        if (hostVocabSoftmax == nullptr) {
            hostVocabSoftmax = (float*)malloc(vocabSize * maxL * sizeof(float));
        }
        cudaMemcpy(hostVocabSoftmax, vocabScores_postSoftmax_DEVICE, 
                   vocabSize * L * sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("\n--- Story %d Predictions (rightEndIndex = %d) ---\n", storyIndex, rightEndIndex);
        
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
            if (correctProb < 0.0001f) {
                correctProb = 0.0001f;
            }
            // Accumulate loss: -log(probability)
            totalLoss += -logf(correctProb);
            numPredictions++;
            
            char currentTokenStr[64];
            char nextTokenStr[64];
            getDisplayToken(currentTokenIdx, currentTokenStr, sizeof(currentTokenStr));
            getDisplayToken(nextTokenIdx, nextTokenStr, sizeof(nextTokenStr));
            printf("%s --> %s (%.2f%%)\n", currentTokenStr, nextTokenStr, correctProb * 100.0f);
        }
        
        float avgLoss = (numPredictions > 0) ? (totalLoss / numPredictions) : 0.0f;
        printf("\n--- Sequence Loss: %.4f (over %d predictions) ---\n", avgLoss, numPredictions);
    }
    
    // ========================================================================
    // Compute gradients for backpropagation
    // ========================================================================
    getGradientsForTraining(leftStartIndex, rightEndIndex, L);
    
    SequenceLossResult result;
    result.totalLoss = totalLoss;
    result.numPredictions = numPredictions;
    return result;
}

// ============================================================================
// MAIN TRAINING LOOP
// ============================================================================

// Learning rate warmup configuration
#define LR_START 9e-6f        // Starting learning rate
#define LR_END 1e-4f          // Final learning rate after warmup
#define LR_WARMUP_STEPS 10000 // Number of iterations for warmup

// Calculate learning rate with linear warmup
static float getLearningRate(int iteration) {
    if (iteration >= LR_WARMUP_STEPS) {
        return LR_END;
    }
    // Linear interpolation from LR_START to LR_END
    float progress = (float)iteration / (float)LR_WARMUP_STEPS;
    return LR_START + progress * (LR_END - LR_START);
}

int runTrainingLoop(void) {
    printf("\n========================================\n");
    printf("Starting Training Loop\n");
    printf("========================================\n\n");
    
    int fileIndex = 1;
    int totalStoriesProcessed = 0;
    int globalIterationCount = 0;  // Track total optimizer steps
    
    // Global cumulative loss tracking
    double globalCumulativeLoss = 0.0;
    long long globalCumulativePredictions = 0;
    
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
        
        // Rest period after loading new JSON to let GPU cool down
        printf("Resting for 20 seconds...\n");
        sleep(20);
        
        // ====================================================================
        // TRAINING LOOP FOR THIS FILE'S STORIES
        // Process stories in batches of batchSize
        // ====================================================================
        
        int numBatches = (numStoriesLoaded + TRAINING_BATCH_SIZE - 1) / TRAINING_BATCH_SIZE;
        printf("Processing %d batches (batch size = %d)...\n", numBatches, TRAINING_BATCH_SIZE);
        
        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            int batchStart = batchIdx * TRAINING_BATCH_SIZE;
            int batchEnd = batchStart + TRAINING_BATCH_SIZE;
            if (batchEnd > numStoriesLoaded) batchEnd = numStoriesLoaded;
            int currentBatchSize = batchEnd - batchStart;
            
            // Determine if we should print this batch (every N batches)
            bool shouldPrintThisBatch = (globalIterationCount % PRINT_EVERY_N_BATCHES == 0);
            
            if (shouldPrintThisBatch) {
                printf("\n==== Batch %d/%d (stories %d to %d) [iter %d] ====\n", 
                       batchIdx + 1, numBatches, batchStart, batchEnd - 1, globalIterationCount + 1);
            }
            
            double batchTotalLoss = 0.0;
            int batchTotalPredictions = 0;
            
            // Inner loop: process each story in the batch
            for (int storyIdxInBatch = 0; storyIdxInBatch < currentBatchSize; storyIdxInBatch++) {
                int globalStoryIdx = batchStart + storyIdxInBatch;
                
                // Process sequence: inference + gradient computation
                // Only print if this is a print batch
                SequenceLossResult seqResult = processSequence(globalStoryIdx, shouldPrintThisBatch);
                batchTotalLoss += seqResult.totalLoss;
                batchTotalPredictions += seqResult.numPredictions;
                
                // Accumulate gradients (reset on first item in batch)
                bool isFirstInBatch = (storyIdxInBatch == 0);
                accumulateGradientsFromLastTrainingStep(isFirstInBatch);
                
                // Wait for user keypress before continuing (commented out for fast training)
                // waitForKeypress();
            }
            
            // Update global cumulative loss (sampled: only from printed batches)
            if (shouldPrintThisBatch) {
                globalCumulativeLoss += batchTotalLoss;
                globalCumulativePredictions += batchTotalPredictions;
            }
            
            // Apply optimizer after batch gradient accumulation
            globalIterationCount++;
            float currentLR = getLearningRate(globalIterationCount);
            apply_adeamix_optimizer(globalIterationCount, currentLR);
            
            if (shouldPrintThisBatch) {
                float batchAvgLoss = (batchTotalPredictions > 0) ? (float)(batchTotalLoss / batchTotalPredictions) : 0.0f;
                float globalAvgLoss = (globalCumulativePredictions > 0) ? (float)(globalCumulativeLoss / globalCumulativePredictions) : 0.0f;
                printf("\n==== Batch %d complete | Batch loss: %.4f (%d preds) | Global loss: %.4f (%lld preds) | Iter %d (lr=%.6f) ====\n", 
                       batchIdx + 1, batchAvgLoss, batchTotalPredictions, globalAvgLoss, globalCumulativePredictions, globalIterationCount, currentLR);
            }
            
            // Save model weights every SAVE_EVERY_N_ITERATIONS
            if (globalIterationCount > 0 && globalIterationCount % SAVE_EVERY_N_ITERATIONS == 0) {
                char saveFilePath[256];
                snprintf(saveFilePath, sizeof(saveFilePath), MODEL_SAVE_PATH, globalIterationCount);
                saveModelWeights(saveFilePath, globalIterationCount);
            }
            /*if (globalIterationCount > 0 && globalIterationCount % 50 == 0) {
                // Rest period after every 50 batches to let GPU cool down
                printf("Resting for 13 seconds...\n");
                sleep(13);
            }*/
        }
        
        printf("Finished processing file %d.\n\n", fileIndex);
        
        // Move to next file
        fileIndex++;
    }
    
    // Final save if training produced work and last iteration wasn't already a save point
    if (globalIterationCount > 0 && globalIterationCount % SAVE_EVERY_N_ITERATIONS != 0) {
        char saveFilePath[256];
        snprintf(saveFilePath, sizeof(saveFilePath), MODEL_SAVE_PATH, globalIterationCount);
        printf("Saving final model weights at iteration %d...\n", globalIterationCount);
        saveModelWeights(saveFilePath, globalIterationCount);
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
