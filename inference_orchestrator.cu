/*
 * inference_orchestrator.cu
 * Contains the text generation inference loop, extracted from main.cu
 */

#include <chrono>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "network_meta.h"
#include "network_globals.h"
#include "load_model.h"
#include "inference.h"
#include "inference_orchestrator.h"

// Comparison function for qsort (descending probability)
static int compareTokenProb(const void* a, const void* b) {
    float diff = ((TokenProb*)b)->prob - ((TokenProb*)a)->prob;
    return (diff > 0) ? 1 : (diff < 0 ? -1 : 0);
}

// Forward declaration from main.cu (loadStoryContext is defined there)
extern int loadStoryContext(const char* storiesPath, int storyIndex, int percentage);

int runInferenceLoop(const char* storiesPath, int storyIndex, int contextPercent,
                     bool skipUserInput, bool verboseOutput) {
    
    // loadStoryContext returns the number of tokens loaded (up to maxL+1).
    // For free inference, L = the number of context tokens = inference positions to start with.
    // We don't need the extra target token here — only teacher-forced training does.
    int L = loadStoryContext(storiesPath, storyIndex, contextPercent);
    if (L <= 0) {
        printf("Failed to load story context.\n");
        return 1;
    }
    
    // If story filled to maxL or beyond, no room to generate
    if (L >= maxL) {
        printf("Sequence already at maxL=%d tokens (loaded %d). No generation needed.\n", maxL, L);
        return 0;
    }
    
    // Allocate host memory for vocabulary scores (allocated at maxL capacity)
    float* vocabScores_postSoftmax = (float*)malloc(vocabSize * maxL * sizeof(float));
    if (!vocabScores_postSoftmax) {
        printf("Error: Failed to allocate memory for vocabulary scores.\n");
        return 1;
    }
    
    // Generation loop
    int tokensGenerated = 0;
    int maxTokensToGenerate = maxL - L;
    
    printf("\n========================================\n");
    printf("Starting text generation (max %d new tokens)\n", maxTokensToGenerate);
    printf("Press Enter to generate next token, or 'q' + Enter to quit\n");
    printf("========================================\n\n");
    
    auto start = std::chrono::high_resolution_clock::now();

    while (L < maxL) {
        runInference(L); // LOAD STORY CONTEXT REPORTED L TOKENS, THIS MUST RUN INFERENCE ON index L-1, TO PREDICT index L
        
        // Copy vocabulary scores from device to host (only the L columns we need)
        cudaMemcpy(vocabScores_postSoftmax, vocabScores_postSoftmax_DEVICE, 
                   vocabSize * L * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Calculate offset for the prediction position (last token = L-1)
        // vocabScores_postSoftmax is column-major: [vocabSize x L]
        // For position L-1 (last token), we want column L-1
        size_t offset = (L - 1) * vocabSize;
        
        // Create array of token probabilities for sorting
        TokenProb* tokenProbs = (TokenProb*)malloc(vocabSize * sizeof(TokenProb));
        for (int i = 0; i < vocabSize; i++) {
            tokenProbs[i].tokenIdx = i;
            tokenProbs[i].prob = vocabScores_postSoftmax[offset + i];
        }
        
        // Sort by probability (descending)
        qsort(tokenProbs, vocabSize, sizeof(TokenProb), compareTokenProb);

        if (verboseOutput) {        
            // Display top-10 most probable tokens
            printf("\n--- Top 10 Most Probable Next Tokens (position %d) ---\n", L);
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
        }
        
        // Get the most probable token
        int nextTokenIdx = tokenProbs[0].tokenIdx;
        const char* nextToken = vocabGetToken(nextTokenIdx);

        if (verboseOutput) {        
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
        }
        
        free(tokenProbs);
        
        if (!skipUserInput) {        
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
        }
        
        // Append the token to the sequence at position L, then grow L
        seqTokenIndices[L] = nextTokenIdx;
        L++;
        tokensGenerated++;
        
        // Update the device memory with the new sequence (L tokens)
        cudaMemcpy(seqTokenIndices_DEVICE, seqTokenIndices, L * sizeof(int), cudaMemcpyHostToDevice);

        if (verboseOutput) {        
            printf("\n[Token added. Total context: %d tokens, generated: %d]\n", 
                   L, tokensGenerated);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Generation took %f ms\n", elapsed.count() * 1000);
    
    printf("\n========================================\n");
    printf("Generation complete. Generated %d new tokens.\n", tokensGenerated);
    printf("Final sequence length: %d tokens\n", L);
    printf("========================================\n\n");
    
    // Display final generated sequence
    printf("=== Final Generated Sequence ===\n");
    for (int i = 0; i < L; i++) {
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
