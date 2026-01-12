/*
 * gradient_testing.cu
 * Module for testing gradient computation accuracy against PyTorch reference gradients
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "cJSON.h"
#include "network_meta.h"
#include "network_globals.h"
#include "load_model.h"
#include "inference.h"
#include "training.h"
#include "gradient_testing.h"

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Number of top predictions to show for each token position
#define TOP_N_PREDICTIONS 3

// Number of token positions to show predictions for (from start of context)
#define NUM_POSITIONS_TO_LOG 20

// Output directory for gradient JSON files
#define GRADIENT_OUTPUT_DIR "./cuda_gradients"

// ============================================================================

// Forward declaration from main.cu
extern int loadStoryContext(const char* storiesPath, int storyIndex, int percentage);

// Structure for sorting token probabilities
typedef struct {
    int tokenIdx;
    float prob;
} TokenProbGT;

// Comparison function for qsort (descending probability)
static int compareTokenProbGT(const void* a, const void* b) {
    float diff = ((TokenProbGT*)b)->prob - ((TokenProbGT*)a)->prob;
    return (diff > 0) ? 1 : (diff < 0 ? -1 : 0);
}

// Helper function to get a display-safe token string
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
// GRADIENT SAVING FUNCTIONS
// ============================================================================

// Save a float array from device memory to a JSON file
// Each number is written on its own line for better text editor performance
static int saveGradientToJSON(float* devicePtr, size_t numElements, const char* filename) {
    // Allocate host memory
    float* hostData = (float*)malloc(numElements * sizeof(float));
    if (!hostData) {
        printf("Error: Failed to allocate host memory for %s\n", filename);
        return 0;
    }
    
    // Copy from device
    cudaError_t err = cudaMemcpy(hostData, devicePtr, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error: cudaMemcpy failed for %s: %s\n", filename, cudaGetErrorString(err));
        free(hostData);
        return 0;
    }
    
    // Build full path
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/%s", GRADIENT_OUTPUT_DIR, filename);
    
    // Open file for writing
    FILE* file = fopen(filepath, "w");
    if (!file) {
        printf("Error: Failed to open %s for writing\n", filepath);
        free(hostData);
        return 0;
    }
    
    // Write JSON array with each number on its own line
    fprintf(file, "[\n");
    for (size_t i = 0; i < numElements; i++) {
        if (i < numElements - 1) {
            fprintf(file, "  %.9g,\n", hostData[i]);
        } else {
            fprintf(file, "  %.9g\n", hostData[i]);
        }
    }
    fprintf(file, "]\n");
    
    fclose(file);
    free(hostData);
    
    printf("  Saved %s (%zu elements)\n", filename, numElements);
    return 1;
}

// Save all gradients from backpropCalculations for a specific transformer index
static int saveTransformerGradients(int tIndex) {
    char filename[256];
    int success = 1;
    
    printf("Saving gradients for transformer %d...\n", tIndex);
    
    // ffn_final_plus_residual: [dim, L]
    snprintf(filename, sizeof(filename), "t%d_ffn_final_plus_residual_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].ffn_final_plus_residual, dim * L, filename);
    
    // ffn_left_weights: [dim, ffnDim]
    snprintf(filename, sizeof(filename), "t%d_ffn_left_weights_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].ffn_left_weights, dim * ffnDim, filename);
    
    // ffn_right_postHadamard: [ffnDim, L]
    snprintf(filename, sizeof(filename), "t%d_ffn_right_postHadamard_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].ffn_right_postHadamard, ffnDim * L, filename);
    
    // ffn_right_1_postSilu: [ffnDim, L]
    snprintf(filename, sizeof(filename), "t%d_ffn_right_1_postSilu_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].ffn_right_1_postSilu, ffnDim * L, filename);
    
    // ffn_right_1_preSilu: [ffnDim, L]
    snprintf(filename, sizeof(filename), "t%d_ffn_right_1_preSilu_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].ffn_right_1_preSilu, ffnDim * L, filename);
    
    // ffn_right_1_weights: [ffnDim, dim]
    snprintf(filename, sizeof(filename), "t%d_ffn_right_1_weights_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].ffn_right_1_weights, ffnDim * dim, filename);
    
    // ffn_right_2: [ffnDim, L]
    snprintf(filename, sizeof(filename), "t%d_ffn_right_2_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].ffn_right_2, ffnDim * L, filename);
    
    // ffn_right_2_weights: [ffnDim, dim]
    snprintf(filename, sizeof(filename), "t%d_ffn_right_2_weights_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].ffn_right_2_weights, ffnDim * dim, filename);
    
    // outputProjPlusResidual_postRMS2_post_gamma: [dim, L]
    snprintf(filename, sizeof(filename), "t%d_outputProjPlusResidual_postRMS2_post_gamma_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].outputProjPlusResidual_postRMS2_post_gamma, dim * L, filename);
    
    // outputProjPlusResidual_postRMS2_pre_gamma: [dim, L] - Note: this field exists in struct but check if allocated
    // Looking at main.cu, this is NOT allocated, so skip it
    
    // rms2_gamma_weights: [dim]
    snprintf(filename, sizeof(filename), "t%d_rms2_gamma_weights_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].rms2_gamma_weights, dim, filename);
    
    // rms2_sigma_scale_x_upGrad_byCol_RMS: [L]
    snprintf(filename, sizeof(filename), "t%d_rms2_sigma_scale_x_upGrad_byCol_RMS.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].rms2_sigma_scale_x_upGrad_byCol_RMS, L, filename);
    
    // rms2_oneOverR_byCol_RMS: [L]
    snprintf(filename, sizeof(filename), "t%d_rms2_oneOverR_byCol_RMS.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].rms2_oneOverR_byCol_RMS, L, filename);
    
    // rms2_oneOverColDimR3_byCol_RMS: [L]
    snprintf(filename, sizeof(filename), "t%d_rms2_oneOverColDimR3_byCol_RMS.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].rms2_oneOverColDimR3_byCol_RMS, L, filename);
    
    // outputProjPlusResidual: [dim, L]
    snprintf(filename, sizeof(filename), "t%d_outputProjPlusResidual_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].outputProjPlusResidual, dim * L, filename);
    
    // valueScaledSoftmaxAttn: [dim, L]
    snprintf(filename, sizeof(filename), "t%d_valueScaledSoftmaxAttn_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].valueScaledSoftmaxAttn, dim * L, filename);
    
    // output_proj_weights: [dim, dim]
    snprintf(filename, sizeof(filename), "t%d_output_proj_weights_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].output_proj_weights, dim * dim, filename);
    
    // values: [dim, L]
    snprintf(filename, sizeof(filename), "t%d_values_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].values, dim * L, filename);
    
    // attnByHead_postSoftmax: [attnHeads, L, L]
    snprintf(filename, sizeof(filename), "t%d_attnByHead_postSoftmax_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].attnByHead_postSoftmax, attnHeads * L * L, filename);
    
    // attnSoftmaxGradSumByCol: [attnHeads, L]
    snprintf(filename, sizeof(filename), "t%d_attnSoftmaxGradSumByCol.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].attnSoftmaxGradSumByCol, attnHeads * L, filename);
    
    // attnKtQByHead: [attnHeads, L, L]
    snprintf(filename, sizeof(filename), "t%d_attnKtQByHead_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].attnKtQByHead, attnHeads * L * L, filename);
    
    // keysPostRoPE: [dim, L]
    snprintf(filename, sizeof(filename), "t%d_keysPostRoPE_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].keysPostRoPE, dim * L, filename);
    
    // keysPreRoPE: [dim, L]
    snprintf(filename, sizeof(filename), "t%d_keysPreRoPE_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].keysPreRoPE, dim * L, filename);
    
    // queriesPostRoPE: [dim, L]
    snprintf(filename, sizeof(filename), "t%d_queriesPostRoPE_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].queriesPostRoPE, dim * L, filename);
    
    // queriesPreRoPE: [dim, L]
    snprintf(filename, sizeof(filename), "t%d_queriesPreRoPE_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].queriesPreRoPE, dim * L, filename);
    
    // value_weights: [dim, dim]
    snprintf(filename, sizeof(filename), "t%d_value_weights_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].value_weights, dim * dim, filename);
    
    // query_weights: [dim, dim]
    snprintf(filename, sizeof(filename), "t%d_query_weights_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].query_weights, dim * dim, filename);
    
    // key_weights: [dim, dim]
    snprintf(filename, sizeof(filename), "t%d_key_weights_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].key_weights, dim * dim, filename);
    
    // rms1_gamma_weights: [dim]
    snprintf(filename, sizeof(filename), "t%d_rms1_gamma_weights_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].rms1_gamma_weights, dim, filename);
    
    // rms1_sigma_scale_x_upGrad_byCol_RMS: [L]
    snprintf(filename, sizeof(filename), "t%d_rms1_sigma_scale_x_upGrad_byCol_RMS.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].rms1_sigma_scale_x_upGrad_byCol_RMS, L, filename);
    
    // rms1_oneOverR_byCol_RMS: [L]
    snprintf(filename, sizeof(filename), "t%d_rms1_oneOverR_byCol_RMS.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].rms1_oneOverR_byCol_RMS, L, filename);
    
    // rms1_oneOverColDimR3_byCol_RMS: [L]
    snprintf(filename, sizeof(filename), "t%d_rms1_oneOverColDimR3_byCol_RMS.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].rms1_oneOverColDimR3_byCol_RMS, L, filename);
    
    // x_postRMS1_post_gamma: [dim, L]
    snprintf(filename, sizeof(filename), "t%d_x_postRMS1_post_gamma_grad.json", tIndex);
    success &= saveGradientToJSON(backpropCalculations[tIndex].x_postRMS1_post_gamma, dim * L, filename);
    
    return success;
}

// Save all global gradients (not per-transformer)
static int saveGlobalGradients(void) {
    int success = 1;
    
    printf("Saving global gradients...\n");
    
    // dLoss_d_vocabScores: [vocabSize, L]
    success &= saveGradientToJSON(dLoss_d_vocabScores, vocabSize * L, "dLoss_d_vocabScores.json");
    
    // dLoss_d_embedding_weights: [dim, vocabSize]
    success &= saveGradientToJSON(dLoss_d_embedding_weights, dim * vocabSize, "dLoss_d_embedding_weights.json");
    
    // dLoss_d_ffn_final_postRMS_postGamma: [dim, L]
    success &= saveGradientToJSON(dLoss_d_ffn_final_postRMS_postGamma, dim * L, "dLoss_d_ffn_final_postRMS_postGamma.json");
    
    // dLoss_d_ffn_final_RMS_gamma_weights: [dim]
    success &= saveGradientToJSON(dLoss_d_ffn_final_RMS_gamma_weights, dim, "dLoss_d_ffn_final_RMS_gamma_weights.json");
    
    return success;
}

int runGradientTests(void) {
    printf("\n");
    printf("============================================================\n");
    printf("          GRADIENT TESTING MODULE\n");
    printf("============================================================\n\n");
    
    // Configuration for gradient testing
    const char* storiesPath = "./tokenizedStories/tokenizedStories_0001.json";
    int storyIndex = 1;
    int contextPercent = 100;
    
    printf("Loading story context...\n");
    printf("  Stories path: %s\n", storiesPath);
    printf("  Story index: %d\n", storyIndex);
    printf("  Context percent: %d%%\n", contextPercent);
    printf("\n");
    
    // Step 1: Load story context
    int rightSeqEndIndex = loadStoryContext(storiesPath, storyIndex, contextPercent);
    if (rightSeqEndIndex < 0) {
        printf("Error: Failed to load story context.\n");
        return 1;
    }
    
    printf("Story context loaded successfully.\n");
    printf("  rightSeqEndIndex: %d\n", rightSeqEndIndex);
    printf("\n");
    
    // Step 2: Run inference
    printf("Running inference...\n");
    runInference();
    printf("Inference completed.\n\n");
    
    // Step 3: Copy vocabulary scores from device to host
    printf("Copying vocabulary scores from device to host...\n");
    
    // Allocate host memory for vocabulary scores
    float* vocabScores_postSoftmax = (float*)malloc(vocabSize * L * sizeof(float));
    if (!vocabScores_postSoftmax) {
        printf("Error: Failed to allocate memory for vocabulary scores.\n");
        return 1;
    }
    
    // Copy from device
    cudaError_t err = cudaMemcpy(vocabScores_postSoftmax, vocabScores_postSoftmax_DEVICE, 
                                  vocabSize * L * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error: cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(vocabScores_postSoftmax);
        return 1;
    }
    
    printf("Vocabulary scores copied successfully.\n");
    printf("  vocabScores_postSoftmax shape: [%d x %d] (vocabSize x L)\n", vocabSize, L);
    printf("\n");
    
    // ========================================================================
    // DETAILED PREDICTIONS FOR EACH TOKEN POSITION
    // ========================================================================
    
    int numPositionsToLog = NUM_POSITIONS_TO_LOG;
    if (numPositionsToLog > rightSeqEndIndex) {
        numPositionsToLog = rightSeqEndIndex;
    }
    
    printf("============================================================\n");
    printf("  TOP-%d PREDICTIONS FOR FIRST %d TOKEN POSITIONS\n", TOP_N_PREDICTIONS, numPositionsToLog);
    printf("============================================================\n\n");
    
    // Allocate array for sorting
    TokenProbGT* tokenProbs = (TokenProbGT*)malloc(vocabSize * sizeof(TokenProbGT));
    if (!tokenProbs) {
        printf("Error: Failed to allocate memory for token sorting.\n");
        free(vocabScores_postSoftmax);
        return 1;
    }
    
    for (int pos = 0; pos < numPositionsToLog; pos++) {
        // Get the actual token at this position
        int actualTokenIdx = seqTokenIndices[pos];
        char actualTokenStr[64];
        getDisplayToken(actualTokenIdx, actualTokenStr, sizeof(actualTokenStr));
        
        // Get the next token (ground truth for prediction at this position)
        int nextTokenIdx = (pos + 1 <= rightSeqEndIndex) ? seqTokenIndices[pos + 1] : -1;
        char nextTokenStr[64];
        if (nextTokenIdx >= 0) {
            getDisplayToken(nextTokenIdx, nextTokenStr, sizeof(nextTokenStr));
        } else {
            snprintf(nextTokenStr, sizeof(nextTokenStr), "(end)");
        }
        
        printf("Position %3d: token='%s' (idx=%d)\n", pos, actualTokenStr, actualTokenIdx);
        printf("  Ground truth next token: '%s' (idx=%d)\n", nextTokenStr, nextTokenIdx);
        printf("  Top-%d predictions:\n", TOP_N_PREDICTIONS);
        
        // Fill the sorting array for this position
        size_t colOffset = pos * vocabSize;
        for (int i = 0; i < vocabSize; i++) {
            tokenProbs[i].tokenIdx = i;
            tokenProbs[i].prob = vocabScores_postSoftmax[colOffset + i];
        }
        
        // Sort by probability (descending)
        qsort(tokenProbs, vocabSize, sizeof(TokenProbGT), compareTokenProbGT);
        
        // Print top-N predictions
        for (int rank = 0; rank < TOP_N_PREDICTIONS && rank < vocabSize; rank++) {
            char predTokenStr[64];
            getDisplayToken(tokenProbs[rank].tokenIdx, predTokenStr, sizeof(predTokenStr));
            
            // Check if this prediction matches the ground truth
            const char* matchIndicator = "";
            if (tokenProbs[rank].tokenIdx == nextTokenIdx) {
                matchIndicator = " <-- MATCH";
            }
            
            printf("    %d. '%s' (idx=%5d) prob=%.4f (%.2f%%)%s\n",
                   rank + 1,
                   predTokenStr,
                   tokenProbs[rank].tokenIdx,
                   tokenProbs[rank].prob,
                   tokenProbs[rank].prob * 100.0f,
                   matchIndicator);
        }
        
        // Also show where the ground truth ranks if not in top-N
        if (nextTokenIdx >= 0) {
            bool foundInTopN = false;
            for (int rank = 0; rank < TOP_N_PREDICTIONS && rank < vocabSize; rank++) {
                if (tokenProbs[rank].tokenIdx == nextTokenIdx) {
                    foundInTopN = true;
                    break;
                }
            }
            if (!foundInTopN) {
                // Find the rank of the ground truth token
                for (int rank = TOP_N_PREDICTIONS; rank < vocabSize; rank++) {
                    if (tokenProbs[rank].tokenIdx == nextTokenIdx) {
                        printf("    ... ground truth '%s' ranked #%d (prob=%.4f, %.2f%%)\n",
                               nextTokenStr, rank + 1, tokenProbs[rank].prob, tokenProbs[rank].prob * 100.0f);
                        break;
                    }
                }
            }
        }
        printf("\n");
    }
    
    free(tokenProbs);
    
    printf("============================================================\n");
    printf("  END OF POSITION-BY-POSITION PREDICTIONS\n");
    printf("============================================================\n\n");
    
    // Clean up vocab scores (no longer needed)
    free(vocabScores_postSoftmax);
    
    // ========================================================================
    // STEP 4: RUN BACKPROPAGATION
    // ========================================================================
    
    printf("============================================================\n");
    printf("          RUNNING BACKPROPAGATION\n");
    printf("============================================================\n\n");
    
    int leftStartIndex = 0;
    int rightEndIndex = 100;  // Testing gradient masking (less than full context length)
    
    printf("Computing gradients with:\n");
    printf("  leftStartIndex: %d\n", leftStartIndex);
    printf("  rightEndIndex: %d\n", rightEndIndex);
    printf("\n");
    
    getGradientsForTraining(leftStartIndex, rightEndIndex);
    
    // Sync to ensure all GPU operations complete
    cudaDeviceSynchronize();
    
    printf("Backpropagation completed.\n\n");
    
    // ========================================================================
    // STEP 5: SAVE GRADIENTS TO JSON FILES
    // ========================================================================
    
    printf("============================================================\n");
    printf("          SAVING GRADIENTS TO JSON FILES\n");
    printf("============================================================\n\n");
    
    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(GRADIENT_OUTPUT_DIR, &st) == -1) {
        printf("Creating directory: %s\n", GRADIENT_OUTPUT_DIR);
        mkdir(GRADIENT_OUTPUT_DIR, 0755);
    }
    
    // Save global gradients
    int saveSuccess = saveGlobalGradients();
    
    // Save gradients for each transformer
    for (int tIndex = 0; tIndex < transformers; tIndex++) {
        saveSuccess &= saveTransformerGradients(tIndex);
    }
    
    printf("\n");
    if (saveSuccess) {
        printf("All gradients saved successfully to %s\n", GRADIENT_OUTPUT_DIR);
    } else {
        printf("Warning: Some gradients failed to save.\n");
    }
    printf("\n");
    
    printf("============================================================\n");
    printf("          GRADIENT TESTING COMPLETE\n");
    printf("============================================================\n");
    printf("\n");
    printf("Gradient files saved to: %s\n", GRADIENT_OUTPUT_DIR);
    printf("Next steps:\n");
    printf("  - Compare CUDA gradients with PyTorch reference gradients\n");
    printf("  - Check ./pytorch_verification/gradients/ for reference files\n");
    printf("\n");
    
    return 0;
}
