/*
nvcc main.cu ./cJSON/cJSON.c inference.cu load_model.cu network_globals.cu \
     -o inference \
     -lcublas \
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

void allocateMemory(bool allocateTraining) {
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

    size_t seqTokenIndices_size = (L + 1) * sizeof(int); 
    seqTokenIndices = (int*)malloc(seqTokenIndices_size);
    for (int i = 0; i < (L + 1); i++) {
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

    cudaMalloc((void**)&ffn_sumByCol_RMS_DEVICE, dim * L * sizeof(float));
    cudaMalloc((void**)&ffn_postRMS_pre_gamma_DEVICE, dim * L * sizeof(float));
    cudaMalloc((void**)&ffn_postRMS_gamma_scaled_DEVICE, dim * L * sizeof(float));    

    cudaMalloc((void**)&vocabScores_DEVICE, vocabSize * L * sizeof(float));
    cudaMalloc((void**)&vocabScores_maxByCol_softmax_DEVICE, L * sizeof(float));
    cudaMalloc((void**)&vocabScores_sumByCol_softmax_DEVICE, L * sizeof(float));
    cudaMalloc((void**)&vocabScores_postSoftmax_DEVICE, vocabSize * L * sizeof(float));

    if (allocateTraining) {
        cudaMalloc((void**)&dLoss_d_vocabScores, vocabSize * L * sizeof(float));
        cudaMalloc((void**)&dLoss_d_embedding_weights, dim * vocabSize * sizeof(float));
        cudaMalloc((void**)&dLoss_d_ffn_final_postRMS_postGamma, dim * L * sizeof(float));
        cudaMalloc((void**)&dLoss_d_ffn_final_RMS_gamma_weights, dim * L * sizeof(float));        

        cudaMalloc((void**)&ffn_final_sigma_scale_x_upGrad_byCol_RMS, L * sizeof(float));
        cudaMalloc((void**)&ffn_final_oneOverR_byCol_RMS, L * sizeof(float));
        cudaMalloc((void**)&ffn_final_oneOverColDimR3_byCol_RMS, L * sizeof(float));

        // implicitly these are gradients (without specifying that in the variable name)
        for (int transformerIndex = 0; transformerIndex < transformers; transformerIndex++) {
            cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_final_plus_residual, dim * L * sizeof(float));

            cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_left_weights, dim * ffnDim * sizeof(float));
            cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_postHadamard, ffnDim * L * sizeof(float));
            
            cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_1_postSilu, ffnDim * L * sizeof(float));
            cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_1_preSilu, ffnDim * L * sizeof(float));
            cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_1_weights, ffnDim * dim * sizeof(float));

            cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_2, ffnDim * L * sizeof(float));
            cudaMalloc((void**)&backpropCalculations[transformerIndex].ffn_right_2_weights, ffnDim * dim * sizeof(float));

            cudaMalloc((void**)&backpropCalculations[transformerIndex].outputProjPlusResidual_postRMS2, dim * L * sizeof(float));
        }
    } 
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
    
    const char* modelNameFromArg = (argc > 1) ? argv[1] : "model";
    //char* modelName = "model_10_lr_5e6";
    //char* modelName = "model_11_lr_1e5";
    //char* modelName = "model_12_lr_4e6";
    char* modelName = "model_13_lr_3e6";
    //char* modelName = "model_14_lr_3e6";
    //char* modelName = "model_15_lr_3e6";
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
    int contextPercent = (argc > 3) ? atoi(argv[3]) : 100;
    
    const char* storiesPath = "./tokenizedStories/tokenizedStories_0001.json";
    int rightSeqEndIndex = loadStoryContext(storiesPath, storyIndex, contextPercent);
    if (rightSeqEndIndex < 0) {
        printf("Failed to load story context.\n");
        return 1;
    }
    
    // Check if sequence is already full (L tokens)
    if (rightSeqEndIndex >= L) {
        printf("Sequence already contains L=%d tokens. No generation needed.\n", L+1);
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
    int maxTokensToGenerate = (L + 1) - (rightSeqEndIndex + 1);  // How many tokens until we hit L total
    bool skipUserInputBeforeGenerating = true;
    bool verboseGenerationOutput = false;
    
    printf("\n========================================\n");
    printf("Starting text generation (max %d new tokens)\n", maxTokensToGenerate);
    printf("Press Enter to generate next token, or 'q' + Enter to quit\n");
    printf("========================================\n\n");
    
    auto start = std::chrono::high_resolution_clock::now();

    while (rightSeqEndIndex <= L - 1) {
        // Run inference
        // printf("rightSeqEndIndex: %d\n", rightSeqEndIndex);
        // printf("seqTokenIndices_DEVICE\n");
        // printIntMatrixToDebugMain(seqTokenIndices_DEVICE, L, L, 30);

        runInference();
        
        // Copy vocabulary scores from device to host
        cudaMemcpy(vocabScores_postSoftmax, vocabScores_postSoftmax_DEVICE, 
                   vocabSize * L * sizeof(float), cudaMemcpyDeviceToHost);
        //printf("vocabScores_postSoftmax_DEVICE");
        //printFloatMatrixToDebugMain(vocabScores_postSoftmax_DEVICE, dim * vocabSize, 3, 10);
        
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

        if (verboseGenerationOutput) {        
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
        }
        
        // Get the most probable token
        int nextTokenIdx = tokenProbs[0].tokenIdx;
        const char* nextToken = vocabGetToken(nextTokenIdx);

        if (verboseGenerationOutput) {        
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
        
        if (!skipUserInputBeforeGenerating) {        
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
        
        // Append the token to the sequence
        rightSeqEndIndex++;
        seqTokenIndices[rightSeqEndIndex] = nextTokenIdx;
        tokensGenerated++;
        
        // Update the device memory with the new sequence
        cudaMemcpy(seqTokenIndices_DEVICE, seqTokenIndices, L * sizeof(int), cudaMemcpyHostToDevice);

        if (verboseGenerationOutput) {        
            printf("\n[Token added. Total context: %d tokens, generated: %d]\n", 
                   rightSeqEndIndex + 1, tokensGenerated);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Generation took %f ms", elapsed.count() * 1000);
    
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