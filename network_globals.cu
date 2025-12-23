#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "network_globals.h"

cublasHandle_t handle = nullptr;

int threadsPerBlock = 256;
float alpha = 1.0f;
float beta = 0.0f;

int* seqTokenIndices = nullptr;
int* seqTokenIndices_DEVICE = nullptr;

float* embedding_weights = nullptr;
float* embedding_weights_DEVICE = nullptr;

float* final_rms_weights = nullptr;
float* final_rms_weights_DEVICE = nullptr;

TransformerWeights transformerWeights[transformers];
TransformerWeights transformerWeights_DEVICE[transformers];

float* preComputedRopeTheta = nullptr;
float* preComputedRopeTheta_DEVICE = nullptr;

float* x_DEVICE = nullptr;

TransformerCalculations_DEVICE transformerCalculations_DEVICE[transformers];

float* vocabScores_DEVICE = nullptr;
float* vocabScores_maxByCol_softmax_DEVICE = nullptr;
float* vocabScores_sumByCol_softmax_DEVICE = nullptr;
float* vocabScores_postSoftmax_DEVICE = nullptr;