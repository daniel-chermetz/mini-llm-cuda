#include "network_globals.h"

cublasHandle_t handle = nullptr;

int threadsPerBlock = 256;
float alpha = 1.0f;
float beta = 0.0f;
float beta_one = 1.0f;

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

float* ffn_sumByCol_RMS_DEVICE = nullptr;
float* ffn_postRMS_pre_gamma_DEVICE = nullptr;
float* ffn_postRMS_post_gamma_DEVICE = nullptr;
float* ffn_postRMS_gamma_scaled_DEVICE = nullptr;

float* vocabScores_DEVICE = nullptr;
float* vocabScores_maxByCol_softmax_DEVICE = nullptr;
float* vocabScores_sumByCol_softmax_DEVICE = nullptr;
float* vocabScores_postSoftmax_DEVICE = nullptr;

/*
### TRAINING ### 
(implicitly on Device)
*/

float* ropeThetaStore_DEVICE = nullptr;

float* dLoss_d_vocabScores = nullptr;
float* dLoss_d_embedding_weights = nullptr;
float* dLoss_d_ffn_final_postRMS_postGamma = nullptr;
float* dLoss_d_ffn_final_RMS_gamma_weights = nullptr;

float* ffn_final_sigma_scale_x_upGrad_byCol_RMS = nullptr;
float* ffn_final_oneOverR_byCol_RMS = nullptr;
float* ffn_final_oneOverColDimR3_byCol_RMS = nullptr;

BackpropCalculations backpropCalculations[transformers];

float* x_DEVICE_grad = nullptr;
