#pragma once

#include <cublas_v2.h>

#include "network_meta.h"

typedef struct { 
    float* rms1_weights; 
    float* query_weights; 
    float* key_weights; 
    float* value_weights; 
    float* output_proj_weights; 
    float* rms2_weights; 
    float* ffn_left_weights; 
    float* ffn_right_1_weights; 
    float* ffn_right_2_weights; 
} TransformerWeights;

typedef struct {
    float* x_sumByCol_RMS1;
    float* x_postRMS1_pre_gamma;
    float* x_postRMS1_post_gamma;
    float* queries;
    float* keys;
    float* values;
    float* queriesPostRoPE;
    float* keysPostRoPE;
    float* attnKtQByHead;
    float* attnKtQByHeadScaledMasked;
    float* attnByHead_maxByCol_softmax;
    float* attnByHead_sumByCol_softmax;
    float* attnByHead_expfCache_softmax;
    float* attnByHead_postSoftmax;
    float* valueScaledSoftmaxAttn;
    float* outputProj;
    float* outputProjPlusResidual;
    float* outputProjPlusResidual_sumByCol_RMS2;
    float* outputProjPlusResidual_postRMS2_pre_gamma;
    float* outputProjPlusResidual_postRMS2_post_gamma;
    float* ffn_right_1_preSilu;
    float* ffn_right_1_postSilu;
    float* ffn_right_2;
    float* ffn_right_postHadamard;
    float* ffn_final;
    float* ffnPlusResidual; // should be called ffn_final_plus_residual
} TransformerCalculations_DEVICE;

// training (implicitly on Device)
// these are gradients (without specifying that in the variable name)
typedef struct {
    float* ffn_final_plus_residual;
    float* ffn_left_weights;
    float* ffn_right_postHadamard;    
    float* ffn_right_1_postSilu;
    float* ffn_right_1_preSilu;
    float* ffn_right_1_weights;    
    float* ffn_right_2;
    float* ffn_right_2_weights;
    float* outputProjPlusResidual_postRMS2_post_gamma;
    float* outputProjPlusResidual_postRMS2_pre_gamma;
    float* rms2_gamma_weights;
    float* rms2_sigma_scale_x_upGrad_byCol_RMS;
    float* rms2_oneOverR_byCol_RMS;
    float* rms2_oneOverColDimR3_byCol_RMS;
    float* outputProjPlusResidual;
    float* output_proj_weights;
    float* valueScaledSoftmaxAttn;
    float* attnByHead_postSoftmax;
    float* attnSoftmaxGradSumByCol;
    float* values;
    float* attnKtQByHead;
    float* keysPostRoPE;
    float* keysPreRoPE;    
    float* queriesPostRoPE;
    float* queriesPreRoPE;
    float* value_weights;
    float* key_weights;
    float* query_weights;
    float* rms1_gamma_weights;
    float* rms1_sigma_scale_x_upGrad_byCol_RMS;
    float* rms1_oneOverR_byCol_RMS;
    float* rms1_oneOverColDimR3_byCol_RMS;
    float* x_postRMS1_post_gamma; 
} BackpropCalculations;

// Optimizer state for transformer layers (gradient accumulation, EMA, variance)
typedef struct {
    float* ffn_left_weights;
    float* ffn_right_1_weights;
    float* ffn_right_2_weights;
    float* rms2_gamma_weights;
    float* output_proj_weights;
    float* value_weights;
    float* query_weights;
    float* key_weights;
    float* rms1_gamma_weights;
} OptimizerTransformerState;

#ifdef __cplusplus
extern "C" {
#endif

extern cublasHandle_t handle;

extern int threadsPerBlock;
extern float alpha;
extern float beta;
extern float beta_one;

extern int* seqTokenIndices;
extern int* seqTokenIndices_DEVICE;

extern float* embedding_weights;
extern float* embedding_weights_DEVICE;

extern float* final_rms_weights;
extern float* final_rms_weights_DEVICE;

extern TransformerWeights transformerWeights[transformers];
extern TransformerWeights transformerWeights_DEVICE[transformers];

extern float* preComputedRopeTheta;
extern float* preComputedRopeTheta_DEVICE;

extern float* x_DEVICE;

extern TransformerCalculations_DEVICE transformerCalculations_DEVICE[transformers];

extern float* ffn_sumByCol_RMS_DEVICE;
extern float* ffn_postRMS_pre_gamma_DEVICE;
extern float* ffn_postRMS_post_gamma_DEVICE;

extern float* vocabScores_DEVICE;
extern float* vocabScores_maxByCol_softmax_DEVICE;
extern float* vocabScores_sumByCol_softmax_DEVICE;
extern float* vocabScores_expfCache_softmax_DEVICE;
extern float* vocabScores_postSoftmax_DEVICE;

/*
### TRAINING ### 
(implicitly on Device)
*/

extern float* ropeThetaStore_DEVICE;

extern float* dLoss_d_vocabScores;
extern float* dLoss_d_embedding_weights;
extern float* dLoss_d_ffn_final_postRMS_postGamma;
extern float* dLoss_d_ffn_final_RMS_gamma_weights;

extern float* ffn_final_sigma_scale_x_upGrad_byCol_RMS;
extern float* ffn_final_oneOverR_byCol_RMS;
extern float* ffn_final_oneOverColDimR3_byCol_RMS;

extern float* x_DEVICE_grad;

extern BackpropCalculations backpropCalculations[transformers];

/*
### OPTIMIZER STATE ###
(implicitly on Device)
*/

// Gradient accumulation
extern float* gradientAccumulation_embedding_weights;
extern float* gradientAccumulation_final_RMS_gamma_weights;
extern OptimizerTransformerState gradientAccumulation[transformers];

// Fast EMA (first moment)
extern float* fastEMA_embedding_weights;
extern float* fastEMA_final_RMS_gamma_weights;
extern OptimizerTransformerState fastEMA[transformers];

// Slow EMA
extern float* slowEMA_embedding_weights;
extern float* slowEMA_final_RMS_gamma_weights;
extern OptimizerTransformerState slowEMA[transformers];

// Variance (second moment)
extern float* variance_embedding_weights;
extern float* variance_final_RMS_gamma_weights;
extern OptimizerTransformerState variance[transformers];

// Beta power stores for bias correction (precomputed 1 - beta^iteration)
extern float* beta1_pow_store;
extern float* beta2_pow_store;
extern float* beta3_pow_store;

/*
### TRAINING DATA STORAGE ###
(on Device - for batch training)
*/

// Storage for training stories: [MAX_TRAINING_STORIES x (L+1)] tokens
// Each story has 257 tokens (padded with ~ if shorter)
extern int* trainingStoryTokens_DEVICE;

// Right end index for each story (0 to L-1, one before last true token)
extern int* trainingStoryRightEndIndices_DEVICE;

#ifdef __cplusplus
}
#endif