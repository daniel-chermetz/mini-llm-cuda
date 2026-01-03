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
    float* ffnPlusResidual;
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
    float* outputProjPlusResidual_postRMS2;
    float* outputProjPlusResidual;
} BackpropCalculations;

#ifdef __cplusplus
extern "C" {
#endif

extern cublasHandle_t handle;

extern int threadsPerBlock;
extern float alpha;
extern float beta;

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
extern float* vocabScores_postSoftmax_DEVICE;

/*
### TRAINING ### 
(implicitly on Device)
*/

extern float* dLoss_d_vocabScores;
extern float* dLoss_d_embedding_weights;
extern float* dLoss_d_ffn_final_postRMS_postGamma;
extern float* dLoss_d_ffn_final_RMS_gamma_weights;

extern float* ffn_final_sigma_scale_x_upGrad_byCol_RMS;
extern float* ffn_final_oneOverR_byCol_RMS;
extern float* ffn_final_oneOverColDimR3_byCol_RMS;

extern BackpropCalculations backpropCalculations[transformers];

#ifdef __cplusplus
}
#endif