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
    // CONFIG_QK_RMS_NORM
    float* query_RMS_weights; 
    float* key_RMS_weights; 
    // CONFIG_QUERY_GATING
    float* gated_query_weights; 
} TransformerWeights;

typedef struct {
    float* x_sumByCol_RMS1;
    float* x_postRMS1_pre_gamma;
    float* x_postRMS1_post_gamma;
    float* queries;
    float* keys;
    float* values;
    // CONFIG_QK_RMS_NORM
    float* queries_RMS_sumByColByHead;
    float* keys_RMS_sumByColByHead;
    float* queries_post_RMS_pre_gamma;
    float* queries_postRMS_post_gamma;
    float* keys_post_RMS_pre_gamma;
    float* keys_postRMS_post_gamma;
    float* queriesPostRoPE;
    float* keysPostRoPE;
    float* attnKtQByHead;
    float* attnKtQByHeadScaledMasked;
    float* attnByHead_maxByCol_softmax;
    float* attnByHead_sumByCol_softmax;
    float* attnByHead_expfCache_softmax;
    float* attnByHead_postSoftmax;
    float* valueScaledSoftmaxAttn;
    // CONFIG_QUERY_GATING
    float* gatedQueries;
    float* gatedQueriesPostSigmoid;
    float* gatedValueScaledSoftmaxAttn;
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
    // CONFIG_QUERY_GATING
    float* gatedValueScaledSoftmaxAttn;
    float* gatedQueriesPostSigmoid;
    float* gatedQueriesPreSigmoid;
    float* gated_query_weights;
    float* attnByHead_postSoftmax;
    float* attnSoftmaxGradSumByCol;
    float* values;
    float* attnKtQByHead;
    float* keysPostRoPE;
    float* keysPreRoPE;    
    float* queriesPostRoPE;
    float* queriesPreRoPE;
    // CONFIG_QK_RMS_NORM
    float* queriesPreRoPE_preRMS;
    float* keysPreRoPE_preRMS;
    float* query_gamma_weights;
    float* key_gamma_weights;
    float* rms_queries_sigma_scale_x_upGrad_byCol_RMS_byHead;
    float* rms_queries_oneOverR_byCol_RMS_byHead;
    float* rms_queries_oneOverHeadDimR3_byCol_RMS_byHead;
    float* rms_keys_sigma_scale_x_upGrad_byCol_RMS_byHead;
    float* rms_keys_oneOverR_byCol_RMS_byHead;
    float* rms_keys_oneOverHeadDimR3_byCol_RMS_byHead;
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
    // CONFIG_QK_RMS_NORM
    float* query_gamma_weights;
    float* key_gamma_weights;
    // CONFIG_QUERY_GATING
    float* gated_query_weights;
} OptimizerTransformerState;

// ============================================================================
// CONFIG_PLE (Per-Layer Embedding) structures
// A PLE layer sits between two transformers. It looks up its own embedding table
// for the sequence tokens (x_ple), gates the combined (token embedding + PLE
// embedding) signal, and adds it onto the upstream transformer's output before
// feeding the downstream transformer.
// ============================================================================

// PLE weights (host mirror + device)
typedef struct {
    float* ple_weights;       // [dim, dim]     projection producing the pre-sigmoid gate
    float* gamma_weights;     // [dim]          per-feature gate scale
    float* embedding_weights; // [dim, vocabSize] PLE-specific token embedding table
} PLEWeights;

// PLE forward calculation buffers (implicitly on Device)
typedef struct {
    float* x_ple;                            // [dim, maxL] PLE embeddings for the sequence
    float* ple_gate_pre_sigmoid_pre_gamma;   // [dim, maxL]
    float* ple_gate_post_sigmoid_pre_gamma;  // [dim, maxL]
    float* ple_gate_post_sigmoid_post_gamma; // [dim, maxL]
    float* sum_ffnPlusResidual_x_ple_gated;  // [dim, maxL] output fed to next transformer
} PLECalculations_DEVICE;

// PLE backprop calculation buffers (implicitly gradients, on Device)
typedef struct {
    float* grad_x_ple;                            // [dim, maxL]
    float* grad_ple_gate_post_sigmoid_post_gamma; // [dim, maxL]
    float* grad_ple_gate_post_sigmoid_pre_gamma;  // [dim, maxL]
    float* grad_ple_gate_pre_sigmoid_pre_gamma;   // [dim, maxL]
    float* gamma_weights;      // [dim]            gradient
    float* ple_weights;        // [dim, dim]       gradient
    float* embedding_weights;  // [dim, vocabSize] gradient (accumulated across the batch via atomicAdd)
} PLEBackpropCalculations;

// PLE optimizer state (gradient accumulation, EMA, variance)
typedef struct {
    float* ple_weights;       // [dim, dim]
    float* gamma_weights;     // [dim]
    float* embedding_weights; // [dim, vocabSize]
} PLEOptimizerState;

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

// CONFIG_PLE weights and forward calculations
extern PLEWeights pleWeights[pleLayers];
extern PLEWeights pleWeights_DEVICE[pleLayers];
extern PLECalculations_DEVICE pleCalculations_DEVICE[pleLayers];

// PLE forward/backward entry points. Kept here because inference.cu and
// training.cu already include this shared declaration header.
int runPLEInference(int pleIndex, int downstreamTransformerIndex, int L);
void getGradientsForPLELayerTraining(int pleIndex, int downstreamTransformerIndex,
                                     int leftStartIndex, int rightEndIndex, int L);

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

// CONFIG_PLE backprop calculations
extern PLEBackpropCalculations ple_backpropCalculations[pleLayers];

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

// CONFIG_PLE optimizer state (gradient accumulation, EMA, variance)
extern PLEOptimizerState pleGradientAccumulation[pleLayers];
extern PLEOptimizerState pleFastEMA[pleLayers];
extern PLEOptimizerState pleSlowEMA[pleLayers];
extern PLEOptimizerState pleVariance[pleLayers];

// CONFIG_PLE: unique token indices touched across a whole batch (host + device).
// Uploaded once per batch (after all sequences run) before the optimizer step so the
// PLE embedding tables are only updated for the tokens that actually appeared.
extern int* unique_seqTokenIndices_batch;
extern int* unique_seqTokenIndices_batch_DEVICE;

// Beta power stores for bias correction (precomputed 1 - beta^iteration)
extern float* beta1_pow_store;
extern float* beta2_pow_store;
extern float* beta3_pow_store;

/*
### TRAINING DATA STORAGE ###
(on Device - for batch training)
*/

// Storage for training stories: [MAX_TRAINING_STORIES x (maxL + 1)] tokens
// Each story has up to (maxL + 1) tokens (unused slots padded with ~)
extern int* trainingStoryTokens_DEVICE;

// Right end index for each story: last position from which a prediction can be made
// For a story with N tokens (capped at maxL+1): rightEndIndex = min(N, maxL+1) - 2
extern int* trainingStoryRightEndIndices_DEVICE;

#ifdef __cplusplus
}
#endif
