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

// ============================================================================
// Linear attention (CONFIG_LINEAR_ATTN_by_tIndex)
// S_t = alpha_t * S_(t-1) + beta_t * k_t v_t^T, evaluated during training in the
// parallel ("attention matrix") form: the L x L K^T Q scores are rescaled by
// exp(sum of log alpha between key and query) * beta_key.
// Queries/keys/values come from the transformer's own projections, then pass through
// a causal depthwise conv (4 taps) + SiLU; queries/keys are then L2-normalized per head.
// Layouts: [dim, L] and [attnHeads, L, L] tensors follow the transformer conventions;
// per-token gate tensors are [L, attnHeads] column-major (index = head * L + token).
// ============================================================================

// Linear-attention weights (device)
typedef struct {
    float* alpha_weights;                      // [dim, attnHeads] x_postRMS1 -> alpha (decay) activation
    float* beta_weights;                       // [dim, attnHeads] x_postRMS1 -> beta (write gate) activation
    float* conv_key_weights;                   // [dim, 4] per channel: channel * 4 + tap (tap 3 = current token)
    float* conv_query_weights;                 // [dim, 4]
    float* conv_value_weights;                 // [dim, 4]
    float* alpha_to_gate_by_head_weights;      // [attnHeads] A_log: alpha = exp(-exp(A_log) * softplus(a + bias))
    float* alpha_to_gate_bias_by_head_weights; // [attnHeads] bias inside the softplus
} LinearAttnWeights;

// Linear-attention forward calculation buffers (implicitly on Device).
// Fields marked "transient" are only read within the same layer's forward pass, so
// every linear-attention layer shares one copy (see memory_allocation.cu).
typedef struct {
    float* keys_post_conv;                             // [dim, L] pre-SiLU
    float* keys_post_conv_post_silu;                   // [dim, L] transient
    float* queries_post_conv;                          // [dim, L] pre-SiLU
    float* queries_post_conv_post_silu;                // [dim, L] transient
    float* values_post_conv;                           // [dim, L] pre-SiLU
    float* values_post_conv_post_silu;                 // [dim, L]
    float* keys_post_l2_norm;                          // [dim, L]
    float* queries_post_l2_norm;                       // [dim, L]
    float* keys_l2_inverse_col_sums_by_col_by_head;    // [L, attnHeads]
    float* queries_l2_inverse_col_sums_by_col_by_head; // [L, attnHeads]
    float* alpha_activation;                           // [L, attnHeads]
    float* beta_activation;                            // [L, attnHeads] transient
    float* alpha_softplus;                             // [L, attnHeads]
    float* alpha_gate;                                 // [L, attnHeads]
    float* beta_gate;                                  // [L, attnHeads]
    float* alpha_prefix_sum;                           // [L, attnHeads] transient
    float* scaled_attnKtQByHead;                       // [attnHeads, L, L]
    float* attn_alpha_beta_gate_scale;                 // [attnHeads, L, L]
} LinearAttnCalculations_DEVICE;

// Linear-attention backprop buffers (implicitly gradients, on Device).
// The weight gradients accumulate across all sequences of a batch (atomicAdd / += /
// cuBLAS beta = 1); they are zeroed once at allocation and re-zeroed by the optimizer
// after every step. All other fields are transient within one layer's backward pass
// and are shared by every linear-attention layer (see memory_allocation.cu).
typedef struct {
    // weight gradients (accumulated across the batch)
    float* alpha_weights;                      // [dim, attnHeads]
    float* beta_weights;                       // [dim, attnHeads]
    float* conv_key_weights;                   // [dim, 4]
    float* conv_query_weights;                 // [dim, 4]
    float* conv_value_weights;                 // [dim, 4]
    float* alpha_to_gate_by_head_weights;      // [attnHeads]
    float* alpha_to_gate_bias_by_head_weights; // [attnHeads]
    // transient activation gradients
    float* scaled_attnKtQByHead;                            // [attnHeads, L, L]
    float* colGradSums;                                     // [L, attnHeads]
    float* rowGradSums;                                     // [L, attnHeads]
    float* alpha_gate;                                      // [L, attnHeads]
    float* beta_gate;                                       // [L, attnHeads]
    float* alpha_activation;                                // [L, attnHeads]
    float* beta_activation;                                 // [L, attnHeads]
    float* keys_features_times_grad_sums_by_col_by_head;    // [L, attnHeads]
    float* queries_features_times_grad_sums_by_col_by_head; // [L, attnHeads]
    float* keys_post_l2_norm;                               // [dim, L]
    float* queries_post_l2_norm;                            // [dim, L]
    float* keys_post_conv_post_silu;                        // [dim, L]
    float* queries_post_conv_post_silu;                     // [dim, L]
    float* values_post_conv_post_silu;                      // [dim, L]
    float* keys_post_conv_pre_silu;                         // [dim, L]
    float* queries_post_conv_pre_silu;                      // [dim, L]
    float* values_post_conv_pre_silu;                       // [dim, L]
} LinearAttnBackpropCalculations;

// Linear-attention optimizer state (EMA / variance per weight; no separate gradient
// accumulation buffers since the backprop weight gradients already accumulate)
typedef struct {
    float* alpha_weights;                      // [dim, attnHeads]
    float* beta_weights;                       // [dim, attnHeads]
    float* conv_key_weights;                   // [dim, 4]
    float* conv_query_weights;                 // [dim, 4]
    float* conv_value_weights;                 // [dim, 4]
    float* alpha_to_gate_by_head_weights;      // [attnHeads]
    float* alpha_to_gate_bias_by_head_weights; // [attnHeads]
} LinearAttnOptimizerState;

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

// Linear-attention weights and forward calculations (indexed by linear-attention layer index)
extern LinearAttnWeights linearAttnWeights_DEVICE[linearAttnLayerSlots];
extern LinearAttnCalculations_DEVICE linearAttnCalculations_DEVICE[linearAttnLayerSlots];

// "Current layer" copies used as shorthands by linear_attention.cu,
// training_linear_attention.cu, inference.cu, training.cu and optimizer.cu
extern TransformerWeights tWeights;
extern TransformerCalculations_DEVICE tCalcs;
extern LinearAttnWeights laWeights;
extern LinearAttnCalculations_DEVICE laCalcs;

// Linear-attention forward/backward entry points (linear_attention.cu, training_linear_attention.cu)
void runLinearAttnPreparationInference(int tIndex, int linearAttnIndex, int L);
void runLinearAttnGateScalingInference(int tIndex, int linearAttnIndex, int L);
void run_linear_attn_gate_unwinding_for_training(int tIndex, int linearAttnIndex, int L);
void run_linear_attn_V_K_Q_unwinding_for_training(int tIndex, int linearAttnIndex, int L);

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

// Linear-attention backprop calculations (indexed by linear-attention layer index)
extern LinearAttnBackpropCalculations linearAttnBackpropCalculations[linearAttnLayerSlots];

// "Current layer" backprop copies (see tCalcs / laCalcs above)
extern BackpropCalculations tBackpropCalculations;
extern LinearAttnBackpropCalculations laBackpropCalculations;

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

// Linear-attention optimizer state (indexed by linear-attention layer index) and the
// "current layer" copies used by optimizer.cu
extern LinearAttnOptimizerState laFastEMA_DEVICE[linearAttnLayerSlots];
extern LinearAttnOptimizerState laSlowEMA_DEVICE[linearAttnLayerSlots];
extern LinearAttnOptimizerState laVariance_DEVICE[linearAttnLayerSlots];
extern LinearAttnOptimizerState laFastEMA;
extern LinearAttnOptimizerState laSlowEMA;
extern LinearAttnOptimizerState laVariance;

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
