#pragma once

#define dim 768
#define dimPairs 384
#define attnHeads 4
#define headDim 192
#define ropeDenomBase 10000
#define ffnDimMultiplier 4
#define ffnDim 3072
#define transformers 16
#define maxL 1280
#define vocabSize 21632

#define TRAINING_BATCH_SIZE 8

// Architecture configuration flags
// The preprocessor substitutes these with the literal true/false at compile time,
// so the corresponding branches are selected (and the other branch optimized out).
#define CONFIG_QK_RMS_NORM false   // Apply RMS norm to queries/keys (per head) before RoPE
#define CONFIG_QUERY_GATING false  // Sigmoid-gate the attention output using a learned query projection
#define CONFIG_PLE false // Per-layer embedding

// Number of PLE (per-layer embedding) layers.
// MUST equal the count of non -1 entries in CONFIG_PLE_post_transformer_by_tIndex below.
#define pleLayers 0

// Placement of PLE layers, indexed by transformer index. A value of -1 means no PLE
// sits at that slot. A value >= 0 is the PLE layer index (starting at 0, incrementing
// by one per enabled PLE layer). Entry [k] = p means PLE layer p runs right after the
// transformer at index k and its gated output feeds the transformer at index k-1, i.e.
// PLE p is sandwiched between transformers k-1 and k (never before the first nor after
// the last transformer).
//
// Sized transformers + 1 (not transformers): the forward pass walks transformers from
// the largest index (the deepest layer, nearest the input) downward, and for each
// transformer tIndex it reads [tIndex + 1] to see whether its input arrived through a
// PLE layer. For the deepest transformer that read lands on index `transformers`, where
// no layer can sit below it, so that final slot is always a -1 placeholder.
//
// Here PLE 0/1/2/3 sit between transformers (3,4), (7,8), (11,12) and
// (15,16) respectively (0-based). In forward order (19 down to 0), this runs
// four transformers before PLE 3, four between every PLE, and four after PLE 0.
// Kept `static const` so every translation unit gets a valid (internal-linkage) copy;
// it is only ever indexed on the host.
static const int CONFIG_PLE_post_transformer_by_tIndex[transformers + 1] = {
    -1, -1, -1, -1,       // 0..3
    -1,                   // [4]  -> PLE 0, between transformers 3 and 4
    -1, -1, -1,           // 5..7
    -1,                   // [8]  -> PLE 1, between transformers 7 and 8
    -1, -1, -1,           // 9..11
    -1,                   // [12] -> PLE 2, between transformers 11 and 12
    -1, -1, -1,           // 13..15
    -1,                   // [16] -> PLE 3, between transformers 15 and 16
    // -1, -1, -1, -1        // 17..20 ([20] is the placeholder below the deepest transformer)
};

// ============================================================================
// Newer configuration: typed compile-time constants instead of #define.
// `constexpr` values are still folded at compile time (usable as array sizes and in
// `if` conditions whose dead branch gets optimized out), but unlike macros they are
// typed, respect scope and can be checked with static_assert.
// ============================================================================

// Rotary position embeddings for the queries/keys of softmax-attention layers.
// Linear-attention layers never apply RoPE (their causal conv and decay gates carry
// positional information).
constexpr bool CONFIG_ROPE = false;

// Placement of linear-attention layers, indexed by transformer index (same indexing
// as transformerWeights / CONFIG_PLE_post_transformer_by_tIndex, i.e. the forward
// pass runs from index transformers - 1 down to 0). A value of -1 keeps standard
// softmax attention at that transformer. A value >= 0 is the linear-attention layer
// index; the indices must be exactly 0, 1, ..., linearAttnLayers - 1 (checked below).
//
// Default: linear and softmax attention alternate, linear at even indices (starting at
// index 0, the last transformer in forward order) and softmax at odd indices (ending at
// index 15, the first transformer in forward order).
constexpr int CONFIG_LINEAR_ATTN_by_tIndex[transformers] = {
     0, -1,  1, -1,       // 0..3
     2, -1,  3, -1,       // 4..7
     4, -1,  5, -1,       // 8..11
     6, -1,  7, -1        // 12..15
};

constexpr int countLinearAttnLayers() {
    int count = 0;
    for (int t = 0; t < transformers; t++) {
        if (CONFIG_LINEAR_ATTN_by_tIndex[t] != -1) {
            count++;
        }
    }
    return count;
}

// Number of linear-attention layers (derived from the placement above).
constexpr int linearAttnLayers = countLinearAttnLayers();

// Array length for per-linear-attention-layer arrays (C++ forbids zero-length arrays,
// so keep one unused slot when no linear-attention layers are configured).
constexpr int linearAttnLayerSlots = linearAttnLayers > 0 ? linearAttnLayers : 1;

constexpr bool linearAttnPlacementIsValid() {
    for (int t = 0; t < transformers; t++) {
        int linearAttnIndex = CONFIG_LINEAR_ATTN_by_tIndex[t];
        if (linearAttnIndex < -1 || linearAttnIndex >= linearAttnLayers) {
            return false;
        }
        for (int other = t + 1; other < transformers; other++) {
            if (linearAttnIndex != -1 && CONFIG_LINEAR_ATTN_by_tIndex[other] == linearAttnIndex) {
                return false;
            }
        }
    }
    return true;
}
static_assert(linearAttnPlacementIsValid(),
              "CONFIG_LINEAR_ATTN_by_tIndex must use each linear-attention index 0..linearAttnLayers-1 exactly once "
              "(and -1 for softmax-attention transformers)");

// AdEMAMix optimizer hyperparameters
#define ADEAMIX_BETA1 0.9f        // Fast EMA decay (like Adam's beta1)
#define ADEAMIX_BETA2 0.999f      // Variance EMA decay (like Adam's beta2)
#define ADEAMIX_BETA3 0.9999f     // Slow EMA decay (AdEMAMix specific)
#define ADEAMIX_ALPHA 5.0f        // Slow EMA contribution scale
#define ADEAMIX_WEIGHT_DECAY 0.0f // Weight decay coefficient
