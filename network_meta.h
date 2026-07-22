#pragma once

#define dim 864
#define dimPairs 432
#define attnHeads 4
#define headDim 216
#define ropeDenomBase 10000
#define ffnDimMultiplier 4
#define ffnDim 3072
#define transformers 20
#define maxL 1280
#define vocabSize 21632

#define TRAINING_BATCH_SIZE 5

// Architecture configuration flags
// The preprocessor substitutes these with the literal true/false at compile time,
// so the corresponding branches are selected (and the other branch optimized out).
#define CONFIG_QK_RMS_NORM false   // Apply RMS norm to queries/keys (per head) before RoPE
#define CONFIG_QUERY_GATING false  // Sigmoid-gate the attention output using a learned query projection
#define CONFIG_PLE false // Per-layer embedding

// Number of PLE (per-layer embedding) layers.
// MUST equal the count of non -1 entries in CONFIG_PLE_post_transformer_by_tIndex below.
#define pleLayers 4

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
     0,                   // [4]  -> PLE 0, between transformers 3 and 4
    -1, -1, -1,           // 5..7
     1,                   // [8]  -> PLE 1, between transformers 7 and 8
    -1, -1, -1,           // 9..11
     2,                   // [12] -> PLE 2, between transformers 11 and 12
    -1, -1, -1,           // 13..15
     3,                   // [16] -> PLE 3, between transformers 15 and 16
    -1, -1, -1, -1        // 17..20 ([20] is the placeholder below the deepest transformer)
};

// AdEMAMix optimizer hyperparameters
#define ADEAMIX_BETA1 0.9f        // Fast EMA decay (like Adam's beta1)
#define ADEAMIX_BETA2 0.999f      // Variance EMA decay (like Adam's beta2)
#define ADEAMIX_BETA3 0.9999f     // Slow EMA decay (AdEMAMix specific)
#define ADEAMIX_ALPHA 5.0f        // Slow EMA contribution scale
#define ADEAMIX_WEIGHT_DECAY 0.01f // Weight decay coefficient
