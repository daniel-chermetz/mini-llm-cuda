#pragma once

#define dim 512
#define dimPairs 256
#define attnHeads 8 
#define headDim 64
#define ropeDenomBase 10000
#define ffnDimMultiplier 4
#define ffnDim 2048
#define transformers 4
#define L 256
#define vocabSize 10097

#define batchSize 16

// AdEMAMix optimizer hyperparameters
#define BETA1 0.9f        // Fast EMA decay (like Adam's beta1)
#define BETA2 0.999f      // Variance EMA decay (like Adam's beta2)
#define BETA3 0.9999f     // Slow EMA decay (AdEMAMix specific)
#define ALPHA 5.0f        // Slow EMA contribution scale
#define WEIGHT_DECAY 0.01f // Weight decay coefficient