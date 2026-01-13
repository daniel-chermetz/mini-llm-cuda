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

#define TRAINING_BATCH_SIZE 16

// AdEMAMix optimizer hyperparameters
#define ADEAMIX_BETA1 0.9f        // Fast EMA decay (like Adam's beta1)
#define ADEAMIX_BETA2 0.999f      // Variance EMA decay (like Adam's beta2)
#define ADEAMIX_BETA3 0.9999f     // Slow EMA decay (AdEMAMix specific)
#define ADEAMIX_ALPHA 5.0f        // Slow EMA contribution scale
#define ADEAMIX_WEIGHT_DECAY 0.01f // Weight decay coefficient