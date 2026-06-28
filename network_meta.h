#pragma once

#define dim 768
#define dimPairs 384
#define attnHeads 6
#define headDim 128
#define ropeDenomBase 10000
#define ffnDimMultiplier 4
#define ffnDim 3072
#define transformers 16
#define maxL 1280
#define vocabSize 20160

#define TRAINING_BATCH_SIZE 8

// AdEMAMix optimizer hyperparameters
#define ADEAMIX_BETA1 0.9f        // Fast EMA decay (like Adam's beta1)
#define ADEAMIX_BETA2 0.999f      // Variance EMA decay (like Adam's beta2)
#define ADEAMIX_BETA3 0.9999f     // Slow EMA decay (AdEMAMix specific)
#define ADEAMIX_ALPHA 5.0f        // Slow EMA contribution scale
#define ADEAMIX_WEIGHT_DECAY 0.01f // Weight decay coefficient