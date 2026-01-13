#pragma once

// Precompute beta power values for bias correction: (1 - beta^iteration)
// Populates indices 1 to numIterationsToCalc in each store
__global__ void preCalcPowBeta(float* beta1_pow_store, float* beta2_pow_store, float* beta3_pow_store, int numIterationsToCalc, float BETA1_, float BETA2_, float BETA3_);

// Apply AdEMAMix optimizer to all model weights
// iterationNum: current iteration (starts at 1)
// learningRate: learning rate for this step
void apply_adeamix_optimizer(int iterationNum, float learningRate);
