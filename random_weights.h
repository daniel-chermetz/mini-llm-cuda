#pragma once

// Initialize all model weights with random values in range [-range, +range]
// (RMS/PLE gammas to 1.0, linear-attention weights per initializeLinearAttnWeights)
void initializeRandomWeights(float range);

// Initialize only the linear-attention weights: alpha/beta projections in [-range, +range],
// conv taps in [-0.5, +0.5], Mamba-2 style decay gate (A_log, softplus bias)
void initializeLinearAttnWeights(float range, unsigned long seed);
