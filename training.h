#pragma once

// Setup RoPE theta store for training (precomputes theta values)
void setupRoPEThetaStore(void);

// Compute gradients for backpropagation
// leftStartIndex: starting token index (inclusive) for gradient computation
// rightEndIndex: ending token index (inclusive) for gradient computation
void getGradientsForTraining(int leftStartIndex, int rightEndIndex);
