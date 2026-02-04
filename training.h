#pragma once

// Setup RoPE theta store for training (precomputes theta values)
void setupRoPEThetaStore(int maxL_);

// Compute gradients for backpropagation
// leftStartIndex: starting token index (inclusive) for gradient computation
// rightEndIndex: ending token index (inclusive) for gradient computation
void getGradientsForTraining(int leftStartIndex, int rightEndIndex, int L);

// Accumulate gradients from the last training step into batch accumulation buffers
// resetGradAccumulation: if true, copy gradients directly (first item in batch)
//                        if false, add gradients to existing accumulation
void accumulateGradientsFromLastTrainingStep(bool resetGradAccumulation);
