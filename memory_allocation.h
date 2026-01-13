#pragma once

// Allocate all GPU and CPU memory for inference and optionally training
// allocateTraining: if true, also allocate memory for gradients, backprop, and optimizer state
void allocateMemory(bool allocateTraining);
