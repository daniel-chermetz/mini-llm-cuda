#pragma once

#include "network_meta.h"  // For maxL

// Maximum number of stories that can be loaded at once
#define MAX_TRAINING_STORIES 11000
// Number of tokens stored per story slot (max story length)
// We load a number of tokens 1 longer than maxL so we'd have a prediction to make when standing at the maxL'th token
#define TOKENS_PER_STORY (maxL + 1)
// Padding token index (~)
#define PADDING_TOKEN_INDEX 10096

// Run the training loop
// This function iterates through all tokenizedStories_XXXX.json files,
// loads them to GPU memory, and processes in batches
// Returns: 0 on success, non-zero on failure
int runTrainingLoop(void);

