#pragma once

// Maximum number of stories that can be loaded at once
#define MAX_TRAINING_STORIES 11000
// Number of tokens per story (L+1 = 257)
#define TOKENS_PER_STORY (L + 1)
// Padding token index (~)
#define PADDING_TOKEN_INDEX 10096

// Run the training loop
// This function iterates through all tokenizedStories_XXXX.json files,
// loads them to GPU memory, and processes in batches
// Returns: 0 on success, non-zero on failure
int runTrainingLoop(void);

