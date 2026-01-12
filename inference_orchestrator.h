#pragma once

// Structure to hold token index and probability for sorting
typedef struct {
    int tokenIdx;
    float prob;
} TokenProb;

// Run the text generation loop
// Parameters:
//   storiesPath: path to the tokenized stories JSON file
//   storyIndex: which story to use (0-based)
//   contextPercent: what percentage of the story to use as context (0-100)
//   skipUserInput: if true, generate tokens without waiting for user input
//   verboseOutput: if true, print detailed generation info
// Returns: 0 on success, non-zero on failure
int runInferenceLoop(const char* storiesPath, int storyIndex, int contextPercent,
                     bool skipUserInput, bool verboseOutput);
