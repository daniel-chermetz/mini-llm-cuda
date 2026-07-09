#pragma once

/*
 * training_orchestrator_instruct_helper.h
 * Helper module for building "instruct" token sequences that are prepended
 * to each story before it is submitted to training.
 *
 * The instruct sequence encodes the story's structural "elements" (category ->
 * value) as a flat token sequence of the form:
 *
 *   [ [ <category tokens> > <value tokens> ] + [ <category tokens> > <value tokens> ] ] \n \n
 *
 * Categories are emitted using their tokenized name (index 2 of
 * element_keys_tokenized_dual.json) and sorted by their priority (index 0),
 * lower priority values coming first.
 */

#include "cJSON/cJSON.h"

// Initialize the instruct category metadata by loading
// ./instruct/element_keys_tokenized_dual.json.
// Returns 0 on success, non-zero on failure.
int instructInit(void);

// Free instruct category metadata allocated by instructInit().
void instructCleanup(void);

// Load the instruct JSON file corresponding to a story file index (1-based).
// Example: fileIndex 3 ->
//   ./instruct/shortenedRoundRobinElementsFlattenedLower/victorian_stories_0003.json
// Returns a cJSON array on success (caller must free via cJSON_Delete), or
// NULL on failure (missing file / parse error).
cJSON* instructLoadFile(int fileIndex);

// Build the flat instruct token-index sequence for a single story.
//   instructRoot     : the cJSON array returned by instructLoadFile()
//   storyIndexInFile : index of the story within the instruct file
//   outTokens        : output buffer that receives vocabulary token indices
//   maxTokens        : capacity of outTokens
// Returns the number of tokens written (>= 0), or -1 on error.
int instructBuildTokens(cJSON* instructRoot, int storyIndexInFile, int* outTokens, int maxTokens);
