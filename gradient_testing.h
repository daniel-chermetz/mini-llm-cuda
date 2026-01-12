#pragma once

// Run gradient verification tests
// This function loads a story, runs inference, and prepares data for gradient comparison
// Returns: 0 on success, non-zero on failure
int runGradientTests(void);
