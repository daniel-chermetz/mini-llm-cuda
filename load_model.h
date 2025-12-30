#pragma once

int loadVocab(const char* vocabPath);
char* vocabGetToken(int index);
int vocabLookup(const char* token);
int loadModel(const char* modelName);

