/*
 * training_orchestrator_instruct_helper.cu
 * Builds "instruct" token sequences that are prepended to stories before training.
 * See training_orchestrator_instruct_helper.h for the high-level format.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "cJSON/cJSON.h"
#include "load_model.h"   // vocabLookup
#include "training_orchestrator_instruct_helper.h"

// ============================================================================
// CONFIGURATION
// ============================================================================

// Path to the tokenized category-key metadata file.
#define INSTRUCT_KEYS_PATH "./instruct/element_keys_tokenized_dual.json"

// Base directory holding the per-file instruct data. The file name matches the
// corresponding story file name (victorian_stories_%04d.json).
#define INSTRUCT_BASE_PATH "./instruct/shortenedRoundRobinElementsFlattenedLower"

// Special structural tokens used to frame the instruct sequence.
#define INSTRUCT_TOK_OPEN     "["
#define INSTRUCT_TOK_CLOSE    "]"
#define INSTRUCT_TOK_ARROW    ">"
#define INSTRUCT_TOK_PLUS     "+"
#define INSTRUCT_TOK_NEWLINE  "\n"

// ============================================================================
// CATEGORY METADATA
// ============================================================================

// Metadata for a single category (key of the "elements" object).
typedef struct {
    char*  name;       // category name (e.g. "Protagonist")
    int    priority;   // ordering priority (lower comes first)
    char** tokens;     // tokenized category name (index 2 of the metadata array)
    int    numTokens;  // number of tokens in the category name
} InstructCategoryMeta;

static InstructCategoryMeta* g_categories = nullptr;
static int g_numCategories = 0;

// Read an entire file into a newly allocated null-terminated buffer.
// Returns the buffer (caller frees) or NULL on failure.
static char* readWholeFile(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        return nullptr;
    }
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (fileSize < 0) {
        fclose(file);
        return nullptr;
    }
    char* buffer = (char*)malloc((size_t)fileSize + 1);
    if (!buffer) {
        fclose(file);
        return nullptr;
    }
    size_t readCount = fread(buffer, 1, (size_t)fileSize, file);
    buffer[readCount] = '\0';
    fclose(file);
    return buffer;
}

int instructInit(void) {
    if (g_categories != nullptr) {
        return 0;  // already initialized
    }

    char* jsonContent = readWholeFile(INSTRUCT_KEYS_PATH);
    if (!jsonContent) {
        printf("Error: Failed to open instruct keys file '%s'.\n", INSTRUCT_KEYS_PATH);
        return -1;
    }

    cJSON* root = cJSON_Parse(jsonContent);
    free(jsonContent);
    if (!root || !cJSON_IsObject(root)) {
        printf("Error: Failed to parse instruct keys JSON as object.\n");
        if (root) cJSON_Delete(root);
        return -1;
    }

    int numCategories = cJSON_GetArraySize(root);
    if (numCategories <= 0) {
        printf("Error: Instruct keys file contains no categories.\n");
        cJSON_Delete(root);
        return -1;
    }

    InstructCategoryMeta* categories =
        (InstructCategoryMeta*)calloc((size_t)numCategories, sizeof(InstructCategoryMeta));
    if (!categories) {
        printf("Error: Failed to allocate instruct category metadata.\n");
        cJSON_Delete(root);
        return -1;
    }

    int stored = 0;
    for (cJSON* entry = root->child; entry != nullptr; entry = entry->next) {
        if (!entry->string || !cJSON_IsArray(entry)) {
            continue;
        }

        cJSON* priorityItem = cJSON_GetArrayItem(entry, 0);
        cJSON* tokensItem    = cJSON_GetArrayItem(entry, 2);
        if (!priorityItem || !cJSON_IsNumber(priorityItem) ||
            !tokensItem   || !cJSON_IsArray(tokensItem)) {
            printf("Warning: Category '%s' has unexpected structure, skipping.\n", entry->string);
            continue;
        }

        int numTokens = cJSON_GetArraySize(tokensItem);
        char** tokens = nullptr;
        if (numTokens > 0) {
            tokens = (char**)calloc((size_t)numTokens, sizeof(char*));
            if (!tokens) {
                printf("Error: Failed to allocate category token array.\n");
                continue;
            }
            for (int t = 0; t < numTokens; t++) {
                cJSON* tokItem = cJSON_GetArrayItem(tokensItem, t);
                const char* tokStr = (tokItem && cJSON_IsString(tokItem)) ? tokItem->valuestring : "";
                tokens[t] = strdup(tokStr);
            }
        }

        categories[stored].name      = strdup(entry->string);
        categories[stored].priority  = priorityItem->valueint;
        categories[stored].tokens    = tokens;
        categories[stored].numTokens = numTokens;
        stored++;
    }

    cJSON_Delete(root);

    g_categories = categories;
    g_numCategories = stored;
    printf("Instruct: loaded %d category definitions from '%s'.\n", stored, INSTRUCT_KEYS_PATH);
    return 0;
}

void instructCleanup(void) {
    if (!g_categories) {
        return;
    }
    for (int i = 0; i < g_numCategories; i++) {
        if (g_categories[i].name) free(g_categories[i].name);
        if (g_categories[i].tokens) {
            for (int t = 0; t < g_categories[i].numTokens; t++) {
                if (g_categories[i].tokens[t]) free(g_categories[i].tokens[t]);
            }
            free(g_categories[i].tokens);
        }
    }
    free(g_categories);
    g_categories = nullptr;
    g_numCategories = 0;
}

// Find category metadata by name. Returns NULL if not present.
static const InstructCategoryMeta* findCategory(const char* name) {
    for (int i = 0; i < g_numCategories; i++) {
        if (strcmp(g_categories[i].name, name) == 0) {
            return &g_categories[i];
        }
    }
    return nullptr;
}

// ============================================================================
// INSTRUCT FILE LOADING
// ============================================================================

cJSON* instructLoadFile(int fileIndex) {
    char filePath[512];
    snprintf(filePath, sizeof(filePath),
             "%s/victorian_stories_%04d.json", INSTRUCT_BASE_PATH, fileIndex);

    char* jsonContent = readWholeFile(filePath);
    if (!jsonContent) {
        printf("Warning: Instruct file not found at '%s'.\n", filePath);
        return nullptr;
    }

    cJSON* root = cJSON_Parse(jsonContent);
    free(jsonContent);
    if (!root) {
        printf("Warning: Failed to parse instruct JSON '%s'.\n", filePath);
        return nullptr;
    }
    if (!cJSON_IsArray(root)) {
        printf("Warning: Instruct JSON '%s' is not an array.\n", filePath);
        cJSON_Delete(root);
        return nullptr;
    }
    return root;
}

// ============================================================================
// INSTRUCT TOKEN CONSTRUCTION
// ============================================================================

// A single element to emit, resolved to its metadata and value tokens.
typedef struct {
    int                         priority;
    int                         origOrder;  // preserves input order for stable sort
    const InstructCategoryMeta* meta;
    cJSON*                      valueArray; // array of value token strings
} ElementEntry;

// Sort by ascending priority; ties broken by original order (stable).
static int compareElementEntries(const void* a, const void* b) {
    const ElementEntry* ea = (const ElementEntry*)a;
    const ElementEntry* eb = (const ElementEntry*)b;
    if (ea->priority != eb->priority) {
        return (ea->priority < eb->priority) ? -1 : 1;
    }
    return (ea->origOrder < eb->origOrder) ? -1 : (ea->origOrder > eb->origOrder ? 1 : 0);
}

// Append a single token string (looked up in the vocabulary) to the output buffer.
// Missing tokens are skipped with a warning. Returns 1 if there is still room,
// 0 if the output buffer is full.
static int emitToken(const char* tok, int* outTokens, int* count, int maxTokens) {
    if (*count >= maxTokens) {
        return 0;
    }
    int idx = vocabLookup(tok);
    if (idx < 0) {
        printf("Warning: instruct token '%s' not found in vocabulary, skipping.\n", tok);
        return 1;
    }
    outTokens[(*count)++] = idx;
    return (*count < maxTokens) ? 1 : 0;
}

int instructBuildTokens(cJSON* instructRoot, int storyIndexInFile, int* outTokens, int maxTokens) {
    if (!instructRoot || !outTokens || maxTokens <= 0) {
        return -1;
    }
    if (!g_categories) {
        printf("Error: instructInit() must be called before instructBuildTokens().\n");
        return -1;
    }

    cJSON* storyObj = cJSON_GetArrayItem(instructRoot, storyIndexInFile);
    if (!storyObj || !cJSON_IsObject(storyObj)) {
        printf("Warning: No instruct entry for story index %d.\n", storyIndexInFile);
        return -1;
    }

    cJSON* elements = cJSON_GetObjectItemCaseSensitive(storyObj, "elements");
    if (!elements || !cJSON_IsObject(elements)) {
        printf("Warning: Instruct entry %d has no 'elements' object.\n", storyIndexInFile);
        return -1;
    }

    int numElements = cJSON_GetArraySize(elements);
    if (numElements <= 0) {
        return 0;  // nothing to prepend
    }

    ElementEntry* entries = (ElementEntry*)calloc((size_t)numElements, sizeof(ElementEntry));
    if (!entries) {
        printf("Error: Failed to allocate instruct element entries.\n");
        return -1;
    }

    int collected = 0;
    int order = 0;
    for (cJSON* el = elements->child; el != nullptr; el = el->next, order++) {
        if (!el->string || !cJSON_IsArray(el)) {
            continue;
        }
        const InstructCategoryMeta* meta = findCategory(el->string);
        if (!meta) {
            printf("Warning: Category '%s' not found in instruct keys, skipping.\n", el->string);
            continue;
        }
        entries[collected].priority   = meta->priority;
        entries[collected].origOrder  = order;
        entries[collected].meta       = meta;
        entries[collected].valueArray = el;
        collected++;
    }

    if (collected == 0) {
        free(entries);
        return 0;
    }

    qsort(entries, (size_t)collected, sizeof(ElementEntry), compareElementEntries);

    int count = 0;
    int room = 1;

    // Opening outer bracket.
    room = emitToken(INSTRUCT_TOK_OPEN, outTokens, &count, maxTokens);

    for (int i = 0; i < collected && room; i++) {
        // Separator between elements.
        if (i > 0) {
            room = emitToken(INSTRUCT_TOK_PLUS, outTokens, &count, maxTokens);
            if (!room) break;
        }

        // Element open bracket.
        room = emitToken(INSTRUCT_TOK_OPEN, outTokens, &count, maxTokens);
        if (!room) break;

        // Category name tokens.
        const InstructCategoryMeta* meta = entries[i].meta;
        for (int t = 0; t < meta->numTokens && room; t++) {
            room = emitToken(meta->tokens[t], outTokens, &count, maxTokens);
        }
        if (!room) break;

        // Arrow separating category from value.
        room = emitToken(INSTRUCT_TOK_ARROW, outTokens, &count, maxTokens);
        if (!room) break;

        // Value tokens.
        cJSON* valueArray = entries[i].valueArray;
        int numValues = cJSON_GetArraySize(valueArray);
        for (int v = 0; v < numValues && room; v++) {
            cJSON* valItem = cJSON_GetArrayItem(valueArray, v);
            if (valItem && cJSON_IsString(valItem)) {
                room = emitToken(valItem->valuestring, outTokens, &count, maxTokens);
            }
        }
        if (!room) break;

        // Element close bracket.
        room = emitToken(INSTRUCT_TOK_CLOSE, outTokens, &count, maxTokens);
    }

    // Closing outer bracket.
    if (room) room = emitToken(INSTRUCT_TOK_CLOSE, outTokens, &count, maxTokens);

    // Two newlines separating the instruct block from the story.
    if (room) room = emitToken(INSTRUCT_TOK_NEWLINE, outTokens, &count, maxTokens);
    if (room) room = emitToken(INSTRUCT_TOK_NEWLINE, outTokens, &count, maxTokens);

    free(entries);
    return count;
}
