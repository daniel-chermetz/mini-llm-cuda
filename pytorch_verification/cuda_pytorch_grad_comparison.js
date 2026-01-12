const fs = require('fs');

// =============================================================================
// CONFIGURATION
// =============================================================================

const RELATIVE_TOLERANCE = 1e-2; // 1%
const PRINT_LIMIT = 0;

// Toggle which stages to test
const TEST_CONFIG = {
    finalLogitStage: true,   // Logits, embedding weights, final RMS
    t0: true,                // Transformer 0 - uppermost (near logits)
    t1: false,                // Transformer 1
    t2: false,                // Transformer 2
    t3: false,                // Transformer 3 - entry point (receives tokens)
};

// Network dimensions (from network_meta.h)
const DIM = 512;
const FFN_DIM = 2048;
const L = 256;
const VOCAB_SIZE = 10097;
const ATTN_HEADS = 8;

// =============================================================================
// TEST CASE DEFINITIONS
// =============================================================================

// Final logit stage gradients
const FINAL_STAGE_TESTS = [
    {
        name: "Vocab Scores (Logits)",
        fileFlat: './cuda_gradients/dLoss_d_vocabScores.json',
        file2D:   './gradients/logits_grad.json',
        rows: VOCAB_SIZE,
        cols: L
    },
    {
        name: "Embedding Weights",
        fileFlat: './cuda_gradients/dLoss_d_embedding_weights.json',
        file2D:   './gradients/token_embeddings_grad.json',
        //file2D:   './gradients/manual_token_embeddings_grad.json',
        rows: DIM,
        cols: VOCAB_SIZE
    },
    {
        name: "FFN Final Post RMS Post Gamma",
        fileFlat: './cuda_gradients/dLoss_d_ffn_final_postRMS_postGamma.json',
        file2D:   './gradients/output_post_rms_final_grad.json',
        rows: DIM,
        cols: L
    },
    {
        name: "Final RMS Gamma Weights",
        fileFlat: './cuda_gradients/dLoss_d_ffn_final_RMS_gamma_weights.json',
        file2D:   './gradients/rms_gamma_final_grad.json',
        rows: DIM,
        cols: 1
    },
];

// Per-transformer gradient mappings
// Both CUDA and PyTorch use same indexing: 0 = uppermost, 3 = entry point
function getTransformerTests(transformerIdx) {
    const t = `t${transformerIdx}`;
    const layer = `layer${transformerIdx}`;
    
    return [
        {
            name: `[T${transformerIdx}] FFN Final Plus Residual`,
            fileFlat: `./cuda_gradients/${t}_ffn_final_plus_residual_grad.json`,
            file2D:   `./gradients/${layer}_post_ffn_residual_grad.json`,
            rows: DIM,
            cols: L
        },
        {
            name: `[T${transformerIdx}] FFN Left Weights`,
            fileFlat: `./cuda_gradients/${t}_ffn_left_weights_grad.json`,
            file2D:   `./gradients/${layer}_W_ffn_left_grad.json`,
            rows: DIM,
            cols: FFN_DIM
        },
        {
            name: `[T${transformerIdx}] FFN Right Post Hadamard`,
            fileFlat: `./cuda_gradients/${t}_ffn_right_postHadamard_grad.json`,
            file2D:   `./gradients/${layer}_ffn_hadamard_grad.json`,
            rows: FFN_DIM,
            cols: L
        },
        {
            name: `[T${transformerIdx}] FFN Right 1 Post Silu`,
            fileFlat: `./cuda_gradients/${t}_ffn_right_1_postSilu_grad.json`,
            file2D:   `./gradients/${layer}_ffn_right_1_post_silu_grad.json`,
            rows: FFN_DIM,
            cols: L
        },
        {
            name: `[T${transformerIdx}] FFN Right 1 Pre Silu`,
            fileFlat: `./cuda_gradients/${t}_ffn_right_1_preSilu_grad.json`,
            file2D:   `./gradients/${layer}_ffn_right_1_pre_silu_grad.json`,
            rows: FFN_DIM,
            cols: L
        },
        {
            name: `[T${transformerIdx}] FFN Right 1 Weights`,
            fileFlat: `./cuda_gradients/${t}_ffn_right_1_weights_grad.json`,
            file2D:   `./gradients/${layer}_W_ffn1_grad.json`,
            rows: FFN_DIM,
            cols: DIM
        },
        {
            name: `[T${transformerIdx}] FFN Right 2`,
            fileFlat: `./cuda_gradients/${t}_ffn_right_2_grad.json`,
            file2D:   `./gradients/${layer}_ffn_right_2_grad.json`,
            rows: FFN_DIM,
            cols: L
        },
        {
            name: `[T${transformerIdx}] FFN Right 2 Weights`,
            fileFlat: `./cuda_gradients/${t}_ffn_right_2_weights_grad.json`,
            file2D:   `./gradients/${layer}_W_ffn2_grad.json`,
            rows: FFN_DIM,
            cols: DIM
        },
        // RMS2 gradients (post-attention, pre-FFN)
        {
            name: `[T${transformerIdx}] Output Proj Plus Residual Post RMS2 Post Gamma`,
            fileFlat: `./cuda_gradients/${t}_outputProjPlusResidual_postRMS2_post_gamma_grad.json`,
            file2D:   `./gradients/${layer}_post_rms2_post_gamma_grad.json`,
            rows: DIM,
            cols: L
        },
        {
            name: `[T${transformerIdx}] RMS2 Gamma Weights`,
            fileFlat: `./cuda_gradients/${t}_rms2_gamma_weights_grad.json`,
            file2D:   `./gradients/${layer}_rms_gamma2_grad.json`,
            rows: DIM,
            cols: 1
        },
        {
            name: `[T${transformerIdx}] Output Proj Plus Residual (pre-RMS2)`,
            fileFlat: `./cuda_gradients/${t}_outputProjPlusResidual_grad.json`,
            file2D:   `./gradients/${layer}_pre_rms2_grad.json`,
            rows: DIM,
            cols: L
        },
        // Attention gradients
        {
            name: `[T${transformerIdx}] Value Scaled Softmax Attn (attn_out_flat)`,
            fileFlat: `./cuda_gradients/${t}_valueScaledSoftmaxAttn_grad.json`,
            file2D:   `./gradients/${layer}_attn_out_flat_grad.json`,
            rows: DIM,
            cols: L
        },
        {
            name: `[T${transformerIdx}] Output Proj Weights (Wo)`,
            fileFlat: `./cuda_gradients/${t}_output_proj_weights_grad.json`,
            file2D:   `./gradients/${layer}_Wo_grad.json`,
            rows: DIM,
            cols: DIM
        },
        {
            name: `[T${transformerIdx}] Values`,
            fileFlat: `./cuda_gradients/${t}_values_grad.json`,
            file2D:   `./gradients/${layer}_v_flat_grad.json`,
            rows: DIM,
            cols: L
        },
        /*{
            name: `[T${transformerIdx}] Attention Probs (postSoftmax)`,
            fileFlat: `./cuda_gradients/${t}_attnByHead_postSoftmax_grad.json`,
            file2D:   `./gradients/${layer}_attn_probs_grad.json`,
            is3D: true,
            transposed: false, // CUDA (row,col) maps to PyTorch [head][col][row]
            dim0: ATTN_HEADS,  // heads
            dim1: L,          // "rows" within each head's LxL matrix
            dim2: L           // "cols" (sequence positions)
        },*/
        {
            name: `[T${transformerIdx}] Attention Scores (KtQ pre-scale)`,
            fileFlat: `./cuda_gradients/${t}_attnKtQByHead_grad.json`,
            file2D:   `./gradients/${layer}_scores_pre_scale_grad.json`,
            is3D: true,
            dim0: ATTN_HEADS,  // heads
            dim1: L,          // "rows" within each head's LxL matrix  
            dim2: L           // "cols" (sequence positions)
        },
        // Q, K, V post-RoPE and pre-RoPE
        // Post-RoPE: PyTorch saves as [ATTN_HEADS, HEAD_DIM, L], CUDA as fused [dim, L]
        {
            name: `[T${transformerIdx}] Keys Post RoPE`,
            fileFlat: `./cuda_gradients/${t}_keysPostRoPE_grad.json`,
            file2D:   `./gradients/${layer}_k_rot_grad.json`,
            isRoPE3D: true,
            dim0: ATTN_HEADS,       // 8 heads
            dim1: DIM / ATTN_HEADS, // HEAD_DIM = 64
            dim2: L                 // 256 positions
        },
        {
            name: `[T${transformerIdx}] Keys Pre RoPE`,
            fileFlat: `./cuda_gradients/${t}_keysPreRoPE_grad.json`,
            file2D:   `./gradients/${layer}_k_flat_grad.json`,
            rows: DIM,
            cols: L
        },
        {
            name: `[T${transformerIdx}] Queries Post RoPE`,
            fileFlat: `./cuda_gradients/${t}_queriesPostRoPE_grad.json`,
            file2D:   `./gradients/${layer}_q_rot_grad.json`,
            isRoPE3D: true,
            dim0: ATTN_HEADS,       // 8 heads
            dim1: DIM / ATTN_HEADS, // HEAD_DIM = 64
            dim2: L                 // 256 positions
        },
        {
            name: `[T${transformerIdx}] Queries Pre RoPE`,
            fileFlat: `./cuda_gradients/${t}_queriesPreRoPE_grad.json`,
            file2D:   `./gradients/${layer}_q_flat_grad.json`,
            rows: DIM,
            cols: L
        },
        // Weight gradients
        {
            name: `[T${transformerIdx}] Value Weights (Wv)`,
            fileFlat: `./cuda_gradients/${t}_value_weights_grad.json`,
            file2D:   `./gradients/${layer}_Wv_grad.json`,
            rows: DIM,
            cols: DIM
        },
        {
            name: `[T${transformerIdx}] Query Weights (Wq)`,
            fileFlat: `./cuda_gradients/${t}_query_weights_grad.json`,
            file2D:   `./gradients/${layer}_Wq_grad.json`,
            rows: DIM,
            cols: DIM
        },
        {
            name: `[T${transformerIdx}] Key Weights (Wk)`,
            fileFlat: `./cuda_gradients/${t}_key_weights_grad.json`,
            file2D:   `./gradients/${layer}_Wk_grad.json`,
            rows: DIM,
            cols: DIM
        },
        // RMS1 gradients (pre-attention)
        {
            name: `[T${transformerIdx}] X Post RMS1 Post Gamma`,
            fileFlat: `./cuda_gradients/${t}_x_postRMS1_post_gamma_grad.json`,
            file2D:   `./gradients/${layer}_post_rms1_post_gamma_grad.json`,
            rows: DIM,
            cols: L
        },
        {
            name: `[T${transformerIdx}] RMS1 Gamma Weights`,
            fileFlat: `./cuda_gradients/${t}_rms1_gamma_weights_grad.json`,
            file2D:   `./gradients/${layer}_rms_gamma_grad.json`,
            rows: DIM,
            cols: 1
        },
    ];
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function loadJSON(filepath) {
    try {
        if (!fs.existsSync(filepath)) {
            return null; // File doesn't exist, skip this test
        }
        console.log(`Loading ${filepath}...`);
        const raw = fs.readFileSync(filepath, 'utf8');
        return JSON.parse(raw);
    } catch (e) {
        console.error(`Error loading ${filepath}:`, e.message);
        return null;
    }
}

function calculateRelativeDiff(a, b) {
    const absA = Math.abs(a);
    const absB = Math.abs(b);
    const diff = Math.abs(a - b);

    if (diff === 0) return 0;
    // Add tiny epsilon to denominator to avoid division by zero
    return diff / (Math.max(absA, absB) + 1e-20);
}

function runComparison3D(testConfig) {
    // For 3D PyTorch tensors stored as [ATTN_HEADS, L, L] (heads, rows, cols)
    // CUDA stores as flat column-major with shape [L rows, ATTN_HEADS*L cols]
    // Columns are grouped by head: cols 0..L-1 for head 0, cols L..2L-1 for head 1, etc.
    // flat_idx = (head * L + pytorch_col) * L + pytorch_row
    console.log(`\n============================================================`);
    console.log(`STARTING TEST (3D): ${testConfig.name}`);
    console.log(`============================================================`);

    const flatDataRaw = loadJSON(testConfig.fileFlat);
    const matrixObj = loadJSON(testConfig.file2D);

    if (flatDataRaw === null) {
        console.log(`[SKIP] CUDA file not found: ${testConfig.fileFlat}`);
        return { skipped: true };
    }
    if (matrixObj === null) {
        console.log(`[SKIP] PyTorch file not found: ${testConfig.file2D}`);
        return { skipped: true };
    }

    const flatArr = Array.isArray(flatDataRaw) ? flatDataRaw : flatDataRaw.data;
    const data3D = matrixObj.data;  // [dim0][dim1][dim2]

    const DIM0 = testConfig.dim0;  // ATTN_HEADS
    const DIM1 = testConfig.dim1;  // L (rows within head)
    const DIM2 = testConfig.dim2;  // L (cols/positions)
    const expectedSize = DIM0 * DIM1 * DIM2;

    if (flatArr.length !== expectedSize) {
        console.error(`[FAIL] Dimension mismatch! Flat array length: ${flatArr.length}, Expected: ${expectedSize}`);
        return { failed: true };
    }

    // Check 3D shape
    const shape0 = data3D.length;
    const shape1 = data3D[0]?.length || 0;
    const shape2 = data3D[0]?.[0]?.length || 0;
    
    if (shape0 !== DIM0 || shape1 !== DIM1 || shape2 !== DIM2) {
        console.error(`[FAIL] Dimension mismatch! 3D shape: [${shape0}, ${shape1}, ${shape2}], Expected: [${DIM0}, ${DIM1}, ${DIM2}]`);
        return { failed: true };
    }

    console.log(`Dimensions confirmed: [${DIM0}, ${DIM1}, ${DIM2}] = ${expectedSize.toLocaleString()} elements`);
    console.log(`CUDA layout: [${DIM1} rows, ${DIM0}*${DIM2} cols] column-major, heads in column blocks`);
    if (testConfig.transposed) {
        console.log(`Transposed mapping: CUDA (row,col) -> PyTorch [head][col][row]`);
    }
    console.log(`Comparing Flat vs 3D...`);
    console.log('-'.repeat(60));

    let mismatches = 0;
    let matches = 0;
    let printedCount = 0;

    let maxAbsDiff = { val: -1, a: 0, b: 0, h: 0, r: 0, c: 0 };
    let maxRelDiff = { val: -1, a: 0, b: 0, h: 0, r: 0, c: 0 };

    // CUDA layout: [L rows, ATTN_HEADS*L cols], column-major
    // Columns grouped by head: head h owns cols [h*L, (h+1)*L)
    // flat_idx = (head * L + col_within_head) * L + row
    //          = head * L * L + col * L + row
    // To invert:
    //   head = floor(i / (L * L))
    //   remainder = i % (L * L)
    //   col = floor(remainder / L)
    //   row = remainder % L
    const L_SQUARED = DIM1 * DIM2;  // L * L
    const isTransposed = testConfig.transposed || false;
    
    for (let i = 0; i < flatArr.length; i++) {
        const head = Math.floor(i / L_SQUARED);
        const remainder = i % L_SQUARED;
        const cudaCol = Math.floor(remainder / DIM1);
        const cudaRow = remainder % DIM1;

        const valFlat = flatArr[i];
        // If transposed, CUDA (row,col) maps to PyTorch [head][col][row]
        const valMatrix = isTransposed ? data3D[head][cudaCol][cudaRow] : data3D[head][cudaRow][cudaCol];

        const absDiff = Math.abs(valFlat - valMatrix);
        const relDiff = calculateRelativeDiff(valFlat, valMatrix);

        if (absDiff > maxAbsDiff.val) {
            maxAbsDiff = { val: absDiff, a: valFlat, b: valMatrix, h: head, r: cudaRow, c: cudaCol };
        }
        if (relDiff > maxRelDiff.val) {
            maxRelDiff = { val: relDiff, a: valFlat, b: valMatrix, h: head, r: cudaRow, c: cudaCol };
        }

        if (relDiff > RELATIVE_TOLERANCE) {
            mismatches++;
            if (printedCount < PRINT_LIMIT) {
                console.log(`[Mismatch] @ (Head: ${head}, CudaRow: ${cudaRow}, CudaCol: ${cudaCol})`);
                console.log(`   Flat: ${valFlat}`);
                console.log(`   3D:   ${valMatrix}`);
                console.log(`   Diff: Abs: ${absDiff.toExponential(4)}, Rel: ${(relDiff * 100).toFixed(4)}%`);
                console.log('');
                printedCount++;
            }
        } else {
            matches++;
        }
    }

    console.log('-'.repeat(60));
    console.log(`RESULT: ${testConfig.name}`);
    console.log(`Total Elements: ${flatArr.length.toLocaleString()}`);
    console.log(`Matches (< 1%): ${matches.toLocaleString()}`);
    console.log(`Mismatches:     ${mismatches.toLocaleString()}`);
    console.log('-'.repeat(60));
    
    console.log('MAX ABSOLUTE DIFFERENCE:');
    console.log(`   Value: ${maxAbsDiff.val}`);
    console.log(`   @ (Head: ${maxAbsDiff.h}, Row: ${maxAbsDiff.r}, Col: ${maxAbsDiff.c})`);
    console.log(`   Flat: ${maxAbsDiff.a} vs 3D: ${maxAbsDiff.b}`);
    console.log('');

    console.log('MAX RELATIVE DIFFERENCE:');
    console.log(`   Value: ${(maxRelDiff.val * 100).toFixed(4)}%`);
    console.log(`   @ (Head: ${maxRelDiff.h}, Row: ${maxRelDiff.r}, Col: ${maxRelDiff.c})`);
    console.log(`   Flat: ${maxRelDiff.a} vs 3D: ${maxRelDiff.b}`);
    console.log('\n');

    return { matches, mismatches, maxAbsDiff, maxRelDiff };
}

function runComparisonRoPE3D(testConfig) {
    // For RoPE gradients:
    // PyTorch: [ATTN_HEADS, HEAD_DIM, L] = [8, 64, 256]
    // CUDA: [dim, L] column-major (fused heads)
    // flat_idx = col * dim + head * HEAD_DIM + headRelRow
    // PyTorch: data3D[head][headRelRow][col]
    console.log(`\n============================================================`);
    console.log(`STARTING TEST (RoPE 3D): ${testConfig.name}`);
    console.log(`============================================================`);

    const flatDataRaw = loadJSON(testConfig.fileFlat);
    const matrixObj = loadJSON(testConfig.file2D);

    if (flatDataRaw === null) {
        console.log(`[SKIP] CUDA file not found: ${testConfig.fileFlat}`);
        return { skipped: true };
    }
    if (matrixObj === null) {
        console.log(`[SKIP] PyTorch file not found: ${testConfig.file2D}`);
        return { skipped: true };
    }

    const flatArr = Array.isArray(flatDataRaw) ? flatDataRaw : flatDataRaw.data;
    const data3D = matrixObj.data;  // [heads][headDim][L]

    const HEADS = testConfig.dim0;     // 8
    const HEAD_DIM = testConfig.dim1;  // 64
    const SEQ_LEN = testConfig.dim2;   // 256
    const TOTAL_DIM = HEADS * HEAD_DIM; // 512
    const expectedSize = TOTAL_DIM * SEQ_LEN;

    if (flatArr.length !== expectedSize) {
        console.error(`[FAIL] Dimension mismatch! Flat array length: ${flatArr.length}, Expected: ${expectedSize}`);
        return { failed: true };
    }

    // Check 3D shape
    const shape0 = data3D.length;
    const shape1 = data3D[0]?.length || 0;
    const shape2 = data3D[0]?.[0]?.length || 0;
    
    if (shape0 !== HEADS || shape1 !== HEAD_DIM || shape2 !== SEQ_LEN) {
        console.error(`[FAIL] Dimension mismatch! 3D shape: [${shape0}, ${shape1}, ${shape2}], Expected: [${HEADS}, ${HEAD_DIM}, ${SEQ_LEN}]`);
        return { failed: true };
    }

    console.log(`CUDA: [${TOTAL_DIM}, ${SEQ_LEN}] column-major fused`);
    console.log(`PyTorch: [${HEADS}, ${HEAD_DIM}, ${SEQ_LEN}]`);
    console.log(`Total elements: ${expectedSize.toLocaleString()}`);
    console.log('-'.repeat(60));

    let mismatches = 0;
    let matches = 0;
    let printedCount = 0;

    let maxAbsDiff = { val: -1, a: 0, b: 0, h: 0, r: 0, c: 0 };
    let maxRelDiff = { val: -1, a: 0, b: 0, h: 0, r: 0, c: 0 };

    // CUDA: flat_idx = col * TOTAL_DIM + row, where row = head * HEAD_DIM + headRelRow
    for (let i = 0; i < flatArr.length; i++) {
        const col = Math.floor(i / TOTAL_DIM);      // position (0-255)
        const row = i % TOTAL_DIM;                   // fused row (0-511)
        const head = Math.floor(row / HEAD_DIM);     // head (0-7)
        const headRelRow = row % HEAD_DIM;           // row within head (0-63)

        const valFlat = flatArr[i];
        const valMatrix = data3D[head][headRelRow][col];

        const absDiff = Math.abs(valFlat - valMatrix);
        const relDiff = calculateRelativeDiff(valFlat, valMatrix);

        if (absDiff > maxAbsDiff.val) {
            maxAbsDiff = { val: absDiff, a: valFlat, b: valMatrix, h: head, r: headRelRow, c: col };
        }
        if (relDiff > maxRelDiff.val) {
            maxRelDiff = { val: relDiff, a: valFlat, b: valMatrix, h: head, r: headRelRow, c: col };
        }

        if (relDiff > RELATIVE_TOLERANCE) {
            mismatches++;
            if (printedCount < PRINT_LIMIT) {
                console.log(`[Mismatch] @ (Head: ${head}, HeadRow: ${headRelRow}, Col: ${col})`);
                console.log(`   Flat: ${valFlat}`);
                console.log(`   3D:   ${valMatrix}`);
                console.log(`   Diff: Abs: ${absDiff.toExponential(4)}, Rel: ${(relDiff * 100).toFixed(4)}%`);
                console.log('');
                printedCount++;
            }
        } else {
            matches++;
        }
    }

    console.log('-'.repeat(60));
    console.log(`RESULT: ${testConfig.name}`);
    console.log(`Total Elements: ${flatArr.length.toLocaleString()}`);
    console.log(`Matches (< 1%): ${matches.toLocaleString()}`);
    console.log(`Mismatches:     ${mismatches.toLocaleString()}`);
    console.log('-'.repeat(60));
    
    console.log('MAX ABSOLUTE DIFFERENCE:');
    console.log(`   Value: ${maxAbsDiff.val}`);
    console.log(`   @ (Head: ${maxAbsDiff.h}, HeadRow: ${maxAbsDiff.r}, Col: ${maxAbsDiff.c})`);
    console.log(`   Flat: ${maxAbsDiff.a} vs 3D: ${maxAbsDiff.b}`);
    console.log('');

    console.log('MAX RELATIVE DIFFERENCE:');
    console.log(`   Value: ${(maxRelDiff.val * 100).toFixed(4)}%`);
    console.log(`   @ (Head: ${maxRelDiff.h}, HeadRow: ${maxRelDiff.r}, Col: ${maxRelDiff.c})`);
    console.log(`   Flat: ${maxRelDiff.a} vs 3D: ${maxRelDiff.b}`);
    console.log('\n');

    return { matches, mismatches, maxAbsDiff, maxRelDiff };
}

function runComparison(testConfig) {
    console.log(`\n============================================================`);
    console.log(`STARTING TEST: ${testConfig.name}`);
    console.log(`============================================================`);

    // 1. Load Data
    const flatDataRaw = loadJSON(testConfig.fileFlat);
    const matrixObj = loadJSON(testConfig.file2D);

    // Check if files exist
    if (flatDataRaw === null) {
        console.log(`[SKIP] CUDA file not found: ${testConfig.fileFlat}`);
        return { skipped: true };
    }
    if (matrixObj === null) {
        console.log(`[SKIP] PyTorch file not found: ${testConfig.file2D}`);
        return { skipped: true };
    }

    const flatArr = Array.isArray(flatDataRaw) ? flatDataRaw : flatDataRaw.data;
    const matrixArr = matrixObj.data;

    // 2. Verification
    const ROWS = testConfig.rows;
    const COLS = testConfig.cols;
    const expectedSize = ROWS * COLS;

    if (flatArr.length !== expectedSize) {
        console.error(`[FAIL] Dimension mismatch! Flat array length: ${flatArr.length}, Expected: ${expectedSize}`);
        return { failed: true };
    }

    // Check 2D shape roughly
    const matrixRows = matrixArr.length;
    const matrixCols = matrixArr[0]?.length || 0;
    
    if (matrixRows !== ROWS || matrixCols !== COLS) {
        console.error(`[FAIL] Dimension mismatch! Matrix shape: [${matrixRows}, ${matrixCols}], Expected: [${ROWS}, ${COLS}]`);
        return { failed: true };
    }

    console.log(`Dimensions confirmed: ${ROWS} rows x ${COLS} cols.`);
    console.log(`Comparing Flat (Column-Major) vs 2D (Row-Major)...`);
    console.log('-'.repeat(60));

    // 3. Comparison Loop
    let mismatches = 0;
    let matches = 0;
    let printedCount = 0;

    let maxAbsDiff = { val: -1, a: 0, b: 0, r: 0, c: 0 };
    let maxRelDiff = { val: -1, a: 0, b: 0, r: 0, c: 0 };

    for (let i = 0; i < flatArr.length; i++) {
        
        // --- coordinate mapping ---
        // Flat Array is Column Major:
        // It stores all rows for Col 0, then all rows for Col 1, etc.
        const col = Math.floor(i / ROWS);
        const row = i % ROWS;

        const valFlat = flatArr[i];
        
        // Matrix is standard Row Major: data[row][col]
        const valMatrix = matrixArr[row][col];

        // --- Analysis ---
        const absDiff = Math.abs(valFlat - valMatrix);
        const relDiff = calculateRelativeDiff(valFlat, valMatrix);

        // Update Global Max Stats
        if (absDiff > maxAbsDiff.val) {
            maxAbsDiff = { val: absDiff, a: valFlat, b: valMatrix, r: row, c: col };
        }
        if (relDiff > maxRelDiff.val) {
            maxRelDiff = { val: relDiff, a: valFlat, b: valMatrix, r: row, c: col };
        }

        // Check Tolerance
        if (relDiff > RELATIVE_TOLERANCE) {
            mismatches++;
            if (printedCount < PRINT_LIMIT) {
                console.log(`[Mismatch] @ (Row: ${row}, Col: ${col})`);
                console.log(`   Flat: ${valFlat}`);
                console.log(`   2D:   ${valMatrix}`);
                console.log(`   Diff: Abs: ${absDiff.toExponential(4)}, Rel: ${(relDiff * 100).toFixed(4)}%`);
                console.log('');
                printedCount++;
            }
        } else {
            matches++;
        }
    }

    // 4. Final Report for this test
    console.log('-'.repeat(60));
    console.log(`RESULT: ${testConfig.name}`);
    console.log(`Total Elements: ${flatArr.length.toLocaleString()}`);
    console.log(`Matches (< 1%): ${matches.toLocaleString()}`);
    console.log(`Mismatches:     ${mismatches.toLocaleString()}`);
    console.log('-'.repeat(60));
    
    console.log('MAX ABSOLUTE DIFFERENCE:');
    console.log(`   Value: ${maxAbsDiff.val}`);
    console.log(`   @ (Row: ${maxAbsDiff.r}, Col: ${maxAbsDiff.c})`);
    console.log(`   Flat: ${maxAbsDiff.a} vs 2D: ${maxAbsDiff.b}`);
    console.log('');

    console.log('MAX RELATIVE DIFFERENCE:');
    console.log(`   Value: ${(maxRelDiff.val * 100).toFixed(4)}%`);
    console.log(`   @ (Row: ${maxRelDiff.r}, Col: ${maxRelDiff.c})`);
    console.log(`   Flat: ${maxRelDiff.a} vs 2D: ${maxRelDiff.b}`);
    console.log('\n');

    return { matches, mismatches, maxAbsDiff, maxRelDiff };
}

// =============================================================================
// MAIN
// =============================================================================

function main() {
    console.log('='.repeat(60));
    console.log('CUDA vs PyTorch GRADIENT COMPARISON');
    console.log('='.repeat(60));
    console.log('\nTest Configuration:');
    console.log(`  Final Logit Stage: ${TEST_CONFIG.finalLogitStage}`);
    console.log(`  T0 (uppermost): ${TEST_CONFIG.t0}`);
    console.log(`  T1: ${TEST_CONFIG.t1}`);
    console.log(`  T2: ${TEST_CONFIG.t2}`);
    console.log(`  T3 (entry point): ${TEST_CONFIG.t3}`);
    console.log('');

    let allTests = [];

    // Add final stage tests if enabled
    if (TEST_CONFIG.finalLogitStage) {
        allTests = allTests.concat(FINAL_STAGE_TESTS);
    }

    // Add transformer tests based on config
    if (TEST_CONFIG.t0) {
        allTests = allTests.concat(getTransformerTests(0));
    }
    if (TEST_CONFIG.t1) {
        allTests = allTests.concat(getTransformerTests(1));
    }
    if (TEST_CONFIG.t2) {
        allTests = allTests.concat(getTransformerTests(2));
    }
    if (TEST_CONFIG.t3) {
        allTests = allTests.concat(getTransformerTests(3));
    }

    // Run all tests and collect summary
    let summary = {
        total: 0,
        passed: 0,
        failed: 0,
        skipped: 0,
        totalElements: 0,
        totalMatches: 0,
        totalMismatches: 0
    };

    allTests.forEach(test => {
        summary.total++;
        let result;
        if (test.isRoPE3D) {
            result = runComparisonRoPE3D(test);
        } else if (test.is3D) {
            result = runComparison3D(test);
        } else {
            result = runComparison(test);
        }
        
        if (result.skipped) {
            summary.skipped++;
        } else if (result.failed) {
            summary.failed++;
        } else {
            summary.passed++;
            summary.totalElements += result.matches + result.mismatches;
            summary.totalMatches += result.matches;
            summary.totalMismatches += result.mismatches;
        }
    });

    // Print summary
    console.log('\n');
    console.log('='.repeat(60));
    console.log('OVERALL SUMMARY');
    console.log('='.repeat(60));
    console.log(`Total Tests:     ${summary.total}`);
    console.log(`Passed:          ${summary.passed}`);
    console.log(`Failed:          ${summary.failed}`);
    console.log(`Skipped:         ${summary.skipped}`);
    console.log('-'.repeat(60));
    console.log(`Total Elements Compared: ${summary.totalElements.toLocaleString()}`);
    console.log(`Total Matches (<1%):     ${summary.totalMatches.toLocaleString()}`);
    console.log(`Total Mismatches:        ${summary.totalMismatches.toLocaleString()}`);
    if (summary.totalElements > 0) {
        const matchRate = (summary.totalMatches / summary.totalElements * 100).toFixed(4);
        console.log(`Match Rate:              ${matchRate}%`);
    }
    console.log('='.repeat(60));
}

main();