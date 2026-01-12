const fs = require('fs');
const path = require('path');

// --- Configuration ---
const FILE_FLAT = './cuda_gradients/dLoss_d_vocabScores.json';
const FILE_2D = './gradients/logits_grad.json';

// Logical dimensions based on your description
// Flat array is Column Major: Blocks of columns, each containing 10097 rows.
const ROWS = 10097; 
const COLS = 256;
const RELATIVE_TOLERANCE = 1e-3; // 1%
const PRINT_LIMIT = 20;

function loadJSON(filepath) {
    try {
        console.log(`Loading ${filepath}...`);
        const raw = fs.readFileSync(filepath, 'utf8');
        return JSON.parse(raw);
    } catch (e) {
        console.error(`Error loading ${filepath}:`, e.message);
        process.exit(1);
    }
}

function calculateRelativeDiff(a, b) {
    const absA = Math.abs(a);
    const absB = Math.abs(b);
    const diff = Math.abs(a - b);

    // Avoid division by zero. If both are near zero, diff is effectively 0.
    // If one is zero and the other is not, relative diff is 1.0 (100%).
    if (diff === 0) return 0;
    
    // Standard relative difference formula: |a-b| / max(|a|,|b|)
    // We add a tiny epsilon to denominator to prevent NaN if both are 0 (handled above, but for safety)
    return diff / (Math.max(absA, absB) + 1e-20);
}

function main() {
    // 1. Load Data
    const flatDataRaw = loadJSON(FILE_FLAT);
    const matrixObj = loadJSON(FILE_2D);

    // Handle flat file structure (it might be a raw array or an object containing data)
    // Assuming the file is just [ ...numbers... ] based on prompt description
    const flatArr = Array.isArray(flatDataRaw) ? flatDataRaw : flatDataRaw.data;

    // Handle 2D file structure
    const matrixArr = matrixObj.data;

    // 2. Verification
    const expectedSize = ROWS * COLS;
    if (flatArr.length !== expectedSize) {
        console.error(`Dimension mismatch! Flat array length: ${flatArr.length}, Expected: ${expectedSize}`);
        return;
    }
    
    // Check 2D shape roughly
    if (matrixArr.length !== ROWS || matrixArr[0].length !== COLS) {
        console.error(`Dimension mismatch! Matrix shape seems to be [${matrixArr.length}, ${matrixArr[0]?.length}], Expected: [${ROWS}, ${COLS}]`);
        return;
    }

    console.log(`Dimensions confirmed: ${ROWS} rows x ${COLS} cols.`);
    console.log(`Comparing... (Threshold: > ${RELATIVE_TOLERANCE * 100}%)`);
    console.log('-'.repeat(60));

    // 3. Comparison Loop
    let mismatches = 0;
    let matches = 0;
    let printedCount = 0;

    let maxAbsDiff = { val: -1, a: 0, b: 0, r: 0, c: 0 };
    let maxRelDiff = { val: -1, a: 0, b: 0, r: 0, c: 0 };

    // We iterate the FLAT array (Column Major)
    // Index i goes from 0 to 2,584,831
    for (let i = 0; i < flatArr.length; i++) {
        
        // --- coordinate mapping ---
        // Since flatArr is Column Major:
        // The first 10097 elements are Column 0.
        // The next 10097 elements are Column 1.
        
        const col = Math.floor(i / ROWS);
        const row = i % ROWS;

        const valFlat = flatArr[i];
        
        // Matrix is standard Row Major [Row][Col]
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

    // 4. Final Report
    console.log('-'.repeat(60));
    console.log('COMPARISON COMPLETE');
    console.log('-'.repeat(60));
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
    console.log(`   Value: ${(maxRelDiff.val * 100).toFixed(2)}%`);
    console.log(`   @ (Row: ${maxRelDiff.r}, Col: ${maxRelDiff.c})`);
    console.log(`   Flat: ${maxRelDiff.a} vs 2D: ${maxRelDiff.b}`);
}

main();