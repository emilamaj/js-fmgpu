
// Version 0: Pure JS implementation for reference (very good for small matrices, N < 200)
function solveLinearSystemJS(A, b) {
    const n = A.length;

    // Create the augmented matrix (A|b)
    let augmentedMatrix = A.map((row, i) => [...row, b[i]]);

    // Gaussian elimination
    for (let i = 0; i < n; i++) {
        // Find the pivot row
        let pivotRow = i;
        for (let j = i + 1; j < n; j++) {
            if (Math.abs(augmentedMatrix[j][i]) > Math.abs(augmentedMatrix[pivotRow][i])) {
                pivotRow = j;
            }
        }

        // Swap rows
        [augmentedMatrix[i], augmentedMatrix[pivotRow]] = [augmentedMatrix[pivotRow], augmentedMatrix[i]];

        // Normalize row
        const factor = augmentedMatrix[i][i];
        for (let j = 0; j < n + 1; j++) {
            augmentedMatrix[i][j] /= factor;
        }

        // Eliminate
        for (let j = 0; j < n; j++) {
            if (j !== i) {
                const factor = augmentedMatrix[j][i];
                for (let k = 0; k < n + 1; k++) {
                    augmentedMatrix[j][k] -= augmentedMatrix[i][k] * factor;
                }
            }
        }
    }

    // Back substitution
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        x[i] = augmentedMatrix[i][n];
        for (let j = i + 1; j < n; j++) {
            x[i] -= augmentedMatrix[i][j] * x[j];
        }
    }

    return x;
}

module.exports = { solveLinearSystemJS };