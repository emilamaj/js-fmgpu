
import { GPU } from 'gpu.js';
const gpu = new GPU();

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

// Version 2: Use of pipelining to minize transfers between CPU and GPU (better for large matrices)
function solveLinearSystemGPU(A, b) {
    const n = A.length;

    // Create the augmented matrix (A|b)
    const augmentedMatrix = A.map((row, i) => [...row, b[i]]);

    // Define GPU kernels:
    // Swap two rows of a matrix
    const swapRowsKernel = gpu.createKernel(function (matrix, row1, row2) {
        if (row1 === row2) return matrix[this.thread.y][this.thread.x];

        if (this.thread.y === row1) {
            return matrix[row2][this.thread.x];
        } else if (this.thread.y === row2) {
            return matrix[row1][this.thread.x];
        } else {
            return matrix[this.thread.y][this.thread.x];
        }
    }).setOutput([n + 1, n]).setPipeline(true);

    // Normalize a row of a matrix
    const normalizeRowKernel = gpu.createKernel(function (matrix, row) {
        if (this.thread.y !== row) return matrix[this.thread.y][this.thread.x];

        const factor = matrix[row][row];
        return matrix[this.thread.y][this.thread.x] / factor;
    }).setOutput([n + 1, n]).setPipeline(true);

    // Eliminate a row of a matrix
    const eliminateKernel = gpu.createKernel(function (matrix, row) {
        if (this.thread.y <= row) return matrix[this.thread.y][this.thread.x];

        const factor = matrix[this.thread.y][row] / matrix[row][row];
        return matrix[this.thread.y][this.thread.x] - factor * matrix[row][this.thread.x];
    }).setOutput([n + 1, n]).setPipeline(true);

    // Transfer the augmented matrix to the GPU
    let gpuMatrix = gpu.createKernel(function (matrix) {
        return matrix[this.thread.y][this.thread.x];
    }).setOutput([n + 1, n]).setPipeline(true)(augmentedMatrix);

    // Gaussian elimination
    for (let i = 0; i < n; i++) {
        // Find the pivot row. Note that we don't need to transfer the matrix back to the CPU
        let pivotRow = i;
        for (let j = i + 1; j < n; j++) {

            if (Math.abs(augmentedMatrix[j][i]) > Math.abs(augmentedMatrix[pivotRow][i])) {
                pivotRow = j;
            }
        }

        // Swap rows
        gpuMatrix = swapRowsKernel(gpuMatrix, i, pivotRow);

        // Normalize row
        gpuMatrix = normalizeRowKernel(gpuMatrix, i);

        // Eliminate
        gpuMatrix = eliminateKernel(gpuMatrix, i);
    }

    // Transfer the final matrix back to the CPU
    const cpuMatrix = gpuMatrix.toArray();

    // Back substitution
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        x[i] = cpuMatrix[i][n];
        for (let j = i + 1; j < n; j++) {
            x[i] -= cpuMatrix[i][j] * x[j];
        }
    }

    return x;
}

module.exports = { solveLinearSystemJS, solveLinearSystemGPU };