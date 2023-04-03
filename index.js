
const { GPU } = require('gpu.js');
const gpu = new GPU();

// Scale a matrix by a scalar
function scale(A, s) {
    const n = A.length;

    // Define GPU kernel
    const scaleKernel = gpu.createKernel(function (A, s) {
        return A[this.thread.y][this.thread.x] * s;
    }).setOutput([n, n]);

    // Apply kernel
    return scaleKernel(A, s);
}

// Add two matrices
function add(A, B) {
    const n = A.length;

    // Define GPU kernel
    const addKernel = gpu.createKernel(function (A, B) {
        return A[this.thread.y][this.thread.x] + B[this.thread.y][this.thread.x];
    }).setOutput([n, n]);

    // Apply kernel
    return addKernel(A, B);
}

// Subtract two matrices
function sub(A, B) {
    const n = A.length;

    // Define GPU kernel
    const subKernel = gpu.createKernel(function (A, B) {
        return A[this.thread.y][this.thread.x] - B[this.thread.y][this.thread.x];
    }).setOutput([n, n]);

    // Apply kernel
    return subKernel(A, B);
}

// Transpose a matrix
function transpose(A) {
    const n = A.length;

    // Define GPU kernel
    const transposeKernel = gpu.createKernel(function (A) {
        return A[this.thread.x][this.thread.y];
    }).setOutput([n, n]);

    // Apply kernel
    return transposeKernel(A);
}

// Hadamard product of two matrices (element-wise multiplication)
function hadamard(A, B) {
    const n = A.length;

    // Define GPU kernel
    const multiplyKernel = gpu.createKernel(function (A, B) {
        return A[this.thread.y][this.thread.x] * B[this.thread.y][this.thread.x];
    }).setOutput([n, n]);

    // Apply kernel
    return multiplyKernel(A, B);
}

// Multiply two matrices
function dot(A, B) {
    const n = A.length;

    // Define GPU kernel
    const multiplyKernel = gpu.createKernel(function (A, B) {
        let sum = 0;
        for (let i = 0; i < this.constants.n; i++) {
            sum += A[this.thread.y][i] * B[i][this.thread.x];
        }
        return sum;
    }).setOutput([n, n]).setConstants({ n });

    // Apply kernel
    return multiplyKernel(A, B);
}

// Calculate the determinant of a matrix in Pure JS
function detJS(A) {
    const n = A.length;

    // Base case: 1x1 matrix
    if (n === 1) return A[0][0];

    // Base case: 2x2 matrix
    if (n === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];

    // Recursive case: nxn matrix
    let det = 0;
    for (let i = 0; i < n; i++) {
        // Create a submatrix by removing the first row and ith column
        let submatrix = A.slice(1).map(row => row.filter((_, j) => j !== i));
        det += A[0][i] * detJS(submatrix) * (i % 2 === 0 ? 1 : -1);
    }
    return det;
}

// Matrix equality
function equals(A, B, epsilon = 1e-6) {
    const n = A.length;

    // Define GPU kernel to return equality matrix C
    const equalKernel = gpu.createKernel(function (A, B) {
        if (Math.abs(A[this.thread.y][this.thread.x] - B[this.thread.y][this.thread.x]) < this.constants.epsil) {
            return 1;
        } else {
            return 0;
        }
    }).setOutput([n, n]).setConstants({ epsil: epsilon });

    // GPU kernel to reduce C to a boolean vector V
    const reduceKernel = gpu.createKernel(function (C) {
        let and = 1;
        for (let i = 0; i < this.constants.size; i++) {
            and = and * C[i];
        }
        return and;
    }).setOutput([n]).setConstants({ size: n });

    // GPU kernel to reduce to a single boolean
    const reduceKernel2 = gpu.createKernel(function (V) {
        let and = 1;
        for (let i = 0; i < this.constants.size; i++) {
            and = and * V[i];
        }
        return and;
    }).setOutput([1]).setConstants({ size: n });

    // Combine kernels
    return gpu.combineKernels(reduceKernel2, reduceKernel, equalKernel, function (A, B) {
        return reduceKernel2(reduceKernel(equalKernel(A, B)));
    })(A, B) == 1;
}

// Version 0: Pure JS implementation for reference (very good for small matrices, N < 200)
function solveLinearSystemSmall(A, b) {
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

// Version 2: Use of pipelining to minize transfers between CPU and GPU (better for large matrices N > 200)
function solveLinearSystem(A, b) {
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

// Matrix inverse GPU
function inverse(A) {
    const n = A.length;

    // Create the augmented matrix (A|I)
    const augmentedMatrix = A.map((row, i) => [...row, ...Array(n).fill(0).map((_, j) => i === j ? 1 : 0)]);

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
    }).setOutput([2 * n, n]).setPipeline(true);

    // Normalize a row of a matrix
    const normalizeRowKernel = gpu.createKernel(function (matrix, row) {
        if (this.thread.y !== row) return matrix[this.thread.y][this.thread.x];

        const factor = matrix[row][row];
        return matrix[this.thread.y][this.thread.x] / factor;
    }).setOutput([2 * n, n]).setPipeline(true);

    // Eliminate a row of a matrix
    const eliminateKernel = gpu.createKernel(function (matrix, row) {
        if (this.thread.y <= row) return matrix[this.thread.y][this.thread.x];

        const factor = matrix[this.thread.y][row] / matrix[row][row];
        return matrix[this.thread.y][this.thread.x] - factor * matrix[row][this.thread.x];
    }).setOutput([2 * n, n]).setPipeline(true);

    // Transfer the augmented matrix to the GPU
    let gpuMatrix = gpu.createKernel(function (matrix) {
        return matrix[this.thread.y][this.thread.x];
    }).setOutput([2 * n, n]).setPipeline(true)(augmentedMatrix);

    // Extract the inverse matrix
    const extractInverseKernel = gpu.createKernel(function (matrix) {
        return matrix[this.thread.y][this.thread.x + this.constants.n];
    }).setOutput([n, n]).setConstants({ n });

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

    // Extract the inverse matrix
    const inverse = extractInverseKernel(gpuMatrix);

    return inverse;
}

// // Element-wise map function. Note that complex expressions for f likely won't compile as a GPU kernel. f shouldn't call other functions or use advanced JS features. Most Math functions should work.
// function map(A, lambda) { // lambda must not be anonymous
//     const n = A.length;

//     let func = function (x) {
//         return lambda(x);
//     }
    
//     // Define GPU kernel
//     const mapKernel = gpu.createKernel(function (A) {
//         return f(A[this.thread.y][this.thread.x]);
//     }).setOutput([n, n]).setFunctions([{
//         name: 'f',
//         source: func.toString(),
//         returnType: 'Number',
//         argumentTypes: ['Number']
//     }]);

//     // Apply kernel
//     return mapKernel(A);
// }

module.exports = {
    scale,
    add,
    sub,
    transpose,
    hadamard,
    dot,
    detJS,
    equals,
    solveLinearSystemSmall,
    solveLinearSystem,
    // inverse,
    // map,
};
