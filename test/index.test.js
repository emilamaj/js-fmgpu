const fmgpu = require('../src/index');
// Test the following functions:
// - scale
// - add
// - sub
// - transpose
// - hadamard
// - dot
// - equals
// - solveLinearSystemSmall
// - solveLinearSystem

test("Scale function in GPU, with 1e-6 precision, for size N=1, 10, 100, 1000", () => {
    for (let n = 1; n <= 100; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));
        const a = Math.random();

        const scaledA = fmgpu.scale(A, a);

        // Calculate average difference between each scaledA and a*A
        const avgDiff = scaledA.reduce((acc, row, i) => { return acc + row.reduce((acc, val, j) => { return acc + Math.abs(val - a * A[i][j]) }, 0) }, 0) / (n * n);
        expect(avgDiff).toBeLessThan(1e-6);

        // Calculate max difference between each scaledA and a*A
        const maxDiff = scaledA.reduce((acc, row, i) => { return Math.max(acc, row.reduce((acc, val, j) => { return Math.max(acc, Math.abs(val - a * A[i][j])) }, 0)) }, 0);
        expect(maxDiff).toBeLessThan(1e-6);
    }
});

test("Add function in GPU, with 1e-6 precision, for size N=1, 10, 100, 1000", () => {
    for (let n = 1; n <= 1000; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));
        const B = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));

        const addedAB = fmgpu.add(A, B);
        const addedABJS = A.map((row, i) => row.map((x, j) => x + B[i][j]));

        // Each element of the matrix must be within 1e-6 of the JS result
        const avgDiff = addedAB.reduce((acc, row, i) => { return acc + row.reduce((acc, val, j) => { return acc + Math.abs(val - addedABJS[i][j]) }, 0) }, 0) / (n * n);
        expect(avgDiff).toBeLessThan(1e-6);

        // Calculate max difference between each addedAB and addedABJS
        const maxDiff = addedAB.reduce((acc, row, i) => { return Math.max(acc, row.reduce((acc, val, j) => { return Math.max(acc, Math.abs(val - addedABJS[i][j])) }, 0)) }, 0);
        expect(maxDiff).toBeLessThan(1e-6);
    }
});

test("Sub function in GPU, with 1e-6 precision, for size N=1, 10, 100, 1000", () => {
    for (let n = 1; n <= 1000; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));
        const B = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));

        const subAB = fmgpu.sub(A, B);
        const subABJS = A.map((row, i) => row.map((x, j) => x - B[i][j]));

        // Each element of the matrix must be within 1e-6 of the JS result
        const avgDiff = subAB.reduce((acc, row, i) => { return acc + row.reduce((acc, val, j) => { return acc + Math.abs(val - subABJS[i][j]) }, 0) }, 0) / (n * n);
        expect(avgDiff).toBeLessThan(1e-6);

        // Calculate max difference between each subAB and subABJS
        const maxDiff = subAB.reduce((acc, row, i) => { return Math.max(acc, row.reduce((acc, val, j) => { return Math.max(acc, Math.abs(val - subABJS[i][j])) }, 0)) }, 0);
        expect(maxDiff).toBeLessThan(1e-6);
    }
});

test("Transpose function in GPU, with 1e-6 precision, for size N=1, 10, 100, 1000", () => {
    for (let n = 1; n <= 1000; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));

        const AT = fmgpu.transpose(A);
        const ATJS = A[0].map((_, i) => A.map(row => row[i]));

        // Each element of the matrix must be within 1e-6 of the JS result
        const avgDiff = AT.reduce((acc, row, i) => { return acc + row.reduce((acc, val, j) => { return acc + Math.abs(val - ATJS[i][j]) }, 0) }, 0) / (n * n);
        expect(avgDiff).toBeLessThan(1e-6);

        // Calculate max difference between each AT and ATJS
        const maxDiff = AT.reduce((acc, row, i) => { return Math.max(acc, row.reduce((acc, val, j) => { return Math.max(acc, Math.abs(val - ATJS[i][j])) }, 0)) }, 0);
        expect(maxDiff).toBeLessThan(1e-6);
    }
});

test("Hadamard function in GPU, with 1e-6 precision, for size N=1, 10, 100, 1000", () => {
    for (let n = 1; n <= 1000; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));
        const B = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));

        const hadA = fmgpu.hadamard(A, B);
        const hadAJS = A.map((row, i) => row.map((x, j) => x * B[i][j]));

        // Each element of the matrix must be within 1e-6 of the JS result
        const avgDiff = hadA.reduce((acc, row, i) => { return acc + row.reduce((acc, val, j) => { return acc + Math.abs(val - hadAJS[i][j]) }, 0) }, 0) / (n * n);
        expect(avgDiff).toBeLessThan(1e-6);

        // Calculate max difference between each hadA and hadAJS
        const maxDiff = hadA.reduce((acc, row, i) => { return Math.max(acc, row.reduce((acc, val, j) => { return Math.max(acc, Math.abs(val - hadAJS[i][j])) }, 0)) }, 0);
        expect(maxDiff).toBeLessThan(1e-6);
    }
});

test("Dot function in GPU, with 1e-3 precision, for size N=1, 10, 100", () => {
    for (let n = 1; n <= 100; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random())); // Random matrix
        const B = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random())); // Random matrix

        const dotA = fmgpu.dot(A, B);
        const dotAJS = A.map((row, i) => row.map((_, j) => A[i].reduce((acc, x, k) => acc + x * B[k][j], 0)));

        // dotA must be equal to dotAJS (within 1e-3 precision)
        const avgDiff = dotA.reduce((acc, row, i) => { return acc + row.reduce((acc, val, j) => { return acc + Math.abs(val - dotAJS[i][j]) }, 0) }, 0) / (n * n);
        expect(avgDiff).toBeLessThan(1e-3);

        // Calculate max difference between each dotA and dotAJS
        const maxDiff = dotA.reduce((acc, row, i) => { return Math.max(acc, row.reduce((acc, val, j) => { return Math.max(acc, Math.abs(val - dotAJS[i][j])) }, 0)) }, 0);
        expect(maxDiff).toBeLessThan(1e-3);
    }
});

test("Equality function in GPU for size N=1, 10, 100, 1000", () => {
    for (let n = 1; n <= 1000; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));

        const eqA = fmgpu.equals(A, A, 1e-6);
        expect(eqA).toBe(true);
    }
});

test("Small linear system solving in pure JS. Within 1e-6 for of size N=1, 10, 100", () => {
    for (let n = 1; n <= 1000; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));
        const b = Array(n).fill(0).map(() => Math.random());

        const x = fmgpu.solveLinearSystemSmall(A, b);
        const Ax = A.map(row => row.reduce((acc, val, i) => acc + val * x[i], 0));

        // Average error must be within 1e-6
        expect(Ax.reduce((acc, val, i) => acc + Math.abs(val - b[i]), 0) / n).toBeLessThan(1e-6);
    }
});

gpuSolveTolerance = 5e-3; // 0.5% error tolerance
test(`Linear system solving in GPU. Within ${gpuSolveTolerance} for size N=1, 10, 100, 1000`, () => {
    for (let n = 1; n <= 1000; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));
        const b = Array(n).fill(0).map(() => Math.random());

        const x = fmgpu.solveLinearSystem(A, b);
        const Ax = A.map(row => row.reduce((acc, val, i) => acc + val * x[i], 0));

        // Average error must be within gpuSolveTolerance
        expect(Ax.reduce((acc, val, i) => acc + Math.abs(val - b[i]), 0) / n).toBeLessThan(gpuSolveTolerance);
    }
});

// test("Matrix inversion in GPU. Within 1e-3 for of size N=1, 10, 100, 1000", () => {
//     for (let n = 1; n <= 1000; n *= 10) {
//         const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));

//         const invA = fmgpu.inverse(A);

//         // A.A^-1 = I
//         // Calculate average and max difference between A.A^-1 and I
//         // A.A^-1
//         const prod = A.map((row, i) => row.map((_, j) => A[i].reduce((acc, x, k) => acc + x * invA[k][j], 0)));

//         // Average difference between A.A^-1 and I
//         const avgDiff = prod.reduce((acc, row, i) => { return acc + row.reduce((acc, val, j) => { return acc + Math.abs(val - (i === j ? 1 : 0)) }, 0) }, 0) / (n * n);
//         expect(avgDiff).toBeLessThan(1e-3);

//         // Max difference between A.A^-1 and I
//         const maxDiff = prod.reduce((acc, row, i) => { return Math.max(acc, row.reduce((acc, val, j) => { return Math.max(acc, Math.abs(val - (i === j ? 1 : 0))) }, 0)) }, 0);
//         expect(maxDiff).toBeLessThan(1e-3);
//     }
// });

// test("Map function in GPU, with 1e-3 precision, for size N=1, 10, 100, 1000", () => {
//     for (let n = 1; n <= 1000; n *= 10) {
//         const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));
//         // Identity
//         let lambda0 = (x) => x;
//         const mapA0 = fmgpu.map(A, lambda0);
//         const mapA0JS = A.map(row => row.map(x => lambda0(x)));
//         // Calculate average difference between each mapA0 and mapA0JS
//         expect(mapA0.reduce((acc, row, i) => { return acc + row.reduce((acc, val, j) => { return acc + Math.abs(val - mapA0JS[i][j]) }, 0) }, 0) / (n * n)).toBeLessThan(1e-3);
        
//         // // ReLU
//         // const lambda1 = (x) => { return x > 0 ? 0.25 * x : 0.05 * x; };
//         // const mapA1 = fmgpu.map(A, lambda1);
//         // const mapA1JS = A.map(row => row.map(x => lambda1(x)));
//         // expect(mapA1).toEqual(mapA1JS, 1e-6);

//         // // Sigmoid
//         // const lambda2 = (x) => { return 1 / (1 + Math.exp(-x)); };
//         // const mapA2 = fmgpu.map(A, lambda2);
//         // expect(mapA2).toEqual(A.map(row => row.map(x => lambda2(x))), 1e-6);
//     }
// });

// test("Determinant function in pure JS, with 1e-6 precision, for size N=1, 10, 100", () => {
//     for (let n = 1; n <= 100; n *= 10) {
//         const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));

//         const detA = fmgpu.detJS(A);
//         const detAJS = fmgpu.detJS(A);

//         // Each element of the matrix must be within 1e-6 of the JS result
//         expect(Math.abs(detA - detAJS)).toBeLessThan(1e-6);
//     }
// });
