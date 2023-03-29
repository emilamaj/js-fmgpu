const { solveLinearSystemJS, solveLinearSystemGPU } = require('../src/index');

test("Linear system solving in pure JS. Within 1e-6 for of size N=1, 10, 100", () => {
    for (let n = 1; n <= 100; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));
        const b = Array(n).fill(0).map(() => Math.random());
        
        const x = solveLinearSystemJS(A, b);
        const Ax = A.map(row => row.reduce((acc, val, i) => acc + val * x[i], 0));

        // Average error must be within 1e-6
        expect(Ax.reduce((acc, val, i) => acc + Math.abs(val - b[i]), 0) / n).toBeLessThan(1e-6);
    }
});

test("Linear system solving in GPU. Within 1e-3 for size N=1, 10, 100, 1000", () => {
    for (let n = 1; n <= 1000; n *= 10) {
        const A = Array(n).fill(0).map(() => Array(n).fill(0).map(() => Math.random()));
        const b = Array(n).fill(0).map(() => Math.random());
        
        const x = solveLinearSystemGPU(A, b);
        const Ax = A.map(row => row.reduce((acc, val, i) => acc + val * x[i], 0));

        // Average error must be within 1e-3
        expect(Ax.reduce((acc, val, i) => acc + Math.abs(val - b[i]), 0) / n).toBeLessThan(1e-3);
    }
});