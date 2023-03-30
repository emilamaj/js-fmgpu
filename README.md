# FMGPU: Fast Matrix operations of the GPU

This library allows to efficiently carry out matrix operations using GPU acceleration.
The following functions are currently available:
## scale
Multiply input matrix by a scalar value.
`B = fmgpu.scale(A, alpha)`

## add
Add two matrices together.
`C = fmgpu.add(A, B)`

## sub
Subtract second matrix from first one
`C = fmgpu.sub(A, B)`

## transpose
Transpose the given matrix
`B = fmgpu.transpose(A)`

## hadamard
Returns Hadamard product, also known as pointwise product between two matrices
`C = fmgpu.hadamard(A, B)`

## dot
Returns the dot product between two matrices.
`C = fmgpu.dot(A, B)`

## detJS (pure JS)
Returns the determinant of a given matrix. Note that this function is not yet GPU-accelerated.
`B = fmgpu.detJS(A)`

## equals
Returns `true` if two matrices are equal (within some epsilon value)
`eq = fmgpu.equals(A, B, 1e-6)`

## solveLinearSystemSmall (pure JS)
Solves a linear system of equations using pure JS code (less overhead, better for smaller matrices, of size < 200)
`x = fmgpu.solveLinearSystemSmall(A, b)`

## solveLinearSystem
Solves a linear system of equations using GPU-accelerated code. (Adapted for matrices of size N > 200)
`x = fmgpu.solveLinearSystem(A, b)`

# Todo
The following functions are to be implemented next:
-det (GPU accelerated)
-map
-inverse
