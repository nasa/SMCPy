from cupyx import jit
import cupy as cp
import numpy as np
import smcpy.utils.global_imports as gi

def matmul_gpu_with_lower_triangular(lower, mat):
    if lower.shape[-1] != mat.shape[0]:
        raise np.linalg.LinAlgError(f"Matrix dimensions {mat.shape} and {lower.shape} "
                                    f"incompatible for multiplication.")

    num_rows = lower.shape[0]
    num_cols = mat.shape[1]
    mid_dim = lower.shape[-1]
    output = gi.num_lib.zeros((num_rows, num_cols))
    blockspergrid = int(np.ceil(num_cols / gi.GPU_THREADS_PER_BLOCK))
    _matmul_lt_gpu_kernel[blockspergrid, gi.GPU_THREADS_PER_BLOCK](lower, mat, output, num_rows, num_cols, mid_dim)
    return output


@jit.rawkernel()
def _matmul_lt_gpu_kernel(lower, mat, output, num_rows, num_cols, mid_dim):
    col = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    if col < num_cols:
        for row in range(num_rows):
            for i in range(row + 1):
                output[row, col] += lower[row, i] * mat[i, col]