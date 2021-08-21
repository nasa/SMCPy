import numpy as num_lib
gpu_matmul = None

USING_GPU = False

def set_use_gpu(use_gpu):
    global USING_GPU
    global num_lib
    global gpu_matmul

    USING_GPU = use_gpu

    if use_gpu:
        import cupy as num_lib
        import smcpy.utils.gpu.matmul_kernel as gpu_matmul
    else:
        import numpy as num_lib


GPU_THREADS_PER_BLOCK = 256