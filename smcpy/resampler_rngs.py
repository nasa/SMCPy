import numpy as np


def standard(mcmc_kernel, N):
    return mcmc_kernel.rng.uniform(0, 1, N)


def stratified(mcmc_kernel, N):
    return 1 / N * (np.arange(N) + mcmc_kernel.rng.uniform(0, 1, N))
