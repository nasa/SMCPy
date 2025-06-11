import numpy as np


def generate_noisy_data(data, std, seed=None):
    is_2d = data.ndim == 2
    if not is_2d:
        raise ValueError

    noise = np.random.default_rng(seed=seed).normal(0, std, data.T.shape).T
    return noise + data
