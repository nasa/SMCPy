import numpy as np


def generate_noisy_data(std, output):
    is_2d = output.ndim == 2
    if not is_2d:
        raise ValueError
    noisy_data = np.random.normal(0, std, output.T.shape).T
    shifted_to_output = noisy_data + output
    return shifted_to_output
