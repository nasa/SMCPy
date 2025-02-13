import numpy as np


def compute_intervals(output, alpha):
    if output.shape == (1, 0):
        return np.array([])
    intervals = []
    for col in output.T:
        x = np.linspace(0, 1, len(col))
        intervals.append(np.interp([1 - alpha / 2, alpha / 2], x, col))
    return np.array(intervals).T
