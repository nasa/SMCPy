import numpy as np


def compute_intervals(output, alpha):
    if alpha > 1 or alpha < 0:
        raise ValueError

    intervals = []
    for col in output.T:
        col.sort()
        x = np.linspace(0, 1, len(col))
        intervals.append(np.interp([1 - alpha / 2, alpha / 2], x, col))
    return np.array(intervals).T
