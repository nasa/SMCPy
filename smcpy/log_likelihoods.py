import numpy as np


def normal(inputs, model, data, args):
    std_dev = args
    if std_dev is None:
        std_dev = inputs[:, -1]
        inputs = inputs[:, :-1]
    var = std_dev ** 2

    output = model(inputs)
    ssqe = np.sum((output - data) ** 2, axis=1)

    term1 = -np.log(2 * np.pi * var) * (output.shape[1] / 2.) 
    term2 = -1 / 2. * ssqe / var
    return (term1 + term2)
