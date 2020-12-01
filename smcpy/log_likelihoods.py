import numpy as np


def normal(inputs, model, data, args):
    std_dev = args
    if std_dev is None:
        std_dev = inputs[:, -1]
        inputs = inputs[:, :-1]
    var = std_dev ** 2

    output = model(inputs)

    return _calc_normal_log_like(output, data, var)


def multisource_normal(inputs, model, data, args):
    output = model(inputs)

    log_likes = []
    start_idx = 0
    for segment_len, segment_std_dev in zip(*args):

        if segment_len == 0:
            continue

        output_segment = output[:, start_idx : start_idx + segment_len]
        data_segment = data[start_idx : start_idx + segment_len]
        log_likes.append(_calc_normal_log_like(output_segment, data_segment,
                                               segment_std_dev ** 2))
        start_idx += segment_len

    return np.sum(log_likes, axis=0)


def _calc_normal_log_like(output, data, var):
    ssqe = np.sum((output - data) ** 2, axis=1)
    term1 = -np.log(2 * np.pi * var) * (output.shape[1] / 2.) 
    term2 = -1 / 2. * ssqe / var
    return (term1 + term2)
