import numpy as np


def normal(inputs, model, data, args):
    '''
    Likelihood function for data with additive, iid errors sampled from a
    normal distribution with mean = 0 and std_dev = args. If args is None,
    assumes that the last column of inputs contains the std_dev value.
    '''
    std_dev = args
    if std_dev is None:
        std_dev = inputs[:, -1]
        inputs = inputs[:, :-1]
    var = std_dev ** 2

    output = model(inputs)

    return _calc_normal_log_like(output, data, var)


def multisource_normal(inputs, model, data, args):
    '''
    Likelihood function for data with additive, iid errors sampled from N 
    normal distributions, each with mean = 0 and std_dev = args[1][n] for
    n = 1, ..., N. The number of data points sampled from each respective
    distribution should be provided in args[1]. Both args[0] and args[1] should
    be aligned and correspond to the ordering of data. That is, if
    args[0] = (2, 3, 1), then len(data) = 6 and the first two data points
    are assumed to be drawn from a Normal distribution w/ std_dev = args[1][0].
    If args[1][n] is None, assumes that the last M columns of inputs contains
    the std_dev values, where M is the total number of Nones in args[1]. The
    std_dev columns should be aligned with args[0] in this case.

    :param args: data segment lengths and corresponding standard deviations from
        Gaussian sample distribution
    :type args: list of two tuples
    '''
    if sum(args[0]) != data.shape[0]:
        raise ValueError("data segments in args[0] must sum to dim of data")

    output = model(inputs)

    args[1], inputs = _process_fixed_and_variable_std(args[1], inputs)

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


def _process_fixed_and_variable_std(std_devs, inputs):
    '''
    Identifies standard deviations to be estimated and pulls appropriate
    samples from the input array.
    '''
    num_nones = std_devs.count(None)

    new_std_devs = []
    for i, std in enumerate(std_devs):
        if std is None:
            new_std_devs.append(inputs[:, -num_nones + i])
        else:
            new_std_devs.append(std)

    new_inputs = inputs[:, :num_nones]

    return tuple(new_std_devs), new_inputs


def _calc_normal_log_like(output, data, var):
    ssqe = np.sum((output - data) ** 2, axis=1)

    term1 = -np.log(2 * np.pi * var) * (output.shape[1] / 2.) 
    term2 = -1 / 2. * ssqe / var

    return (term1 + term2)
