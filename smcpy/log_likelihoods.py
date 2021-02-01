import numpy as np


class Normal:

    def __init__(self, model, data, args):
        '''
        Likelihood function for data with additive, iid errors sampled from a
        normal distribution with mean = 0 and std_dev = args. If args is None,
        assumes that the last column of inputs contains the std_dev value.
        '''
        self._model = model
        self._data = data
        self._args = args

    def __call__(self, inputs):
        std_dev = self._args
        if std_dev is None:
            std_dev = inputs[:, -1]
            inputs = inputs[:, :-1]
        var = std_dev ** 2

        output = self._model(inputs)

        return self._calc_normal_log_like(output, self._data, var)

    @staticmethod
    def _calc_normal_log_like(output, data, var):
        ssqe = np.sum((output - data) ** 2, axis=1)
    
        term1 = -np.log(2 * np.pi * var) * (output.shape[1] / 2.) 
        term2 = -1 / 2. * ssqe / var
    
        return (term1 + term2)


class MultiSourceNormal(Normal):

    def __init__(self, model, data, args):
        '''
        Likelihood function for data with additive, iid errors sampled from N 
        normal distributions, each with mean = 0 and std_dev = args[1][n] for
        n = 1, ..., N. The number of data points sampled from each respective
        distribution should be provided in args[1]. Both args[0] and args[1]
        should be aligned and correspond to the ordering of data. That is, if
        args[0] = (2, 3, 1), then len(data) = 6 and the first two data points
        are assumed to be drawn from a Normal distribution w/
        std_dev = args[1][0]. If args[1][n] is None, assumes that the last M
        columns of inputs contains the std_dev values, where M is the total
        number of Nones in args[1]. The std_dev columns should be aligned with
        args[0] in this case.

        :param args: data segment lengths and corresponding standard deviations
            N from Gaussian distribution
        :type args: list of two tuples, each of length N
        '''
        self._model = model
        self._data = data
        self._args = args
        self._num_nones = self._args[1].count(None)

        if sum(args[0]) != data.shape[0]:
            raise ValueError("data segments in args[0] must sum to dim of data")

    def __call__(self, inputs):

        std_devs, inputs = self._process_fixed_and_variable_std(inputs)

        output = self._model(inputs)

        log_likes = []
        start_idx = 0
        for segment_len, segment_std_dev in zip(*[self._args[0], std_devs]):

            if segment_len == 0:
                continue

            output_segment = output[:, start_idx : start_idx + segment_len]
            data_segment = self._data[start_idx : start_idx + segment_len]
            log_likes.append(self._calc_normal_log_like(output_segment,
                                                        data_segment,
                                                        segment_std_dev ** 2))
            start_idx += segment_len

        return np.sum(log_likes, axis=0)


    def _process_fixed_and_variable_std(self, inputs):
        '''
        Identifies standard deviations to be estimated and pulls appropriate
        samples from the input array.
        '''
        std_devs = self._args[1]
        new_std_devs = []
        j = 0
        for i, std in enumerate(std_devs):
            if std is None:
                new_std_devs.append(inputs[:, -self._num_nones + j])
                j += 1
            else:
                new_std_devs.append(std)
    
        new_inputs = inputs.copy()
        if self._num_nones > 0:
            new_inputs = new_inputs[:, :-self._num_nones]
    
        return tuple(new_std_devs), new_inputs
