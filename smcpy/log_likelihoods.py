import numpy as np


class BaseLogLike:
    def __init__(self, model, data, args):
        self._model = model
        self._data = data
        self._args = args

        self.set_model_wrapper(lambda model, x: model(x))

    def _get_output(self, inputs):
        output = self._model_wrapper(self._model, inputs)
        if np.isnan(output).any():
            raise ValueError("nan in model output.")
        return output

    def set_model_wrapper(self, wrapper):
        self._model_wrapper = wrapper


class Normal(BaseLogLike):
    def __init__(self, model, data, args):
        """
        Likelihood function for data with additive, iid errors sampled from a
        normal distribution with mean = 0 and std_dev = args. If args is None,
        assumes that the last column of inputs contains the std_dev value.
        """
        super().__init__(model, data, args)

    def __call__(self, inputs):
        std_dev = self._args
        if std_dev is None:
            std_dev = inputs[:, -1]
            inputs = inputs[:, :-1]
        var = std_dev**2

        output = self._get_output(inputs)

        return self._calc_normal_log_like(output, self._data, var)

    @staticmethod
    def _calc_normal_log_like(output, data, var):
        ssqe = np.sum((output - data) ** 2, axis=1)

        term1 = -np.log(2 * np.pi * var) * (output.shape[1] / 2.0)
        term2 = -1 / 2.0 * ssqe / var

        return term1 + term2


class MultiSourceNormal(Normal):
    def __init__(self, model, data, args):
        """
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
        """
        super().__init__(model, data, args)
        self._num_nones = self._args[1].count(None)

        if sum(args[0]) != data.shape[0]:
            raise ValueError("data segments in args[0] must sum to dim of data")

    def __call__(self, inputs):
        std_devs, inputs = self._process_fixed_and_variable_std(inputs)

        output = self._get_output(inputs)

        log_likes = []
        start_idx = 0
        for segment_len, segment_std_dev in zip(*[self._args[0], std_devs]):
            if segment_len == 0:
                continue

            output_segment = output[:, start_idx : start_idx + segment_len]
            data_segment = self._data[start_idx : start_idx + segment_len]
            log_likes.append(
                self._calc_normal_log_like(
                    output_segment, data_segment, segment_std_dev**2
                )
            )
            start_idx += segment_len

        return np.sum(log_likes, axis=0)

    def _process_fixed_and_variable_std(self, inputs):
        """
        Identifies standard deviations to be estimated and pulls appropriate
        samples from the input array.
        """
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
            new_inputs = new_inputs[:, : -self._num_nones]

        return tuple(new_std_devs), new_inputs


class MVNormal(BaseLogLike):
    def __init__(self, model, data, args):
        """
        Likelihood function for data with additive, iid errors sampled from N-D
        multivariate normal distribution with mean = [0] * N and covariances =
        args[n] for n = 1, ..., N where, for example, a 4-D covariance
        matrix is defined as:

                      [0 1 2 3]
                cov = [1 4 5 6]  where 0, ..., 9 represent entries in args
                      [2 5 7 8]
                      [3 6 8 9]

        If args[n] is None, assumes that the last M columns of inputs
        contains covariance samples, where M is the total number of Nones in
        args (Nones do not have to be contiguous).

        Data array should have shape (number of snapshots, number of features)
        where snapshots are independent simultaneous measurements of all the
        data features. Features, for example, might be sensor readings at
        different locations. Model should output array with shape (number of
        samples/particles, number of features).

        :param args: covariances of the N-D multivariate normal
        :type args: list, len(args) = N * (N + 1) / 2
        """
        super().__init__(model, data, args)
        self._num_nones = self._args.count(None)

    def __call__(self, inputs):
        cov_arg_array, inputs = self._process_fixed_and_variable_covar(inputs)
        cov_matrices = np.tile(
            self._get_cov(cov_arg_array), (self._data.shape[0], 1, 1, 1)
        )

        data = np.expand_dims(self._data, 1)
        output = self._get_output(inputs)
        error = output - data
        error = np.expand_dims(error, 2)
        errorT = np.transpose(error, axes=(0, 1, 3, 2))

        term1 = -self._data.shape[1] / 2 * np.log(2 * np.pi)
        term2 = -1 / 2 * np.log(np.linalg.det(cov_matrices))
        term2 = np.expand_dims(term2, (2, 3))
        term3 = np.matmul(np.matmul(error, np.linalg.inv(cov_matrices)), errorT)

        log_likes = term1 + term2 + -(1 / 2) * term3
        log_likes = np.sum(log_likes, axis=0)

        return log_likes[:, :, 0]

    def _get_cov(self, cov_args):
        d = int((np.sqrt(1 + 8 * cov_args.shape[1]) - 1) / 2)
        covs = np.zeros((cov_args.shape[0], d, d))
        p, q = np.triu_indices(d)
        covs[:, p, q] = cov_args
        covs += np.transpose(np.triu(covs, 1), axes=[0, 2, 1])
        return covs

    def _process_fixed_and_variable_covar(self, inputs):
        covars = np.tile(self._args, (inputs.shape[0], 1))
        j = 0
        for i, arg in enumerate(self._args):
            if arg is None:
                covars[:, i] = inputs[:, -self._num_nones + j]
                j += 1

        new_inputs = inputs.copy()
        if self._num_nones > 0:
            new_inputs = new_inputs[:, : -self._num_nones]

        return covars, new_inputs


class MVNRandomEffects(BaseLogLike):
    def __init__(self, model, data, args):
        """
        Likelihood function for random effects.

        :param model: computational model
        :type model: callable
        :param data: data for the random effects with shape (num rand effects,
            num data features)
        :type data: 2D array
        :param args: total and random effects covariances and standard devs
        :type args: tuple (x, y)
        """

        super().__init__(model, data, args)

        self._randeffs = None
        self._init_randeffs()

        self._n_total_eff_nones = self._args[0].count(None)
        self._n_rand_eff_nones = self._args[1].count(None)
        self._n_nones = self._n_total_eff_nones + self._n_rand_eff_nones

    def __call__(self, inputs):
        processed_inputs = self._process_fixed_and_variable_std(inputs)

        total_eff_inputs = processed_inputs[0]
        rand_eff_inputs = processed_inputs[1]
        rand_eff_model_inputs = processed_inputs[2]

        total_eff_log_like = np.full((inputs.shape[0], 1), -np.inf)
        for i in range(inputs.shape[0]):
            reff_data_array = np.array([ni[i] for ni in rand_eff_model_inputs])
            total_eff = MVNormal(
                self._total_effects_model, reff_data_array, self._args[0]
            )

            in_ = total_eff_inputs[i].reshape(1, -1)
            total_eff_log_like[i, 0] = total_eff(in_).item()

        iterable = zip(rand_eff_inputs, self._randeffs)
        log_like = [np.c_[d(in_)] for in_, d in iterable]
        log_like.append(total_eff_log_like)

        return np.sum(log_like, axis=0)

    def set_model_wrapper(self, wrapper):
        self._model_wrapper = wrapper
        self._init_randeffs()

    def _init_randeffs(self):
        self._randeffs = []
        models = self._verify_model_type()
        for i, arg_i in enumerate(self._args[1]):
            self._randeffs.append(Normal(models[i], self._data[i], arg_i))
            self._randeffs[-1].set_model_wrapper(self._model_wrapper)

    def _verify_model_type(self):
        if isinstance(self._model, list):
            if len(self._args[1]) != len(self._model):
                raise ValueError(
                    "If multiple models provided, number must be ",
                    "equal to number of random effects.",
                )
            return self._model
        return [self._model] * len(self._args[1])

    def _process_fixed_and_variable_std(self, inputs):
        num_randeff = len(self._args[1])

        model_inputs = inputs
        if self._n_nones > 0:
            model_inputs = model_inputs[:, : -self._n_nones]
            like_args = inputs[:, -self._n_nones :]
            total_eff_args = like_args[:, : self._n_total_eff_nones]
            rand_eff_args = like_args[:, self._n_total_eff_nones :]

        split_inputs = np.array_split(model_inputs, num_randeff + 1, axis=1)
        total_eff_model_inputs = split_inputs[0]
        rand_eff_model_inputs = split_inputs[1:]

        total_eff_inputs = total_eff_model_inputs.copy()
        rand_eff_inputs = rand_eff_model_inputs.copy()

        if self._n_total_eff_nones > 0:
            total_eff_inputs = np.c_[total_eff_model_inputs, total_eff_args]

        if self._n_rand_eff_nones > 0:
            arg_idx = [i for i, val in enumerate(self._args[1]) if val == None]
            for i, j in enumerate(arg_idx):
                rand_eff_inputs[j] = np.c_[
                    rand_eff_model_inputs[j], rand_eff_args[:, i]
                ]

        return (total_eff_inputs, rand_eff_inputs, rand_eff_model_inputs)

    @staticmethod
    def _total_effects_model(inputs):
        return inputs
