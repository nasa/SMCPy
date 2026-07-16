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


class MVNormalEmulatorUncertainty(BaseLogLike):
    def __init__(self, model, data, args):
        """
        Multivariate-normal likelihood for a surrogate/emulator model that
        returns BOTH a predictive mean and a predictive covariance for each
        input (particle).

        The emulator (e.g., a GP) outputs, for each particle, a mean vector
        mu_GP (length d) and a d x d predictive covariance Sigma_GP. This is
        combined with a measurement covariance Sigma_meas via:

            y_obs | theta ~ N( mu_GP(theta),  Sigma_meas + Sigma_GP(theta) )

        valid when measurement error and emulator error are independent.

        Measurement covariance handling (via `args`):
          - args is a scalar (float/int) -> interpreted as a standard
                                             deviation; Sigma_meas = args^2 * I_d.
          - args is a (d, d) matrix       -> used directly as the full
                                             measurement covariance Sigma_meas
                                             (e.g., noise projected into PC
                                             space, generally non-diagonal).
          - args is None                  -> the measurement noise std_dev is
                                             ESTIMATED as an SMC parameter. The
                                             LAST column of `inputs` is taken as
                                             std_dev, and Sigma_meas =
                                             std_dev^2 * I_d.

        Shapes:
          data : (1, d)                          # single measurement snapshot
          model returns:
            mean : (num_particles, d)
            cov  : (num_particles, d, d)          # Sigma_GP

        :param args: scalar std dev (applied as args**2 * I_d), a (d, d)
            covariance matrix (used directly), or None to estimate an
            isotropic measurement std_dev from the last input column.
        """
        super().__init__(model, data, args)

        self._d = self._data.shape[0]
        self._estimate_meas = self._args is None

        if self._estimate_meas:
            self._meas_cov = None            # built per-particle in __call__
        elif np.isscalar(self._args):
            # scalar std dev -> Sigma_meas = args^2 * I_d
            self._meas_cov = float(self._args) ** 2 * np.eye(self._d)
        else:
            # full covariance matrix, used directly
            self._meas_cov = np.asarray(self._args, dtype=float)
            if self._meas_cov.shape != (self._d, self._d):
                raise ValueError(
                    f"matrix args must be ({self._d}, {self._d}), "
                    f"got {self._meas_cov.shape}"
                )
            if not np.allclose(self._meas_cov, self._meas_cov.T):
                raise ValueError("matrix args (covariance) must be symmetric.")

    def _get_output(self, inputs):
        """Emulator returns (mean, cov). Validate and return both."""
        mean, cov = self._model_wrapper(self._model, inputs)
        if np.isnan(mean).any() or np.isnan(cov).any():
            raise ValueError("nan in model output.")
        return np.asarray(mean), np.asarray(cov)

    def _build_meas_cov(self, inputs):
        """
        Returns (meas_cov, model_inputs).
          - scalar/matrix case: meas_cov shape (1, d, d), broadcasts over
            particles; inputs passed through unchanged.
          - estimate case: strips last input column as std_dev, builds
            per-particle covariance std_dev^2 * I_d, shape (P, d, d).
        """
        if not self._estimate_meas:
            return self._meas_cov[None, :, :], inputs

        std_dev = inputs[:, -1]                      # (P,)
        model_inputs = inputs[:, :-1]                # drop std_dev column
        var = std_dev ** 2                           # (P,)
        eye = np.eye(self._d)[None, :, :]            # (1, d, d)
        meas_cov = var[:, None, None] * eye          # (P, d, d)
        return meas_cov, model_inputs

    def __call__(self, inputs):
        # ------------------------------------------------------------------
        # 0. Build measurement covariance (and strip std_dev if estimating).
        # ------------------------------------------------------------------
        meas_cov, model_inputs = self._build_meas_cov(inputs)

        # ------------------------------------------------------------------
        # 1. Emulator prediction: mean (P, d) and GP covariance (P, d, d)
        # ------------------------------------------------------------------
        mu_gp, sigma_gp = self._get_output(model_inputs)   # (P, d), (P, d, d)
        num_particles, d = mu_gp.shape

        # ------------------------------------------------------------------
        # 2. Total covariance = Sigma_meas + Sigma_GP   (per particle)
        # ------------------------------------------------------------------
        total_cov = meas_cov + sigma_gp                    # (P, d, d)

        inv_cov = np.linalg.inv(total_cov)                 # (P, d, d)
        sign, logdet = np.linalg.slogdet(total_cov)        # (P,), (P,)
        if np.any(sign <= 0):
            raise ValueError("total covariance is not positive definite.")

        # ------------------------------------------------------------------
        # 3. Residual: single data snapshot (1, d) vs emulator mean (P, d).
        # ------------------------------------------------------------------
        error = self._data - mu_gp                          # (P, d)

        # ------------------------------------------------------------------
        # 4. Quadratic form: error^T Sigma^-1 error per particle.
        # ------------------------------------------------------------------
        tmp = np.einsum("pij,pj->pi", inv_cov, error)       # (P, d)
        quad = np.einsum("pi,pi->p", error, tmp)            # (P,)

        # ------------------------------------------------------------------
        # 5. MVN log density.
        # ------------------------------------------------------------------
        term1 = -0.5 * d * np.log(2 * np.pi)                # scalar
        term2 = -0.5 * logdet                               # (P,)
        term3 = -0.5 * quad                                 # (P,)

        return term1 + term2 + term3                        # (P,)