import numpy as np

from scipy.stats import invwishart


class ImproperUniform:
    """
    Improper uniform prior distribution over the provided bounds. Sampling from
    this distribution is not possible and thus a proposal distribution should be
    used (see smcpy.proposals).
    """

    def __init__(self, lower_bound=None, upper_bound=None):
        self._lower = lower_bound
        self._upper = upper_bound

        if self._lower is None:
            self._lower = -np.inf

        if self._upper is None:
            self._upper = np.inf

    def logpdf(self, x):
        """
        :param x: input array
        :type x: 1D or 2D array; if 2D, must squeeze to 1D
        """
        array_x = np.array(x).squeeze()

        if array_x.ndim > 1:
            raise ValueError("Input array must be 1D or must squeeze to 1D")

        log_pdf = np.full(array_x.size, -np.inf)
        in_bounds = (array_x >= self._lower) & (array_x <= self._upper)
        return np.where(in_bounds, 0, log_pdf)


class InvWishart:
    def __init__(self, dof, scale):
        """
        :param scale: scale matrix
        :param dof: degrees of freedom (if == scale.shape[0], noninformative)
        """
        self._cov_dim = scale.shape[0]
        self._dim = int(self._cov_dim * (self._cov_dim + 1) / 2)
        self._invwishart = invwishart(dof, scale)

    @property
    def dim(self):
        return self._dim

    def rvs(self, num_samples, random_state=None):
        """
        :param num_samples: number of samples to return
        :type num_samples: int
        """
        cov = self._invwishart.rvs(num_samples, random_state=random_state)
        if num_samples == 1:
            cov = np.expand_dims(cov, 0)
        idx1, idx2 = np.triu_indices(self._cov_dim)
        return cov[:, idx1, idx2]

    def pdf(self, x):
        """
        :param x: input array
        :type x: 2D array (# samples, # unique covariances)
        """
        covs = self._assemble_covs(x)
        try:
            return self._invwishart.pdf(covs).reshape(-1, 1)
        except np.linalg.LinAlgError:
            return np.zeros((x.shape[0], 1))

    def logpdf(self, x):
        """
        :param x: input array
        :type x: 2D array (# samples, # unique covariances)
        """
        covs = self._assemble_covs(x)
        try:
            return self._invwishart.logpdf(covs).reshape(-1, 1)
        except np.linalg.LinAlgError:
            return np.full((x.shape[0], 1), -np.inf)

    def _assemble_covs(self, samples):
        covs = np.zeros((samples.shape[0], self._cov_dim, self._cov_dim))
        idx1, idx2 = np.triu_indices(self._cov_dim)
        covs[:, idx1, idx2] = samples
        covs += np.transpose(np.triu(covs, 1), axes=(0, 2, 1))
        return np.transpose(covs, axes=(1, 2, 0))


class ImproperCov:
    """
    Improper uniform prior distribution over all positive definite covariance
    matrices. If sampling using the rvs method, will return samples from an
    Inverse Wishart distribution with degrees of freedom dof and scale matrix S.
    """

    def __init__(self, num_cov_cols, dof=None, S=None):
        self._ncols = num_cov_cols
        self._dim = int(num_cov_cols * (num_cov_cols + 1) / 2)
        self._iw_dof = dof if dof is not None else num_cov_cols
        self._iw_scale = S if S is not None else np.eye(self._ncols)

    @property
    def dim(self):
        return self._dim

    def logpdf(self, samples):
        self._check_samples_shape(samples)
        covs = self._assemble_covs(samples)
        pdf = np.array([self._is_pos_semidef(x) for x in covs]).reshape(-1, 1)
        pdf = np.where(pdf == 0, -np.inf, pdf)
        pdf[pdf != -np.inf] = 0
        return pdf

    def rvs(self, num_samples, random_state=None):
        cov = invwishart(self._iw_dof, self._iw_scale).rvs(
            num_samples, random_state=random_state
        )
        idx1, idx2 = np.triu_indices(self._ncols)
        return cov[:, idx1, idx2]

    def _assemble_covs(self, samples):
        covs = np.zeros((samples.shape[0], self._ncols, self._ncols))
        idx1, idx2 = np.triu_indices(self._ncols)
        covs[:, idx1, idx2] = samples
        covs += np.transpose(np.triu(covs, 1), axes=(0, 2, 1))
        return covs

    @staticmethod
    def _is_pos_semidef(x):
        try:
            np.linalg.cholesky(x)
            return 1
        except np.linalg.LinAlgError:
            return 0

    def _check_samples_shape(self, cov):
        if len(cov.shape) != 2 or cov.shape[1] != self._dim:
            raise ValueError(
                "Covariance must be 3D array w/ index 0 "
                "representing the sample index and indices 1 and "
                "2 representing the 2D covariance indices (which "
                "should also be equal size)"
            )


class ImproperConstrainedUniform:
    def __init__(self, constraint_function, dim=None, bounds=None, max_rvs_tries=1000):
        """
        :param constraint_function: a function that takes in an array of
            parameters with shape = (num_samples, num_params) and returns
            a boolean array-like with length num_samples. Prior probability will
            be 1.0 if constraint returns True and 0.0 if False unless point
            is outside bounds then prior probability is 0.0. See "bounds"
            for additional information.
        :type constraint_function: callable
        :param dim: [optional] number of parameters; calculated from "bounds"
            if provided. Either "dim" or "bounds" must be provided.
        :type dim: int
        :param bounds: [optional] an array with shape (2, num_params) defining
            the lower and upper bounds for each parameter, respectively. If not
            provided, "dim" must be provided and only the constraint will be
            used to determine prior pdf (0 if False or 1 if True).
            Parameter vectors falling outside of the bounds will result in a
            prior pdf of 0.0 and 1.0 otherwise.
        :type bounds: 2D array
        :param max_rvs_tries: [optional; default=1000] number of tries to
            generate random samples that satisfy the provided constraint.
            Samples are generated uniformly within bounds and then pruned via
            rejection if violating the constraint; this process is repeated
            until the requested number of samples satisfying the constraint
            have been generated.
        """
        self._constraint_function = constraint_function
        self._bounds = bounds
        self._dim = dim
        self._max_rvs_tries = max_rvs_tries

        self._verify_inputs()

    def logpdf(self, samples):
        constr = self._constraint_function(samples).astype(int).reshape(-1, 1)
        if self._bounds is not None:
            bnd = self._are_within_bounds(samples)
            constr *= bnd
        constr = np.where(constr == 0, -np.inf, constr)
        constr[constr != -np.inf] = 0
        return constr

    def rvs(self, num_samples, random_state=None):
        self._raise_error_if_rng_is_not_numpy_generator(random_state)
        if self._bounds_are_finite():
            return self._get_constrained_samples(
                num_samples,
                rng=(np.random.default_rng() if random_state is None else random_state),
            )
        raise NotImplementedError("rvs cannot be used without finite bounds.")

    def _get_constrained_samples(self, num_samples, rng):
        samples = []
        tries = 0

        while len(samples) < num_samples and tries < self._max_rvs_tries:
            candidates = rng.uniform(*self._bounds, (num_samples, self.dim))
            samples += list(candidates[self._constraint_function(candidates)])
            tries += 1

        self._raise_error_if_less_than_requested(samples, num_samples)

        return np.array(samples)[:num_samples]

    def _are_within_bounds(self, samples):
        in_bounds = (self._bounds[0] <= samples).all(axis=1) * (
            self._bounds[1] >= samples
        ).all(axis=1)
        return in_bounds.astype(int).reshape(-1, 1)

    def _bounds_are_finite(self):
        if self._bounds is None:
            return False
        return (self._bounds > -np.inf).all() and (self._bounds < np.inf).all()

    @staticmethod
    def _raise_error_if_less_than_requested(samples, num_samples):
        if len(samples) < num_samples:
            raise ValueError(
                "Max rvs tries exceeded; " f"{len(samples)}/{num_samples} generated"
            )

    def _verify_inputs(self):
        if self._dim is None and self._bounds is None:
            raise ValueError('Either "bounds" or "dim" must be provided.')

    @property
    def dim(self):
        if self._dim:
            return self._dim
        return self._bounds.shape[1]

    @staticmethod
    def _raise_error_if_rng_is_not_numpy_generator(rng):
        if rng is not None and not isinstance(rng, np.random._generator.Generator):
            raise TypeError("Random number generator must be a numpy generator.")
        return None
