import numpy as np

from scipy.stats import invwishart


class ImproperUniform:

    def __init__(self, lower_bound=None, upper_bound=None):
        self._lower = lower_bound
        self._upper = upper_bound

        if self._lower is None:
            self._lower = -np.inf

        if self._upper is None:
            self._upper = np.inf

    def pdf(self, x):
        '''
        :param x: input array
        :type x: 1D or 2D array; if 2D, must squeeze to 1D
        '''
        array_x = np.array(x).squeeze()

        if array_x.ndim > 1:
            raise ValueError('Input array must be 1D or must squeeze to 1D')

        prior_pdf = np.zeros(array_x.size)
        in_bounds = (array_x >= self._lower) & (array_x <= self._upper)
        return np.add(prior_pdf, 1, out=prior_pdf, where=in_bounds)



class InvWishart:

    def __init__(self, dof, scale):
        '''
        :param scale: scale matrix
        :param dof: degrees of freedom (if == scale.shape[0], noninformative)
        '''
        self._cov_dim = scale.shape[0]
        self._dim = int(self._cov_dim * (self._cov_dim + 1) / 2)
        self._invwishart = invwishart(dof, scale)

    @property
    def dim(self):
        return self._dim

    def rvs(self, num_samples):
        '''
        :param num_samples: number of samples to return
        :type num_samples: int
        '''
        cov = self._invwishart.rvs(num_samples)
        idx1, idx2 = np.triu_indices(self._cov_dim)
        return cov[:, idx1, idx2]

    def pdf(self, x):
        '''
        :param x: input array
        :type x: 2D array (# samples, # unique covariances)
        '''
        covs = self._assemble_covs(x)
        try:
            return self._invwishart.pdf(covs)
        except np.linalg.LinAlgError:
            return np.zeros((x.shape[0], 1))

    def _assemble_covs(self, samples):
        covs = np.zeros((samples.shape[0], self._cov_dim, self._cov_dim))
        idx1, idx2 = np.triu_indices(self._cov_dim)
        covs[:, idx1, idx2] = samples
        covs += np.transpose(np.triu(covs, 1), axes=(0, 2, 1))
        return np.transpose(covs, axes=(1, 2, 0))

class ImproperCov:
    '''
    Improper uniform prior distribution over all positive definite covariance
    matrices. If sampling using the rvs method, will return samples from an
    Inverse Wishart distribution with degrees of freedom dof and scale matrix S.
    '''
    def __init__(self, num_cov_cols, dof=None, S=None):
        self._ncols = num_cov_cols
        self._dim = int(num_cov_cols * (num_cov_cols + 1) / 2)
        self._iw_dof = dof if dof is not None else num_cov_cols
        self._iw_scale = S if S is not None else np.eye(self._ncols)

    @property
    def dim(self):
        return self._dim

    def pdf(self, samples):
        self._check_samples_shape(samples)
        covs = self._assemble_covs(samples)
        pdf = np.array([self._is_pos_semidef(x) for x in covs]).reshape(-1, 1)
        return pdf

    def rvs(self, num_samples):
        cov = invwishart(self._iw_dof, self._iw_scale).rvs(num_samples)
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
        except np.linalg.linalg.LinAlgError:
            return 0

    def _check_samples_shape(self, cov):
        if len(cov.shape) != 2 or cov.shape[1] != self._dim:
            raise ValueError('Covariance must be 3D array w/ index 0 '
                             'representing the sample index and indices 1 and '
                             '2 representing the 2D covariance indices (which '
                             'should also be equal size)')


class ConstrainedUniform:

    def __init__(self, constraint_function, dim=None, bounds=None, seed=None,
                 max_rvs_tries=1000):
        self._constraint_function = constraint_function
        self._bounds = bounds
        self._dim = dim
        self._seed = seed
        self._max_rvs_tries = max_rvs_tries

        self._verify_inputs()

    def pdf(self, samples):
        constr = self._constraint_function(samples).astype(int).reshape(-1, 1)
        if self._bounds is not None:
            bnd = self._are_within_bounds(samples)
            constr *= bnd
        return constr

    def rvs(self, num_samples):
        if self._bounds_are_finite():
            return self._get_constrained_samples(num_samples)
        raise NotImplementedError('rvs cannot be used without finite bounds.')

    def _get_constrained_samples(self, num_samples):
        rng = np.random.default_rng(seed=self._seed)
        samples = []
        tries = 0

        while len(samples) < num_samples and tries < self._max_rvs_tries:
            candidates = rng.uniform(*self._bounds, (num_samples, self.dim))
            samples += list(candidates[self._constraint_function(candidates)])
            tries += 1

        self._raise_error_if_less_than_requested(samples, num_samples)

        return np.array(samples)[:num_samples]

    def _are_within_bounds(self, samples):
        in_bounds = (self._bounds[0] <= samples).all(axis=1) * \
                    (self._bounds[1] >= samples).all(axis=1)
        return in_bounds.astype(int).reshape(-1, 1)

    def _bounds_are_finite(self):
        if self._bounds is None:
            return False
        return (self._bounds > -np.inf).all() and (self._bounds < np.inf).all()

    @staticmethod
    def _raise_error_if_less_than_requested(samples, num_samples):
        if len(samples) < num_samples:
            raise ValueError('Max rvs tries exceeded; '
                             f'{len(samples)}/{num_samples} generated')


    def _verify_inputs(self):
        if self._dim is None and self._bounds is None:
            raise ValueError('Either "bounds" or "dim" must be provided.')

    @property
    def dim(self):
        if self._dim:
            return self._dim
        return self._bounds.shape[1]
