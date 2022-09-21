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