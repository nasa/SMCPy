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

    def __init__(self, cov_dim):
        self._cov_dim = cov_dim
        self._dim = cov_dim * (cov_dim + 1) / 2
        self._invwishart = invwishart(cov_dim, np.eye(cov_dim))

    @property
    def dim(self):
        return self._dim

    def rvs(self, num_samples):
        cov = self._invwishart.rvs(num_samples)
        idx1, idx2 = np.triu_indices(self._cov_dim)
        return cov[:, idx1, idx2]

    def pdf(self, x):
        '''
        :param x: input array
        :type x: 2D array (# samples, # unique covariances)
        '''
        covs = self._assemble_covs(x)
        return self._invwishart.pdf(covs)

    def _assemble_covs(self, samples):
        covs = np.zeros((samples.shape[0], self._cov_dim, self._cov_dim))
        idx1, idx2 = np.triu_indices(self._cov_dim)
        covs[:, idx1, idx2] = samples
        covs += np.transpose(np.triu(covs, 1), axes=(0, 2, 1))
        return np.transpose(covs, axes=(1, 2, 0))
