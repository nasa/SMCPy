import numpy as np


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
