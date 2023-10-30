import abc
import numpy as np

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class PathBase:

    def __init__(self):
        self._phi = 0

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi <= self._phi: 
            raise ValueError('phi updates must be monotonic.')
        self._phi = phi

    @abc.abstractmethod
    def __call__(self, inputs, log_like, log_prior):
        return None


class GeometricPath(PathBase):

    def __init__(self, proposal=None):
        super().__init__()
        self._proposal = proposal

    def __call__(self, inputs, log_like, log_prior):
        log_p = self._proposal.log_pdf(inputs) if self._proposal else log_prior
        return np.sum(np.hstack((
            log_like * self.phi,
            log_prior * min(1, self.phi),
            log_p * max(0, 1 - self.phi)
            )), axis=1)