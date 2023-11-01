import abc
import numpy as np

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class PathBase:

    def __init__(self, proposal):
        self._phi = 0
        self._previous_phi = None
        self._proposal = proposal

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._previous_phi = self.phi
        if phi <= self._previous_phi: 
            raise ValueError('phi updates must be monotonic.')
        self._phi = phi

    @property
    def delta_phi(self):
        return self.phi - self._previous_phi

    @property
    def proposal(self):
        return self._proposal

    @abc.abstractmethod
    def log_pdf(self, inputs, log_like, log_prior):
        return None

    @abc.abstractmethod
    def inc_weights(self, inputs, log_like, log_prior, delta_phi):
        return None


class GeometricPath(PathBase):

    def __init__(self, proposal=None, required_phi=1):
        super().__init__(proposal)
        self._lambda = None
        self.required_phi_list = required_phi

    @property
    def required_phi_list(self):
        return self._required_phi_list

    @required_phi_list.setter
    def required_phi_list(self, phi):
        if isinstance(phi, float) or isinstance(phi, int):
            phi = [phi]
        self._required_phi_list = sorted([p for p in phi if p < 1])
        self._lambda = min(self._required_phi_list + [1])

    def log_pdf(self, inputs, log_like, log_prior):
        log_p = self._proposal.log_pdf(inputs) if self._proposal else log_prior
        return np.sum(np.hstack((
            log_like * self.phi,
            log_prior * min(1, self.phi / self._lambda),
            log_p * max(0, (self._lambda - self.phi) / self._lambda)
        )), axis=1)

    def inc_weights(self, inputs, log_like, log_prior, delta_phi):
        log_p = self._proposal.log_pdf(inputs) if self._proposal else log_prior
        return np.sum(np.hstack((
            log_like, log_prior, -log_p
        )), axis=1, keepdims=True) * delta_phi