import abc
import numpy as np

from ..paths import GeometricPath

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta("ABC", (object,), {"__slots__": ()})


class MCMCKernel(ABC):

    def __init__(self, mcmc_object, param_order, path, rng=None):
        self._mcmc = mcmc_object
        self._param_order = param_order
        self.path = GeometricPath() if path is None else path
        self.rng = np.random.default_rng() if rng is None else rng

    def has_proposal(self):
        return self.path.proposal

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng):
        self.set_mcmc_rng(rng)
        self._rng = rng

    @abc.abstractmethod
    def mutate_particles(self, particles, num_samples, proposal_cov, phi):
        return new_param_values, new_log_likelihoods

    @abc.abstractmethod
    def sample_from_prior(self, num_samples):
        return param_values

    @abc.abstractmethod
    def sample_from_proposal(self, num_samples):
        return param_values

    @abc.abstractmethod
    def get_log_likelihoods(self, param_dict):
        return log_likelihoods

    @abc.abstractmethod
    def get_log_priors(self, param_dict):
        return log_priors

    @abc.abstractmethod
    def set_mcmc_rng(self, rng):
        return None
