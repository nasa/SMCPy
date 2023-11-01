import abc

from ..paths import GeometricPath

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class MCMCKernel(ABC):

    def __init__(self, mcmc_object, param_order, path):
        self._mcmc = mcmc_object
        self._param_order = param_order
        self.path = GeometricPath() if path is None else path

    def has_proposal(self):
        return self.path.proposal

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
