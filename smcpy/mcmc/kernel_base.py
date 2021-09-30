import abc

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class MCMCKernel(ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def mutate_particles(self, particles, num_samples, proposal_cov, phi):
        return new_param_values, new_log_likelihoods

    @abc.abstractmethod
    def get_log_likelihoods(self, param_dict):
        return log_likelihoods

    @abc.abstractmethod
    def get_log_priors(self, param_dict):
        return log_priors
