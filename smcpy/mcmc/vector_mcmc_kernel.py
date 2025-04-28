from copy import copy
import numpy as np

from .kernel_base import KernelBase


class VectorMCMCKernel(KernelBase):
    def __init__(self, vector_mcmc_object, param_order, path=None, rng=None):
        super().__init__(vector_mcmc_object, param_order, path, rng)
        self._mcmc.evaluate_log_posterior = self.path.logpdf
        self.param_order = tuple(str(param) for param in param_order)

    def mutate_particles(self, param_dict, num_samples, cov):
        param_array = self._conv_param_dict_to_array(param_dict)
        param_array, log_likes = self._mcmc.smc_metropolis(
            param_array, num_samples, cov
        )
        param_dict = self._conv_param_array_to_dict(param_array)
        return param_dict, log_likes

    def sample_from_prior(self, num_samples):
        param_array = self._mcmc.sample_from_priors(num_samples)
        return self._conv_param_array_to_dict(param_array)

    def sample_from_proposal(self, num_samples):
        param_array = self.path.proposal.rvs(num_samples, random_state=self.rng)
        return self._conv_param_array_to_dict(param_array)

    def get_log_likelihoods(self, param_dict):
        param_array = self._conv_param_dict_to_array(param_dict)
        return self._mcmc.evaluate_log_likelihood(param_array)

    def get_log_priors(self, param_dict):
        param_array = self._conv_param_dict_to_array(param_dict)
        log_priors = self._mcmc.evaluate_log_priors(param_array)
        return np.sum(log_priors, axis=1).reshape(-1, 1)

    def set_mcmc_rng(self, rng):
        self._mcmc.rng = rng
