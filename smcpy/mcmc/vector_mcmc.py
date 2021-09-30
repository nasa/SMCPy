import numpy as np

from .mcmc_base import MCMCBase
from ..log_likelihoods import Normal


class VectorMCMC(MCMCBase):
    
    def __init__(self, model, data, log_like_args=None,
                 log_like_func=Normal, debug=False):
        '''
        :param model: maps inputs to outputs
        :type model: callable
        :param data: data corresponding to model outputs
        :type data: 1D array
        :param priors: random variable objects with a pdf and rvs method (e.g.
            scipy stats random variable objects)
        :type priors: list of objects
        :param log_like_args: any fixed parameters that define the likelihood
            function (e.g., standard deviation for a Gaussian likelihood).
        :type log_like_args: 1D array or None
        :param log_like_func: log likelihood function that takes inputs, model,
            data, and hyperparameters and returns log likelihoods
        :type log_like_func: callable
        '''
        super().__init__(model, data, log_like_args, log_like_func)

    def evaluate_model(self, inputs):
        return self._eval_model(inputs)
